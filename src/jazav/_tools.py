"""`verify` and `materialize` for virtual arrays."""

from __future__ import annotations

import itertools
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage import LocalStore

from jazav import _storeio
from jazav._codec import ReferenceCodec
from jazav._references import ReferenceFormatError, decode_reference

if TYPE_CHECKING:
    import os

    from zarr.abc.store import Store


@dataclass(frozen=True)
class Problem:
    """One thing `verify` found wrong with a reference array."""

    array: str
    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.array}: [{self.kind}] {self.message}"


@dataclass
class VerifyReport:
    """Outcome of `verify`."""

    arrays: list[str] = field(default_factory=list)
    n_sources: int = 0
    n_chunks: int = 0
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no problem was found."""
        return not self.problems

    def __str__(self) -> str:
        lines = [str(problem) for problem in self.problems]
        status = "OK" if self.ok else f"{len(self.problems)} problem(s)"
        lines.append(
            f"{status}: {len(self.arrays)} reference array(s), "
            f"{self.n_sources} source(s), {self.n_chunks} chunk(s)."
        )
        return "\n".join(lines)


def _reference_codec(array: zarr.Array) -> ReferenceCodec | None:
    metadata = array.metadata
    if not isinstance(metadata, ArrayV3Metadata):
        return None
    if len(metadata.codecs) == 1 and isinstance(metadata.codecs[0], ReferenceCodec):
        return metadata.codecs[0]
    return None


def find_reference_arrays(node: zarr.Group | zarr.Array) -> list[zarr.Array]:
    """Return every array below `node` whose only codec is the reference codec."""
    if isinstance(node, zarr.Array):
        return [node] if _reference_codec(node) is not None else []
    return [
        member
        for _, member in node.members(max_depth=None)
        if isinstance(member, zarr.Array) and _reference_codec(member) is not None
    ]


def _open(
    target: str | os.PathLike[str] | zarr.Group | zarr.Array,
    mode: Literal["r", "r+"],
) -> Any:
    if isinstance(target, zarr.Group | zarr.Array):
        return target
    return zarr.open(target, mode=mode)


async def _list_chunk_keys(store: Store, prefix: str) -> list[str]:
    chunk_prefix = f"{prefix}/c/" if prefix else "c/"
    return [key async for key in store.list_prefix(chunk_prefix)]


def _verify_array(array: zarr.Array, report: VerifyReport, *, deep: bool) -> None:
    codec = _reference_codec(array)
    assert codec is not None
    name = array.path or "/"
    report.arrays.append(name)
    report.n_sources += len(codec.sources)
    store = _storeio.unwrap_store(array.store_path.store)
    prefix = array.store_path.path

    def problem(kind: str, message: str) -> None:
        report.problems.append(Problem(name, kind, message))

    for index, source in enumerate(codec.sources):
        try:
            source_store, key = _storeio.resolve(store, prefix, source)
            actual = sync(_storeio.source_size(source_store, key))
        except ReferenceFormatError as e:
            problem("missing-source", str(e))
            continue
        if actual is not None and actual != codec.sizes[index]:
            problem(
                "size-mismatch",
                f"source {source!r} is {actual} bytes, expected "
                f"{codec.sizes[index]}; the file was truncated or replaced.",
            )

    prototype = default_buffer_prototype()
    keys = sync(_list_chunk_keys(store, prefix))
    for key in keys:
        payload = sync(store.get(key, prototype=prototype))
        if payload is None:  # pragma: no cover - listed keys exist
            continue
        try:
            reference = decode_reference(payload.to_bytes())
        except ReferenceFormatError as e:
            problem("bad-chunk", f"{key}: {e}")
            continue
        if reference.source >= len(codec.sources):
            problem(
                "out-of-range",
                f"{key} references source #{reference.source}, but only "
                f"{len(codec.sources)} source(s) are declared.",
            )
        elif reference.offset + reference.length > codec.sizes[reference.source]:
            problem(
                "out-of-range",
                f"{key} references bytes [{reference.offset}, "
                f"{reference.offset + reference.length}) of "
                f"{codec.sources[reference.source]!r}, which is only "
                f"{codec.sizes[reference.source]} bytes.",
            )
    report.n_chunks += len(keys)

    if deep:
        for region in _chunk_regions(array.shape, array.chunks):
            try:
                array[region]
            except Exception as e:
                problem("decode-error", f"chunk at {region}: {e}")


def verify(
    target: str | os.PathLike[str] | zarr.Group | zarr.Array, *, deep: bool = False
) -> VerifyReport:
    """Check that the reference arrays under `target` can be read.

    Checks that every source exists with the declared size and that every chunk
    payload is a well-formed reference into a declared source. With
    `deep=True`, every chunk is also decoded.

    Data problems are collected in the report, never raised.

    Raises:
        FileNotFoundError, zarr errors: if `target` is not an existing zarr
            group or array.
    """
    node = _open(target, "r")
    report = VerifyReport()
    for array in find_reference_arrays(node):
        _verify_array(array, report, deep=deep)
    return report


def _chunk_regions(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> list[tuple[slice, ...]]:
    counts = [-(-s // c) for s, c in zip(shape, chunks, strict=True)]
    return [
        tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, s, c in zip(index, shape, chunks, strict=True)
        )
        for index in itertools.product(*(range(n) for n in counts))
    ]


def materialize(
    target: str | os.PathLike[str] | zarr.Group,
    *,
    chunks: tuple[int, ...] | None = None,
    compressors: Any = "auto",
) -> list[str]:
    """Rewrite every reference array under `target` as a regular array in place.

    Each reference array is copied chunk by chunk into a sibling array
    `<name>-materializing`, then swapped in with two directory renames. If the
    process dies between the renames, a `<name>-virtual-old` directory is left
    next to the array: rename it back to recover the references.

    Args:
        target: Path of a zarr group on a local filesystem (an OME-Zarr root),
            or an open group.
        chunks: Chunk shape of the materialized arrays; the reference array's
            chunks by default.
        compressors: Passed to `zarr.Group.create_array`.

    Returns:
        On-disk paths of the rewritten arrays, empty when nothing was virtual.

    Raises:
        ReferenceFormatError: if `target` is not a local group, or `verify`
            reports a problem; nothing is rewritten in that case.
    """
    group = _open(target, "r+")
    if not isinstance(group, zarr.Group):
        raise ReferenceFormatError(
            f"materialize needs a zarr group, got {type(group).__name__} at "
            f"{target!s}. Pass the group that contains the array."
        )
    store = _storeio.unwrap_store(group.store_path.store)
    if not isinstance(store, LocalStore):
        raise ReferenceFormatError(
            f"materialize rewrites directories in place and needs a LocalStore, "
            f"got {type(store).__name__}. Open the zarr through a local path."
        )
    report = verify(group)
    if not report.ok:
        raise ReferenceFormatError(
            f"refusing to materialize {target!s}: the references cannot all be "
            f"read.\n{report}"
        )

    base = store.root.resolve() / group.store_path.path
    rewritten: list[str] = []
    for array in find_reference_arrays(group):
        name = array.path[len(group.path) :].lstrip("/") if group.path else array.path
        temp_name = f"{name}-materializing"
        metadata = array.metadata
        assert isinstance(metadata, ArrayV3Metadata)  # guaranteed by _reference_codec
        temp = group.create_array(
            name=temp_name,
            shape=array.shape,
            chunks=chunks or array.chunks,
            dtype=array.dtype,
            fill_value=array.fill_value,
            dimension_names=metadata.dimension_names,
            attributes=dict(array.attrs),
            compressors=compressors,
            overwrite=True,
        )
        for region in _chunk_regions(temp.shape, temp.chunks):
            temp[region] = array[region]

        current = base / name
        backup = base / f"{name}-virtual-old"
        current.rename(backup)
        (base / temp_name).rename(current)
        shutil.rmtree(backup)

        reopened = zarr.open_array(current, mode="r")
        if _reference_codec(reopened) is not None:  # pragma: no cover - canary
            raise ReferenceFormatError(
                f"{current} still carries {ReferenceCodec.__name__} after "
                "materializing; this is a bug in jazav."
            )
        rewritten.append(str(current))
    return rewritten
