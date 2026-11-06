"""Helpers that create a reference array and write its reference tables."""

from __future__ import annotations

import asyncio
import math
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync

from jazav._codec import CODEC_NAME, ReferenceCodec
from jazav._references import (
    DEFAULT_MAX_CHUNKS_PER_SHARD,
    ReferenceFormatError,
    check_relative_path,
    chunk_grid,
    chunks_per_shard,
    default_shard_shape,
    encode_table,
    locate_chunk,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Mapping

    from zarr.abc.codec import Codec

_WRITE_BATCH = 512
"""Shard tables written per `asyncio.gather`, to cap open file handles."""


def relative_source_path(
    array_dir: str | os.PathLike[str], source_path: str | os.PathLike[str]
) -> str:
    """Return `source_path` relative to `array_dir` as a `/`-separated path.

    Both paths are expanded and resolved the way `LocalStore` resolves its root,
    so the result matches what the codec joins onto the array's key prefix.

    Raises:
        ReferenceFormatError: if the paths are on different drives or the
            result violates the source path rules.
    """
    array = Path(array_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    try:
        relative = posixpath.relpath(source.as_posix(), array.as_posix())
    except ValueError as e:  # different drives on Windows
        raise ReferenceFormatError(
            f"source {source} cannot be expressed relative to {array}: {e}. The "
            "virtual array and its sources must share a filesystem root."
        ) from e
    check_relative_path(relative)
    return relative


def reference_codec(array: zarr.Array) -> ReferenceCodec | None:
    """The array's reference codec, or `None` if the array is not virtual."""
    metadata = array.metadata
    if not isinstance(metadata, ArrayV3Metadata):
        return None
    if len(metadata.codecs) == 1 and isinstance(metadata.codecs[0], ReferenceCodec):
        return metadata.codecs[0]
    return None


def create_reference_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Iterable[int],
    chunks: Iterable[int],
    dtype: Any,
    fill_value: Any,
    dimension_names: Iterable[str] | None,
    sources: Iterable[str],
    sizes: Iterable[int],
    codecs: Iterable[Codec | dict[str, Any]],
    shards: Iterable[int] | None = None,
    max_chunks_per_shard: int = DEFAULT_MAX_CHUNKS_PER_SHARD,
) -> zarr.Array:
    """Create (or replace) the array `name` of `group` as a reference array.

    `chunks` is the inner chunk shape: the shape of one stored block. The
    array's zarr chunks are shards, `shards` or `default_shard_shape(...)`
    (the whole array unless it has more than `max_chunks_per_shard` chunks),
    each holding one reference table. The reference codec is the array's only
    codec and the chunk key encoding is the default one with the `/`
    separator, as the format requires. The caller passes `sources` in their
    final order (see `normalize_sources`), since table entries index into it.

    Raises:
        ReferenceFormatError: if `shards` is not a multiple of `chunks`, or the
            installed zarr does not route a lone array->bytes codec through
            partial decoding, which would make the array unreadable.
    """
    shape = tuple(shape)
    chunks = tuple(chunks)
    shard_shape = (
        tuple(shards)
        if shards is not None
        else default_shard_shape(shape, chunks, max_chunks_per_shard)
    )
    chunks_per_shard(shard_shape, chunks)
    array = group.create_array(
        name=name,
        shape=shape,
        chunks=shard_shape,
        dtype=dtype,
        fill_value=fill_value,
        dimension_names=None if dimension_names is None else tuple(dimension_names),
        chunk_key_encoding={"name": "default", "separator": "/"},
        serializer=ReferenceCodec(
            chunk_shape=chunks, sources=sources, sizes=sizes, codecs=codecs
        ),
        filters=None,
        compressors=None,
        overwrite=True,
    )
    if not array._async_array.codec_pipeline.supports_partial_decode:
        raise ReferenceFormatError(
            f"zarr {zarr.__version__} does not route {CODEC_NAME} through partial "
            "decoding, so the reference array would be unreadable. Pin a zarr "
            "release whose codec pipeline supports partial decoding for a lone "
            "array->bytes codec."
        )
    return array


def write_references(
    array: zarr.Array, refs: Mapping[tuple[int, ...], tuple[int, int, int]]
) -> None:
    """Write the reference tables of `array` from a mapping of inner chunks.

    Every shard that has at least one entry in `refs` is written as a whole
    table; inner chunks without an entry read as the fill value, and shards
    without any entry are not written at all. Call it once with all the
    references of the array.

    Args:
        array: An array created by `create_reference_array`.
        refs: Maps an inner chunk grid index to `(source, offset, length)`.

    Raises:
        ReferenceFormatError: if `array` is not a reference array, an index is
            outside the chunk grid, a source index is unknown or a byte range
            exceeds the declared source size. Nothing is written in that case.
    """
    codec = reference_codec(array)
    if codec is None:
        first = array.metadata.codecs[0] if getattr(array.metadata, "codecs", None) else None
        raise ReferenceFormatError(
            f"array {array.path!r} is not a reference array (its first codec is "
            f"{type(first).__name__}); create it with create_reference_array."
        )
    grid = chunk_grid(array.shape, codec.chunk_shape)
    per_shard = chunks_per_shard(array.chunks, codec.chunk_shape)
    n_inner = math.prod(per_shard)

    tables: dict[tuple[int, ...], dict[int, tuple[int, int, int]]] = {}
    for index, (source, offset, length) in refs.items():
        if len(index) != len(grid) or any(
            not 0 <= i < n for i, n in zip(index, grid, strict=True)
        ):
            raise ReferenceFormatError(
                f"chunk index {index} is outside the chunk grid {grid} of array "
                f"{array.path!r}."
            )
        if not 0 <= source < len(codec.sources):
            raise ReferenceFormatError(
                f"chunk {index} references source #{source}, but array "
                f"{array.path!r} declares {len(codec.sources)} source(s)."
            )
        if offset + length > codec.sizes[source]:
            raise ReferenceFormatError(
                f"chunk {index} references bytes [{offset}, {offset + length}) of "
                f"{codec.sources[source]!r}, which is only {codec.sizes[source]} "
                "bytes."
            )
        shard, flat = locate_chunk(index, per_shard)
        tables.setdefault(shard, {})[flat] = (source, offset, length)

    payloads = [
        (array.metadata.encode_chunk_key(shard), encode_table(entries, n_inner))
        for shard, entries in tables.items()
    ]
    prototype = default_buffer_prototype()

    async def _write() -> None:
        for start in range(0, len(payloads), _WRITE_BATCH):
            await asyncio.gather(
                *(
                    (array.store_path / key).set(prototype.buffer.from_bytes(payload))
                    for key, payload in payloads[start : start + _WRITE_BATCH]
                )
            )

    sync(_write())
