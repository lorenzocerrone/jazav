"""Helpers that create a reference array and write its chunk references."""

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
    ReferenceFormatError,
    check_relative_path,
    encode_reference,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Mapping

    from zarr.abc.codec import Codec

_WRITE_BATCH = 512
"""Chunk references written per `asyncio.gather`, to cap open file handles."""


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
) -> zarr.Array:
    """Create (or replace) the array `name` of `group` as a reference array.

    The reference codec is the array's only codec and the chunk key encoding is
    the default one with the `/` separator, as the format requires. The caller
    passes `sources` in their final order (see `normalize_sources`), since chunk
    payloads index into it.

    Raises:
        ReferenceFormatError: if the installed zarr does not route a lone
            array->bytes codec through partial decoding, which would make the
            array unreadable.
    """
    array = group.create_array(
        name=name,
        shape=tuple(shape),
        chunks=tuple(chunks),
        dtype=dtype,
        fill_value=fill_value,
        dimension_names=None if dimension_names is None else tuple(dimension_names),
        chunk_key_encoding={"name": "default", "separator": "/"},
        serializer=ReferenceCodec(sources=sources, sizes=sizes, codecs=codecs),
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
    """Write one 20-byte chunk per entry of `refs` into `array`.

    Args:
        array: An array created by `create_reference_array`.
        refs: Maps a chunk grid index to `(source, offset, length)`.

    Raises:
        ReferenceFormatError: if `array` is not a reference array, an index is
            outside the chunk grid, a source index is unknown or a byte range
            exceeds the declared source size. Nothing is written in that case.
    """
    metadata = array.metadata
    codec = (
        metadata.codecs[0]
        if isinstance(metadata, ArrayV3Metadata) and metadata.codecs
        else None
    )
    if not isinstance(codec, ReferenceCodec):
        raise ReferenceFormatError(
            f"array {array.path!r} is not a reference array (its first codec is "
            f"{type(codec).__name__}); create it with create_reference_array."
        )
    grid = tuple(
        math.ceil(s / c) for s, c in zip(array.shape, array.chunks, strict=True)
    )
    payloads: list[tuple[str, bytes]] = []
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
        key = array.metadata.encode_chunk_key(index)
        payloads.append((key, encode_reference(source, offset, length)))

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
