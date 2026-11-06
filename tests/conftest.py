"""Shared fixtures: synthetic raw planes and a helper that writes a virtual array."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import LocalStore

import jazav
from jazav import _references as refs

BYTES_LE: dict[str, Any] = {"name": "bytes", "configuration": {"endian": "little"}}


@pytest.fixture
def data() -> np.ndarray:
    return np.arange(2 * 16 * 24, dtype="<u2").reshape(2, 16, 24)


def raw_files(directory: Path, data: np.ndarray) -> None:
    """Write one raw little-endian file per plane of `data`."""
    directory.mkdir(parents=True, exist_ok=True)
    for c in range(data.shape[0]):
        (directory / f"plane{c}.bin").write_bytes(data[c].tobytes())


def _default_size(
    store: Any,
    path: str,
    index: int,
    source: str,
    references: dict[tuple[int, ...], tuple[int, int, int]],
) -> int:
    if isinstance(store, LocalStore):
        candidate = (Path(store.root) / path / source).resolve()
        if candidate.is_file():
            return candidate.stat().st_size
    ends = [off + ln for src, off, ln in references.values() if src == index]
    return max(ends, default=1)


def write_virtual_array(
    store: Any,
    *,
    path: str = "0",
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    sources: list[str],
    references: dict[tuple[int, ...], tuple[int, int, int]],
    codecs: list[dict[str, Any]] | None = None,
    sizes: list[int] | None = None,
    fill_value: Any = 0,
    shards: tuple[int, ...] | None = None,
) -> zarr.Array:
    """Create a group holding one virtual array and write its reference tables.

    `sizes` default to the real file size when the source exists under the
    store, else to the end of the referenced byte range. Without `shards` the
    whole array is one shard.
    """
    group = zarr.open_group(store, mode="w", zarr_format=3)
    if sizes is None:
        sizes = [
            _default_size(store, path, i, s, references) for i, s in enumerate(sources)
        ]
    array = jazav.create_reference_array(
        group,
        path,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
        dimension_names=None,
        sources=sources,
        sizes=sizes,
        codecs=codecs or [BYTES_LE],
        shards=shards,
    )
    jazav.write_references(array, references)
    return array


def write_raw_shard(array: zarr.Array, shard: tuple[int, ...], payload: bytes) -> None:
    """Write `payload` as the stored key of shard `shard`, bypassing validation."""
    key = array.metadata.encode_chunk_key(shard)
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync((array.store_path / key).set(buffer))


def write_table(
    array: zarr.Array, references: dict[tuple[int, ...], tuple[int, int, int]]
) -> None:
    """Write whole tables from inner-chunk references, bypassing validation.

    Unlike `jazav.write_references`, source indices and byte ranges are not
    checked, so tests can plant corrupt entries.
    """
    codec = jazav.reference_codec(array)
    assert codec is not None
    per_shard = refs.chunks_per_shard(array.chunks, codec.chunk_shape)
    tables: dict[tuple[int, ...], dict[int, tuple[int, int, int]]] = {}
    for index, reference in references.items():
        shard, flat = refs.locate_chunk(index, per_shard)
        tables.setdefault(shard, {})[flat] = reference
    for shard, entries in tables.items():
        write_raw_shard(array, shard, refs.encode_table(entries, math.prod(per_shard)))
