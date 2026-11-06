"""Reference format version 2 (pure Python, no zarr import).

A virtual array is a zarr v3 array whose stored chunks are **shards**: each
stored key holds a *reference table*, one 20-byte entry per inner chunk of the
shard, in C order over the shard's inner chunk grid. An entry is little-endian:

| offset | size | field           | meaning                                       |
|--------|------|-----------------|-----------------------------------------------|
| 0      | 4    | `uint32 source` | index into the codec's `sources` table        |
| 4      | 8    | `uint64 offset` | byte offset of the stored block in the source |
| 12     | 8    | `uint64 length` | byte length of the stored block               |

An entry of twenty `0xFF` bytes marks an inner chunk that has no block: it
reads as the fill value. A shard key that does not exist reads as fill for all
of its inner chunks. The table has exactly `20 * n` bytes for a shard of `n`
inner chunks, including shards that extend past the array's edge.

The sources are named once per array, in the codec configuration, as
`/`-separated paths relative to the array's key prefix (the "directory" that
holds the array's `zarr.json`). They may climb above the zarr root with `..`
segments; absolute paths and URLs are rejected so that metadata can never make
a reader fetch an arbitrary location.
"""

from __future__ import annotations

import itertools
import math
import posixpath
import struct
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

REFERENCE_SIZE = 20
"""Byte size of one table entry."""

FORMAT_VERSION = 2
"""Version of the reference format described in this module."""

MISSING_ENTRY = b"\xff" * REFERENCE_SIZE
"""Table entry of an inner chunk without a block; it reads as the fill value."""

DEFAULT_MAX_CHUNKS_PER_SHARD = 16384
"""Default upper bound on the inner chunks of one shard (a 320 KiB table)."""

_REF = struct.Struct("<IQQ")
_MAX_U32 = 2**32 - 1
_MAX_U64 = 2**64 - 1


class ReferenceFormatError(ValueError):
    """A malformed reference table, source path, chunk key or source file."""


class ChunkReference(NamedTuple):
    """One decoded table entry."""

    source: int
    offset: int
    length: int


def encode_reference(source: int, offset: int, length: int) -> bytes:
    """Pack a reference into its 20-byte table entry.

    Raises:
        ReferenceFormatError: if a field does not fit its integer type or the
            byte range overflows.
    """
    if not 0 <= source < _MAX_U32:
        raise ReferenceFormatError(
            f"source index {source} does not fit in uint32 (0..{_MAX_U32 - 1})."
        )
    if not (0 <= offset <= _MAX_U64 and 0 < length <= _MAX_U64):
        raise ReferenceFormatError(
            f"byte range (offset={offset}, length={length}) must have a "
            "non-negative offset and a positive length that fit in uint64."
        )
    if offset + length > _MAX_U64:
        raise ReferenceFormatError(
            f"byte range (offset={offset}, length={length}) overflows uint64."
        )
    return _REF.pack(source, offset, length)


def decode_reference(payload: bytes | memoryview) -> ChunkReference | None:
    """Unpack a 20-byte table entry; `None` for the missing-chunk entry.

    Raises:
        ReferenceFormatError: if the entry has the wrong size, uses the reserved
            source index without being the missing-chunk entry, or its byte
            range overflows.
    """
    if len(payload) != REFERENCE_SIZE:
        raise ReferenceFormatError(
            f"a reference entry must be exactly {REFERENCE_SIZE} bytes, got "
            f"{len(payload)}."
        )
    if bytes(payload) == MISSING_ENTRY:
        return None
    source, offset, length = _REF.unpack(payload)
    if source == _MAX_U32:
        raise ReferenceFormatError(
            f"source index {_MAX_U32} is reserved for the missing-chunk entry, "
            "but the entry is not all 0xFF bytes."
        )
    if length == 0:
        raise ReferenceFormatError("a reference entry must have a positive length.")
    if offset + length > _MAX_U64:
        raise ReferenceFormatError(
            f"byte range (offset={offset}, length={length}) overflows uint64."
        )
    return ChunkReference(source, offset, length)


def table_size(n_chunks: int) -> int:
    """Byte size of the reference table of a shard with `n_chunks` inner chunks."""
    return REFERENCE_SIZE * n_chunks


def encode_table(entries: Mapping[int, tuple[int, int, int]], n_chunks: int) -> bytes:
    """Build a shard's table from `{flat inner index: (source, offset, length)}`.

    Inner chunks absent from `entries` get the missing-chunk entry.

    Raises:
        ReferenceFormatError: if an index is outside `0..n_chunks-1` or an entry
            does not encode.
    """
    table = bytearray(MISSING_ENTRY * n_chunks)
    for flat, (source, offset, length) in entries.items():
        if not 0 <= flat < n_chunks:
            raise ReferenceFormatError(
                f"inner chunk index {flat} is outside a shard of {n_chunks} chunks."
            )
        start = flat * REFERENCE_SIZE
        table[start : start + REFERENCE_SIZE] = encode_reference(source, offset, length)
    return bytes(table)


def decode_table(
    payload: bytes | memoryview, n_chunks: int
) -> list[ChunkReference | None]:
    """Unpack a shard's table into one entry per inner chunk, in C order.

    Raises:
        ReferenceFormatError: if the table does not have exactly
            `table_size(n_chunks)` bytes, or an entry is malformed.
    """
    expected = table_size(n_chunks)
    if len(payload) != expected:
        raise ReferenceFormatError(
            f"a reference table for {n_chunks} inner chunk(s) must be exactly "
            f"{expected} bytes, got {len(payload)}. The stored key is not a "
            f"format-{FORMAT_VERSION} table; the virtual array was written with "
            "a different tool or has been corrupted."
        )
    view = memoryview(payload)
    return [
        decode_reference(view[i * REFERENCE_SIZE : (i + 1) * REFERENCE_SIZE])
        for i in range(n_chunks)
    ]


# -- shard geometry ---------------------------------------------------------


def chunks_per_shard(
    shard_shape: Iterable[int], chunk_shape: Iterable[int]
) -> tuple[int, ...]:
    """Inner chunks along each axis of a shard.

    Raises:
        ReferenceFormatError: if the shard shape is not a positive multiple of
            the chunk shape along every axis.
    """
    shard = tuple(shard_shape)
    chunk = tuple(chunk_shape)
    if len(shard) != len(chunk):
        raise ReferenceFormatError(
            f"shard shape {shard} and chunk shape {chunk} differ in rank."
        )
    for axis, (s, c) in enumerate(zip(shard, chunk, strict=True)):
        if c <= 0 or s <= 0 or s % c != 0:
            raise ReferenceFormatError(
                f"shard edge {s} on axis {axis} is not a positive multiple of the "
                f"inner chunk edge {c}."
            )
    return tuple(s // c for s, c in zip(shard, chunk, strict=True))


def chunk_grid(shape: Iterable[int], chunk_shape: Iterable[int]) -> tuple[int, ...]:
    """Number of inner chunks along each axis of an array."""
    return tuple(
        math.ceil(s / c) for s, c in zip(tuple(shape), tuple(chunk_shape), strict=True)
    )


def locate_chunk(
    chunk_index: Iterable[int], per_shard: tuple[int, ...]
) -> tuple[tuple[int, ...], int]:
    """Map an inner chunk grid index to `(shard index, flat index in the table)`."""
    index = tuple(chunk_index)
    shard = tuple(i // n for i, n in zip(index, per_shard, strict=True))
    inner = tuple(i % n for i, n in zip(index, per_shard, strict=True))
    return shard, flat_index(inner, per_shard)


def flat_index(inner: Iterable[int], per_shard: tuple[int, ...]) -> int:
    """C-order position of an inner chunk coordinate within its shard's table."""
    flat = 0
    for i, n in zip(tuple(inner), per_shard, strict=True):
        flat = flat * n + i
    return flat


def inner_coords(per_shard: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    """Inner chunk coordinates of a shard in C order (table order)."""
    return itertools.product(*(range(n) for n in per_shard))


def default_shard_shape(
    shape: Iterable[int],
    chunk_shape: Iterable[int],
    max_chunks: int = DEFAULT_MAX_CHUNKS_PER_SHARD,
) -> tuple[int, ...]:
    """Pick a shard shape: the whole array when its table stays small.

    The whole chunk grid is one shard while it has at most `max_chunks` inner
    chunks. Above that, the grid is split along the slowest axes first, leaving
    the last two axes (typically y and x) whole, so a plane read touches one
    table. The two fastest axes are split only when they alone exceed the bound.
    The result is always a multiple of the chunk shape and may exceed the array
    shape on the last shard.
    """
    chunk = tuple(chunk_shape)
    grid = list(chunk_grid(shape, chunk))
    if max_chunks < 1:
        raise ReferenceFormatError(f"max_chunks must be positive, got {max_chunks}.")
    for axis in range(len(grid)):  # slowest axis first, y and x last
        if math.prod(grid) <= max_chunks:
            break
        others = math.prod(grid) // grid[axis]
        grid[axis] = max(1, max_chunks // others)
    return tuple(g * c for g, c in zip(grid, chunk, strict=True))


# -- paths ------------------------------------------------------------------


def check_relative_path(path: str) -> None:
    """Reject anything but a relative, `/`-separated path without `.` segments.

    Accepted: `raw/tile.tif`, `../../raw/tile.tif`. Rejected: absolute paths
    (`/a`), URLs and Windows drives (`s3://b/a`, `C:/a`), backslashes, empty
    segments (`a//b`) and `.` segments.

    Raises:
        ReferenceFormatError: if `path` violates one of these rules.
    """
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or "://" in path
        or ":" in path.split("/")[0]
    ):
        raise ReferenceFormatError(
            f"source path {path!r} must be a relative '/'-separated path "
            "(no absolute paths, URLs, drive letters or backslashes)."
        )
    if any(segment in ("", ".") for segment in path.split("/")):
        raise ReferenceFormatError(
            f"source path {path!r} has an empty or '.' segment; normalize it "
            "first (e.g. 'raw/a.tif', not './raw//a.tif')."
        )


def normalize_sources(sources: Iterable[str]) -> tuple[str, ...]:
    """Validate, deduplicate and sort source paths for a new reference array.

    The codec itself never reorders `sources` (table entries index into it), so
    the canonical order is fixed here, before any table is written.
    """
    unique = sorted(set(sources))
    for source in unique:
        check_relative_path(source)
    return tuple(unique)


def resolve_key(base_prefix: str, relative: str) -> tuple[int, str]:
    """Join `relative` onto the key prefix `base_prefix` (a store "directory").

    Returns `(levels_above_root, key)`: how many levels the result escapes
    above the store root (0 means it stays inside) and the key relative to that
    possibly higher root.
    """
    check_relative_path(relative)
    joined = posixpath.normpath(posixpath.join("/" + base_prefix.strip("/"), relative))
    # normpath clamps at "/", so count the escapes ourselves.
    depth = len([p for p in base_prefix.strip("/").split("/") if p])
    up = 0
    for segment in relative.split("/"):
        if segment == "..":
            if depth == 0:
                up += 1
            else:
                depth -= 1
        else:
            depth += 1
    return up, joined.lstrip("/")


def array_prefix_from_chunk_key(chunk_key: str, ndim: int) -> str:
    """Strip the default v3 chunk-key suffix `c/i/j/...` from a shard key.

    Raises:
        ReferenceFormatError: if the key does not end with `c` followed by
            `ndim` indices, i.e. the array does not use the `default` chunk
            key encoding with the `/` separator.
    """
    parts = chunk_key.split("/")
    if len(parts) < ndim + 1 or parts[-(ndim + 1)] != "c":
        raise ReferenceFormatError(
            f"cannot derive the array path from chunk key {chunk_key!r}: the "
            "reference codec requires the 'default' chunk key encoding with the "
            "'/' separator."
        )
    return "/".join(parts[: -(ndim + 1)])
