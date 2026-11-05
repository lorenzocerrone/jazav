"""Reference format version 1 (pure Python, no zarr import).

Every chunk of a virtual array is exactly 20 bytes, little-endian:

| offset | size | field           | meaning                                       |
|--------|------|-----------------|-----------------------------------------------|
| 0      | 4    | `uint32 source` | index into the codec's `sources` table        |
| 4      | 8    | `uint64 offset` | byte offset of the stored block in the source |
| 12     | 8    | `uint64 length` | byte length of the stored block               |

The sources are named once per array, in the codec configuration, as
`/`-separated paths relative to the array's key prefix (the "directory" that
holds the array's `zarr.json`). They may climb above the zarr root with `..`
segments; absolute paths and URLs are rejected so that metadata can never make
a reader fetch an arbitrary location.
"""

from __future__ import annotations

import posixpath
import struct
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

REFERENCE_SIZE = 20
"""Byte size of one chunk payload."""

FORMAT_VERSION = 1
"""Version of the reference format described in this module."""

_REF = struct.Struct("<IQQ")
_MAX_U32 = 2**32 - 1
_MAX_U64 = 2**64 - 1


class ReferenceFormatError(ValueError):
    """A malformed chunk reference, source path, chunk key or source file."""


class ChunkReference(NamedTuple):
    """One decoded chunk payload."""

    source: int
    offset: int
    length: int


def encode_reference(source: int, offset: int, length: int) -> bytes:
    """Pack a reference into its 20-byte chunk payload.

    Raises:
        ReferenceFormatError: if a field does not fit its integer type or the
            byte range overflows.
    """
    if not 0 <= source <= _MAX_U32:
        raise ReferenceFormatError(
            f"source index {source} does not fit in uint32 (0..{_MAX_U32})."
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


def decode_reference(payload: bytes | memoryview) -> ChunkReference:
    """Unpack a 20-byte chunk payload.

    Raises:
        ReferenceFormatError: if the payload has the wrong size or the byte
            range overflows.
    """
    if len(payload) != REFERENCE_SIZE:
        raise ReferenceFormatError(
            f"a reference chunk must be exactly {REFERENCE_SIZE} bytes, got "
            f"{len(payload)}. The chunk file is not a format-{FORMAT_VERSION} "
            "reference; the virtual array was written with a different tool or "
            "has been corrupted."
        )
    source, offset, length = _REF.unpack(payload)
    if offset + length > _MAX_U64:
        raise ReferenceFormatError(
            f"byte range (offset={offset}, length={length}) overflows uint64."
        )
    return ChunkReference(source, offset, length)


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

    The codec itself never reorders `sources` (chunk payloads index into it), so
    the canonical order is fixed here, before any chunk is written.
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
    """Strip the default v3 chunk-key suffix `c/i/j/...` from a chunk key.

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
