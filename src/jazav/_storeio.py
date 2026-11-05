"""Store-relative byte access for the reference codec.

This is the only module that relies on zarr-python implementation details
rather than public codec interfaces:

1. The `ByteGetter` handed to `_decode_partial_single` is in practice a
   `zarr.storage.StorePath` exposing `.store` and `.path`. The public
   protocol only promises `get()` for the current chunk key. zarr's own
   sharding codec relies on the same attributes.
2. ngio wraps stores in a `WrapperStore` subclass; the inner store is
   `._store`. Parent-store resolution needs the innermost store.
3. When a relative source path climbs above the store root, a parent store is
   derived for the two built-in store types that have a location:
   `LocalStore` (`.root`) and `FsspecStore` (`.fs` + `.path`). Other stores
   (`MemoryStore`, `ZipStore`) must be opened at a root that contains the
   sources.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any

from zarr.abc.store import RangeByteRequest, Store
from zarr.storage import FsspecStore, LocalStore, WrapperStore

from jazav._references import ReferenceFormatError, resolve_key

__all__ = [
    "parent_store",
    "read_range",
    "resolve",
    "resolve_key",
    "source_size",
    "store_path_of",
    "unwrap_store",
]

if TYPE_CHECKING:
    from zarr.core.buffer import Buffer, BufferPrototype


def unwrap_store(store: Store) -> Store:
    """Return the innermost store behind any `WrapperStore` layers."""
    while isinstance(store, WrapperStore):
        store = store._store
    return store


def store_path_of(byte_getter: Any) -> tuple[Store, str]:
    """Return `(innermost store, chunk key)` for a chunk's `ByteGetter`."""
    store = getattr(byte_getter, "store", None)
    path = getattr(byte_getter, "path", None)
    if not isinstance(store, Store) or not isinstance(path, str):
        raise ReferenceFormatError(
            "the reference codec needs a store-backed ByteGetter (a zarr "
            f"StorePath) to resolve references, got {type(byte_getter).__name__}. "
            "Nesting the codec inside sharding or decoding through a custom "
            "ByteGetter is not supported."
        )
    return unwrap_store(store), path


def parent_store(store: Store, levels: int) -> Store:
    """Return a read-only store rooted `levels` directories above `store`."""
    if levels == 0:
        return store
    if isinstance(store, LocalStore):
        root = store.root.resolve()
        for _ in range(levels):
            root = root.parent
        return LocalStore(root, read_only=True)
    if isinstance(store, FsspecStore):
        path = store.path.rstrip("/")
        for _ in range(levels):
            parent = posixpath.dirname(path)
            if parent == "" or parent == path:
                raise ReferenceFormatError(
                    f"a source path climbs {levels} level(s) above the store "
                    f"root {store.path!r}, which leaves the bucket. The virtual "
                    "array and its source files must live in the same bucket."
                )
            path = parent
        return FsspecStore(store.fs, path=path, read_only=True)
    raise ReferenceFormatError(
        f"a source path climbs {levels} level(s) above the root of a "
        f"{type(store).__name__}, which has no location. Open the dataset "
        "through a LocalStore or FsspecStore rooted at a directory that "
        "contains both the zarr and the source files."
    )


def resolve(store: Store, base_prefix: str, relative: str) -> tuple[Store, str]:
    """Resolve `relative` against `base_prefix` within `store`."""
    up, key = resolve_key(base_prefix, relative)
    return parent_store(store, up), key


def _missing_source(store: Store, key: str) -> ReferenceFormatError:
    return ReferenceFormatError(
        f"referenced source {key!r} was not found in {store}. The source "
        "file was moved or deleted; virtual arrays and their source files "
        "must be moved together."
    )


async def source_size(store: Store, key: str) -> int | None:
    """Byte size of `key` in `store`, or `None` if the store cannot tell cheaply.

    Only `LocalStore` and `FsspecStore` answer with a stat; the base
    `Store.getsize` reads the whole key, which must never happen for a source
    file.

    Raises:
        ReferenceFormatError: if the source does not exist.
    """
    if not isinstance(store, LocalStore | FsspecStore):
        return None
    try:
        return await store.getsize(key)
    except FileNotFoundError as e:
        raise _missing_source(store, key) from e


async def read_range(
    store: Store, key: str, prototype: BufferPrototype, offset: int, length: int
) -> Buffer:
    """Read exactly `[offset, offset + length)` of `key`; strict on size."""
    buf = await store.get(
        key, prototype=prototype, byte_range=RangeByteRequest(offset, offset + length)
    )
    if buf is None:
        raise _missing_source(store, key)
    if len(buf) != length:
        raise ReferenceFormatError(
            f"short read from {key!r}: expected {length} bytes at offset "
            f"{offset}, got {len(buf)}. The source file is truncated or was "
            "rewritten since the virtual array was created."
        )
    return buf
