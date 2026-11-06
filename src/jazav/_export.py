"""Export a virtual OME-Zarr as a kerchunk reference manifest for web viewers.

Browsers cannot load the reference codec, but vizarr (through zarrita.js) and
fsspec open kerchunk manifests: a JSON document that maps every key of a zarr
hierarchy to inline content or to a byte range of a URL. The manifest written
here is the image as served from one base URL:

- every `zarr.json` is inlined, the virtual arrays' ones rewritten with the
  nested codecs and the inner chunk shape, so readers see plain arrays;
- every other file of the hierarchy (pyramid chunks, tables, labels) is
  `[url, 0, size]`, fetched live from the directory;
- every inner chunk of a virtual array is `[source url, offset, length]`, the
  reference table resolved against the array's URL.

Nothing but metadata is copied, so the manifest is one directory walk plus the
tables. It fixes the *set of keys* and the file sizes: a chunk written where
none existed, a new pyramid level or label, or a file rewritten at another size
need a new manifest. Regenerate it whenever the directory is published.

Whole-file entries carry an explicit size because zarrita's reference store
treats a missing length as zero and rejects the request; fsspec would accept
`[url]` alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from jazav import _storeio
from jazav._references import (
    ReferenceFormatError,
    chunks_per_shard,
    decode_table,
    inner_coords,
)
from jazav._tools import _open, find_reference_arrays
from jazav._writer import reference_codec

if TYPE_CHECKING:
    import os

    import zarr
    from zarr.abc.store import Store

MANIFEST_VERSION = 1
"""kerchunk manifest version written: `{"version": 1, "refs": {...}}`."""


def normalize_base_url(base_url: str) -> str:
    """Return `base_url` with a trailing slash, checking it has a scheme.

    Raises:
        ReferenceFormatError: if `base_url` is not an absolute URL.
    """
    parts = urlsplit(base_url)
    if not parts.scheme or (not parts.netloc and parts.scheme != "file"):
        raise ReferenceFormatError(
            f"base URL {base_url!r} must be absolute (e.g. https://host/data/img.zarr/"
            " or file:///data/img.zarr/): web readers resolve nothing relative to "
            "the manifest."
        )
    return base_url if base_url.endswith("/") else base_url + "/"


def default_manifest_path(target: str | os.PathLike[str]) -> Path:
    """`<name>.refs.json` next to `target` (`img.zarr` -> `img.refs.json`)."""
    path = Path(target)
    stem = path.name[: -len(".zarr")] if path.name.endswith(".zarr") else path.name
    return path.with_name(f"{stem}.refs.json")


async def _list_keys(store: Store, prefix: str) -> list[str]:
    keys = [key async for key in store.list_prefix(f"{prefix}/" if prefix else "")]
    return sorted(keys)


def _relative(key: str, prefix: str) -> str:
    return key[len(prefix) :].lstrip("/") if prefix else key


def _join(prefix: str, name: str) -> str:
    """Join a store key prefix and a name; the root prefix is the empty string."""
    return f"{prefix}/{name}" if prefix else name


def _shard_index(key: str, ndim: int) -> tuple[int, ...]:
    """Parse `c/i/j/...` (default encoding, separator `/`) into a shard index.

    Raises:
        ReferenceFormatError: if the key has another form.
    """
    parts = key.split("/")
    if (
        len(parts) != ndim + 1
        or parts[0] != "c"
        or not all(p.isdigit() for p in parts[1:])
    ):
        raise ReferenceFormatError(
            f"shard key {key!r} is not a default-encoded chunk key of rank {ndim}."
        )
    return tuple(int(p) for p in parts[1:])


def _belongs_to_virtual_array(key: str, virtual_paths: list[str]) -> bool:
    return any(
        key == _join(path, "zarr.json") or key.startswith(_join(path, "c/"))
        for path in virtual_paths
    )


def export_references(
    target: str | os.PathLike[str] | zarr.Group | zarr.Array,
    base_url: str,
    *,
    out: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build the kerchunk manifest of the hierarchy at `target` served at `base_url`.

    Args:
        target: Path of a zarr group or array, or an open one.
        base_url: Absolute URL at which `target` itself is (or will be) served.
            Source files are resolved against the arrays' URLs with the same
            relative paths they have on disk, so the raw files must be served
            in the same layout relative to the zarr.
        out: Where to write the manifest as JSON; nothing is written if `None`.

    Returns:
        The manifest, `{"version": 1, "refs": {key: entry}}`.

    Raises:
        ReferenceFormatError: if `base_url` is not absolute or a reference
            table is malformed.
    """
    base = normalize_base_url(base_url)
    node = _open(target, "r")
    store = _storeio.unwrap_store(node.store_path.store)
    prefix = node.store_path.path
    prototype = default_buffer_prototype()

    virtual = find_reference_arrays(node)
    virtual_paths = [array.store_path.path for array in virtual]

    refs: dict[str, Any] = {}
    for key in sync(_list_keys(store, prefix)):
        if _belongs_to_virtual_array(key, virtual_paths):
            continue
        relative = _relative(key, prefix)
        if relative.rsplit("/", 1)[-1] == "zarr.json":
            payload = sync(store.get(key, prototype=prototype))
            if payload is not None:  # pragma: no branch - listed keys exist
                refs[relative] = payload.to_bytes().decode("utf-8")
            continue
        refs[relative] = [base + relative, 0, sync(store.getsize(key))]

    for array in virtual:
        codec = reference_codec(array)
        assert codec is not None
        path = array.store_path.path
        relative_path = _relative(path, prefix)
        array_url = urljoin(base, f"{relative_path}/") if relative_path else base
        key_prefix = f"{relative_path}/" if relative_path else ""

        payload = sync(store.get(_join(path, "zarr.json"), prototype=prototype))
        if payload is None:  # pragma: no cover - the array was just opened
            raise ReferenceFormatError(f"{path!r}: zarr.json disappeared")
        metadata = json.loads(payload.to_bytes())
        metadata["codecs"] = [c.to_dict() for c in codec.codecs]
        metadata["chunk_grid"]["configuration"]["chunk_shape"] = list(codec.chunk_shape)
        refs[f"{key_prefix}zarr.json"] = json.dumps(metadata)

        per_shard = chunks_per_shard(array.chunks, codec.chunk_shape)
        coords = list(inner_coords(per_shard))
        for shard_key in sync(_list_keys(store, _join(path, "c"))):
            table_bytes = sync(store.get(shard_key, prototype=prototype))
            if table_bytes is None:  # pragma: no cover - listed keys exist
                continue
            shard = _shard_index(_relative(shard_key, path), len(per_shard))
            try:
                table = decode_table(table_bytes.to_bytes(), len(coords))
            except ReferenceFormatError as e:
                raise ReferenceFormatError(f"shard {shard_key!r}: {e}") from e
            for inner, reference in zip(coords, table, strict=True):
                if reference is None:
                    continue
                if reference.source >= len(codec.sources):
                    raise ReferenceFormatError(
                        f"shard {shard_key!r}, inner chunk {inner}: source "
                        f"#{reference.source} is not declared."
                    )
                chunk = [
                    s * n + i for s, n, i in zip(shard, per_shard, inner, strict=True)
                ]
                chunk_key = key_prefix + "c/" + "/".join(str(i) for i in chunk)
                source_url = urljoin(array_url, codec.sources[reference.source])
                refs[chunk_key] = [source_url, reference.offset, reference.length]

    manifest = {"version": MANIFEST_VERSION, "refs": refs}
    if out is not None:
        Path(out).write_text(json.dumps(manifest, indent=1))
    return manifest
