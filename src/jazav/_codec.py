"""`jazav.reference`: a read-only zarr v3 reference codec.

Decoding one chunk:

```text
chunk key  --(20 bytes: source, offset, length)-->  byte range in a source file
store.get(source, byte_range)                   -->  stored block bytes
nested codec chain (e.g. [bytes])               -->  chunk array
```

The codec must be the array's **only** codec (serializer, no filters, no
compressors), exactly like `sharding_indexed`: zarr-python hands a codec the
store-backed `ByteGetter` it needs to read another key only through the
partial-decode hook, and only when the array->bytes codec stands alone. The
real decoding chain for one stored block lives in the nested `codecs`
configuration and may contain any zarr codecs, e.g. `transpose` for blocks
stored in another axis order or `zstd`/`numcodecs.zlib` for compressed blocks.

Configuration:

```json
{"name": "jazav.reference",
 "configuration": {
   "version": 1,
   "sources": ["../../../raw/A01_f0.tif", "../../../raw/A01_f1.tif"],
   "sizes": [8388744, 8388744],
   "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}}
```

`sources` are relative to the array's key prefix (see `jazav._references`);
`sizes` are their byte sizes at write time, checked once per source before the
first read so a truncated or replaced file is refused instead of decoded into
garbage. The array is read-only: writing a chunk raises.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, cast

from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin, Codec
from zarr.core.common import parse_named_configuration
from zarr.core.metadata.v3 import parse_codecs
from zarr.registry import get_pipeline_class, register_codec

from jazav import _storeio
from jazav._references import (
    FORMAT_VERSION,
    REFERENCE_SIZE,
    ReferenceFormatError,
    array_prefix_from_chunk_key,
    check_relative_path,
    decode_reference,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.abc.codec import CodecPipeline
    from zarr.abc.store import ByteGetter, Store
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.common import JSON
    from zarr.core.indexing import SelectorTuple

CODEC_NAME = "jazav.reference"
"""Codec name in `zarr.json` and in the `zarr.codecs` entry point. Frozen."""


def _installed_version() -> str:
    from jazav import __version__

    return __version__


@dataclass(frozen=True)
class ReferenceCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Resolve 20-byte chunk references into byte ranges of source files.

    Args:
        sources: Relative paths of the source files, indexed by the chunk
            payload's `source` field.
        sizes: Byte size of each source when the references were written.
        codecs: Nested zarr v3 codec chain that decodes one stored block
            into the chunk array.
        version: Reference format version; only `1` exists.
    """

    is_fixed_size = True  # ty: ignore[invalid-attribute-override]

    sources: tuple[str, ...]
    sizes: tuple[int, ...]
    codecs: tuple[Codec, ...]
    version: int
    _pipelines: dict[ArraySpec, CodecPipeline] = field(
        init=False, repr=False, compare=False, hash=False
    )
    _size_checks: dict[int, asyncio.Future[None]] = field(
        init=False, repr=False, compare=False, hash=False
    )
    _parent_stores: dict[int, Store] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __init__(
        self,
        *,
        sources: Iterable[str],
        sizes: Iterable[int],
        codecs: Iterable[Codec | dict[str, JSON]],
        version: int = FORMAT_VERSION,
    ) -> None:
        sources = tuple(sources)
        sizes = tuple(sizes)
        if version != FORMAT_VERSION:
            raise ReferenceFormatError(
                f"unsupported reference format version {version}: jazav "
                f"{_installed_version()} reads version {FORMAT_VERSION}. "
                "Upgrade jazav."
            )
        if not sources:
            raise ReferenceFormatError(
                f"{CODEC_NAME} needs at least one source; got sources={sources!r}."
            )
        if len(set(sources)) != len(sources):
            raise ReferenceFormatError(f"duplicate entries in sources {sources!r}.")
        for source in sources:
            check_relative_path(source)
        if len(sizes) != len(sources):
            raise ReferenceFormatError(
                f"sizes has {len(sizes)} entries but sources has {len(sources)}; "
                "every source needs its byte size."
            )
        for source, size in zip(sources, sizes, strict=True):
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ReferenceFormatError(
                    f"size of source {source!r} must be a non-negative integer, "
                    f"got {size!r}."
                )
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "codecs", parse_codecs(codecs))
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "_pipelines", {})
        object.__setattr__(self, "_size_checks", {})
        object.__setattr__(self, "_parent_stores", {})

    # Pickling (dask with the processes scheduler or distributed). The caches
    # are rebuilt on the receiving side.
    def __getstate__(self) -> dict[str, JSON]:
        return self.to_dict()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state["configuration"])  # type: ignore[misc]

    # -- metadata ---------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME)
        for key in ("sources", "sizes", "codecs"):
            if key not in configuration:
                raise ReferenceFormatError(
                    f"{CODEC_NAME} configuration is missing {key!r}; a format-"
                    f"{FORMAT_VERSION} configuration has version, sources, sizes "
                    "and codecs."
                )
        return cls(
            sources=cast("Iterable[str]", configuration["sources"]),
            sizes=cast("Iterable[int]", configuration["sizes"]),
            codecs=cast("Iterable[dict[str, JSON]]", configuration["codecs"]),
            version=cast("int", configuration.get("version", FORMAT_VERSION)),
        )

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": CODEC_NAME,
            "configuration": {
                "version": self.version,
                "sources": list(self.sources),
                "sizes": list(self.sizes),
                "codecs": [codec.to_dict() for codec in self.codecs],
            },
        }

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return REFERENCE_SIZE

    def _nested_pipeline(self, chunk_spec: ArraySpec) -> CodecPipeline:
        pipeline = self._pipelines.get(chunk_spec)
        if pipeline is None:
            pipeline = (
                get_pipeline_class()
                .from_codecs(self.codecs)
                .evolve_from_array_spec(chunk_spec)
            )
            self._pipelines[chunk_spec] = pipeline
        return pipeline

    def _resolve_source(
        self, store: Store, prefix: str, source: int
    ) -> tuple[Store, str]:
        up, key = _storeio.resolve_key(prefix, self.sources[source])
        parent = self._parent_stores.get(up)
        if parent is None:
            parent = _storeio.parent_store(store, up)
            self._parent_stores[up] = parent
        return parent, key

    async def _stat_source(self, store: Store, key: str, source: int) -> None:
        actual = await _storeio.source_size(store, key)
        expected = self.sizes[source]
        if actual is not None and actual != expected:
            raise ReferenceFormatError(
                f"source {key!r} is {actual} bytes but the virtual array was "
                f"written against {expected} bytes. The file was truncated or "
                "replaced; restore the original file or rebuild the virtual array."
            )

    async def _check_source_size(self, store: Store, key: str, source: int) -> None:
        """Compare the source's size with `sizes` once, even under concurrency.

        Chunks of one source decode concurrently, so the first caller starts the
        stat and every other caller awaits the same future.
        """
        pending = self._size_checks.get(source)
        if pending is None:
            pending = asyncio.ensure_future(self._stat_source(store, key, source))
            self._size_checks[source] = pending
        await pending

    # -- reading ----------------------------------------------------------
    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        store, chunk_key = _storeio.store_path_of(byte_getter)
        prototype = chunk_spec.prototype

        payload = await byte_getter.get(prototype=prototype)
        if payload is None:
            return None  # chunk absent: zarr fills with fill_value
        reference = decode_reference(payload.to_bytes())
        if reference.source >= len(self.sources):
            raise ReferenceFormatError(
                f"chunk {chunk_key!r} references source #{reference.source}, but "
                f"the array declares only {len(self.sources)} source(s)."
            )
        size = self.sizes[reference.source]
        if reference.offset + reference.length > size:
            raise ReferenceFormatError(
                f"chunk {chunk_key!r} references bytes "
                f"[{reference.offset}, {reference.offset + reference.length}) of "
                f"source {self.sources[reference.source]!r}, which is only {size} "
                "bytes. The reference is corrupt or the sizes table is wrong."
            )

        prefix = array_prefix_from_chunk_key(chunk_key, len(chunk_spec.shape))
        source_store, source_key = self._resolve_source(store, prefix, reference.source)
        await self._check_source_size(source_store, source_key, reference.source)
        data = await _storeio.read_range(
            source_store, source_key, prototype, reference.offset, reference.length
        )
        (decoded,) = await self._nested_pipeline(chunk_spec).decode(
            [(data, chunk_spec)]
        )
        if decoded is None:
            raise ReferenceFormatError(
                f"the nested codecs returned no data for chunk {chunk_key!r} "
                f"({source_key!r}, offset={reference.offset}, "
                f"length={reference.length})."
            )
        if tuple(decoded.shape) != tuple(chunk_spec.shape):
            raise ReferenceFormatError(
                f"chunk {chunk_key!r} decoded to shape {tuple(decoded.shape)}, "
                f"expected the chunk shape {tuple(chunk_spec.shape)}. The stored "
                "block does not match the array's chunk grid."
            )
        return decoded[selection]

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # Only reached when the codec is combined with other codecs, which
        # bypasses partial decoding. Without a store, references cannot resolve.
        raise ReferenceFormatError(
            f"{CODEC_NAME} must be the array's only codec (no filters or "
            "compressors): it resolves references through the partial-decode "
            "path, which zarr uses only for a lone array->bytes codec."
        )

    async def _encode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        raise NotImplementedError(
            f"{CODEC_NAME} is read-only: level 0 of a virtual array references "
            "the original files and cannot be written. Run `jazav materialize` "
            "to turn it into a regular, writable array."
        )


# Make the codec resolvable even when the entry point of an editable install is
# stale; registering the same class twice is harmless.
register_codec(CODEC_NAME, ReferenceCodec)
