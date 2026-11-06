"""`jazav.reference`: a read-only zarr v3 reference-table codec.

The array's zarr chunks are shards. Each stored shard key holds a table with
one 20-byte `(source, offset, length)` entry per inner chunk (see
`jazav._references`). Decoding a selection of one shard:

```text
shard key   --(table: 20 bytes per inner chunk)-->  byte ranges in source files
coalesce adjacent ranges per source, store.get(source, byte_range) each
nested codec chain (e.g. [bytes]) per inner chunk  -->  inner chunk arrays
```

The codec must be the array's **only** codec (serializer, no filters, no
compressors), exactly like `sharding_indexed`: zarr-python hands a codec the
store-backed `ByteGetter` it needs to read other keys only through the
partial-decode hook, and only when the array->bytes codec stands alone. The
decoding chain for one stored block lives in the nested `codecs` configuration
and may contain any zarr codecs, e.g. `transpose` for blocks stored in another
axis order or `zstd`/`numcodecs.zlib` for compressed blocks.

Configuration:

```json
{"name": "jazav.reference",
 "configuration": {
   "version": 2,
   "chunk_shape": [1, 1, 2160, 2560],
   "sources": ["../../../raw/A01_f0.tif", "../../../raw/A01_f1.tif"],
   "sizes": [8388744, 8388744],
   "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}}
```

`chunk_shape` is the inner chunk shape; the array's `chunk_grid` is the shard
shape and must be a multiple of it. `sources` are relative to the array's key
prefix; `sizes` are their byte sizes at write time, checked once per source
before the first read so a truncated or replaced file is refused instead of
decoded into garbage. The array is read-only: writing a chunk raises.
"""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Self, cast

from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin, Codec
from zarr.abc.store import (
    ByteGetter,
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.array_spec import ArraySpec
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import concurrent_map, parse_named_configuration
from zarr.core.config import config as zarr_config
from zarr.core.indexing import get_indexer
from zarr.core.metadata.v3 import parse_codecs
from zarr.registry import get_pipeline_class, register_codec

from jazav import _storeio
from jazav._references import (
    FORMAT_VERSION,
    ChunkReference,
    ReferenceFormatError,
    array_prefix_from_chunk_key,
    check_relative_path,
    chunks_per_shard,
    decode_table,
    flat_index,
    table_size,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from zarr.abc.codec import CodecPipeline
    from zarr.abc.store import Store
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.common import JSON
    from zarr.core.indexing import SelectorTuple

CODEC_NAME = "jazav.reference"
"""Codec name in `zarr.json` and in the `zarr.codecs` entry point. Frozen."""

MAX_GAP_BYTES = 1 << 20
"""Two blocks of one source closer than this are read with one range request."""

MAX_COALESCED_BYTES = 64 << 20
"""Upper bound on one coalesced range request."""

TABLE_CACHE_SIZE = 64
"""Decoded reference tables kept per codec instance (one instance per array)."""


def _installed_version() -> str:
    from jazav import __version__

    return __version__


@dataclass(frozen=True)
class _BlockGetter(ByteGetter):
    """In-memory `ByteGetter` for one inner chunk; `None` means fill value.

    Also implements zarr's `SyncByteGetter` protocol (`get_sync`), so a
    synchronous codec pipeline can take its fast path on inner chunks.
    """

    blocks: Mapping[tuple[int, ...], Buffer]
    coords: tuple[int, ...]

    def get_sync(
        self, prototype: BufferPrototype | None = None, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        value = self.blocks.get(self.coords)
        if value is None or byte_range is None:
            return value
        size = len(value)
        if isinstance(byte_range, RangeByteRequest):
            return value[byte_range.start : byte_range.end]
        if isinstance(byte_range, OffsetByteRequest):
            return value[byte_range.offset :]
        if isinstance(byte_range, SuffixByteRequest):
            return value[max(0, size - byte_range.suffix) :]
        raise TypeError(f"unexpected byte range {byte_range!r}")  # pragma: no cover

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        return self.get_sync(prototype, byte_range)


@dataclass(frozen=True)
class _Pending:
    """One inner chunk to read: where it is and how it maps into the output."""

    coords: tuple[int, ...]
    reference: ChunkReference


@dataclass(frozen=True)
class ReferenceCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Resolve a shard's reference table into byte ranges of source files.

    Args:
        chunk_shape: Inner chunk shape; every table entry describes one such
            chunk, and the array's chunk grid (the shard shape) must be a
            multiple of it.
        sources: Relative paths of the source files, indexed by the entries'
            `source` field.
        sizes: Byte size of each source when the references were written.
        codecs: Nested zarr v3 codec chain that decodes one stored block
            into one inner chunk array.
        version: Reference format version; only `2` exists.
    """

    is_fixed_size = True  # ty: ignore[invalid-attribute-override]

    chunk_shape: tuple[int, ...]
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
    _tables: OrderedDict[str, asyncio.Future[list[ChunkReference | None] | None]] = (
        field(init=False, repr=False, compare=False, hash=False)
    )

    def __init__(
        self,
        *,
        chunk_shape: Iterable[int],
        sources: Iterable[str],
        sizes: Iterable[int],
        codecs: Iterable[Codec | dict[str, JSON]],
        version: int = FORMAT_VERSION,
    ) -> None:
        chunk_shape = tuple(chunk_shape)
        sources = tuple(sources)
        sizes = tuple(sizes)
        if version != FORMAT_VERSION:
            raise ReferenceFormatError(
                f"unsupported reference format version {version}: jazav "
                f"{_installed_version()} reads version {FORMAT_VERSION}. "
                "Rebuild the virtual array with this jazav, or install the jazav "
                "that wrote it."
            )
        if not chunk_shape or any(
            not isinstance(c, int) or isinstance(c, bool) or c <= 0 for c in chunk_shape
        ):
            raise ReferenceFormatError(
                f"chunk_shape must be a non-empty tuple of positive integers, got "
                f"{chunk_shape!r}."
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
        object.__setattr__(self, "chunk_shape", chunk_shape)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "codecs", parse_codecs(codecs))
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "_pipelines", {})
        object.__setattr__(self, "_size_checks", {})
        object.__setattr__(self, "_parent_stores", {})
        object.__setattr__(self, "_tables", OrderedDict())

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
        for key in ("chunk_shape", "sources", "sizes", "codecs"):
            if key not in configuration:
                raise ReferenceFormatError(
                    f"{CODEC_NAME} configuration is missing {key!r}; a format-"
                    f"{FORMAT_VERSION} configuration has version, chunk_shape, "
                    "sources, sizes and codecs."
                )
        return cls(
            chunk_shape=cast("Iterable[int]", configuration["chunk_shape"]),
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
                "chunk_shape": list(self.chunk_shape),
                "sources": list(self.sources),
                "sizes": list(self.sizes),
                "codecs": [codec.to_dict() for codec in self.codecs],
            },
        }

    def validate(self, *, shape: tuple[int, ...], dtype: Any, chunk_grid: Any) -> None:
        if len(self.chunk_shape) != len(shape):
            raise ReferenceFormatError(
                f"chunk_shape {self.chunk_shape} and array shape {shape} differ in "
                "rank."
            )
        shard_shape = getattr(chunk_grid, "chunk_shape", None)
        if shard_shape is not None:
            chunks_per_shard(shard_shape, self.chunk_shape)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        inner = self._inner_spec(array_spec)
        evolved = tuple(c.evolve_from_array_spec(array_spec=inner) for c in self.codecs)
        if evolved != self.codecs:
            return replace(self, codecs=evolved)
        return self

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        return table_size(math.prod(chunks_per_shard(chunk_spec.shape, self.chunk_shape)))

    # -- helpers ----------------------------------------------------------
    def _inner_spec(self, shard_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            shape=self.chunk_shape,
            dtype=shard_spec.dtype,
            fill_value=shard_spec.fill_value,
            config=shard_spec.config,
            prototype=shard_spec.prototype,
        )

    def _nested_pipeline(self, inner_spec: ArraySpec) -> CodecPipeline:
        pipeline = self._pipelines.get(inner_spec)
        if pipeline is None:
            pipeline = (
                get_pipeline_class()
                .from_codecs(self.codecs)
                .evolve_from_array_spec(inner_spec)
            )
            self._pipelines[inner_spec] = pipeline
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

        Shards of one source decode concurrently, so the first caller starts the
        stat and every other caller awaits the same future.
        """
        pending = self._size_checks.get(source)
        if pending is None:
            pending = asyncio.ensure_future(self._stat_source(store, key, source))
            self._size_checks[source] = pending
        await pending

    async def _fetch_table(
        self, byte_getter: ByteGetter, shard_key: str, n_chunks: int, prototype: Any
    ) -> list[ChunkReference | None] | None:
        payload = await byte_getter.get(prototype=prototype)
        if payload is None:
            return None
        try:
            return decode_table(payload.to_bytes(), n_chunks)
        except ReferenceFormatError as e:
            raise ReferenceFormatError(f"shard {shard_key!r}: {e}") from e

    async def _load_table(
        self, byte_getter: ByteGetter, shard_key: str, n_chunks: int, prototype: Any
    ) -> list[ChunkReference | None] | None:
        """Read and decode a shard's table once; later reads hit the cache.

        Level 0 is read-only, so a table never changes under a codec instance.
        A failed fetch is forgotten, so the next read retries it.
        """
        pending = self._tables.get(shard_key)
        if pending is None:
            pending = asyncio.ensure_future(
                self._fetch_table(byte_getter, shard_key, n_chunks, prototype)
            )
            self._tables[shard_key] = pending
            while len(self._tables) > TABLE_CACHE_SIZE:
                self._tables.popitem(last=False)
        else:
            self._tables.move_to_end(shard_key)
        try:
            return await pending
        except Exception:
            self._tables.pop(shard_key, None)
            raise

    def _check_reference(
        self, shard_key: str, coords: tuple[int, ...], reference: ChunkReference
    ) -> None:
        if reference.source >= len(self.sources):
            raise ReferenceFormatError(
                f"shard {shard_key!r}, inner chunk {coords}: references source "
                f"#{reference.source}, but the array declares only "
                f"{len(self.sources)} source(s)."
            )
        size = self.sizes[reference.source]
        if reference.offset + reference.length > size:
            raise ReferenceFormatError(
                f"shard {shard_key!r}, inner chunk {coords}: references bytes "
                f"[{reference.offset}, {reference.offset + reference.length}) of "
                f"source {self.sources[reference.source]!r}, which is only {size} "
                "bytes. The reference is corrupt or the sizes table is wrong."
            )

    async def _read_blocks(
        self,
        store: Store,
        prefix: str,
        pending: list[_Pending],
        prototype: BufferPrototype,
    ) -> dict[tuple[int, ...], Buffer]:
        """Read every pending block, coalescing neighbours within one source."""
        by_source: dict[int, list[_Pending]] = {}
        for item in pending:
            by_source.setdefault(item.reference.source, []).append(item)

        requests: list[tuple[Store, str, int, int, list[_Pending]]] = []
        for source, items in by_source.items():
            source_store, key = self._resolve_source(store, prefix, source)
            await self._check_source_size(source_store, key, source)
            items.sort(key=lambda item: item.reference.offset)
            group: list[_Pending] = []
            start = end = 0
            for item in items:
                ref = item.reference
                if group and (
                    ref.offset > end + MAX_GAP_BYTES
                    or max(end, ref.offset + ref.length) - start > MAX_COALESCED_BYTES
                ):
                    requests.append((source_store, key, start, end - start, group))
                    group = []
                if not group:
                    start, end = ref.offset, ref.offset + ref.length
                else:
                    end = max(end, ref.offset + ref.length)
                group.append(item)
            if group:
                requests.append((source_store, key, start, end - start, group))

        async def read(
            source_store: Store, key: str, start: int, length: int, group: list[_Pending]
        ) -> list[tuple[tuple[int, ...], Buffer]]:
            data = await _storeio.read_range(source_store, key, prototype, start, length)
            return [
                (
                    item.coords,
                    data[
                        item.reference.offset - start : item.reference.offset
                        - start
                        + item.reference.length
                    ],
                )
                for item in group
            ]

        results = await concurrent_map(
            requests, read, zarr_config.get("async.concurrency")
        )
        return dict(block for result in results for block in result)

    # -- reading ----------------------------------------------------------
    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, shard_spec: ArraySpec
    ) -> NDBuffer | None:
        store, shard_key = _storeio.store_path_of(byte_getter)
        prototype = shard_spec.prototype
        per_shard = chunks_per_shard(shard_spec.shape, self.chunk_shape)

        table = await self._load_table(
            byte_getter, shard_key, math.prod(per_shard), prototype
        )
        if table is None:
            return None  # shard absent: zarr fills with fill_value

        indexer = get_indexer(
            selection,
            shape=shard_spec.shape,
            chunk_grid=ChunkGrid.from_sizes(shard_spec.shape, self.chunk_shape),
        )
        projections = list(indexer)
        pending: list[_Pending] = []
        seen: set[tuple[int, ...]] = set()
        for coords, *_ in projections:
            if coords in seen:
                continue
            seen.add(coords)
            reference = table[flat_index(coords, per_shard)]
            if reference is not None:
                self._check_reference(shard_key, coords, reference)
                pending.append(_Pending(coords, reference))

        prefix = array_prefix_from_chunk_key(shard_key, len(shard_spec.shape))
        blocks = await self._read_blocks(store, prefix, pending, prototype)

        out = prototype.nd_buffer.empty(
            shape=indexer.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )
        inner_spec = self._inner_spec(shard_spec)
        await self._nested_pipeline(inner_spec).read(
            [
                (
                    _BlockGetter(blocks, coords),
                    inner_spec,
                    chunk_selection,
                    out_selection,
                    is_complete,
                )
                for coords, chunk_selection, out_selection, is_complete in projections
            ],
            out,
        )
        if hasattr(indexer, "sel_shape"):
            return out.reshape(indexer.sel_shape)
        return out

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
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
