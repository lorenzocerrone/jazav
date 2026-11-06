# Step 1: reference-table codec (format 2)

Context file for the first roadmap step. It records the findings and decisions
of the 2026-11-06 design session so the work can be picked up without the
conversation. Steps 2 and 3 have their own files next to this one.

## Problem

Format 1 stored one 20-byte reference per level-0 chunk key. Reading a chunk
took two dependent store requests: the reference, then the byte range in the
source file. Measured on a local disk in a container, 256 strips of 128 KiB:

| read                                        | best time |
|---------------------------------------------|-----------|
| format 1 virtual array                      | 239 ms    |
| plain uncompressed zarr, same chunks        | 127 ms    |
| source range reads alone, no reference hop  | 95 ms     |
| one `open`+`read` of a 20-byte reference    | 20 µs     |

The extra cost on local disk is not I/O but a second asyncio-to-thread hop and
a second pass through zarr's store machinery per chunk. On object storage every
hop is a round trip and a billable request, so the request count doubles. On
NFS, Lustre or GPFS one small file per chunk is the classic metadata
bottleneck, the same problem zarr's sharding was created for.

A batch override of `decode_partial` (fetching all references of a batch in
one call) was ruled out: zarr's `codec_pipeline.batch_size` defaults to 1, so
the codec sees one chunk per call unless every reader changes a global config.
Zarr's own sharding codec is immune because the shard hook receives the whole
selection over the shard. That is what format 2 inherits.

## Decisions (agreed)

1. **Format 2 replaces format 1.** No compatibility. Same codec name
   `jazav.reference`, `version: 2`, a format-1 array is refused with a message
   to rebuild.
2. **One table per shard.** The array's zarr chunk is the shard; the stored key
   holds `20 * n` bytes, one entry per inner chunk, C order over the shard's
   inner chunk grid. Entry is `<IQQ`: `uint32 source`, `uint64 offset`,
   `uint64 length`. Twenty `0xFF` bytes mark a missing inner chunk (fill value).
   A missing shard key reads as fill for all its chunks. The table is always
   full size, also on edge shards.
3. **Default shard shape:** the whole array while it has at most 16,384 inner
   chunks (a 320 KiB table); above that split along the slowest axes first,
   leaving y and x whole so a plane read touches one table. `shards` and
   `max_chunks_per_shard` are writer arguments. Table sizes for reference:
   1,024 chunks = 20 KiB, 16,384 = 320 KiB, 131,072 = 2.5 MiB.
4. **Codec configuration** gains `chunk_shape` (inner chunk shape); the array's
   `chunk_grid` is the shard shape and must be a multiple of it. `sources`,
   `sizes`, `codecs` (nested chain per inner chunk) unchanged.
5. **Read path:** table decoded once per shard and cached in the codec instance
   (bounded LRU, 64 shards; level 0 is read-only so no invalidation); ranges
   coalesced per source (gap up to 1 MiB, request up to 64 MiB); inner chunks
   decoded through zarr's own pipeline via an in-memory `ByteGetter`, as the
   sharding codec does, so integer, orthogonal and fancy indexing behave the
   same. Size check per source once, as before.
6. **Entry stays `source, offset, length`.** Per-source checksums (Icechunk and
   refkv have them) and multi-range chunks (refkv `Concat`) are deferred; both
   are compatible additions.
7. **Public API:** `create_reference_array(..., chunks=inner, shards=None,
   max_chunks_per_shard=16384)`; `write_references` writes whole tables for the
   shards it has entries for; new `reference_codec(array)` helper; `verify`
   counts shards and chunks separately and reports `bad-table`; `materialize`
   defaults to the inner chunk shape and copies in blocks bounded by 256 MiB.

## Draft measurement

The draft codec on the same 256-strip read: **68 ms** best (plain zarr 123 ms
in the same run). Coalescing turned 256 range reads into one; the table is one
20-byte-per-chunk file read once.

## zarr-python facts relied on (checked in 3.2.1 and 3.4.2)

- A codec reaches the store only through `ArrayBytesCodecPartialDecodeMixin.
  _decode_partial_single(byte_getter, selection, chunk_spec)`, and only when it
  is the array's sole codec. `byte_getter` is a `StorePath` with `.store` and
  `.path` (semi-private; the sharding codec relies on the same).
- `get_indexer(selection, shape, chunk_grid=ChunkGrid.from_sizes(...))`,
  `ArraySpec(shape, dtype, fill_value, config, prototype)`, `ArraySpec.order`,
  `concurrent_map`, `config.get("async.concurrency")` (default 10) exist in
  both versions with the same signatures.
- `CodecPipeline.read(batch, out)` with `(ByteGetter, spec, chunk_selection,
  out_selection, is_complete)` tuples handles writing decoded inner chunks into
  the output buffer; a getter returning `None` yields the fill value. zarr 3.4's
  fused pipeline takes a sync fast path when the getter implements `get_sync`.
- `Array.chunks` reports the shard shape for a custom codec (only
  `sharding_indexed` is special-cased to report inner chunks). Dask built on
  `arr.chunks` therefore sees shards; document this and point to
  `reference_codec(array).chunk_shape`.
- Validation hook: `Codec.validate(*, shape, dtype, chunk_grid)`; the chunk grid
  object exposes `chunk_shape` for regular grids.

## Work remaining for this step

- Update `tests/` for the new layout (conftest helpers write shard tables, not
  chunk files; version 2 in metadata tests; `bad-table` kind; default shard =
  whole array means keys like `0/c/0/0/0/0` hold the table).
- Regenerate `samples/tiff_grid` with `python examples/tiff_grid.py --out DIR`
  and copy `DIR/raw/tiff` and `DIR/tiff/virtual.zarr` back; `tests/test_sample.py`
  pins the format.
- `examples/_common.py` passes `plan.chunks` as inner chunks; shards default.
- README: format section rewritten as the spec (this is the document for the
  vizarr conversation), reading/writing examples, shard rule, `arr.chunks`
  caveat. CHANGELOG entry.
- Run `pytest` in both environments (zarr 3.2.1 on Python 3.12, latest zarr on
  3.13), `ruff`, `ty`.

## Parked (not this step)

- ngio hooks to regenerate derived artifacts (manifests, RFC-9 zips) on writes
  that add keys; ngio marking level 0 read-only with a clear message.
- refkv's `zarrify` as an optional planner front end (parsers for ~two dozen
  formats, Rust core, wasm); its storage format is alpha and not adopted.
- Checksums per source, multi-range entries.
