# Step 3: vizarr proof of concept with a reference-resolving store

Context file for the third roadmap step. Work happens in the user's fork of
hms-dbmi/vizarr; depends on step 1 (format 2) and on the README spec section
written there. Step 2's manifest export is the alternative for the hosted
viewer; this step needs no manifest.

## Goal

A static vizarr build that opens a virtual OME-Zarr directory as served, with
level 0 resolved in the browser: `?source=jazav://https://host/image.zarr`.
Then propose it upstream; the user is in contact with the vizarr maintainers.

## Why a store and not a codec

- zarrita.js exposes a mutable codec `registry` (`registry.set(name, () =>
  import(...))`, entry `{ fromConfig(config, meta), kind: "array_to_bytes" }`),
  and `decode` may be async, so a browser codec could fetch ranges itself. But a
  codec receives only the chunk bytes and its configuration: no store, no key,
  no array URL, so relative source paths cannot be resolved inside it.
- zarrita has no partial-decode hook for custom codecs; only `sharding_indexed`
  is special-cased. A codec for the table format would have to fetch the whole
  shard for one tile. Unusable for format 2.
- vizarr bundles zarrita (`zarrita ~0.6`, `@zarrita/storage ~0.1.4` in
  package.json, app build with no externals). It registers its own `jpeg2k`
  codec in `src/codecs/register.ts` and exposes no registration hook, so a codec
  registered by a page lands in a different module instance.
- `addImage({ source })` accepts `string | zarr.Readable`; `normalizeStore` in
  `src/utils.ts` does `zarr.root(source)` for objects and, for strings, picks
  `OmeTiffStore` (`ome-tiff://`), `ReferenceStore` (`ref://` or `.json`) or
  `FetchStore`, then wraps in an LRU store (`src/lru-store.ts`). A new prefix
  belongs next to those.
- Python widget (`python/src/vizarr/_widget.py`) proxies raw key bytes to the
  hosted build, so it gains the feature only once the hosted build has it.

## The store

Implement zarrita's `AsyncReadable` (from `@zarrita/storage` `types.ts`):
`get(key: AbsolutePath, opts?) => Promise<Uint8Array | undefined>` and optional
`getRange(key, { offset, length } | { suffixLength }, opts?)`. Keys arrive with a
leading slash, e.g. `/0/c/0/0/0/0`. Wrap a `FetchStore` on the dataset URL:

1. Forward every key to the `FetchStore`.
2. On `<prefix>/zarr.json` whose codecs are exactly `[jazav.reference]`:
   remember `{ chunk_shape, sources, sizes, codecs, shard_shape }` by array
   prefix, and return the metadata rewritten with `codecs` = nested chain and
   `chunk_grid.chunk_shape` = inner `chunk_shape`, so zarrita sees an ordinary
   array with inner chunks.
3. On a chunk key under a remembered prefix (default encoding, `c/` separator
   `/`): compute the shard key from the inner chunk index and the shard shape,
   fetch the table once (cache per shard key), take the entry at the C-order
   index, return `undefined` for the all-`0xFF` entry (fill), else resolve the
   relative source against the array's URL (`new URL(source, arrayUrl + "/")`)
   and issue one range request (`Range: bytes=offset-(offset+length-1)`),
   returning those bytes. Honour the optional `sizes` check with a `HEAD` or the
   `Content-Range` total if cheap.
4. If a chunk key arrives before its metadata (unlikely, zarrita needs the
   metadata to compute keys), fetch `<prefix>/zarr.json` lazily.
5. Everything else passes through: pyramid levels, labels, tables.

Roughly a hundred lines of TypeScript, no new dependency. Table parsing: 20
bytes per entry, little-endian `u32, u64, u64`, `DataView.getBigUint64`.

## vizarr integration

- Smallest: a `jazav://` prefix in `normalizeStore`, mirroring `ome-tiff://`.
- Nicer: detection, open with `FetchStore`, swap in the wrapper when an array's
  metadata names `jazav.reference`.
- `FetchStore` facts: `resolve(root, path)` appends the key without its leading
  slash to the base URL (trailing slash added); 404 returns `undefined`; 200 or
  206 returns bytes; `getRange` sets the `Range` header; suffix requests use a
  `HEAD` first unless `useSuffixRequest`.

## Demo and test

- Serve the committed sample `samples/tiff_grid` (after step 1 regenerates it)
  with range requests and CORS; Python's `http.server` has no range support,
  use e.g. Caddy, nginx, or a small Node/`serve` with ranges enabled.
- Hosting requirements are the same as any zarr in vizarr plus ranges and CORS
  on the raw files.
- Playwright is available in the cloud session, but reaching the hosted vizarr
  build from the sandbox was not verified; plan on running the browser test on
  the user's machine.

## What to bring to the maintainers

- The frozen format 2 spec from the README, including the relative path rules
  and the nested codec semantics.
- Registering `jazav.reference` as a zarr v3 extension codec through the
  zarr-extensions process, so another implementation reading it is normal.
- A demo they can click, and the request-count argument: one table read per
  shard, then one range per inner chunk, coalescing optional.
- Precedent: vizarr already maintains two stores of this kind (OME-TIFF,
  kerchunk) and one non-standard codec (jpeg2k).
- Likely objections: committing to a codec that exists in one Python package,
  and format stability. Both are answered by the spec and the extension
  registration.

## Also ask them

Whether they would take an Icechunk store built on icechunk-js, as a data
point on how they weigh new store types; this does not change jazav's plan.
