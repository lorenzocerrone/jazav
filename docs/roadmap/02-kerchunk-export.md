# Step 2: kerchunk manifest export

Context file for the second roadmap step. Depends on step 1 (format 2 tables).

## Why export at all

Web viewers cannot load a custom codec. vizarr opens exactly three source
forms (checked in `src/utils.ts` of hms-dbmi/vizarr, 2026-11-06):

- a plain zarr URL (`FetchStore`),
- `ome-tiff://<url>` (a vizarr-internal `OmeTiffStore`),
- `ref://<url>` or any URL ending in `.json`: fetched and opened through
  `@zarrita/storage`'s `ReferenceStore.fromSpec`, i.e. a kerchunk manifest.

So a manifest is the only way the hosted vizarr shows virtual level 0 today,
with no fork and no server-side code. Step 3 (a store in a vizarr fork) is the
alternative that needs no manifest.

## What the manifest must contain

To the viewer the manifest *is* the store: a key that is not listed does not
exist, and zarrita's `ReferenceStore` has no fall-through to the real
directory. So the manifest must list the whole image, not only level 0:

- root `zarr.json` (multiscales, omero), every level's `zarr.json`,
- every pyramid chunk and, if wanted, labels,
- level-0 chunks as byte ranges into the raw files.

Level-0 `zarr.json` must be rewritten: codecs = the nested chain from the
codec configuration (`bytes`, `transpose` + `bytes`, `bytes` + `zstd`, ...),
chunk grid = the **inner** chunk shape, since each inner chunk becomes one
manifest key.

## URL-only form (decided)

Write every key except level-0 chunks as a whole-file URL entry `[url]`, never
inline the contents. Then:

- metadata and pyramid pixels are fetched live from the directory, so rewriting
  a chunk or changing attributes never goes stale;
- only the *set of keys* is frozen: a new pyramid level, a new label, or a
  chunk written where none existed, requires regeneration;
- generation is one directory walk plus the level-0 tables, with no pixel reads,
  cheap enough to run on every publish.

Level-0 entries are `[url, offset, length]` where `url` is the raw file's URL
under the base URL. Inline values, if ever needed, use the `base64:` prefix.

## zarrita `ReferenceStore` facts (from `packages/@zarrita-storage/src/ref.ts`)

- `ReferenceStore.fromUrl(url, opts)` and `fromSpec(obj, opts)`; `opts.target`
  is a fallback for entries with a null URL; `opts.fetch` can rewrite requests.
- Entry forms: `Uint8Array`, inline string, `[url]`, `[url, offset, length]`;
  `base64:` strings decoded upfront.
- **No relative URL resolution and no templating.** Entry URLs are used as is,
  so the manifest is tied to one base URL. Write absolute URLs.
- Range requests use `Range: bytes=start-end`; `composeRange` throws when a
  requested range falls outside the entry's window.
- `s3://` and `gs://` entry URIs are rewritten to HTTPS by `resolveUri`.

## Python side

fsspec's reference filesystem reads the same manifest:
`zarr.open_group("reference://", storage_options={"fo": "refs.json"}, mode="r")`
works with zarr 3.4 (tested). ngio accepts a `FsspecStore` or `FSMap` over it
with `mode="r"` (tested; default `r+` fails because the store is read-only).
Use this for the export's own tests, since the browser cannot be driven from the
cloud session reliably. kerchunk itself is not needed; the reference filesystem
lives in fsspec.

Experiments that informed the design (ngio 1.1.3, fsspec 2026.9, zarr 3.4.2):

- Building directly on a reference store works with `compressors=None` on
  level 0, but 10 MiB of pyramid pixels landed inline and the saved JSON was
  13.7 MiB (base64). Writes are in memory until `fs.save_json`. Not alive.
- A hybrid directory plus manifest works for reading, but opening the directory
  alone by path returned level 0 as all fill values silently, and the manifest
  goes stale on every write that adds keys. Hence the URL-only form and
  regeneration at publish time.

## Hosting requirements (document in README)

- CORS on the manifest, the zarr files and the raw files:
  `Access-Control-Allow-Origin`, `Range` allowed as a request header,
  `Content-Range` exposed.
- HTTP range requests on the raw files. Python's `http.server` does not support
  them; nginx, Caddy, S3, GCS and CDNs do.
- The raw files must be published at the URLs written into the manifest.
  Virtualization moves the read, it does not remove the need to serve them.

## Command shape

`jazav export-refs image.zarr --base-url https://host/data/plate.zarr/B/03/0/
[--raw-base-url ...] [--out image.refs.json]`

- Walks the group; every file becomes `[base_url + relative path]`.
- For each reference array: reads the tables, emits one key per present inner
  chunk, `[raw_url, offset, length]`, where `raw_url` is the source path
  resolved against the array URL (relative paths may climb above the zarr root,
  so a separate `--raw-base-url` or resolution against the parent of the base
  URL is needed; decide and document).
- Rewrites the level-0 `zarr.json` inline (small, and it must differ from the
  one on disk).
- One manifest per image by default; a plate manifest is possible but grows by
  roughly a hundred bytes per key.
- Optional flat kerchunk form (plain key to value dict) is the safest for
  zarrita; fsspec also reads `{"version": 1, "refs": {...}}`.

Usage with the hosted viewer:
`https://hms-dbmi.github.io/vizarr/?source=https://host/.../image.refs.json`.

## Where regeneration should live later (parked)

In ngio, on the operations that add keys (pyramid consolidation, adding a
label), if the group carries a published base URL in its attributes; or in the
publish step that syncs the directory to the host. Related to the general ngio
need for hooks that regenerate derived artifacts, e.g. RFC-9 zipped OME-Zarr.

## Alternatives considered and rejected for this step

- **Icechunk**: a versioned transactional store with native virtual chunks and
  browser readers (icechunk-js); solves a different problem and would make the
  whole dataset an Icechunk repository, unreadable by path-based tools.
- **refkv / vzip**: a reference value type bound to a ZIP file with a Rust
  `zarrify` parser core; self-described proof of concept, format version 0,
  written-once archives, its own benchmarks show 125 bytes per reference versus
  kerchunk Parquet's 3. Not an export target now; watch the overlap with RFC-9
  zipped OME-Zarr (`.ozx`), and note a standard ZIP reader would decode a
  reference payload as pixels unless guarded.
- **VirtualiZarr**: a producer of manifests; its store is read-only. Could later
  turn jazav tables into Icechunk repositories if anyone needs that.
