# Changelog

## Unreleased

### Features

- `samples/tiff_grid`: a committed virtual OME-Zarr (2x2 grid of uncompressed
  TIFF tiles, 64 KiB in total) that `tests/test_sample.py` reads back, so a
  change that breaks the on-disk format fails the test suite.

- Reference format version 2: level 0 of a virtual array is stored as shards,
  each holding a table of 20-byte `(source, offset, length)` entries, one per
  inner chunk, into the original files. Replaces format 1 (one 20-byte chunk file
  per chunk), which needed two dependent store requests per chunk; format-1
  arrays are refused and must be rebuilt.
- `jazav.reference`, a read-only zarr v3 array-to-bytes codec that reads a
  shard's table once, coalesces adjacent byte ranges per source and resolves
  the references at read time, discovered through the `zarr.codecs` entry point.
- Writer helpers `create_reference_array` (with `shards=` and a default of one
  shard per array up to 16,384 chunks), `write_references`, `reference_codec`
  and `relative_source_path`.
- `docs/roadmap/`: context files for the reference-table codec, the kerchunk
  manifest export and the vizarr store proof of concept.
- `verify` and `materialize` as functions and as the `jazav` console script.
- `examples/tiff_grid.py`: virtualize a grid of uncompressed TIFF and `.npy` tiles
  into an OME-Zarr built with ngio; the reference example.
- `examples/leica_lif.py`, `examples/zeiss_czi.py`, `examples/nikon_nd2.py`:
  virtualize Leica LIF mosaics, Zeiss CZI scenes (uncompressed, ZSTD0 and ZSTD1
  subblocks) and Nikon ND2 positions, reporting files that cannot be referenced
  in place with a reason.
- `napari` pixi environment with `examples/napari_view.py` and a headless test
  that reads the virtual TIFF example through napari-ome-zarr.
