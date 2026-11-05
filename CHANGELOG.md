# Changelog

## Unreleased

### Features

- Reference format version 1: every level-0 chunk of a virtual array is a 20-byte
  `(source, offset, length)` reference into an original file.
- `jazav.reference`, a read-only zarr v3 array-to-bytes codec that resolves the
  references at read time, discovered through the `zarr.codecs` entry point.
- Writer helpers `create_reference_array`, `write_references` and
  `relative_source_path`.
- `verify` and `materialize` as functions and as the `jazav` console script.
- `examples/tiff_grid.py`: virtualize a grid of uncompressed TIFF and `.npy` tiles
  into an OME-Zarr built with ngio; the reference example.
- `examples/leica_lif.py`, `examples/zeiss_czi.py`, `examples/nikon_nd2.py`:
  virtualize Leica LIF mosaics, Zeiss CZI scenes (uncompressed, ZSTD0 and ZSTD1
  subblocks) and Nikon ND2 positions, reporting files that cannot be referenced
  in place with a reason.
- `napari` pixi environment with `examples/napari_view.py` and a headless test
  that reads the virtual TIFF example through napari-ome-zarr.
