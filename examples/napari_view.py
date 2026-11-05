"""Open one or more virtual OME-Zarr images in napari.

napari reads them through napari-ome-zarr and ome-zarr-py, i.e. through
zarr-python, so level 0 is resolved from the original files by the installed
`jazav` codec without any napari-side change. Run from the `napari` pixi
environment: `pixi run -e napari napari-tiff` builds the TIFF example and opens
it, or `pixi run -e napari python examples/napari_view.py PATH [PATH ...]`.
The script only adds a printout of the layers; `napari --plugin napari-ome-zarr
PATH` opens the same image through the same plugin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import napari


def main(argv: list[str] | None = None) -> int:
    """Open each path as an OME-Zarr layer set and start the napari event loop."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="OME-Zarr image roots")
    args = parser.parse_args(argv)
    viewer = napari.Viewer(title="jazav virtual OME-Zarr")
    for path in args.paths:
        if not path.exists():
            print(f"skipping {path}: not found", file=sys.stderr)
            continue
        layers = viewer.open(str(path), plugin="napari-ome-zarr")
        for layer in layers:
            shape = layer.data[0].shape if layer.multiscale else layer.data.shape
            print(f"{path}: {layer.name} {shape}")
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
