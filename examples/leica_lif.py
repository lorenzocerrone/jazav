"""Virtualize Leica LIF images (checked with liffile 2026.10.14).

What the sample showed: a LIF memory block is one contiguous, uncompressed
array in the dimension order `image.dims`; with the mosaic dimension `M`
slowest, every mosaic tile is one contiguous range of `prod(tile shape) *
itemsize` bytes at `memory_block.offset + m * tile_nbytes`. Tiles become level-0
chunks placed on the `FieldX`/`FieldY` grid of `TileScanInfo` (overlaps between
tiles are ignored: the grid is the layout). A `transpose` codec is emitted when
the on-disk order differs from the canonical `t, c, z, y, x` order; the sample
is stored `C, Z, Y, X` and needs none. One OME-Zarr per LIF image.

Not virtualizable: images whose memory block is not exactly the pixel array
(compressed or multi-frame XLIF/LOF layouts), dimensions outside `TCZYX`
(samples `S`, histograms `H`), a mosaic dimension that is not the slowest.

Run: `pixi run -e dev example-lif [FILE ...]`.
"""

from __future__ import annotations

import math
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any

import liffile
import numpy as np
from _common import Block, Input, NotVirtualizable, VirtualPlan, main

SAMPLES = Path(
    "/Users/locerr/Projects/Converters/fractal-lif-converters/tests/data/Leica-LIF/raw"
)
CANONICAL = "TCZYX"


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_")


def _spacing(coords: dict[str, Any], axis: str) -> float:
    values = coords.get(axis)
    if values is None or len(values) < 2 or values.dtype.kind not in "fi":
        return 1.0
    return abs(float(values[1] - values[0])) * 1e6  # metres -> micrometres


def describe_image(path: Path, index: int) -> tuple[VirtualPlan, np.ndarray]:
    """Plan one chunk per mosaic tile of LIF image `index`."""
    with liffile.LifFile(path, squeeze=False) as lif:
        image = lif.images[index]
        label = f"{path.name}#{image.name}"
        dims, shape, dtype = tuple(image.dims), tuple(image.shape), image.dtype
        if dtype.kind not in "uif":
            raise NotVirtualizable(
                f"{label}: dtype {dtype} is not a plain numeric type"
            )
        block = image.memory_block
        nbytes = math.prod(shape) * dtype.itemsize
        if block.offset < 0 or block.size != nbytes:
            raise NotVirtualizable(
                f"{label}: memory block ({block.size} bytes at {block.offset}) is not "
                f"one contiguous uncompressed array of {nbytes} bytes"
            )
        if "M" in dims and dims.index("M") != 0:
            raise NotVirtualizable(f"{label}: mosaic dimension M is not the slowest")
        has_mosaic = dims[0] == "M"
        tile_dims = dims[1:] if has_mosaic else dims
        tile_shape = shape[1:] if has_mosaic else shape
        unsupported = sorted(set(tile_dims) - set(CANONICAL))
        if unsupported:
            raise NotVirtualizable(f"{label}: unsupported dimension(s) {unsupported}")

        n_tiles = shape[0] if has_mosaic else 1
        if n_tiles > 1:
            scan = image.tilescan
            if scan is None or len(scan.tiles) != n_tiles:
                raise NotVirtualizable(
                    f"{label}: {n_tiles} mosaic tiles but no matching TileScanInfo"
                )
            fields = [(int(t["field_y"]), int(t["field_x"])) for t in scan.tiles]
        else:
            fields = [(0, 0)]
        if len(set(fields)) != len(fields):
            raise NotVirtualizable(f"{label}: two tiles share a FieldX/FieldY cell")

        axes = tuple(d.lower() for d in CANONICAL if d in tile_dims)
        stored = tuple(d.lower() for d in tile_dims)
        chunk = tuple(tile_shape[stored.index(a)] for a in axes)
        ny = max(fy for fy, _ in fields) + 1
        nx = max(fx for _, fx in fields) + 1
        full = tuple(
            c * (ny if a == "y" else nx if a == "x" else 1)
            for a, c in zip(axes, chunk, strict=True)
        )
        endian = "big" if dtype.byteorder == ">" else "little"
        codecs: list[dict[str, Any]] = [
            {"name": "bytes", "configuration": {"endian": endian}}
        ]
        if stored != axes:
            order = [axes.index(s) for s in stored]
            codecs.insert(0, {"name": "transpose", "configuration": {"order": order}})

        tile_nbytes = math.prod(tile_shape) * dtype.itemsize
        blocks = [
            Block(
                index=tuple(fy if a == "y" else fx if a == "x" else 0 for a in axes),
                source=path.resolve(),
                offset=block.offset + m * tile_nbytes,
                length=tile_nbytes,
            )
            for m, (fy, fx) in enumerate(fields)
        ]

        data = image.asarray()
        truth = np.zeros(full, dtype=dtype.newbyteorder("="))
        to_canonical = [stored.index(a) for a in axes]
        for m, (fy, fx) in enumerate(fields):
            tile = np.transpose(data[m] if has_mosaic else data, to_canonical)
            region = tuple(
                slice(fy * c, (fy + 1) * c)
                if a == "y"
                else slice(fx * c, (fx + 1) * c)
                if a == "x"
                else slice(None)
                for a, c in zip(axes, chunk, strict=True)
            )
            truth[region] = tile

        coords = image.coords
        names = coords.get("C")
        channel_labels = (
            [str(n) for n in names]
            if names is not None and names.dtype.kind in "US"
            else None
        )
        plan = VirtualPlan(
            shape=full,
            chunks=chunk,
            dtype=truth.dtype.name,
            codecs=codecs,
            blocks=blocks,
            axes=axes,
            pixelsize=_spacing(coords, "X"),
            z_spacing=_spacing(coords, "Z"),
            channel_labels=channel_labels,
        )
        return plan, truth


def inputs_of(path: Path) -> list[Input]:
    """One input per image series of the LIF file."""
    with liffile.LifFile(path, squeeze=False) as lif:
        return [
            (f"{path.stem}_{i}_{_safe(image.name)}", partial(describe_image, path, i))
            for i, image in enumerate(lif.images)
        ]


if __name__ == "__main__":
    sys.exit(main("leica_lif", inputs_of, sorted(SAMPLES.glob("*.lif"))))
