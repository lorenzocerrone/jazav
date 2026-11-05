"""Virtualize a 2x2 grid of uncompressed TIFF tiles, and the same grid as `.npy`.

This is the reference example. Only contiguous, uncompressed TIFF series qualify:
`series.dataoffset` is set and `page.compression == 1`; the byte range starts at
`dataoffset` and spans the whole series, the byte order comes from
`tif.byteorder`. For `.npy` the pixels start right after the header (`f.tell()`
after reading it) and must be in C order. A zlib-compressed copy of the grid is
included to show a "no" row.

Run: `pixi run -e dev example-tiff` (or `python examples/tiff_grid.py --out DIR`).
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import tifffile
from _common import Block, NotVirtualizable, Row, VirtualPlan, run

TILE = 64
GRID = (2, 2)
AXES = ("c", "z", "y", "x")
_TILE_NAME = re.compile(r"tile_y(\d+)_x(\d+)\.(tif|tiff|npy)$")


@dataclass(frozen=True)
class RawBlock:
    """Where one file's pixels live and how to decode them."""

    offset: int
    length: int
    shape: tuple[int, ...]
    dtype: np.dtype
    endian: str


def make_tiles(root: Path) -> list[Path]:
    """Write the grid as uncompressed TIFF, `.npy` and zlib TIFF; return the dirs."""
    rng = np.random.default_rng(0)
    full = rng.integers(0, 2**16, (GRID[0] * TILE, GRID[1] * TILE), dtype=np.uint16)
    dirs = {kind: root / kind for kind in ("tiff", "npy", "tiff_zlib")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    for iy in range(GRID[0]):
        for ix in range(GRID[1]):
            tile = full[iy * TILE : (iy + 1) * TILE, ix * TILE : (ix + 1) * TILE]
            name = f"tile_y{iy}_x{ix}"
            tifffile.imwrite(dirs["tiff"] / f"{name}.tif", tile)
            np.save(dirs["npy"] / f"{name}.npy", tile)
            tifffile.imwrite(
                dirs["tiff_zlib"] / f"{name}.tif", tile, compression="zlib"
            )
    return list(dirs.values())


def describe_tiff(path: Path) -> RawBlock:
    """Byte range of an uncompressed, contiguous TIFF series."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        page = series.keyframe
        if page.compression != 1:
            raise NotVirtualizable(
                f"{path.name}: compressed TIFF ({page.compression!s})"
            )
        if series.dataoffset is None:
            raise NotVirtualizable(
                f"{path.name}: TIFF pixels are not one contiguous block"
            )
        dtype = np.dtype(series.dtype)
        if dtype.kind not in "uif":
            raise NotVirtualizable(
                f"{path.name}: dtype {dtype} is not a plain numeric type"
            )
        return RawBlock(
            offset=int(series.dataoffset),
            length=math.prod(series.shape) * dtype.itemsize,
            shape=tuple(int(s) for s in series.shape),
            dtype=dtype.newbyteorder("="),
            endian="big" if tif.byteorder == ">" else "little",
        )


def describe_npy(path: Path) -> RawBlock:
    """Byte range of the pixels of a C-ordered `.npy` file."""
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise NotVirtualizable(f"{path.name}: unsupported .npy format {version}")
        offset = f.tell()
    if fortran:
        raise NotVirtualizable(f"{path.name}: Fortran-ordered .npy")
    if dtype.hasobject or dtype.fields is not None:
        raise NotVirtualizable(
            f"{path.name}: dtype {dtype} is not a plain numeric type"
        )
    endian = (
        "big"
        if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big")
        else "little"
    )
    return RawBlock(
        offset=offset,
        length=math.prod(shape) * dtype.itemsize,
        shape=tuple(int(s) for s in shape),
        dtype=dtype.newbyteorder("="),
        endian=endian,
    )


def describe_grid(directory: Path) -> tuple[VirtualPlan, np.ndarray]:
    """Plan one chunk per tile of a `tile_y{i}_x{j}.{tif,npy}` directory."""
    blocks: list[Block] = []
    tiles: dict[tuple[int, int], np.ndarray] = {}
    first: RawBlock | None = None
    for path in sorted(directory.iterdir()):
        match = _TILE_NAME.search(path.name)
        if match is None:
            continue
        iy, ix = int(match[1]), int(match[2])
        raw = describe_npy(path) if path.suffix == ".npy" else describe_tiff(path)
        if first is None:
            first = raw
        elif (raw.shape, raw.dtype, raw.endian) != (
            first.shape,
            first.dtype,
            first.endian,
        ):
            raise NotVirtualizable(
                f"{path.name} differs from the first tile in shape, dtype or byte order"
            )
        blocks.append(Block((0, 0, iy, ix), path.resolve(), raw.offset, raw.length))
        tiles[(iy, ix)] = (
            np.load(path) if path.suffix == ".npy" else tifffile.imread(path)
        )
    if first is None:
        raise NotVirtualizable(f"{directory.name}: no tiles found")
    ny = max(iy for iy, _ in tiles) + 1
    nx = max(ix for _, ix in tiles) + 1
    th, tw = first.shape
    truth = np.zeros((1, 1, ny * th, nx * tw), dtype=first.dtype)
    for (iy, ix), tile in tiles.items():
        truth[0, 0, iy * th : (iy + 1) * th, ix * tw : (ix + 1) * tw] = tile
    plan = VirtualPlan(
        shape=truth.shape,
        chunks=(1, 1, th, tw),
        dtype=first.dtype.name,
        codecs=[{"name": "bytes", "configuration": {"endian": first.endian}}],
        blocks=blocks,
        axes=AXES,
        pixelsize=0.65,
        channel_labels=["DAPI"],
    )
    return plan, truth


def build(out_dir: Path) -> list[Row]:
    """Write the tiles under `out_dir/raw` and virtualize them under `out_dir`."""
    dirs = make_tiles(out_dir / "raw")
    return run([(d.name, partial(describe_grid, d)) for d in dirs], out_dir)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from _common import print_table

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("_examples_out") / "tiff_grid")
    args = parser.parse_args(argv)
    print_table(build(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
