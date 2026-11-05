"""The committed sample must keep reading: it pins the on-disk format.

`samples/tiff_grid` was built by `examples/tiff_grid.py` (seed 0); the tiles are
uncompressed TIFFs whose pixel block the level-0 chunks reference.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr

import jazav

SAMPLE = (
    Path(__file__).parent.parent / "samples" / "tiff_grid" / "tiff" / "virtual.zarr"
)
TILE, GRID = 64, (2, 2)


@pytest.fixture(scope="module")
def truth() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 2**16, (GRID[0] * TILE, GRID[1] * TILE), dtype=np.uint16)


def test_sample_is_committed_with_its_sources():
    assert SAMPLE.is_dir(), f"missing sample at {SAMPLE}"
    codec = jazav.ReferenceCodec.from_dict(
        zarr.open_array(SAMPLE / "0", mode="r").metadata.codecs[0].to_dict()
    )
    assert codec.version == 1
    assert codec.sources == tuple(
        f"../../../raw/tiff/tile_y{iy}_x{ix}.tif" for iy in range(2) for ix in range(2)
    )
    for src, size in zip(codec.sources, codec.sizes, strict=True):
        assert (SAMPLE / "0" / src).resolve().stat().st_size == size


def test_sample_verifies():
    report = jazav.verify(SAMPLE, deep=True)
    assert report.ok, str(report)
    assert (report.n_sources, report.n_chunks) == (4, 4)


def test_sample_level0_reads_the_tiles(truth):
    level0 = zarr.open_array(SAMPLE / "0", mode="r")
    assert level0.shape == (1, 1, 128, 128)
    np.testing.assert_array_equal(level0[0, 0], truth)


def test_sample_pyramid_levels_are_real_pixels():
    group = zarr.open_group(SAMPLE, mode="r")
    for level in ("1", "2"):
        array = group[level]
        assert isinstance(array, zarr.Array)
        names = [c.to_dict()["name"] for c in array.metadata.codecs]
        assert jazav.CODEC_NAME not in names
        assert array[:].max() > 0
