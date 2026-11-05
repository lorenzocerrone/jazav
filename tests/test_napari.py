"""napari-ome-zarr reads the virtual TIFF example through jazav, without a GUI."""

import sys
from pathlib import Path

import numpy as np
import pytest

napari_ome_zarr = pytest.importorskip("napari_ome_zarr")
ngio = pytest.importorskip("ngio")
pytest.importorskip("tifffile")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_examples import _load_example  # noqa: E402


def _layers(path: Path) -> list:
    reader = napari_ome_zarr.napari_get_reader(str(path))
    assert reader is not None, f"napari-ome-zarr does not recognise {path}"
    return reader(str(path))


def test_napari_reader_reads_the_virtual_image(tmp_path: Path) -> None:
    tiff_grid = _load_example("tiff_grid")
    tiff_grid.build(tmp_path)
    virtual = tmp_path / "tiff" / "virtual.zarr"
    copy = tmp_path / "tiff" / "copy.zarr"

    expected = ngio.open_ome_zarr_container(copy).get_image().get_array()
    layers = _layers(virtual)
    images = [layer for layer in layers if layer[2] == "image"]
    assert images, layers
    data, meta, _ = images[0]
    levels = data if isinstance(data, list) else [data]
    assert meta.get("multiscale", len(levels) > 1) or len(levels) == 1
    level0 = np.asarray(levels[0])
    np.testing.assert_array_equal(level0.reshape(expected.shape), expected)
    for level, path in zip(levels[1:], ("1", "2"), strict=False):
        lower = ngio.open_ome_zarr_container(copy).get_image(path=path).get_array()
        np.testing.assert_array_equal(np.asarray(level).reshape(lower.shape), lower)
