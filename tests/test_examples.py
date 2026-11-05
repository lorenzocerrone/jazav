"""End-to-end check of `examples/tiff_grid.py`, the reference example."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import jazav

pytest.importorskip("tifffile")
ngio = pytest.importorskip("ngio")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> ModuleType:
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_tiff_grid_end_to_end(tmp_path: Path) -> None:
    tiff_grid = _load_example("tiff_grid")
    rows = {row.file: row for row in tiff_grid.build(tmp_path)}
    assert rows["tiff"].virtualizable and rows["npy"].virtualizable
    assert not rows["tiff_zlib"].virtualizable
    assert "compressed" in rows["tiff_zlib"].reason
    assert rows["tiff"].virtual_size < rows["tiff"].copy_size

    virtual = tmp_path / "tiff" / "virtual.zarr"
    copy = tmp_path / "tiff" / "copy.zarr"
    assert jazav.verify(virtual).ok
    expected = ngio.open_ome_zarr_container(copy).get_image().get_array()
    container = ngio.open_ome_zarr_container(virtual)
    np.testing.assert_array_equal(container.get_image().get_array(), expected)
    assert set(container.list_tables()) >= {"FOV_ROI_table", "well_ROI_table"}
    assert len(container.get_table("FOV_ROI_table").rois()) == 4

    # Materialize, drop the sources, and the image still reads unchanged.
    assert jazav.materialize(virtual) == [str((virtual / "0").resolve())]
    shutil.rmtree(tmp_path / "raw")
    container = ngio.open_ome_zarr_container(virtual)
    np.testing.assert_array_equal(container.get_image().get_array(), expected)
    for path in container.level_paths[1:]:
        np.testing.assert_array_equal(
            container.get_image(path=path).get_array(),
            ngio.open_ome_zarr_container(copy).get_image(path=path).get_array(),
        )
    assert jazav.verify(virtual).arrays == []
