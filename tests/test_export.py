"""`export_references` writes a kerchunk manifest that fsspec reads back."""

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import jazav
from jazav._cli import main
from jazav._export import default_manifest_path, normalize_base_url
from tests.conftest import raw_files, write_virtual_array

fsspec = pytest.importorskip("fsspec")


@pytest.fixture
def image(tmp_path: Path, data: np.ndarray) -> Path:
    """A group with a virtual level 0 (one chunk missing) and a concrete level 1."""
    raw_files(tmp_path / "raw", data)
    out = tmp_path / "out.zarr"
    write_virtual_array(
        LocalStore(out),
        shape=data.shape,
        chunks=(1, 8, 24),
        dtype="uint16",
        sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
        references={
            (0, 0, 0): (0, 0, 384),
            (0, 1, 0): (0, 384, 384),
            (1, 0, 0): (1, 0, 384),
        },
    )
    group = zarr.open_group(out, mode="r+")
    group.attrs["ome"] = {"version": "0.5", "note": "test"}
    level1 = group.create_array(
        "1", shape=(2, 8, 12), chunks=(1, 8, 12), dtype="uint16", compressors=None
    )
    level1[:] = data[:, ::2, ::2]
    return out


def _open_manifest(manifest: dict) -> zarr.Group:
    fs = fsspec.filesystem("reference", fo=manifest, asynchronous=True)
    from zarr.storage import FsspecStore

    return zarr.open_group(FsspecStore(fs, path="", read_only=True), mode="r")


class TestExportReferences:
    def test_manifest_layout(self, image: Path, data: np.ndarray) -> None:
        base = image.as_uri() + "/"
        manifest = jazav.export_references(image, base)
        assert manifest["version"] == 1
        refs = manifest["refs"]

        root = json.loads(refs["zarr.json"])  # metadata is inlined
        assert root["attributes"]["ome"]["note"] == "test"

        level0 = json.loads(refs["0/zarr.json"])  # rewritten for plain readers
        assert level0["codecs"] == [
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
        assert level0["chunk_grid"]["configuration"]["chunk_shape"] == [1, 8, 24]

        raw0 = (image.parent / "raw" / "plane0.bin").as_uri()
        raw1 = (image.parent / "raw" / "plane1.bin").as_uri()
        assert refs["0/c/0/0/0"] == [raw0, 0, 384]
        assert refs["0/c/0/1/0"] == [raw0, 384, 384]
        assert refs["0/c/1/0/0"] == [raw1, 0, 384]
        assert "0/c/1/1/0" not in refs  # missing inner chunk: fill value

        level1_chunk = image / "1" / "c" / "0" / "0" / "0"
        assert refs["1/c/0/0/0"] == [
            base + "1/c/0/0/0",
            0,
            level1_chunk.stat().st_size,
        ]
        assert json.loads(refs["1/zarr.json"])["shape"] == [2, 8, 12]
        # No table keys and no reference codec leak into the manifest.
        assert not any(key.startswith("0/c/0/0/0/") for key in refs)
        assert "jazav.reference" not in json.dumps(manifest)

    def test_fsspec_reads_the_manifest(self, image: Path, data: np.ndarray) -> None:
        manifest = jazav.export_references(image, image.as_uri())
        group = _open_manifest(manifest)
        expected = data.copy()
        expected[1, 8:] = 0
        np.testing.assert_array_equal(group["0"][:], expected)
        np.testing.assert_array_equal(group["1"][:], data[:, ::2, ::2])
        assert group.attrs["ome"]["note"] == "test"

    def test_array_target_and_out_file(self, image: Path, data: np.ndarray) -> None:
        out = image.parent / "level0.refs.json"
        manifest = jazav.export_references(image / "0", image.as_uri() + "/0", out=out)
        assert json.loads(out.read_text()) == manifest
        assert sorted(manifest["refs"]) == [
            "c/0/0/0",
            "c/0/1/0",
            "c/1/0/0",
            "zarr.json",
        ]
        fs = fsspec.filesystem("reference", fo=manifest, asynchronous=True)
        from zarr.storage import FsspecStore

        array = zarr.open_array(FsspecStore(fs, path="", read_only=True), mode="r")
        np.testing.assert_array_equal(array[0], data[0])

    def test_rejects_relative_base_url(self, image: Path) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="must be absolute"):
            jazav.export_references(image, "data/out.zarr/")
        assert normalize_base_url("https://h/x") == "https://h/x/"
        assert normalize_base_url("file:///x/") == "file:///x/"

    def test_rejects_bad_table(self, image: Path) -> None:
        (image / "0" / "c" / "0" / "0" / "0").write_bytes(b"\0" * 8)
        with pytest.raises(jazav.ReferenceFormatError, match="exactly 80 bytes"):
            jazav.export_references(image, image.as_uri())


class TestCli:
    def test_default_output_path(self) -> None:
        assert default_manifest_path("a/b/img.zarr") == Path("a/b/img.refs.json")
        assert default_manifest_path("a/img") == Path("a/img.refs.json")

    def test_export_refs(self, image: Path, capsys: pytest.CaptureFixture) -> None:
        assert main(["export-refs", str(image), "--base-url", image.as_uri()]) == 0
        out = image.parent / "out.refs.json"
        assert out.is_file()
        assert f"wrote {out}" in capsys.readouterr().out
        assert main(["export-refs", str(image), "--base-url", "no-scheme/"]) == 1
        assert "must be absolute" in capsys.readouterr().err
