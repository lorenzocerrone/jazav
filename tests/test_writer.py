import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import jazav
from tests.conftest import BYTES_LE, raw_files, write_virtual_array


class TestRelativeSourcePath:
    def test_sibling_and_climb(self, tmp_path: Path) -> None:
        array_dir = tmp_path / "out.zarr" / "0"
        assert (
            jazav.relative_source_path(array_dir, tmp_path / "raw" / "a.tif")
            == "../../raw/a.tif"
        )
        assert jazav.relative_source_path(array_dir, array_dir / "a.tif") == "a.tif"

    def test_expands_user_and_resolves_symlinks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = tmp_path / "real"
        (real / "raw").mkdir(parents=True)
        (real / "raw" / "a.bin").write_bytes(b"x")
        link = tmp_path / "link"
        os.symlink(real, link, target_is_directory=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert (
            jazav.relative_source_path("~/link/out.zarr/0", link / "raw" / "a.bin")
            == "../../raw/a.bin"
        )


class TestCreateReferenceArray:
    def test_zarr_json_layout(self, tmp_path: Path) -> None:
        group = zarr.open_group(tmp_path / "out.zarr", mode="w", zarr_format=3)
        jazav.create_reference_array(
            group,
            "0",
            shape=(2, 16, 24),
            chunks=(1, 16, 24),
            dtype="uint16",
            fill_value=0,
            dimension_names=("c", "y", "x"),
            sources=["../../raw/a.bin"],
            sizes=[768],
            codecs=[BYTES_LE],
        )
        metadata = json.loads((tmp_path / "out.zarr" / "0" / "zarr.json").read_text())
        assert metadata["codecs"] == [
            {
                "name": "jazav.reference",
                "configuration": {
                    "version": 2,
                    "chunk_shape": [1, 16, 24],
                    "sources": ["../../raw/a.bin"],
                    "sizes": [768],
                    "codecs": [BYTES_LE],
                },
            }
        ]
        # The zarr chunk grid is the shard: the whole array by default.
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [2, 16, 24]
        assert metadata["chunk_key_encoding"] == {
            "name": "default",
            "configuration": {"separator": "/"},
        }
        assert metadata["dimension_names"] == ["c", "y", "x"]

    def test_explicit_shards(self, tmp_path: Path) -> None:
        group = zarr.open_group(tmp_path / "out.zarr", mode="w", zarr_format=3)
        array = jazav.create_reference_array(
            group,
            "0",
            shape=(2, 16, 24),
            chunks=(1, 8, 24),
            shards=(1, 16, 24),
            dtype="uint16",
            fill_value=0,
            dimension_names=None,
            sources=["a.bin"],
            sizes=[1536],
            codecs=[BYTES_LE],
        )
        assert array.chunks == (1, 16, 24)
        codec = jazav.reference_codec(array)
        assert codec is not None and codec.chunk_shape == (1, 8, 24)
        with pytest.raises(jazav.ReferenceFormatError, match="not a positive multiple"):
            jazav.create_reference_array(
                group,
                "1",
                shape=(2, 16, 24),
                chunks=(1, 8, 24),
                shards=(1, 12, 24),
                dtype="uint16",
                fill_value=0,
                dimension_names=None,
                sources=["a.bin"],
                sizes=[1536],
                codecs=[BYTES_LE],
            )

    def test_default_shards_follow_the_chunk_bound(self, tmp_path: Path) -> None:
        group = zarr.open_group(tmp_path / "out.zarr", mode="w", zarr_format=3)
        array = jazav.create_reference_array(
            group,
            "0",
            shape=(4, 16, 24),
            chunks=(1, 8, 24),
            max_chunks_per_shard=2,
            dtype="uint16",
            fill_value=0,
            dimension_names=None,
            sources=["a.bin"],
            sizes=[1],
            codecs=[BYTES_LE],
        )
        assert array.chunks == (1, 16, 24)

    def test_replaces_an_existing_array(self, tmp_path: Path) -> None:
        group = zarr.open_group(tmp_path / "out.zarr", mode="w", zarr_format=3)
        group.create_array("0", shape=(4, 4), chunks=(2, 2), dtype="uint8")
        array = jazav.create_reference_array(
            group,
            "0",
            shape=(4, 4),
            chunks=(4, 4),
            dtype="uint8",
            fill_value=0,
            dimension_names=None,
            sources=["a.bin"],
            sizes=[16],
            codecs=[BYTES_LE],
        )
        assert isinstance(array.metadata.codecs[0], jazav.ReferenceCodec)
        assert group["0"].chunks == (4, 4)

    def test_reference_codec_helper(self, tmp_path: Path) -> None:
        group = zarr.open_group(tmp_path / "out.zarr", mode="w", zarr_format=3)
        plain = group.create_array("p", shape=(4,), chunks=(4,), dtype="uint8")
        assert jazav.reference_codec(plain) is None


class TestWriteReferences:
    def test_writes_one_table_per_shard(self, tmp_path: Path, data: np.ndarray):
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        table = tmp_path / "out.zarr" / "0" / "c" / "0" / "0" / "0"
        assert table.stat().st_size == 40  # two inner chunks, one table
        assert jazav.decode_table(table.read_bytes(), 2) == [
            jazav.ChunkReference(0, 0, data[0].nbytes),
            None,
        ]
        assert array.nchunks_initialized == 1

    def test_groups_references_into_shards(self, tmp_path: Path, data: np.ndarray):
        write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 8, 24),
            shards=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
            references={
                (0, 0, 0): (0, 0, 384),
                (0, 1, 0): (0, 384, 384),
                (1, 1, 0): (1, 384, 384),
            },
        )
        keys = sorted(
            str(p.relative_to(tmp_path / "out.zarr" / "0"))
            for p in (tmp_path / "out.zarr" / "0" / "c").rglob("*")
            if p.is_file()
        )
        assert keys == ["c/0/0/0", "c/1/0/0"]
        second = (tmp_path / "out.zarr" / "0" / "c" / "1" / "0" / "0").read_bytes()
        assert jazav.decode_table(second, 2) == [
            None,
            jazav.ChunkReference(1, 384, 384),
        ]

    @pytest.mark.parametrize(
        ("refs", "message"),
        [
            ({(2, 0, 0): (0, 0, 8)}, "outside the chunk grid"),
            ({(0, 0): (0, 0, 8)}, "outside the chunk grid"),
            ({(0, 0, 0): (1, 0, 8)}, "source #1"),
            ({(0, 0, 0): (0, 760, 16)}, "which is only 768"),
        ],
    )
    def test_rejects_bad_references_before_writing(
        self, tmp_path: Path, refs: dict, message: str
    ) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match=message):
            write_virtual_array(
                LocalStore(tmp_path / "out.zarr"),
                shape=(2, 16, 24),
                chunks=(1, 16, 24),
                dtype="uint16",
                sources=["../../raw/plane0.bin"],
                sizes=[768],
                references=refs,
            )
        assert not (tmp_path / "out.zarr" / "0" / "c").exists()

    def test_rejects_regular_arrays(self, tmp_path: Path) -> None:
        group = zarr.open_group(tmp_path / "out.zarr", mode="w", zarr_format=3)
        array = group.create_array("0", shape=(4,), chunks=(4,), dtype="uint8")
        with pytest.raises(jazav.ReferenceFormatError, match="not a reference array"):
            jazav.write_references(array, {(0,): (0, 0, 4)})


class TestMoving:
    def _build(self, root: Path, data: np.ndarray) -> None:
        raw_files(root / "raw", data)
        write_virtual_array(
            LocalStore(root / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
            references={
                (0, 0, 0): (0, 0, data[0].nbytes),
                (1, 0, 0): (1, 0, data[1].nbytes),
            },
        )

    def test_moving_zarr_and_sources_together_still_reads(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        self._build(tmp_path / "a", data)
        shutil.move(tmp_path / "a", tmp_path / "b")
        via_group = zarr.open_group(tmp_path / "b" / "out.zarr", mode="r")["0"]
        np.testing.assert_array_equal(via_group[:], data)
        direct = zarr.open_array(tmp_path / "b" / "out.zarr" / "0", mode="r")
        np.testing.assert_array_equal(direct[:], data)

    def test_moving_the_zarr_alone_fails_naming_the_source(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        self._build(tmp_path / "a", data)
        (tmp_path / "c").mkdir()
        shutil.move(tmp_path / "a" / "out.zarr", tmp_path / "c" / "out.zarr")
        array = zarr.open_group(tmp_path / "c" / "out.zarr", mode="r")["0"]
        with pytest.raises(jazav.ReferenceFormatError, match=r"raw/plane0\.bin"):
            array[0]
        report = jazav.verify(tmp_path / "c" / "out.zarr")
        assert not report.ok
        assert {p.kind for p in report.problems} == {"missing-source"}
        assert "raw/plane0.bin" in str(report)
