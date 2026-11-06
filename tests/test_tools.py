import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore

import jazav
from jazav import _tools
from jazav._cli import main
from tests.conftest import raw_files, write_raw_shard, write_table, write_virtual_array


@pytest.fixture
def virtual(tmp_path: Path, data: np.ndarray) -> Path:
    raw_files(tmp_path / "raw", data)
    write_virtual_array(
        LocalStore(tmp_path / "out.zarr"),
        shape=data.shape,
        chunks=(1, 16, 24),
        dtype="uint16",
        sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
        references={
            (0, 0, 0): (0, 0, data[0].nbytes),
            (1, 0, 0): (1, 0, data[1].nbytes),
        },
    )
    return tmp_path / "out.zarr"


TABLE = Path("0") / "c" / "0" / "0" / "0"
"""Key of the single shard of the `virtual` fixture (two inner chunks)."""


class TestVerify:
    def test_ok_report(self, virtual: Path) -> None:
        report = jazav.verify(virtual)
        assert report.ok
        assert report.arrays == ["0"]
        assert (report.n_sources, report.n_shards, report.n_chunks) == (2, 1, 2)
        assert str(report).startswith("OK: 1 reference array(s)")
        assert "1 shard(s), 2 chunk(s)" in str(report)

    def test_accepts_group_and_array_objects(self, virtual: Path) -> None:
        group = zarr.open_group(virtual, mode="r")
        assert jazav.verify(group).ok
        assert jazav.verify(group["0"]).ok
        assert jazav.verify(virtual / "0").arrays == ["/"]

    def test_regular_arrays_are_ignored(self, tmp_path: Path) -> None:
        group = zarr.open_group(tmp_path / "plain.zarr", mode="w", zarr_format=3)
        group.create_array("0", shape=(4,), chunks=(4,), dtype="uint8")
        report = jazav.verify(tmp_path / "plain.zarr")
        assert report.ok and report.arrays == []

    def test_missing_source(self, virtual: Path) -> None:
        (virtual.parent / "raw" / "plane1.bin").unlink()
        report = jazav.verify(virtual)
        assert [p.kind for p in report.problems] == ["missing-source"]
        assert "raw/plane1.bin" in report.problems[0].message

    def test_size_mismatch(self, virtual: Path) -> None:
        source = virtual.parent / "raw" / "plane0.bin"
        source.write_bytes(source.read_bytes()[:-1])
        report = jazav.verify(virtual)
        assert [p.kind for p in report.problems] == ["size-mismatch"]
        assert "767 bytes, expected 768" in report.problems[0].message

    def test_bad_table(self, virtual: Path) -> None:
        (virtual / TABLE).write_bytes(b"\0" * 16)
        report = jazav.verify(virtual)
        assert [p.kind for p in report.problems] == ["bad-table"]
        assert "exactly 40 bytes, got 16" in report.problems[0].message
        assert report.n_chunks == 0

    def test_out_of_range_entries(self, virtual: Path) -> None:
        array = zarr.open_array(virtual / "0", mode="r+")
        write_table(array, {(0, 0, 0): (0, 0, 768), (1, 0, 0): (1, 4, 768)})
        report = jazav.verify(virtual)
        assert [p.kind for p in report.problems] == ["out-of-range"]
        assert "inner chunk (1, 0, 0)" in report.problems[0].message
        write_table(array, {(1, 0, 0): (9, 0, 768)})
        assert "source #9" in str(jazav.verify(virtual))

    def test_deep_decodes_every_chunk(self, virtual: Path) -> None:
        array = zarr.open_array(virtual / "0", mode="r+")
        write_table(array, {(0, 0, 0): (0, 0, 768), (1, 0, 0): (1, 0, 760)})
        assert jazav.verify(virtual).ok
        report = jazav.verify(virtual, deep=True)
        assert [p.kind for p in report.problems] == ["decode-error"]

    def test_memory_store_climb_is_a_problem_not_an_exception(
        self, data: np.ndarray
    ) -> None:
        array = write_virtual_array(
            MemoryStore(),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        report = jazav.verify(array)
        assert [p.kind for p in report.problems] == ["missing-source"]
        assert "MemoryStore" in report.problems[0].message


class TestReadRegions:
    def test_whole_shard_when_it_fits(self, virtual: Path) -> None:
        array = zarr.open_array(virtual / "0", mode="r")
        assert _tools._read_regions(array) == [
            (slice(0, 2), slice(0, 16), slice(0, 24))
        ]

    def test_shards_are_split_on_inner_chunks(
        self, virtual: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        array = zarr.open_array(virtual / "0", mode="r")
        monkeypatch.setattr(_tools, "READ_BLOCK_BYTES", 16 * 24 * 2)
        regions = _tools._read_regions(array)
        assert regions == [
            (slice(0, 1), slice(0, 16), slice(0, 24)),
            (slice(1, 2), slice(0, 16), slice(0, 24)),
        ]


class TestMaterialize:
    def test_rewrites_level_in_place(self, virtual: Path, data: np.ndarray) -> None:
        rewritten = jazav.materialize(virtual)
        assert rewritten == [str((virtual / "0").resolve())]
        metadata = json.loads((virtual / "0" / "zarr.json").read_text())
        assert all(c["name"] != "jazav.reference" for c in metadata["codecs"])
        assert sorted(p.name for p in virtual.iterdir()) == ["0", "zarr.json"]
        # The sources are no longer needed.
        for source in (virtual.parent / "raw").iterdir():
            source.unlink()
        array = zarr.open_group(virtual, mode="r")["0"]
        np.testing.assert_array_equal(array[:], data)
        assert array.chunks == (1, 16, 24)  # the inner chunks, not the shard
        assert jazav.verify(virtual).arrays == []
        assert jazav.materialize(virtual) == []

    def test_custom_chunks_and_fill_value(self, virtual: Path, data: np.ndarray):
        # Drop one inner chunk so the fill value shows up in the materialized array.
        array = zarr.open_array(virtual / "0", mode="r+")
        write_table(array, {(0, 0, 0): (0, 0, data[0].nbytes)})
        jazav.materialize(virtual, chunks=(2, 8, 8))
        array = zarr.open_array(virtual / "0", mode="r")
        assert array.chunks == (2, 8, 8)
        np.testing.assert_array_equal(array[0], data[0])
        np.testing.assert_array_equal(array[1], 0)

    def test_refuses_when_verify_fails(self, virtual: Path) -> None:
        (virtual.parent / "raw" / "plane1.bin").unlink()
        with pytest.raises(jazav.ReferenceFormatError, match="missing-source"):
            jazav.materialize(virtual)
        assert (virtual / TABLE).stat().st_size == 40

    def test_refuses_non_local_stores_and_bare_arrays(
        self, virtual: Path, data: np.ndarray
    ) -> None:
        store = MemoryStore()
        write_virtual_array(
            store,
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        with pytest.raises(jazav.ReferenceFormatError, match="LocalStore"):
            jazav.materialize(zarr.open_group(store, mode="r+"))
        with pytest.raises(jazav.ReferenceFormatError, match="needs a zarr group"):
            jazav.materialize(virtual / "0")


class TestCli:
    def test_verify_exit_codes(self, virtual: Path, capsys: pytest.CaptureFixture):
        assert main(["verify", str(virtual)]) == 0
        assert "OK" in capsys.readouterr().out
        (virtual.parent / "raw" / "plane0.bin").unlink()
        assert main(["verify", "--deep", str(virtual)]) == 1
        assert "missing-source" in capsys.readouterr().out

    def test_materialize(self, virtual: Path, capsys: pytest.CaptureFixture) -> None:
        assert main(["materialize", str(virtual)]) == 0
        assert "materialized" in capsys.readouterr().out
        assert main(["materialize", str(virtual)]) == 0
        assert "nothing to materialize" in capsys.readouterr().out
        assert main(["materialize", str(virtual / "0")]) == 1
        assert "error:" in capsys.readouterr().err


def test_write_raw_shard_helper_targets_the_shard_key(virtual: Path) -> None:
    array = zarr.open_array(virtual / "0", mode="r+")
    write_raw_shard(array, (0, 0, 0), b"x")
    assert (virtual / TABLE).read_bytes() == b"x"
