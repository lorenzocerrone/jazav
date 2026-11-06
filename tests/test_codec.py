import pickle
import re
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.core.sync import sync
from zarr.storage import LocalStore, MemoryStore, WrapperStore

import jazav
from jazav import _codec, _storeio
from tests.conftest import (
    BYTES_LE,
    raw_files,
    write_raw_shard,
    write_table,
    write_virtual_array,
)


def make_codec(**overrides):
    kwargs = {
        "chunk_shape": (1, 16, 24),
        "sources": ["../../raw/a.bin"],
        "sizes": [768],
        "codecs": [BYTES_LE],
    }
    kwargs.update(overrides)
    return jazav.ReferenceCodec(**kwargs)


class TestReferenceCodecMetadata:
    def test_to_dict_matches_documented_layout(self) -> None:
        codec = make_codec()
        assert codec.to_dict() == {
            "name": "jazav.reference",
            "configuration": {
                "version": 2,
                "chunk_shape": [1, 16, 24],
                "sources": ["../../raw/a.bin"],
                "sizes": [768],
                "codecs": [BYTES_LE],
            },
        }
        assert jazav.ReferenceCodec.from_dict(codec.to_dict()) == codec

    def test_from_dict_requires_every_field(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="missing 'sizes'"):
            jazav.ReferenceCodec.from_dict(
                {
                    "name": "jazav.reference",
                    "configuration": {
                        "chunk_shape": [1],
                        "sources": ["a"],
                        "codecs": [BYTES_LE],
                    },
                }
            )
        with pytest.raises(jazav.ReferenceFormatError, match="missing 'chunk_shape'"):
            jazav.ReferenceCodec.from_dict(
                {
                    "name": "jazav.reference",
                    "configuration": {"sources": ["a"], "sizes": [1], "codecs": []},
                }
            )

    def test_rejects_bad_configuration(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="at least one source"):
            make_codec(sources=[], sizes=[])
        with pytest.raises(jazav.ReferenceFormatError, match="duplicate"):
            make_codec(sources=["a", "a"], sizes=[1, 1])
        with pytest.raises(jazav.ReferenceFormatError, match="relative"):
            make_codec(sources=["/abs"], sizes=[1])
        with pytest.raises(jazav.ReferenceFormatError, match="sizes has 2"):
            make_codec(sources=["a"], sizes=[1, 2])
        with pytest.raises(jazav.ReferenceFormatError, match="non-negative"):
            make_codec(sources=["a"], sizes=[-1])
        with pytest.raises(jazav.ReferenceFormatError, match="chunk_shape"):
            make_codec(chunk_shape=(0, 16, 24))
        with pytest.raises(jazav.ReferenceFormatError, match="chunk_shape"):
            make_codec(chunk_shape=())

    def test_format_1_is_refused_naming_installed_jazav(self) -> None:
        with pytest.raises(
            jazav.ReferenceFormatError,
            match=re.escape(f"version 1: jazav {jazav.__version__} reads version 2"),
        ):
            make_codec(version=1)

    def test_pickle_roundtrip(self) -> None:
        codec = make_codec(sources=["a.bin", "b.bin"], sizes=[1, 2])
        assert pickle.loads(pickle.dumps(codec)) == codec

    def test_validate_checks_the_shard_grid(self) -> None:
        codec = make_codec(chunk_shape=(1, 8, 24))
        with pytest.raises(jazav.ReferenceFormatError, match="rank"):
            codec.validate(shape=(16, 24), dtype=None, chunk_grid=None)

        class Grid:
            chunk_shape = (1, 12, 24)

        with pytest.raises(jazav.ReferenceFormatError, match="not a positive multiple"):
            codec.validate(shape=(2, 16, 24), dtype=None, chunk_grid=Grid())


class TestReferenceCodecReads:
    def test_sources_inside_the_store(self, tmp_path: Path, data: np.ndarray) -> None:
        # The sources are written after the group: `mode="w"` truncates the store.
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../raw/plane0.bin", "../raw/plane1.bin"],
            references={
                (0, 0, 0): (0, 0, data[0].nbytes),
                (1, 0, 0): (1, 0, data[1].nbytes),
            },
        )
        raw_files(tmp_path / "out.zarr" / "raw", data)
        np.testing.assert_array_equal(array[:], data)
        np.testing.assert_array_equal(array[1, 3:7, 10:12], data[1, 3:7, 10:12])

    def test_sources_above_the_store_root(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
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
        via_group = zarr.open_group(tmp_path / "out.zarr", mode="r")["0"]
        np.testing.assert_array_equal(via_group[:], data)
        direct = zarr.open_array(tmp_path / "out.zarr" / "0", mode="r")
        np.testing.assert_array_equal(direct[:], data)

    def test_missing_entry_reads_fill_value(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
            fill_value=7,
        )
        np.testing.assert_array_equal(array[0], data[0])
        np.testing.assert_array_equal(array[1], np.full_like(data[1], 7))

    def test_missing_shard_reads_fill_value(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 8, 24),
            shards=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, y, 0): (0, y * 384, 384) for y in range(2)},
            fill_value=3,
        )
        assert sorted(
            p.name for p in (tmp_path / "out.zarr" / "0" / "c").iterdir()
        ) == ["0"]
        np.testing.assert_array_equal(array[0], data[0])
        np.testing.assert_array_equal(array[1], np.full_like(data[1], 3))

    def test_many_inner_chunks_per_shard_and_edge_shards(self, tmp_path: Path) -> None:
        # 5 x 7 inner chunks of (2, 3) over a (9, 20) array: ragged edges on both
        # axes, shards of 2 x 2 inner chunks, so the last shards extend past the
        # array and the table is still full size.
        rng = np.random.default_rng(1)
        full = rng.integers(0, 2**16, (10, 21), dtype="<u2")  # padded to the grid
        (tmp_path / "raw").mkdir()
        blocks = {}
        offset = 0
        payload = bytearray()
        for iy in range(5):
            for ix in range(7):
                block = full[iy * 2 : iy * 2 + 2, ix * 3 : ix * 3 + 3]
                blocks[(iy, ix)] = (0, offset, block.nbytes)
                payload += block.tobytes()
                offset += block.nbytes
        (tmp_path / "raw" / "blocks.bin").write_bytes(payload)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=(9, 20),
            chunks=(2, 3),
            shards=(4, 6),
            dtype="uint16",
            sources=["../../raw/blocks.bin"],
            references=blocks,
        )
        assert array.chunks == (4, 6)
        np.testing.assert_array_equal(array[:], full[:9, :20])
        np.testing.assert_array_equal(array[3:8, 5:19], full[3:8, 5:19])
        np.testing.assert_array_equal(array[8], full[8, :20])
        np.testing.assert_array_equal(
            array[[0, 8], :][:, [1, 19]], full[[0, 8]][:, [1, 19]]
        )
        report = jazav.verify(tmp_path / "out.zarr", deep=True)
        assert report.ok, str(report)
        assert (report.n_shards, report.n_chunks) == (12, 35)

    def test_adjacent_blocks_are_read_with_one_range_request(
        self, tmp_path: Path, data: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 4, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
            references={
                (c, y, 0): (c, y * 192, 192) for c in range(2) for y in range(4)
            },
        )
        calls: list[tuple[str, int, int]] = []
        original = _storeio.read_range

        async def counting(store, key, prototype, offset, length):  # type: ignore[no-untyped-def]
            calls.append((key, offset, length))
            return await original(store, key, prototype, offset, length)

        monkeypatch.setattr(_storeio, "read_range", counting)
        np.testing.assert_array_equal(array[:], data)
        assert sorted(calls) == [("raw/plane0.bin", 0, 768), ("raw/plane1.bin", 0, 768)]
        calls.clear()
        np.testing.assert_array_equal(array[1, 9:14], data[1, 9:14])
        assert calls == [("raw/plane1.bin", 384, 384)]

    def test_far_apart_blocks_are_not_coalesced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "raw").mkdir()
        gap = b"\0" * 32
        a = np.arange(16, dtype="<u2")
        b = np.arange(16, 32, dtype="<u2")
        (tmp_path / "raw" / "f.bin").write_bytes(a.tobytes() + gap + b.tobytes())
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=(2, 16),
            chunks=(1, 16),
            dtype="uint16",
            sources=["../../raw/f.bin"],
            references={(0, 0): (0, 0, 32), (1, 0): (0, 64, 32)},
        )
        calls: list[tuple[int, int]] = []
        original = _storeio.read_range

        async def counting(store, key, prototype, offset, length):  # type: ignore[no-untyped-def]
            calls.append((offset, length))
            return await original(store, key, prototype, offset, length)

        monkeypatch.setattr(_storeio, "read_range", counting)
        monkeypatch.setattr(_codec, "MAX_GAP_BYTES", 16)
        np.testing.assert_array_equal(array[:], np.stack([a, b]))
        assert sorted(calls) == [(0, 32), (64, 32)]

    def test_table_is_read_once_per_shard(
        self, tmp_path: Path, data: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 8, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
            references={
                (c, y, 0): (c, y * 384, 384) for c in range(2) for y in range(2)
            },
        )
        decoded: list[int] = []
        original = _codec.decode_table

        def counting(payload, n):  # type: ignore[no-untyped-def]
            decoded.append(n)
            return original(payload, n)

        monkeypatch.setattr(_codec, "decode_table", counting)
        for _ in range(3):
            np.testing.assert_array_equal(array[0, 2:5], data[0, 2:5])
            np.testing.assert_array_equal(array[1], data[1])
        assert decoded == [4]

    def test_corrupt_table_raises(self, tmp_path: Path, data: np.ndarray) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        write_raw_shard(array, (0, 0, 0), b"\0" * 16)
        with pytest.raises(
            jazav.ReferenceFormatError, match="exactly 40 bytes, got 16"
        ):
            array[0]
        # A failed table fetch is not cached: fixing the table makes reads work.
        write_table(array, {(0, 0, 0): (0, 0, data[0].nbytes)})
        np.testing.assert_array_equal(array[0], data[0])

    def test_source_index_out_of_range_raises(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={},
        )
        write_table(array, {(0, 0, 0): (5, 0, data[0].nbytes)})
        with pytest.raises(jazav.ReferenceFormatError, match="source #5"):
            array[0]

    def test_reference_beyond_declared_size_raises(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={},
        )
        write_table(array, {(0, 0, 0): (0, 2, data[0].nbytes)})
        with pytest.raises(jazav.ReferenceFormatError, match="which is only"):
            array[0]

    def test_wrong_length_raises(self, tmp_path: Path, data: np.ndarray) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes - 2)},
        )
        with pytest.raises(Exception, match=r"shape|short read"):
            array[0]

    def test_missing_source_file_raises(self, tmp_path: Path, data: np.ndarray) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/missing.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        with pytest.raises(jazav.ReferenceFormatError, match="moved or deleted"):
            array[0]

    def test_truncated_source_is_refused(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes - 4)},
        )
        source = tmp_path / "raw" / "plane0.bin"
        source.write_bytes(source.read_bytes()[:-2])
        with pytest.raises(
            jazav.ReferenceFormatError, match="was truncated or replaced"
        ):
            array[0]

    def test_replaced_source_of_other_size_is_refused(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        source = tmp_path / "raw" / "plane0.bin"
        source.write_bytes(source.read_bytes() + b"\0")
        with pytest.raises(jazav.ReferenceFormatError, match="is 769 bytes"):
            array[0]

    def test_size_check_runs_once_per_source(
        self, tmp_path: Path, data: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 8, 24),
            shards=(1, 8, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
            references={
                (c, y, 0): (c, y * 8 * 24 * 2, 8 * 24 * 2)
                for c in range(2)
                for y in range(2)
            },
        )
        calls: list[str] = []
        original = _storeio.source_size

        async def counting(store, key):  # type: ignore[no-untyped-def]
            calls.append(key)
            return await original(store, key)

        monkeypatch.setattr(_storeio, "source_size", counting)
        np.testing.assert_array_equal(array[:], data)
        np.testing.assert_array_equal(array[:], data)
        assert sorted(calls) == ["raw/plane0.bin", "raw/plane1.bin"]

    def test_memory_store_cannot_climb_above_root(self, data: np.ndarray) -> None:
        array = write_virtual_array(
            MemoryStore(),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        with pytest.raises(jazav.ReferenceFormatError, match="MemoryStore"):
            array[0]

    def test_memory_store_sources_inside_root_read_without_size_check(
        self, data: np.ndarray
    ) -> None:
        store = MemoryStore()
        array = write_virtual_array(
            store,
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../raw/plane0.bin"],
            references={},
            sizes=[data[0].nbytes + 1],  # wrong on purpose: MemoryStore is unchecked
        )
        write_table(array, {(0, 0, 0): (0, 0, data[0].nbytes)})
        from zarr.core.buffer import default_buffer_prototype

        sync(
            store.set(
                "raw/plane0.bin",
                default_buffer_prototype().buffer.from_bytes(data[0].tobytes()),
            )
        )
        np.testing.assert_array_equal(array[0], data[0])

    def test_wrapper_store_is_unwrapped(self, tmp_path: Path, data: np.ndarray) -> None:
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
        wrapped = WrapperStore(LocalStore(tmp_path / "out.zarr", read_only=True))
        array = zarr.open_group(wrapped, mode="r")["0"]
        np.testing.assert_array_equal(array[:], data)

    def test_nested_transpose_chain_decodes_interleaved_blocks(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        # Stored as (y, x, c) with channels interleaved, like ND2 frames.
        interleaved = np.ascontiguousarray(data.transpose(1, 2, 0))
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "frame.bin").write_bytes(interleaved.tobytes())
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=data.shape,
            dtype="uint16",
            sources=["../../raw/frame.bin"],
            references={(0, 0, 0): (0, 0, interleaved.nbytes)},
            codecs=[
                {"name": "transpose", "configuration": {"order": [1, 2, 0]}},
                BYTES_LE,
            ],
        )
        np.testing.assert_array_equal(array[:], data)

    def test_nested_zstd_chain_decodes_compressed_blocks(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        numcodecs = pytest.importorskip("numcodecs")
        compressed = [numcodecs.Zstd().encode(data[c].tobytes()) for c in range(2)]
        (tmp_path / "raw").mkdir()
        blob = b"".join(compressed)
        (tmp_path / "raw" / "planes.zst").write_bytes(blob)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/planes.zst"],
            references={
                (0, 0, 0): (0, 0, len(compressed[0])),
                (1, 0, 0): (0, len(compressed[0]), len(compressed[1])),
            },
            codecs=[BYTES_LE, {"name": "zstd", "configuration": {"level": 0}}],
        )
        np.testing.assert_array_equal(array[:], data)

    def test_partial_decode_path_is_used(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        assert array._async_array.codec_pipeline.supports_partial_decode
        # A successful read proves the partial path: the plain path raises.
        np.testing.assert_array_equal(array[0], data[0])
        codec = array.metadata.codecs[0]
        assert isinstance(codec, jazav.ReferenceCodec)
        with pytest.raises(jazav.ReferenceFormatError, match="only codec"):
            sync(codec._decode_single(None, None))  # type: ignore[arg-type]

    def test_reads_through_the_fused_pipeline(
        self, tmp_path: Path, data: np.ndarray
    ) -> None:
        import zarr.core.codec_pipeline as pipelines

        if not hasattr(pipelines, "FusedCodecPipeline"):
            pytest.skip("this zarr has no FusedCodecPipeline")
        raw_files(tmp_path / "raw", data)
        write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 8, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
            references={
                (c, y, 0): (c, y * 384, 384) for c in range(2) for y in range(2)
            },
        )
        with zarr.config.set(
            {"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}
        ):
            array = zarr.open_array(tmp_path / "out.zarr" / "0", mode="r")
            np.testing.assert_array_equal(array[:], data)
            np.testing.assert_array_equal(array[1, 5:11, 3], data[1, 5:11, 3])

    def test_array_is_read_only(self, tmp_path: Path, data: np.ndarray) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        # A whole-shard write goes straight to encode; a partial write first
        # reads the shard through the plain decode path. Both say read-only.
        with pytest.raises(NotImplementedError, match="read-only"):
            array[:] = 1
        with pytest.raises(
            (NotImplementedError, jazav.ReferenceFormatError), match="read-only"
        ):
            array[0] = 1
        np.testing.assert_array_equal(array[0], data[0])
