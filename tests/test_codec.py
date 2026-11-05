import pickle
import re
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.core.sync import sync
from zarr.storage import LocalStore, MemoryStore, WrapperStore

import jazav
from jazav import _storeio
from tests.conftest import BYTES_LE, raw_files, write_raw_chunk, write_virtual_array


class TestReferenceCodecMetadata:
    def test_to_dict_matches_documented_layout(self) -> None:
        codec = jazav.ReferenceCodec(
            sources=["../../raw/a.bin"], sizes=[768], codecs=[BYTES_LE]
        )
        assert codec.to_dict() == {
            "name": "jazav.reference",
            "configuration": {
                "version": 1,
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
                    "configuration": {"sources": ["a"], "codecs": [BYTES_LE]},
                }
            )

    def test_rejects_bad_configuration(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="at least one source"):
            jazav.ReferenceCodec(sources=[], sizes=[], codecs=[BYTES_LE])
        with pytest.raises(jazav.ReferenceFormatError, match="duplicate"):
            jazav.ReferenceCodec(sources=["a", "a"], sizes=[1, 1], codecs=[BYTES_LE])
        with pytest.raises(jazav.ReferenceFormatError, match="relative"):
            jazav.ReferenceCodec(sources=["/abs"], sizes=[1], codecs=[BYTES_LE])
        with pytest.raises(jazav.ReferenceFormatError, match="sizes has 2"):
            jazav.ReferenceCodec(sources=["a"], sizes=[1, 2], codecs=[BYTES_LE])
        with pytest.raises(jazav.ReferenceFormatError, match="non-negative"):
            jazav.ReferenceCodec(sources=["a"], sizes=[-1], codecs=[BYTES_LE])

    def test_unknown_version_names_installed_jazav(self) -> None:
        with pytest.raises(
            jazav.ReferenceFormatError,
            match=re.escape(f"version 2: jazav {jazav.__version__}"),
        ):
            jazav.ReferenceCodec(sources=["a"], sizes=[1], codecs=[BYTES_LE], version=2)

    def test_pickle_roundtrip(self) -> None:
        codec = jazav.ReferenceCodec(
            sources=["a.bin", "b.bin"], sizes=[1, 2], codecs=[BYTES_LE]
        )
        assert pickle.loads(pickle.dumps(codec)) == codec


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

    def test_missing_chunk_reads_fill_value(
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

    def test_corrupt_chunk_file_raises(self, tmp_path: Path, data: np.ndarray) -> None:
        raw_files(tmp_path / "raw", data)
        array = write_virtual_array(
            LocalStore(tmp_path / "out.zarr"),
            shape=data.shape,
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin"],
            references={(0, 0, 0): (0, 0, data[0].nbytes)},
        )
        (tmp_path / "out.zarr" / "0" / "c" / "0" / "0" / "0").write_bytes(b"\0" * 16)
        with pytest.raises(jazav.ReferenceFormatError, match="exactly 20 bytes"):
            array[0]

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
        write_raw_chunk(array, (0, 0, 0), jazav.encode_reference(5, 0, data[0].nbytes))
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
        write_raw_chunk(array, (0, 0, 0), jazav.encode_reference(0, 2, data[0].nbytes))
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
        write_raw_chunk(array, (0, 0, 0), jazav.encode_reference(0, 0, data[0].nbytes))
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
            chunks=(1, 16, 24),
            dtype="uint16",
            sources=["../../raw/plane0.bin", "../../raw/plane1.bin"],
            references={
                (0, 0, 0): (0, 0, data[0].nbytes),
                (1, 0, 0): (1, 0, data[1].nbytes),
            },
        )
        with zarr.config.set(
            {"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}
        ):
            array = zarr.open_array(tmp_path / "out.zarr" / "0", mode="r")
            np.testing.assert_array_equal(array[:], data)

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
        with pytest.raises(NotImplementedError, match="read-only"):
            array[0] = 1
