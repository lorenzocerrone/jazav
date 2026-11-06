import pytest

import jazav
from jazav import _references as refs


class TestReferenceEntry:
    def test_roundtrip(self) -> None:
        payload = jazav.encode_reference(3, 1024, 2048)
        assert len(payload) == jazav.REFERENCE_SIZE == 20
        assert jazav.decode_reference(payload) == jazav.ChunkReference(3, 1024, 2048)

    def test_missing_entry_decodes_to_none(self) -> None:
        assert len(jazav.MISSING_ENTRY) == 20
        assert jazav.decode_reference(jazav.MISSING_ENTRY) is None

    def test_reserved_source_index_is_rejected(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="uint32"):
            jazav.encode_reference(2**32 - 1, 0, 1)
        almost_missing = b"\xff" * 4 + b"\0" * 16
        with pytest.raises(jazav.ReferenceFormatError, match="reserved"):
            jazav.decode_reference(almost_missing)

    @pytest.mark.parametrize("size", [16, 19, 21, 0])
    def test_decode_rejects_wrong_size(self, size: int) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="exactly 20 bytes"):
            jazav.decode_reference(b"\0" * size)

    def test_decode_rejects_zero_length(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="positive length"):
            jazav.decode_reference(b"\0" * 20)

    def test_encode_rejects_out_of_range(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="uint32"):
            jazav.encode_reference(2**32, 0, 1)
        with pytest.raises(jazav.ReferenceFormatError, match="positive length"):
            jazav.encode_reference(0, 0, 0)
        with pytest.raises(jazav.ReferenceFormatError, match="overflows"):
            jazav.encode_reference(0, 2**64 - 1, 2)


class TestReferenceTable:
    def test_roundtrip_with_missing_entries(self) -> None:
        table = jazav.encode_table({0: (0, 0, 8), 2: (1, 8, 8)}, 3)
        assert len(table) == refs.table_size(3) == 60
        assert table[20:40] == jazav.MISSING_ENTRY
        assert jazav.decode_table(table, 3) == [
            jazav.ChunkReference(0, 0, 8),
            None,
            jazav.ChunkReference(1, 8, 8),
        ]

    def test_encode_rejects_index_outside_shard(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="outside a shard of 3"):
            jazav.encode_table({3: (0, 0, 8)}, 3)

    def test_decode_rejects_wrong_table_size(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="exactly 60 bytes"):
            jazav.decode_table(b"\xff" * 40, 3)


class TestShardGeometry:
    def test_chunks_per_shard(self) -> None:
        assert refs.chunks_per_shard((2, 16, 24), (1, 8, 24)) == (2, 2, 1)
        with pytest.raises(jazav.ReferenceFormatError, match="not a positive multiple"):
            refs.chunks_per_shard((2, 12, 24), (1, 8, 24))
        with pytest.raises(jazav.ReferenceFormatError, match="rank"):
            refs.chunks_per_shard((2, 16), (1, 8, 24))

    def test_chunk_grid_rounds_up(self) -> None:
        assert refs.chunk_grid((2, 17, 24), (1, 8, 24)) == (2, 3, 1)

    def test_locate_chunk_and_flat_index_are_c_order(self) -> None:
        per_shard = (2, 3, 1)
        assert refs.locate_chunk((0, 0, 0), per_shard) == ((0, 0, 0), 0)
        assert refs.locate_chunk((1, 2, 0), per_shard) == ((0, 0, 0), 5)
        assert refs.locate_chunk((3, 4, 0), per_shard) == ((1, 1, 0), 4)
        assert list(refs.inner_coords((2, 2, 1))) == [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 0),
        ]

    def test_default_shard_shape_is_whole_array_when_small(self) -> None:
        assert refs.default_shard_shape((2, 17, 24), (1, 8, 24)) == (2, 24, 24)
        assert refs.default_shard_shape((3, 10, 2160, 2560), (1, 1, 2160, 2560)) == (
            3,
            10,
            2160,
            2560,
        )

    def test_default_shard_shape_splits_slowest_axes_first(self) -> None:
        # 100 x 100 x 10 x 10 chunks = 1e6; the bound trims t, then c, keeps y, x.
        chunk = (1, 1, 256, 256)
        shape = (100, 100, 2560, 2560)
        shards = refs.default_shard_shape(shape, chunk, max_chunks=16384)
        assert shards[2:] == (2560, 2560)
        assert shards[0] == 1
        assert shards[1] == 100  # 100 * 10 * 10 = 10,000 <= 16,384
        assert refs.default_shard_shape(shape, chunk, max_chunks=1000) == (
            1,
            10,
            2560,
            2560,
        )

    def test_default_shard_shape_splits_spatial_axes_only_when_needed(self) -> None:
        chunk = (64, 64)
        shards = refs.default_shard_shape((64 * 200, 64 * 200), chunk, max_chunks=100)
        assert shards == (64, 64 * 100)
        with pytest.raises(jazav.ReferenceFormatError, match="positive"):
            refs.default_shard_shape((64, 64), chunk, max_chunks=0)


class TestPaths:
    @pytest.mark.parametrize(
        "path", ["/abs/a.tif", "s3://b/a.tif", "C:/a.tif", "a\\b", "a//b", "./a", ""]
    )
    def test_check_relative_path_rejects(self, path: str) -> None:
        with pytest.raises(jazav.ReferenceFormatError):
            jazav.check_relative_path(path)

    @pytest.mark.parametrize("path", ["a.tif", "raw/a.tif", "../../raw/a.tif"])
    def test_check_relative_path_accepts(self, path: str) -> None:
        jazav.check_relative_path(path)

    def test_normalize_sources_dedupes_and_sorts(self) -> None:
        assert jazav.normalize_sources(["b", "a", "b"]) == ("a", "b")
        with pytest.raises(jazav.ReferenceFormatError, match="relative"):
            jazav.normalize_sources(["/abs"])

    def test_resolve_key(self) -> None:
        assert refs.resolve_key("0", "../raw/t.tif") == (0, "raw/t.tif")
        assert refs.resolve_key("0", "../../raw/t.tif") == (1, "raw/t.tif")
        assert refs.resolve_key("A/01/0/0", "../../../../raw/t.tif") == (0, "raw/t.tif")
        assert refs.resolve_key("", "../../x") == (2, "x")

    def test_array_prefix_from_chunk_key(self) -> None:
        assert refs.array_prefix_from_chunk_key("A/01/0/0/c/0/1/2", 3) == "A/01/0/0"
        assert refs.array_prefix_from_chunk_key("c/0/1", 2) == ""
        with pytest.raises(jazav.ReferenceFormatError, match="default"):
            refs.array_prefix_from_chunk_key("0/0.1.2", 3)
