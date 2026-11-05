import pytest

import jazav
from jazav import _references as refs


class TestReferenceFormat:
    def test_roundtrip(self) -> None:
        payload = jazav.encode_reference(3, 1024, 2048)
        assert len(payload) == jazav.REFERENCE_SIZE == 20
        assert jazav.decode_reference(payload) == jazav.ChunkReference(3, 1024, 2048)

    @pytest.mark.parametrize("size", [16, 19, 21, 0])
    def test_decode_rejects_wrong_size(self, size: int) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="exactly 20 bytes"):
            jazav.decode_reference(b"\0" * size)

    def test_encode_rejects_out_of_range(self) -> None:
        with pytest.raises(jazav.ReferenceFormatError, match="uint32"):
            jazav.encode_reference(2**32, 0, 1)
        with pytest.raises(jazav.ReferenceFormatError, match="positive length"):
            jazav.encode_reference(0, 0, 0)
        with pytest.raises(jazav.ReferenceFormatError, match="overflows"):
            jazav.encode_reference(0, 2**64 - 1, 2)

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
