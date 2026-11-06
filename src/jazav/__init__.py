"""jazav: virtual OME-Zarr through a zarr v3 reference-table codec.

Level 0 of a virtual array is stored as shards, each holding a table of 20-byte
references into the original microscopy files; `ReferenceCodec`
(`jazav.reference`) resolves them at read time. Importing this package registers
the codec; installed, the `zarr.codecs` entry point makes zarr find it without
an import.
"""

try:
    from jazav._version import __version__
except ImportError:  # running from a checkout without a build step
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("jazav")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from jazav._codec import CODEC_NAME, ReferenceCodec
from jazav._references import (
    DEFAULT_MAX_CHUNKS_PER_SHARD,
    FORMAT_VERSION,
    MISSING_ENTRY,
    REFERENCE_SIZE,
    ChunkReference,
    ReferenceFormatError,
    check_relative_path,
    decode_reference,
    decode_table,
    default_shard_shape,
    encode_reference,
    encode_table,
    normalize_sources,
)
from jazav._tools import Problem, VerifyReport, materialize, verify
from jazav._writer import (
    create_reference_array,
    reference_codec,
    relative_source_path,
    write_references,
)

__all__ = [
    "CODEC_NAME",
    "DEFAULT_MAX_CHUNKS_PER_SHARD",
    "FORMAT_VERSION",
    "MISSING_ENTRY",
    "REFERENCE_SIZE",
    "ChunkReference",
    "Problem",
    "ReferenceCodec",
    "ReferenceFormatError",
    "VerifyReport",
    "__version__",
    "check_relative_path",
    "create_reference_array",
    "decode_reference",
    "decode_table",
    "default_shard_shape",
    "encode_reference",
    "encode_table",
    "materialize",
    "normalize_sources",
    "reference_codec",
    "relative_source_path",
    "verify",
    "write_references",
]
