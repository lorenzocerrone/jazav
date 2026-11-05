"""jazav: virtual OME-Zarr through a zarr v3 reference codec.

Level-0 chunks of a virtual array are 20-byte references into the original
microscopy files; `ReferenceCodec` (`jazav.reference`) resolves them at read
time. Importing this package registers the codec; installed, the `zarr.codecs`
entry point makes zarr find it without an import.
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
    FORMAT_VERSION,
    REFERENCE_SIZE,
    ChunkReference,
    ReferenceFormatError,
    check_relative_path,
    decode_reference,
    encode_reference,
    normalize_sources,
)
from jazav._tools import Problem, VerifyReport, materialize, verify
from jazav._writer import create_reference_array, relative_source_path, write_references

__all__ = [
    "CODEC_NAME",
    "FORMAT_VERSION",
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
    "encode_reference",
    "materialize",
    "normalize_sources",
    "relative_source_path",
    "verify",
    "write_references",
]
