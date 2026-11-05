"""Virtualize Zeiss CZI images (checked with czifile 2026.10.14, headers only).

Every non-pyramid subblock is one stored block: one level-0 chunk of the
subblock's `C, Z, Y, X` (and `T`) extent, placed by `entry.start` relative to
the scene origin. Byte ranges come from the subblock segment header
(`data_offset`, `data_size`), the pixels are never read through czifile except
to build the ground truth.

- `UNCOMPRESSED`: `[bytes]` at `data_offset`.
- `ZSTD` (ZSTD0): `[bytes, zstd]` at `data_offset`.
- `ZSTDHDR` (ZSTD1): a header of `n = raw[0]` bytes, then a plain zstd frame.
  The header's chunk of type 1 carries the hi-lo byte-unpack flag; when it is
  off (always for 8-bit, as in the samples) the block is `[bytes, zstd]` at
  `data_offset + n`. When it is on, the block is refused.
- JPEG, JPEG-XR, LZW, CHUNKED and pyramid subblocks are refused.

One OME-Zarr per scene. Run: `pixi run -e dev example-czi [FILE ...]`.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from typing import Any

import czifile
import numpy as np
from _common import Block, Input, NotVirtualizable, VirtualPlan, main
from czifile import CziCompressionType

SAMPLES = Path(
    "/Users/locerr/Projects/Converters/fractal-czi-converters/tests/data/Zeiss-CZI/raw"
)
CANONICAL = "TCZYX"


def _scaling(czi: Any) -> dict[str, float]:
    """Pixel size per axis in micrometres from the metadata XML (1.0 if absent)."""
    out = {"X": 1.0, "Y": 1.0, "Z": 1.0}
    try:
        root = ET.fromstring(czi.metadata())
    except Exception:
        return out
    for item in root.iter("Distance"):
        axis = item.get("Id")
        value = item.findtext("Value")
        if axis is not None and axis in out and value:
            out[axis] = float(value) * 1e6
    return out


def _byte_range(czi: Any, entry: Any) -> tuple[int, int, list[dict[str, Any]]]:
    segment = entry.read_segment_data(czi)
    offset, size = int(segment.data_offset), int(segment.data_size)
    little = {"name": "bytes", "configuration": {"endian": "little"}}
    compression = entry.compression
    if compression == CziCompressionType.UNCOMPRESSED:
        return offset, size, [little]
    zstd = {"name": "zstd", "configuration": {"level": 0, "checksum": False}}
    if compression == CziCompressionType.ZSTD:
        return offset, size, [little, zstd]
    if compression == CziCompressionType.ZSTDHDR:
        header = bytes(czi._read_at(offset, 1))
        n = header[0]
        chunks = bytes(czi._read_at(offset + 1, n - 1)) if n > 1 else b""
        position = 0
        while position < len(chunks):
            chunk_type = chunks[position]
            if chunk_type == 1:  # HiLoByteUnpack
                if position + 1 < len(chunks) and chunks[position + 1] & 1:
                    raise NotVirtualizable(
                        "ZSTD1 subblock with hi-lo byte unpacking (no zarr codec)"
                    )
                position += 2
            else:
                raise NotVirtualizable(f"ZSTD1 header chunk type {chunk_type}")
        return offset + n, size - n, [little, zstd]
    raise NotVirtualizable(f"subblock compression {compression.name}")


def describe_scene(path: Path, scene: int) -> tuple[VirtualPlan, np.ndarray]:
    """Plan one chunk per subblock of `scene`."""
    label = f"{path.name}#scene{scene}"
    with czifile.CziFile(path) as czi:
        entries = [
            e
            for e in czi.filtered_subblock_directory
            if e.scene_index == scene and not e.is_pyramid
        ]
        if not entries:
            raise NotVirtualizable(f"{label}: no full-resolution subblocks")
        first = entries[0]
        dims = tuple(first.dims)
        for entry in entries:
            if tuple(entry.dims) != dims or tuple(entry.shape) != tuple(first.shape):
                raise NotVirtualizable(f"{label}: subblocks differ in dims or shape")
            if entry.pixel_type != first.pixel_type:
                raise NotVirtualizable(f"{label}: subblocks differ in pixel type")
        extents = dict(zip(dims, first.shape, strict=True))
        if extents.get("S", 1) != 1:
            raise NotVirtualizable(f"{label}: multi-sample (RGB) pixels")
        unsupported = sorted(set(dims) - set(CANONICAL) - {"S", "H"})
        if unsupported:
            raise NotVirtualizable(f"{label}: unsupported dimension(s) {unsupported}")
        dtype = np.dtype(first.dtype)
        if dtype.kind not in "uif":
            raise NotVirtualizable(f"{label}: pixel type {first.pixel_type.name}")

        starts = [dict(zip(dims, e.start, strict=True)) for e in entries]
        origin = {d: min(s.get(d, 0) for s in starts) for d in CANONICAL}
        has_t = (
            max(s.get("T", 0) + extents.get("T", 1) for s in starts) - origin["T"]
        ) > 1
        axes = tuple(d.lower() for d in CANONICAL if d != "T" or has_t)
        chunk = tuple(extents.get(a.upper(), 1) for a in axes)

        blocks: list[Block] = []
        indices: list[tuple[int, ...]] = []
        codecs: list[dict[str, Any]] | None = None
        for entry, start in zip(entries, starts, strict=True):
            index = []
            for a, c in zip(axes, chunk, strict=True):
                rel = start.get(a.upper(), 0) - origin[a.upper()]
                if rel % c:
                    raise NotVirtualizable(
                        f"{label}: subblock at {a}={start.get(a.upper())} is not on "
                        f"the {c}-pixel chunk grid"
                    )
                index.append(rel // c)
            offset, length, entry_codecs = _byte_range(czi, entry)
            if codecs is None:
                codecs = entry_codecs
            elif entry_codecs != codecs:
                raise NotVirtualizable(f"{label}: subblocks differ in compression")
            blocks.append(Block(tuple(index), path.resolve(), offset, length))
            indices.append(tuple(index))
        assert codecs is not None
        grid = tuple(max(i[k] for i in indices) + 1 for k in range(len(axes)))
        full = tuple(g * c for g, c in zip(grid, chunk, strict=True))

        truth = np.zeros(full, dtype=dtype)
        for entry, index in zip(entries, indices, strict=True):
            pixels = np.asarray(entry.asimage(czi).asarray()).reshape(chunk)
            region = tuple(
                slice(i * c, (i + 1) * c) for i, c in zip(index, chunk, strict=True)
            )
            truth[region] = pixels

        scaling = _scaling(czi)
        plan = VirtualPlan(
            shape=full,
            chunks=chunk,
            dtype=dtype.name,
            codecs=codecs,
            blocks=blocks,
            axes=axes,
            pixelsize=scaling["X"],
            z_spacing=scaling["Z"],
        )
        return plan, truth


def inputs_of(path: Path) -> list[Input]:
    """One input per scene of the CZI file."""
    with czifile.CziFile(path) as czi:
        scenes = sorted({e.scene_index for e in czi.filtered_subblock_directory})
    return [
        (f"{path.stem}_scene{scene}", partial(describe_scene, path, scene))
        for scene in scenes
    ]


if __name__ == "__main__":
    sys.exit(main("zeiss_czi", inputs_of, sorted(SAMPLES.glob("*.czi"))))
