"""Virtualize Nikon ND2 images (checked with nd2 0.12.0).

Frames are stored `(y, x, c)` with channels interleaved, so one frame is one
level-0 chunk of shape `(c, 1, y, x)` decoded through
`[transpose(order=[1, 2, 3, 0]), bytes]` (the stored block is read as
`(z=1, y, x, c)`). Frame byte offsets come from the private
`ModernReader._frame_offsets` (chunkmap offset + 24 + 4072 bytes), which is
what `nd2` itself reads from. Without compression the frame is
`heightPx * widthBytes` bytes.

- `compressionType is None`: `[transpose, bytes]`.
- `"lossless"`: one zlib stream per frame, `[transpose, bytes, numcodecs.zlib]`;
  the stream starts 8 bytes into the frame chunk's data. Unverified: no sample.
- Refused: `"lossy"`, padded rows (`widthBytes != widthPx * bytesPerComponent *
  componentCount`), multi-component (RGB) channels, legacy ND2.

One OME-Zarr per position `P`. Run: `pixi run -e dev example-nd2 [FILE ...]`.
"""

from __future__ import annotations

import struct
import sys
from functools import partial
from pathlib import Path
from typing import Any

import nd2
import numpy as np
from _common import Block, Input, NotVirtualizable, VirtualPlan, main

SAMPLES = Path(
    "/Users/locerr/Projects/Converters/fractal-nd2-converters/tests/data/Nikon-ND2/raw"
)
CHUNK_MAGIC = 0x0ABECEDA


def _zlib_stream(fh: Any, chunk_offset: int) -> tuple[int, int]:
    """Byte range of the zlib stream inside the ND2 chunk at `chunk_offset`."""
    fh.seek(chunk_offset)
    magic, name_length, data_length = struct.unpack("<IIQ", fh.read(16))
    if magic != CHUNK_MAGIC:
        raise NotVirtualizable(f"bad chunk magic {magic:#x} at {chunk_offset}")
    data_start = chunk_offset + 16 + name_length
    return data_start + 8, data_length - 8


def describe_position(
    path: Path, position: int | None
) -> tuple[VirtualPlan, np.ndarray]:
    """Plan one chunk per frame of `position` (None when the file has none)."""
    label = path.name if position is None else f"{path.name}#P{position}"
    with nd2.ND2File(path) as f:
        if f.is_legacy:
            raise NotVirtualizable(f"{label}: legacy ND2 (JPEG2000)")
        attrs = f.attributes
        compression = attrs.compressionType
        if compression == "lossy":
            raise NotVirtualizable(f"{label}: lossy compression")
        bytes_per_component = attrs.bitsPerComponentInMemory // 8
        n_components = attrs.componentCount
        n_channels = attrs.channelCount or 1
        width, height = attrs.widthPx or 0, attrs.heightPx
        if attrs.widthBytes != width * bytes_per_component * n_components:
            raise NotVirtualizable(
                f"{label}: padded rows (widthBytes={attrs.widthBytes}, widthPx={width})"
            )
        if n_components != n_channels:
            raise NotVirtualizable(f"{label}: multi-component (RGB) channels")
        dtype = np.dtype(f.dtype)
        if dtype.kind not in "uif":
            raise NotVirtualizable(
                f"{label}: dtype {dtype} is not a plain numeric type"
            )

        reader: Any = f._rdr  # nd2 0.12.0 ModernReader (private API)
        offsets: dict[int, int] = reader._frame_offsets
        loops = reader.loop_indices()
        sizes = dict(f.sizes)
        n_t, n_z = sizes.get("T", 1), sizes.get("Z", 1)
        axes = ("t", "c", "z", "y", "x") if n_t > 1 else ("c", "z", "y", "x")
        stored = ("t", "z", "y", "x", "c") if n_t > 1 else ("z", "y", "x", "c")
        order = [axes.index(s) for s in stored]
        chunk = tuple(
            n_channels if a == "c" else height if a == "y" else width if a == "x" else 1
            for a in axes
        )
        full = tuple(
            n_t if a == "t" else n_z if a == "z" else c
            for a, c in zip(axes, chunk, strict=True)
        )
        codecs: list[dict[str, Any]] = [
            {"name": "transpose", "configuration": {"order": order}},
            {"name": "bytes", "configuration": {"endian": "little"}},
        ]
        if compression == "lossless":
            codecs.append({"name": "numcodecs.zlib", "configuration": {"level": 6}})

        blocks: list[Block] = []
        frame_nbytes = height * attrs.widthBytes
        for seq, loop in enumerate(loops):
            if loop.get("P", 0) != (position or 0):
                continue
            offset = offsets.get(seq)
            if offset is None:
                continue  # missing frame: the chunk stays absent and reads as 0
            index = tuple(
                loop.get("T", 0) if a == "t" else loop.get("Z", 0) if a == "z" else 0
                for a in axes
            )
            if compression == "lossless":
                chunk_offset, _ = reader.chunkmap[f"ImageDataSeq|{seq}!".encode()]
                start, length = _zlib_stream(reader._fh, chunk_offset)
            else:
                start, length = offset, frame_nbytes
            blocks.append(Block(index, path.resolve(), start, length))

        data = f.asarray(position=position) if "P" in sizes else f.asarray()
        dims = [d.lower() for d in sizes if d != "P"]
        for a in axes:
            if a not in dims:
                data = np.expand_dims(data, 0)
                dims.insert(0, a)
        truth = np.ascontiguousarray(np.transpose(data, [dims.index(a) for a in axes]))
        assert truth.shape == full, (truth.shape, full)

        voxel = f.voxel_size()
        channels = [c.channel.name for c in f.metadata.channels or []]
        plan = VirtualPlan(
            shape=full,
            chunks=chunk,
            dtype=dtype.name,
            codecs=codecs,
            blocks=blocks,
            axes=axes,
            pixelsize=float(voxel.x) or 1.0,
            z_spacing=float(voxel.z) or 1.0,
            channel_labels=channels if len(channels) == n_channels else None,
        )
        return plan, truth


def inputs_of(path: Path) -> list[Input]:
    """One input per position of the ND2 file."""
    with nd2.ND2File(path) as f:
        n_positions = f.sizes.get("P")
    if n_positions is None:
        return [(path.stem, partial(describe_position, path, None))]
    return [
        (f"{path.stem}_P{p}", partial(describe_position, path, p))
        for p in range(n_positions)
    ]


def _default_files() -> list[Path]:
    return sorted(SAMPLES.rglob("*.nd2"))


if __name__ == "__main__":
    sys.exit(main("nikon_nd2", inputs_of, _default_files()))
