"""Shared helpers for the example scripts.

Each example describes a file as a `VirtualPlan`: one `Block` per level-0 chunk,
each block being exactly one stored block of one source file. `run` builds a
virtual OME-Zarr from the plan with ngio next to a copying conversion of the
same pixels, checks that both read identically, verifies the references and
collects one `Row` per input file. Files that cannot be referenced in place are
reported with a reason, never raised: `NotVirtualizable` is a result, any other
exception is a bug.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from ngio import Roi, RoiTable, create_empty_ome_zarr, open_ome_zarr_container

import jazav


class NotVirtualizable(Exception):
    """The file cannot be referenced in place; `reason` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Block:
    """One level-0 chunk: its grid index and the byte range it references."""

    index: tuple[int, ...]
    source: Path
    offset: int
    length: int


@dataclass
class VirtualPlan:
    """Everything needed to build the virtual level 0 of one image."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    codecs: list[dict[str, Any]]
    blocks: list[Block]
    axes: tuple[str, ...]
    pixelsize: float = 1.0
    z_spacing: float = 1.0
    channel_labels: list[str] | None = None

    def check(self) -> None:
        """Enforce the ingestion rule: one block per chunk, no overlap."""
        grid = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks, strict=True))
        seen: dict[tuple[int, ...], Path] = {}
        for block in self.blocks:
            if len(block.index) != len(grid) or any(
                not 0 <= i < n for i, n in zip(block.index, grid, strict=True)
            ):
                raise NotVirtualizable(
                    f"block {block.index} of {block.source.name} is outside the "
                    f"chunk grid {grid}"
                )
            if block.index in seen:
                raise NotVirtualizable(
                    f"{block.source.name} and {seen[block.index].name} both cover "
                    f"chunk {block.index}: overlapping blocks"
                )
            seen[block.index] = block.source


def _create(out: Path, plan: VirtualPlan, levels: int) -> Any:
    if out.exists():
        shutil.rmtree(out)
    return create_empty_ome_zarr(
        store=str(out),
        shape=plan.shape,
        pixelsize=plan.pixelsize,
        z_spacing=plan.z_spacing,
        levels=levels,
        axes_names=plan.axes,
        channels_meta=plan.channel_labels,
        chunks=plan.chunks,
        dtype=plan.dtype,
        ngff_version="0.5",  # zarr v3: the reference codec needs v3 metadata
        overwrite=True,
    )


def _add_tables(container: Any, plan: VirtualPlan) -> None:
    """Add `well_ROI_table` (whole image) and `FOV_ROI_table` (one ROI per tile)."""
    image = container.get_image()
    container.add_table(
        "well_ROI_table", container.build_image_roi_table("well"), overwrite=True
    )
    y, x = plan.axes.index("y"), plan.axes.index("x")
    tiles = sorted({(block.index[y], block.index[x]) for block in plan.blocks})
    rois = [
        Roi.from_values(
            {
                "y": slice(iy * plan.chunks[y], (iy + 1) * plan.chunks[y]),
                "x": slice(ix * plan.chunks[x], (ix + 1) * plan.chunks[x]),
            },
            name=f"FOV_{n}",
            space="pixel",
        ).to_world(image.pixel_size)
        for n, (iy, ix) in enumerate(tiles)
    ]
    container.add_table("FOV_ROI_table", RoiTable(rois=rois), overwrite=True)


def build_virtual(out: Path, plan: VirtualPlan, levels: int = 3) -> None:
    """Build a virtual OME-Zarr at `out` from `plan`."""
    container = _create(out, plan, levels)
    level0 = container.level_paths[0]
    files = {block.source for block in plan.blocks}
    relative = {src: jazav.relative_source_path(out / level0, src) for src in files}
    sources = jazav.normalize_sources(relative.values())
    inverse = {rel: src for src, rel in relative.items()}
    sizes = [inverse[rel].stat().st_size for rel in sources]
    index = {rel: i for i, rel in enumerate(sources)}

    # Replace the regular level 0 ngio created, on the same group and path.
    group = zarr.open_group(out, mode="r+")
    array = jazav.create_reference_array(
        group,
        level0,
        shape=plan.shape,
        chunks=plan.chunks,
        dtype=plan.dtype,
        fill_value=0,
        dimension_names=plan.axes,
        sources=sources,
        sizes=sizes,
        codecs=plan.codecs,
    )
    jazav.write_references(
        array,
        {
            block.index: (index[relative[block.source]], block.offset, block.length)
            for block in plan.blocks
        },
    )
    # Only now bind the image: ngio binds the zarr array when the image object is
    # created. The pyramid is built by reading level 0 through the codec.
    container = open_ome_zarr_container(out)
    container.get_image().consolidate(mode="auto")
    _add_tables(container, plan)


def build_copy(
    out: Path, plan: VirtualPlan, truth: np.ndarray, levels: int = 3
) -> None:
    """Build a regular OME-Zarr at `out` holding `truth` (axes as in `plan`)."""
    container = _create(out, plan, levels)
    image = container.get_image()
    image.set_array(truth, axes_order=plan.axes)
    image.consolidate(mode="auto")
    _add_tables(container, plan)


def compare(virtual: Path, copy: Path, truth: np.ndarray, plan: VirtualPlan) -> float:
    """Assert both containers read identically; returns the level-0 read time."""
    v = open_ome_zarr_container(virtual)
    c = open_ome_zarr_container(copy)
    start = time.perf_counter()
    level0 = v.get_image().get_array(axes_order=plan.axes)
    read_s = time.perf_counter() - start
    np.testing.assert_array_equal(level0, truth)
    plain = zarr.open_array(virtual / v.level_paths[0], mode="r")[:]
    np.testing.assert_array_equal(plain, truth)
    for path in v.level_paths[1:]:
        np.testing.assert_array_equal(
            v.get_image(path=path).get_array(), c.get_image(path=path).get_array()
        )
    return read_s


@dataclass
class Row:
    """One line of the report."""

    file: str
    virtualizable: bool
    reason: str
    virtual_size: int | None = None
    copy_size: int | None = None
    level0_read_s: float | None = None


def dir_size(path: Path) -> int:
    """Total size of the files below `path`."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def fmt_bytes(n: int | None) -> str:
    """Human-readable byte count."""
    if n is None:
        return "-"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GiB"


def print_table(rows: Sequence[Row]) -> None:
    """Print the report table."""
    header = ("file", "virtual", "reason", "virtual size", "copy size", "L0 read")
    body = [
        (
            row.file,
            "yes" if row.virtualizable else "no",
            row.reason,
            fmt_bytes(row.virtual_size),
            fmt_bytes(row.copy_size),
            "-" if row.level0_read_s is None else f"{row.level0_read_s * 1000:.0f} ms",
        )
        for row in rows
    ]
    widths = [max(len(line[i]) for line in [header, *body]) for i in range(6)]
    for line in [header, tuple("-" * w for w in widths), *body]:
        print("  ".join(cell.ljust(w) for cell, w in zip(line, widths, strict=True)))


Describe = Callable[[], tuple[VirtualPlan, np.ndarray]]
"""Produces the plan and the vendor reader's pixels for one image."""

Input = tuple[str, Describe]
"""A report name and how to describe that image."""


def run(inputs: Sequence[Input], out_dir: Path, levels: int = 3) -> list[Row]:
    """Virtualize every input, compare with a copy, verify, collect one row each."""
    rows: list[Row] = []
    for name, describe in inputs:
        try:
            plan, truth = describe()
            plan.check()
        except NotVirtualizable as e:
            rows.append(Row(name, False, e.reason))
            continue
        virtual = out_dir / name / "virtual.zarr"
        copy = out_dir / name / "copy.zarr"
        build_virtual(virtual, plan, levels)
        build_copy(copy, plan, truth, levels)
        read_s = compare(virtual, copy, truth, plan)
        report = jazav.verify(virtual)
        if not report.ok:
            raise AssertionError(f"verify failed on {virtual}:\n{report}")
        rows.append(Row(name, True, "ok", dir_size(virtual), dir_size(copy), read_s))
    return rows


def main(
    name: str,
    inputs_of: Callable[[Path], Sequence[Input]],
    default_files: Sequence[Path],
    argv: Sequence[str] | None = None,
) -> int:
    """Command-line entry point shared by the vendor examples.

    `inputs_of(file)` lists the images of one file; a file that cannot be opened
    at all should raise `NotVirtualizable`, which becomes one "no" row.
    """
    parser = argparse.ArgumentParser(description=f"virtualize {name} files")
    parser.add_argument("files", nargs="*", type=Path, default=list(default_files))
    parser.add_argument("--out", type=Path, default=Path("_examples_out") / name)
    args = parser.parse_args(argv)
    inputs: list[Input] = []
    rows: list[Row] = []
    for file in args.files:
        if not file.exists():
            print(f"skipping {file}: not found", file=sys.stderr)
            continue
        try:
            inputs.extend(inputs_of(file))
        except NotVirtualizable as e:
            rows.append(Row(file.name, False, e.reason))
    rows.extend(run(inputs, args.out))
    print_table(rows)
    return 0
