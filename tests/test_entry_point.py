"""zarr must resolve `jazav.reference` through the entry point, without an import."""

import importlib.metadata
import subprocess
import sys

import jazav


def test_entry_point_resolves_without_importing_jazav_first() -> None:
    names = {ep.name for ep in importlib.metadata.entry_points(group="zarr.codecs")}
    assert jazav.CODEC_NAME in names, (
        "the installed jazav has no zarr.codecs entry point; editable installs "
        "bake entry_points.txt at install time, run `pixi install` again"
    )
    script = (
        "import sys, zarr\n"
        "assert 'jazav' not in sys.modules\n"
        "from zarr.registry import get_codec_class\n"
        f"cls = get_codec_class({jazav.CODEC_NAME!r})\n"
        "assert cls.__module__ == 'jazav._codec', cls\n"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
