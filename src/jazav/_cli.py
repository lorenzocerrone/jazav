"""The `jazav` console script: `jazav verify <zarr>` and `jazav materialize <zarr>`."""

from __future__ import annotations

import argparse
import sys

from jazav._references import ReferenceFormatError
from jazav._tools import materialize, verify


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jazav", description="Inspect and convert virtual OME-Zarr images."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser(
        "verify", help="check that every reference array under <zarr> can be read"
    )
    check.add_argument("zarr", help="path of a zarr group or array")
    check.add_argument(
        "--deep", action="store_true", help="also decode every chunk through the codec"
    )

    rewrite = commands.add_parser(
        "materialize",
        help="rewrite every reference array under <zarr> as a regular array",
    )
    rewrite.add_argument("zarr", help="path of a zarr group on a local filesystem")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the console script; returns the exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            report = verify(args.zarr, deep=args.deep)
            print(report)
            return 0 if report.ok else 1
        rewritten = materialize(args.zarr)
        if rewritten:
            print("materialized:\n  " + "\n  ".join(rewritten))
        else:
            print(f"nothing to materialize under {args.zarr}")
        return 0
    except ReferenceFormatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
