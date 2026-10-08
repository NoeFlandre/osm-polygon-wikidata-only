"""Thin CLI module entry point.

This exists so that ``python -m osm_polygon_wikidata_only`` and
``osm-polygon-wikidata-only`` (via pyproject script) both work.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from osm_polygon_wikidata_only.version import PACKAGE_VERSION

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence


def run(argv: Sequence[str] | None = None) -> int:
    """Parse the lightweight CLI surface before loading a command handler."""
    args_in = sys.argv[1:] if argv is None else list(argv)
    if args_in[:1] == ["--version"]:
        print(f"osm-polygon-wikidata-only {PACKAGE_VERSION}")
        raise SystemExit(0)

    from .dispatch import parse_and_dispatch  # noqa: PLC0415
    from .parser import build_parser  # noqa: PLC0415

    return parse_and_dispatch(build_parser(), args_in, _load_and_run_parsed)


def _load_and_run_parsed(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Import the processing handlers only once the command line has parsed."""
    from .commands import run_parsed  # noqa: PLC0415

    return run_parsed(parser, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
