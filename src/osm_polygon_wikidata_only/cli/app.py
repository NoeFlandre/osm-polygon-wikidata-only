"""Thin CLI module entry point.

This exists so that ``python -m osm_polygon_wikidata_only`` and
``osm-polygon-wikidata-only`` (via pyproject script) both work.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from osm_polygon_wikidata_only.version import PACKAGE_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence


def run(argv: Sequence[str] | None = None) -> int:
    """Parse the lightweight CLI surface before loading a command handler."""
    args_in = sys.argv[1:] if argv is None else list(argv)
    if args_in[:1] == ["--version"]:
        print(f"osm-polygon-wikidata-only {PACKAGE_VERSION}")
        raise SystemExit(0)

    from .parser import build_parser  # noqa: PLC0415

    parser = build_parser()
    args = parser.parse_args(args_in)

    from .tools import dispatch_tool  # noqa: PLC0415

    tool_status = dispatch_tool(args)
    if tool_status is not None:
        return tool_status

    from .commands import run_parsed  # noqa: PLC0415

    return run_parsed(parser, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
