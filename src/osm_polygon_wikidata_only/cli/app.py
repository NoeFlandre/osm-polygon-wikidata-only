"""Thin CLI module entry point.

This exists so that ``python -m osm_polygon_wikidata_only`` and
``osm-polygon-wikidata-only`` (via pyproject script) both work.
"""

from __future__ import annotations

import sys


def run() -> int:
    """Parse the lightweight CLI surface before loading a command handler."""
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        from osm_polygon_wikidata_only import __version__ as package_version  # noqa: PLC0415

        print(f"osm-polygon-wikidata-only {package_version}")
        raise SystemExit(0)

    from .parser import build_parser  # noqa: PLC0415

    parser = build_parser()
    args = parser.parse_args()

    from .tools import dispatch_tool  # noqa: PLC0415

    tool_status = dispatch_tool(args)
    if tool_status is not None:
        return tool_status

    from .commands import run_parsed  # noqa: PLC0415

    return run_parsed(parser, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
