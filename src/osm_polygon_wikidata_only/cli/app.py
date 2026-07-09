"""Thin CLI module entry point.

This exists so that ``python -m osm_polygon_wikidata_only`` and
``osm-polygon-wikidata-only`` (via pyproject script) both work.
"""

from __future__ import annotations

import sys

from .commands import main


def run() -> int:
    return main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
