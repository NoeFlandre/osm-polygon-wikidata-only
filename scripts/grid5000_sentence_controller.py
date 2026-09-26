"""Run and resume the local Grid5000 sentence-splitting controller.

Compatibility shim for ``osm-polygon-wikidata-only grid5000 controller``.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.cli.grid5000 import controller_main as main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
