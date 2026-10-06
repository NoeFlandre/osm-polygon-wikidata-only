"""Allow ``python -m osm_polygon_wikidata_only``."""

from __future__ import annotations

from osm_polygon_wikidata_only.cli.app import run

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
