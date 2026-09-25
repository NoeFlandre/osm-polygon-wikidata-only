"""Run one CUDA-required sentence batch on a reserved Grid5000 node.

Compatibility shim for ``osm-polygon-wikidata-only grid5000 job``.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.cli.grid5000 import job_main as main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
