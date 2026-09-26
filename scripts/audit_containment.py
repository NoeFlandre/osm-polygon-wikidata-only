"""Read-only audit of configured whole-file containment retirements.

Compatibility shim for ``osm-polygon-wikidata-only audit-containment``.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.cli.audit_containment import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
