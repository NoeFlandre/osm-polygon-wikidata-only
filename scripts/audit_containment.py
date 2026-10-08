"""Read-only audit of configured whole-file containment retirements.

Compatibility shim for ``osm-polygon-wikidata-only audit-containment``.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.cli.audit_containment import main
from osm_polygon_wikidata_only.cli.errors import report_deprecated

__all__ = ["main"]

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        report_deprecated(
            "scripts/audit_containment.py", "osm-polygon-wikidata-only audit-containment"
        )
