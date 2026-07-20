"""CLI: run the deterministic join-integrity pass against a data root.

This subcommand runs :func:`enforce_all_regions` against an
already-staged data root (typically the local mirror of the HF
snapshot that was downloaded for re-publication). It rewrites only
the parquets that contain rejected rows and emits a deterministic
``integrity_audit.json`` summarising the rejections.

The pass is idempotent: re-running it against a clean dataset is a
no-op (no parquet is touched, the audit file is byte-identical for
the same input set).

Usage:

    osm-polygon-wikidata-only-enforce-integrity \
        --data-root /Volumes/Seagate M3/projects/osm-polygon-wikidata-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.integrity import enforce_all_regions
from osm_polygon_wikidata_only.config.paths import resolve_data_root

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osm-polygon-wikidata-only-enforce-integrity",
        description=(
            "Deterministic join-integrity enforcement (Path A). Reject "
            "polygon_articles rows whose wikidata does not match the "
            "canonical polygons table, and wikivoyage documents whose "
            "wikidata is absent from polygons (with cascading section "
            "drops). Emits an audit JSON."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Path to the data root. Falls back to the OSM_POLYGON_DATA_ROOT "
            "environment variable, then to the recommended local path."
        ),
    )
    parser.add_argument(
        "--audit-filename",
        type=str,
        default="integrity_audit.json",
        help="Name of the audit JSON inside <data-root>/processed/integrity/.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_root = resolve_data_root(
        explicit=args.data_root, repo_root=Path(__file__).resolve().parents[3]
    )
    LOGGER.info("Running integrity pass against data root: %s", data_root.path)
    report = enforce_all_regions(
        data_root,
        audit_filename=args.audit_filename,
    )
    LOGGER.info(
        "Integrity pass complete: %d polygon_articles rejected, "
        "%d wikivoyage_documents rejected, %d wikivoyage_sections cascaded.",
        report.total_polygon_articles_rejected,
        report.total_wikivoyage_documents_rejected,
        report.total_wikivoyage_sections_cascaded,
    )
    LOGGER.info("Audit written to %s", report.audit_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
