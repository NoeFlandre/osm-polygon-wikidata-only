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
        --data-root /path/to/osm-polygon-data [--dry-run] [--json]

``--dry-run`` computes the rejection counts without rewriting any table
or writing the audit. Data-root, IO and data-contract errors are reported
as one line on stderr with exit status 1 instead of a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.integrity import IntegrityReport, enforce_all_regions
from osm_polygon_wikidata_only.cli.parser import add_enforce_integrity_arguments
from osm_polygon_wikidata_only.config.paths import DataRootError, repository_root, resolve_data_root
from osm_polygon_wikidata_only.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)

PROG = "osm-polygon-wikidata-only-enforce-integrity"
EXIT_FAILURE = 1

# Expected operator-facing failures: an unusable data root, missing or
# unreadable parquets, and data-contract violations (ValueError, which also
# covers pyarrow's ArrowInvalid). Anything else is a bug and propagates.
_EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    DataRootError,
    OSError,
    ValueError,
    pa.ArrowException,
)


def _build_parser(prog: str = PROG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Deterministic join-integrity enforcement (Path A). Reject "
            "polygon_articles rows whose wikidata does not match the "
            "canonical polygons table, and wikivoyage documents whose "
            "wikidata is absent from polygons (with cascading section "
            "drops). Emits an audit JSON."
        ),
    )
    add_enforce_integrity_arguments(parser)
    return parser


def _summary(report: IntegrityReport, *, dry_run: bool) -> dict[str, object]:
    """Return the machine-readable report summary."""
    return {
        "dry_run": dry_run,
        "audit_path": None if dry_run else str(report.audit_path),
        "polygon_articles_rejected": report.total_polygon_articles_rejected,
        "wikivoyage_documents_rejected": report.total_wikivoyage_documents_rejected,
        "wikivoyage_sections_cascaded": report.total_wikivoyage_sections_cascaded,
    }


def _log_summary(report: IntegrityReport, *, dry_run: bool) -> None:
    prefix = "Integrity dry run" if dry_run else "Integrity pass"
    LOGGER.info(
        "%s complete: %d polygon_articles rejected, "
        "%d wikivoyage_documents rejected, %d wikivoyage_sections cascaded.",
        prefix,
        report.total_polygon_articles_rejected,
        report.total_wikivoyage_documents_rejected,
        report.total_wikivoyage_sections_cascaded,
    )
    if dry_run:
        LOGGER.info("Dry run: no table rewritten and no audit written.")
    else:
        LOGGER.info("Audit written to %s", report.audit_path)


def execute(args: argparse.Namespace, *, prog: str = PROG) -> int:
    """Run the integrity pass for already-parsed *args*."""
    configure_logging(args.log_level)
    try:
        data_root = resolve_data_root(explicit=args.data_root, repo_root=repository_root())
        LOGGER.info("Running integrity pass against data root: %s", data_root.path)
        report = enforce_all_regions(
            data_root,
            audit_filename=args.audit_filename,
            dry_run=args.dry_run,
        )
    except _EXPECTED_ERRORS as error:
        print(f"{prog}: error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    _log_summary(report, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(_summary(report, dry_run=args.dry_run), sort_keys=True))
    return 0


def run(argv: list[str] | None = None) -> int:
    return execute(_build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
