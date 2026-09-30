"""Deterministic post-write integrity enforcement for the published shards.

This module owns the focused, idempotent integrity pass that runs
after the per-PBF and per-region writers have materialized their
parquet files. It enforces the join contract that the downstream
consumer (:mod:`osm_polygon_sentence_relevance.joins`) relies on:

* :func:`enforce_polygon_articles_integrity` rejects every
  ``polygon_articles`` row whose ``wikidata`` does not match the
  canonical wikidata of the linked polygon in the shard's
  ``polygons`` table. Path A only: rows are dropped, never
  rewritten. An Italy-shaped defect (``italy-latest:way:845321022``
  where ``polygon_articles.wikidata = Q30901095`` but
  ``polygons.wikidata = Q134675336``) is captured here.

* :func:`enforce_wikivoyage_integrity` rejects every
  ``wikivoyage/documents`` row whose ``wikidata`` is not in the
  shard's polygon wikidata set, and cascades the rejection to the
  dependent rows in ``wikivoyage/sections``. The eight wikivoyage
  defects (australia, bahamas, brazil-nordeste,
  canada-prince-edward-island, canada-yukon, chile, mexico,
  rheinland-pfalz) all have 1-12 wikivoyage documents whose QIDs
  do not appear in their shard's polygons.

The functions are pure with respect to the filesystem state they
read: they only consult the canonical ``polygons`` parquet for a
shard and the sidecar parquets they are about to rewrite. They
never read network, never compute wikidata aliases, never call
Wikimedia APIs. Unknown integrity violations -- e.g. a polygon_articles
row referencing a polygon_id not present in the shard's
``polygons`` table, or a missing input file -- fail loudly and
propagate the underlying error; this module does not silently
coerce data.

Every rejection is captured as a deterministic record (sorted by
the relevant identifier) and the per-region rejection summary is
merged into the existing manifest entries so the audit metadata
travels with the dataset. The rejection record schema is::

    {
        "shard": "<stem>",
        "source_table": "polygon_articles" | "wikivoyage_documents",
        "identifier": "<polygon_id>" | "<document_id>",
        "wikidata": "<qid>",
        "expected": "<canonical qid from polygons>" | null,
        "reason": "wikidata_mismatch_with_polygon_master"
                | "wikidata_absent_from_polygons",
        "cascaded_sections": <int>
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.integrity_io import (
    read_polygon_wikidata_set,
    read_table_required,
)
from osm_polygon_wikidata_only.augmentation.integrity_models import (
    INTEGRITY_CONTRACT_VERSION,
    REASON_POLYGON_ARTICLES_MISMATCH,
    REASON_WIKIVOYAGE_ABSENT,
    IntegrityReport,
    PolygonArticlesIntegrityResult,
    RejectionRecord,
    WikivoyageIntegrityResult,
)
from osm_polygon_wikidata_only.augmentation.polygon_articles_integrity import (
    enforce_polygon_articles_integrity,
)
from osm_polygon_wikidata_only.augmentation.rejection_ledger import attach_cascade_counts
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    section_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_parquet, atomic_write_text
from osm_polygon_wikidata_only.utils.time import utc_now_iso as _utc_now_iso

# Preserve the former private read names for callers of this module while
# keeping cross-module dependencies on the explicit integrity I/O interface.
_read_polygon_wikidata_set = read_polygon_wikidata_set
_read_table_required = read_table_required

# ---------------------------------------------------------------------------
# enforce_wikivoyage_integrity
# ---------------------------------------------------------------------------


def _partition_wikivoyage_documents(
    stem: str,
    rows: list[dict[str, Any]],
    valid_qids: set[str],
) -> tuple[list[dict[str, Any]], set[str], list[RejectionRecord]]:
    """Split Wikivoyage documents into retained rows and rejected identities."""
    retained: list[dict[str, Any]] = []
    rejected_ids: set[str] = set()
    rejections: list[RejectionRecord] = []
    for row in rows:
        document_id = str(row.get("document_id", ""))
        wikidata = str(row.get("wikidata", ""))
        if wikidata not in valid_qids:
            rejected_ids.add(document_id)
            rejections.append(
                RejectionRecord(
                    shard=stem,
                    source_table="wikivoyage_documents",
                    identifier=document_id,
                    wikidata=wikidata,
                    expected=None,
                    reason=REASON_WIKIVOYAGE_ABSENT,
                    cascaded_sections=0,
                )
            )
            continue
        retained.append({column: row.get(column) for column in DOCUMENT_COLUMNS})
    return retained, rejected_ids, rejections


def _partition_wikivoyage_sections(
    rows: list[dict[str, Any]],
    rejected_document_ids: set[str],
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    """Drop sections cascading from rejected Wikivoyage documents."""
    retained: list[dict[str, Any]] = []
    cascaded_count = 0
    cascades_by_document: dict[str, int] = {}
    for row in rows:
        document_id = str(row.get("document_id", ""))
        if document_id in rejected_document_ids:
            cascaded_count += 1
            cascades_by_document[document_id] = cascades_by_document.get(document_id, 0) + 1
            continue
        retained.append({column: row.get(column) for column in SECTION_COLUMNS})
    return retained, cascaded_count, cascades_by_document


def _load_wikivoyage_integrity_inputs(
    data_root: DataRoot,
    stem: str,
) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Load the shard paths and rows needed for Wikivoyage validation."""
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    documents_path = data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    sections_path = data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    valid_qids = _read_polygon_wikidata_set(polygons_path)
    documents_table = _read_table_required(
        documents_path,
        label="wikivoyage/documents",
        columns=DOCUMENT_COLUMNS,
    )
    sections_table = (
        _read_table_required(
            sections_path,
            label="wikivoyage/sections",
            columns=SECTION_COLUMNS,
        )
        if sections_path.is_file()
        else pa.table({column: [] for column in SECTION_COLUMNS})
    )
    return (
        documents_path,
        sections_path,
        documents_table.to_pylist(),
        sections_table.to_pylist(),
        valid_qids,
    )


def _write_wikivoyage_integrity_tables(
    documents_path: Path,
    sections_path: Path,
    retained_documents: list[dict[str, Any]],
    retained_sections: list[dict[str, Any]],
    *,
    rewrite_documents: bool,
    rewrite_sections: bool,
) -> None:
    """Atomically rewrite only the Wikivoyage tables that changed."""
    if rewrite_documents:
        atomic_write_parquet(
            documents_path,
            _table_from_rows(retained_documents, DOCUMENT_COLUMNS, document_schema()),
        )
    if rewrite_sections:
        atomic_write_parquet(
            sections_path,
            _table_from_rows(retained_sections, SECTION_COLUMNS, section_schema()),
        )


def _table_from_rows(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    schema: pa.Schema,
) -> pa.Table:
    """Build a schema-preserving table, including the empty-row case."""
    if rows:
        return pa.Table.from_pylist(rows, schema=schema)
    return pa.table({column: [] for column in columns}, schema=schema)


def enforce_wikivoyage_integrity(
    data_root: DataRoot, stem: str, *, dry_run: bool = False
) -> WikivoyageIntegrityResult:
    """Reject every wikivoyage document whose wikidata is absent from
    the shard's polygons, and cascade the rejection to its sections.

    The polygons parquet is the source of truth for the valid
    wikidata QID set. Sections whose ``document_id`` belongs to a
    rejected document are dropped; the cascade count is recorded in
    each rejection record.

    When at least one document or section is rejected the
    ``wikivoyage/documents`` and ``wikivoyage/sections`` parquets
    are atomically rewritten. When nothing is rejected the parquets
    are left untouched (byte-identical). ``dry_run=True`` never
    rewrites either table.
    """
    documents_path, sections_path, documents_rows, sections_rows, valid_qids = (
        _load_wikivoyage_integrity_inputs(data_root, stem)
    )
    retained_documents, rejected_document_ids, rejections = _partition_wikivoyage_documents(
        stem, documents_rows, valid_qids
    )
    retained_sections, cascaded_count, cascades_by_document = _partition_wikivoyage_sections(
        sections_rows, rejected_document_ids
    )
    rejections = attach_cascade_counts(rejections, cascades_by_document)

    original_document_count = len(documents_rows)
    retained_document_count = len(retained_documents)
    rejected_document_count = len(rejections)
    original_section_count = len(sections_rows)
    retained_section_count = len(retained_sections)
    rewritten_documents = rejected_document_count > 0
    rewritten_sections = cascaded_count > 0

    _write_wikivoyage_integrity_tables(
        documents_path,
        sections_path,
        retained_documents,
        retained_sections,
        rewrite_documents=rewritten_documents and not dry_run,
        rewrite_sections=rewritten_sections and not dry_run,
    )

    rejections_tuple = tuple(
        sorted(rejections, key=lambda record: (record.identifier, record.wikidata))
    )
    return WikivoyageIntegrityResult(
        shard=stem,
        original_document_count=original_document_count,
        retained_document_count=retained_document_count,
        rejected_document_count=rejected_document_count,
        original_section_count=original_section_count,
        retained_section_count=retained_section_count,
        cascaded_section_count=cascaded_count,
        rewritten_documents=rewritten_documents,
        rewritten_sections=rewritten_sections,
        rejections=rejections_tuple,
    )


# ---------------------------------------------------------------------------
# enforce_all_regions
# ---------------------------------------------------------------------------


def _collect_integrity_results(
    data_root: DataRoot,
    stems: list[str],
    *,
    dry_run: bool = False,
) -> tuple[list[PolygonArticlesIntegrityResult], list[WikivoyageIntegrityResult]]:
    """Run available integrity checks for each requested shard."""
    polygon_results: list[PolygonArticlesIntegrityResult] = []
    wikivoyage_results: list[WikivoyageIntegrityResult] = []
    for stem in stems:
        links_path = data_root.processed_links / f"{stem}.parquet"
        if links_path.is_file():
            polygon_results.append(
                enforce_polygon_articles_integrity(data_root, stem, dry_run=dry_run)
            )
        wikivoyage_documents_path = (
            data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
        )
        if wikivoyage_documents_path.is_file():
            wikivoyage_results.append(
                enforce_wikivoyage_integrity(data_root, stem, dry_run=dry_run)
            )
    return polygon_results, wikivoyage_results


def _integrity_audit_payload(
    report: IntegrityReport,
    polygon_results: list[PolygonArticlesIntegrityResult],
    wikivoyage_results: list[WikivoyageIntegrityResult],
) -> dict[str, Any]:
    """Build the deterministic JSON payload for an integrity report."""
    return {
        "contract_version": INTEGRITY_CONTRACT_VERSION,
        "generated_at": _utc_now_iso(),
        "polygon_articles": [result.to_dict() for result in polygon_results],
        "wikivoyage": [result.to_dict() for result in wikivoyage_results],
        "totals": {
            "polygon_articles_rejected": report.total_polygon_articles_rejected,
            "wikivoyage_documents_rejected": report.total_wikivoyage_documents_rejected,
            "wikivoyage_sections_cascaded": report.total_wikivoyage_sections_cascaded,
            "shards_with_rejections": sorted(
                {
                    result.shard
                    for result in polygon_results + wikivoyage_results
                    if result.rejections
                }
            ),
        },
    }


def enforce_all_regions(
    data_root: DataRoot,
    *,
    stems: list[str] | None = None,
    audit_filename: str = "integrity_audit.json",
    dry_run: bool = False,
) -> IntegrityReport:
    """Run both integrity checks across every shard and emit a
    deterministic audit JSON.

    When *stems* is ``None`` the union of polygon stems is used
    (the canonical intersection of polygons and either
    polygon_articles or wikivoyage/documents). The audit JSON is
    written to ``<data_root>/processed/integrity/<audit_filename>``
    with deterministic key order.

    With ``dry_run=True`` the report is computed but nothing is written:
    no parquet is rewritten and the audit JSON is not created.
    """
    if stems is None:
        stems = sorted(path.stem for path in data_root.processed_polygons.glob("*.parquet"))

    polygon_results, wikivoyage_results = _collect_integrity_results(
        data_root, stems, dry_run=dry_run
    )

    polygon_results.sort(key=lambda result: result.shard)
    wikivoyage_results.sort(key=lambda result: result.shard)

    report = IntegrityReport(
        contract_version=INTEGRITY_CONTRACT_VERSION,
        polygon_articles=tuple(polygon_results),
        wikivoyage=tuple(wikivoyage_results),
        audit_path=data_root.processed / "integrity" / audit_filename,
    )

    if dry_run:
        return report
    audit_dir = report.audit_path.parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = _integrity_audit_payload(report, polygon_results, wikivoyage_results)
    atomic_write_text(report.audit_path, json.dumps(audit_payload, indent=2, sort_keys=True) + "\n")
    return report


__all__ = [
    "INTEGRITY_CONTRACT_VERSION",
    "REASON_POLYGON_ARTICLES_MISMATCH",
    "REASON_WIKIVOYAGE_ABSENT",
    "IntegrityReport",
    "PolygonArticlesIntegrityResult",
    "RejectionRecord",
    "WikivoyageIntegrityResult",
    "enforce_all_regions",
    "enforce_polygon_articles_integrity",
    "enforce_wikivoyage_integrity",
]
