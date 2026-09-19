"""Audit Wikidata-backed polygon data and classify recovery state.

The orchestration and receipt boundary stay here for compatibility; tabular
input validation, upstream resolution, and receipt codecs live in focused
siblings.
"""

# ruff: noqa: F401

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.steps import sha256_file
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.enrichment.wikidata.models import (
    WikidataClient,
    WikidataEntity,
)

from .audit_entities import (
    affected_qid_entry,
    blocked_region_result,
    build_qid_result,
    classified_region_result,
    classify_qid,
    classify_region,
    collect_qid_data,
    collect_region_qid_data,
    eligible_sitelinks,
    filtered_sitelinks,
    global_qid_results,
    limit_sitelinks,
    progress_checkpoint,
    resolve_entities,
)
from .audit_inputs import (
    index_document_rows,
    index_polygon_rows,
    link_identity,
    linked_polygon_qids,
    load_region_rows,
    missing_polygon_links,
    orphan_fact_ids,
    read_rows,
    reference_column,
    region_paths,
    remember_link_identity,
    require_schema,
    resolve_link_target,
    validate_link_row,
    validate_polygon_qid,
)
from .audit_receipts import (
    INDEX_RELATIVE_PATH,
    RECOVERY_CONTRACT_VERSION,
    decode_receipts,
    fingerprints_match,
    load_receipts,
    parse_receipt_classifications,
    parse_receipt_entries,
    parse_receipt_polygon_ids,
    receipt_fields,
    receipt_from_result,
    receipt_maps,
    receipt_needs_repair,
    receipt_result,
    record_recovery_receipt,
    reuse_receipt,
    save_receipts,
    store_receipt,
    validate_receipt_classifications,
)
from .audit_runtime import (
    classify_scoped_regions,
    eligible_sitelinks_by_qid,
    emit_audit_complete,
    record_current_receipt,
    save_changed_receipts,
    validate_batch_size,
    validation_progress,
)
from .audit_scanning import ScanHooks, scan_regions, validation_qids
from .audit_types import RegionRows, RegionScan, ScanError
from .models import (
    QidAuditResult,
    RecoveryAuditResult,
    RecoveryClassification,
    RegionAuditResult,
)

LOGGER = logging.getLogger(__name__)
_UPSTREAM_BATCH_WINDOW = 3

_RegionScan = RegionScan
_RegionRows = RegionRows
_ScanError = ScanError
_INDEX_RELATIVE_PATH = INDEX_RELATIVE_PATH
_region_paths = region_paths
_load_region_rows = load_region_rows
_index_polygon_rows = index_polygon_rows
_index_document_rows = index_document_rows
_linked_polygon_qids = linked_polygon_qids
_missing_polygon_links = missing_polygon_links
_orphan_fact_ids = orphan_fact_ids
_require_schema = require_schema
_read_rows = read_rows
_link_identity = link_identity
_reference_key = reference_column
_remember_link_identity = remember_link_identity
_resolve_link_target = resolve_link_target
_validate_link_row = validate_link_row
_validate_polygon_qid = validate_polygon_qid
_classify_region = classify_region
_classified_region_result = classified_region_result
_blocked_region_result = blocked_region_result
_affected_qid_entry = affected_qid_entry
_classify_qid = classify_qid
_global_qid_results = global_qid_results
_collect_qid_data = collect_qid_data
_collect_region_qid_data = collect_region_qid_data
_build_qid_result = build_qid_result
_eligible_sitelinks = eligible_sitelinks
_filtered_sitelinks = filtered_sitelinks
_limit_sitelinks = limit_sitelinks
_progress_checkpoint = progress_checkpoint
_load_receipts = load_receipts
_decode_receipts = decode_receipts
_reuse_receipt = reuse_receipt
_receipt_fields = receipt_fields
_fingerprints_match = fingerprints_match
_receipt_maps = receipt_maps
_parse_receipt_entries = parse_receipt_entries
_parse_receipt_classifications = parse_receipt_classifications
_parse_receipt_polygon_ids = parse_receipt_polygon_ids
_receipt_needs_repair = receipt_needs_repair
_receipt_from_result = receipt_from_result
_validate_receipt_classifications = validate_receipt_classifications
_receipt_result = receipt_result
_store_receipt = store_receipt
_save_receipts = save_receipts
_validate_batch_size = validate_batch_size
_validation_progress = validation_progress
_eligible_sitelinks_by_qid = eligible_sitelinks_by_qid
_save_changed_receipts = save_changed_receipts
_emit_audit_complete = emit_audit_complete
_record_current_receipt = record_current_receipt
_validation_qids = validation_qids


def _resolve_entities(
    client: WikidataClient,
    qids: list[str],
    *,
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, WikidataEntity | None], int]:
    """Preserve the audit module's patchable ``as_completed`` seam."""
    return resolve_entities(
        client,
        qids,
        batch_size=batch_size,
        progress=progress,
        as_completed_fn=as_completed,
    )


def audit_wikidata_integrity(
    data_root: DataRoot,
    stems: list[str] | tuple[str, ...],
    client: WikidataClient,
    *,
    batch_size: int = 50,
    languages: tuple[str, ...] | None = None,
    max_articles_per_qid: int | None = None,
    log: Callable[[str], None] | None = None,
) -> RecoveryAuditResult:
    """Audit every scoped polygon QID using identities and authoritative outcomes."""
    _validate_batch_size(batch_size)
    emit = log or LOGGER.info
    scoped_stems = tuple(sorted(set(stems)))
    started_at = time.monotonic()
    emit(f"Wikidata integrity audit started: {len(scoped_stems)} finalized regions")
    index_path = data_root.cache / _INDEX_RELATIVE_PATH
    receipts, contract_matches = _load_receipts(index_path)
    scans, reused_results = _scan_regions(
        data_root,
        scoped_stems,
        receipts,
        contract_matches=contract_matches,
        started_at=started_at,
        emit=emit,
    )
    validation_qids = _validation_qids(scans)
    emit(
        "Wikidata integrity audit local scan complete: "
        f"{len(scoped_stems)} regions; {len(validation_qids)} QIDs require upstream validation; "
        f"{time.monotonic() - started_at:.0f}s elapsed"
    )

    entities, cache_hits = _resolve_entities(
        client,
        validation_qids,
        batch_size=batch_size,
        progress=_validation_progress(emit, batch_size, started_at),
    )
    eligible_sitelinks = _eligible_sitelinks_by_qid(
        entities,
        languages=languages,
        max_articles_per_qid=max_articles_per_qid,
    )

    region_results, changed_receipts = _classify_scoped_regions(
        scoped_stems,
        scans,
        reused_results,
        entities,
        eligible_sitelinks,
        receipts,
    )

    _save_changed_receipts(index_path, receipts, changed_receipts)

    qid_results = _global_qid_results(region_results, eligible_sitelinks)
    _emit_audit_complete(
        emit,
        region_results,
        qid_results,
        cache_hits,
        validation_qids,
        len(scoped_stems),
        started_at,
    )
    return RecoveryAuditResult(
        regions=tuple(region_results),
        qids=tuple(qid_results),
        upstream_validation_count=len(validation_qids),
        authoritative_cache_hits=cache_hits,
    )


def _region_fingerprints(data_root: DataRoot, stem: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (label, _fingerprint_path(path, required))
        for label, path, required in _region_paths(data_root, stem)
    )


def _fingerprint_path(path: Path, required: bool) -> str:
    if path.is_file():
        return sha256_file(path)
    return _missing_fingerprint(path, required)


def _missing_fingerprint(path: Path, required: bool) -> str:
    if required:
        raise FileNotFoundError(f"Required recovery input is missing: {path}")
    return ""


def _scan_regions(
    data_root: DataRoot,
    scoped_stems: tuple[str, ...],
    receipts: Mapping[str, object],
    *,
    contract_matches: bool,
    started_at: float,
    emit: Callable[[str], None],
) -> tuple[dict[str, _RegionScan], dict[str, RegionAuditResult]]:
    hooks = ScanHooks(
        region_fingerprints=_region_fingerprints,
        scan_region=_scan_region,
        progress_checkpoint=lambda completed, total, *, every: _progress_checkpoint(
            completed,
            total,
            every=every,
        ),
    )
    return scan_regions(
        data_root,
        scoped_stems,
        receipts,
        contract_matches=contract_matches,
        started_at=started_at,
        emit=emit,
        hooks=hooks,
    )


def _classify_scoped_regions(
    scoped_stems: tuple[str, ...],
    scans: Mapping[str, _RegionScan],
    reused_results: Mapping[str, RegionAuditResult],
    entities: Mapping[str, WikidataEntity | None],
    eligible_sitelinks: Mapping[str, tuple[tuple[str, str], ...]],
    receipts: dict[str, object],
) -> tuple[list[RegionAuditResult], bool]:
    return classify_scoped_regions(
        scoped_stems,
        scans,
        reused_results,
        entities,
        eligible_sitelinks,
        receipts,
        classify_region_fn=_classify_region,
    )


def _scan_region(
    data_root: DataRoot,
    stem: str,
    fingerprints: tuple[tuple[str, str], ...],
) -> _RegionScan:
    region_rows = _load_region_rows(data_root, stem)
    polygons, polygon_ids_by_qid = _index_polygon_rows(region_rows.polygon_rows)
    documents_by_article, documents_by_id, orphan_document_ids = _index_document_rows(
        region_rows.document_rows,
        polygon_ids_by_qid,
    )
    linked_polygon_qids = _linked_polygon_qids(
        region_rows.link_rows,
        canonical_links=region_rows.canonical_links,
        polygons=polygons,
        documents_by_article=documents_by_article,
        documents_by_id=documents_by_id,
        orphan_document_ids=orphan_document_ids,
    )
    missing = _missing_polygon_links(polygon_ids_by_qid, linked_polygon_qids)
    orphan_fact_ids = _orphan_fact_ids(region_rows.fact_rows, set(polygon_ids_by_qid))
    return _RegionScan(
        stem,
        fingerprints,
        tuple((qid, tuple(sorted(ids))) for qid, ids in sorted(polygon_ids_by_qid.items())),
        missing,
        tuple(sorted(orphan_fact_ids)),
        tuple(sorted(orphan_document_ids)),
    )


def record_region_recovery_receipt(
    data_root: DataRoot,
    stem: str,
    classifications: Mapping[str, RecoveryClassification],
) -> RegionAuditResult:
    fingerprints = _region_fingerprints(data_root, stem)
    scan = _scan_region(data_root, stem, fingerprints)
    return record_recovery_receipt(
        data_root,
        stem,
        classifications,
        fingerprints=fingerprints,
        scan=scan,
    )


__all__ = [
    "RECOVERY_CONTRACT_VERSION",
    "audit_wikidata_integrity",
    "record_region_recovery_receipt",
]
