"""Compatibility facade for the staged Wikidata enrichment recovery workflow.

Private imports intentionally preserve the historical module seams while the
implementation is split into input, artifact, merge, and output modules.
"""

# ruff: noqa: F401

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import as_completed
from typing import Any

from osm_polygon_wikidata_only.augmentation.steps import AugmentationClient
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.audit import (
    audit_wikidata_integrity,
    record_region_recovery_receipt,
)
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot

from .checkpoints import RecoveryBatchArtifacts, RecoveryCheckpointStore
from .models import (
    RecoveryRepairError,
    RecoveryRepairResult,
    RegionAuditResult,
)
from .progress import RecoveryProgress
from .repair_artifacts import (
    RECOVERY_BATCH_WINDOW,
)
from .repair_artifacts import (
    build_and_checkpoint as _build_and_checkpoint,
)
from .repair_artifacts import (
    build_batch_artifacts as _build_batch_artifacts,
)
from .repair_artifacts import (
    collect_recovery_futures as _collect_recovery_futures,
)
from .repair_artifacts import execute_recovery_batches as _execute_recovery_batches_impl
from .repair_artifacts import (
    recovery_qid_batches as _recovery_qid_batches,
)
from .repair_artifacts import (
    run_missing_recovery_batches as _run_missing_recovery_batches,
)
from .repair_fetch import (
    RECOVERY_NETWORK_WORKERS,
)
from .repair_fetch import (
    advance_missing_document as _advance_missing_document,
)
from .repair_fetch import (
    document_from_recovery_article as _document_from_recovery_article,
)
from .repair_fetch import (
    eligible_sitelinks as _eligible_sitelinks,
)
from .repair_fetch import (
    fetch_missing_documents as _fetch_missing_documents,
)
from .repair_fetch import (
    fetch_missing_documents_parallel as _fetch_missing_documents_parallel,
)
from .repair_fetch import (
    fetch_qid_documents as _fetch_qid_documents,
)
from .repair_fetch import (
    fetch_recovery_article as _fetch_recovery_article,
)
from .repair_fetch import (
    fetch_recovery_document as _fetch_recovery_document,
)
from .repair_fetch import (
    language_sitelinks as _language_sitelinks,
)
from .repair_fetch import (
    limit_sitelinks as _limit_sitelinks,
)
from .repair_fetch import (
    missing_recovery_article as _missing_recovery_article,
)
from .repair_fetch import (
    parse_recovery_document as _parse_recovery_document,
)
from .repair_fetch import (
    parse_recovery_documents as _parse_recovery_documents,
)
from .repair_fetch import (
    resolve_entities as _resolve_entities,
)
from .repair_fetch import (
    resolved_entity_map as _resolved_entity_map,
)
from .repair_fetch import (
    sections_for_new_documents as _sections_for_new_documents,
)
from .repair_fetch import (
    select_new_documents as _select_new_documents,
)
from .repair_fetch import (
    validate_recovery_fetch as _validate_recovery_fetch,
)
from .repair_fields import (
    apply_best_language_links as _apply_best_language_links,
)
from .repair_fields import (
    has_article_text as _has_article_text,
)
from .repair_fields import (
    preferred_language as _preferred_language,
)
from .repair_fields import (
    recompute_affected_polygon_fields as _recompute_affected_polygon_fields,
)
from .repair_fields import (
    recompute_polygon_row as _recompute_polygon_row,
)
from .repair_fields import (
    recompute_polygon_rows as _recompute_polygon_rows,
)
from .repair_fields import (
    summarize_polygon_links as _summarize_polygon_links,
)
from .repair_inputs import (
    load_repair_inputs as _load_repair_inputs,
)
from .repair_inputs import (
    load_repair_links as _load_repair_links,
)
from .repair_inputs import (
    orphan_article_ids as _orphan_article_ids,
)
from .repair_inputs import (
    repair_checkpoint_store as _repair_checkpoint_store,
)
from .repair_inputs import (
    retain_documents as _retain_documents,
)
from .repair_inputs import (
    retain_facts as _retain_facts,
)
from .repair_inputs import (
    retain_links as _retain_links,
)
from .repair_inputs import (
    retain_repair_rows as _retain_repair_rows,
)
from .repair_inputs import (
    retain_sections as _retain_sections,
)
from .repair_merge import (
    affected_polygon_ids as _affected_polygon_ids,
)
from .repair_merge import (
    append_merge_rows as _append_merge_rows,
)
from .repair_merge import (
    flatten_batch_rows as _flatten_batch_rows,
)
from .repair_merge import (
    flatten_recovery_batches as _flatten_recovery_batches,
)
from .repair_merge import (
    merge_repair_outputs as _merge_repair_outputs,
)
from .repair_merge import (
    merge_repair_tables as _merge_repair_tables,
)
from .repair_merge import (
    merge_rows as _merge_rows,
)
from .repair_merge import (
    persisted_repair_links as _persisted_repair_links,
)
from .repair_merge import (
    removed_section_ids as _removed_section_ids,
)
from .repair_merge import (
    repair_change_flags as _repair_change_flags,
)
from .repair_merge import (
    sort_repair_tables as _sort_repair_tables,
)
from .repair_merge import (
    terminal_classification as _terminal_classification,
)
from .repair_merge import (
    terminal_classifications as _terminal_classifications,
)
from .repair_merge import (
    validate_merge_rows as _validate_merge_rows,
)
from .repair_merge import (
    validate_merged_repair as _validate_merged_repair,
)
from .repair_outputs import persist_repair_outputs as _persist_repair_outputs_impl
from .repair_outputs import (
    polygon_qids as _polygon_qids,
)
from .repair_outputs import (
    processed_manifest_statistics as _processed_manifest_statistics,
)
from .repair_outputs import (
    stage_augmentation_manifest as _stage_augmentation_manifest,
)
from .repair_outputs import (
    stage_manifests as _stage_manifests,
)
from .repair_outputs import (
    stage_processed_manifest as _stage_processed_manifest,
)
from .repair_outputs import (
    stage_repair_tables as _stage_repair_tables,
)
from .repair_outputs import (
    staged_repair_paths as _staged_repair_paths,
)
from .repair_types import RepairInputs as _RepairInputs
from .repair_types import RepairOutputs as _RepairOutputs
from .transaction import recover_interrupted_transactions


def _execute_recovery_batches(
    *,
    stem: str,
    affected_qids: tuple[str, ...],
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None = None,
    batch_window: int = RECOVERY_BATCH_WINDOW,
) -> list[RecoveryBatchArtifacts]:
    """Preserve the historic monkeypatch seam for ``as_completed``."""
    return _execute_recovery_batches_impl(
        stem=stem,
        affected_qids=affected_qids,
        checkpoint_store=checkpoint_store,
        build_batch=build_batch,
        emit=emit,
        scheduler_snapshot=scheduler_snapshot,
        batch_window=batch_window,
        as_completed_fn=as_completed,
    )


def _build_repair_outputs(
    data_root: DataRoot,
    region: RegionAuditResult,
    inputs: _RepairInputs,
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    augmentation_client: AugmentationClient,
    settings: Settings,
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None,
) -> tuple[_RepairOutputs, RecoveryCheckpointStore]:
    """Build repaired rows from durable recovery batches before staging."""
    affected_qids = tuple(sorted(region.affected_qids))
    checkpoint_store = _repair_checkpoint_store(data_root, region, inputs, settings, affected_qids)

    def build_batch(
        batch_qids: tuple[str, ...],
        progress: RecoveryProgress,
    ) -> RecoveryBatchArtifacts:
        return _build_batch_artifacts(
            batch_qids,
            existing_documents=inputs.retained_documents,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            augmentation_client=augmentation_client,
            settings=settings,
            progress=progress,
        )

    completed_batches = _execute_recovery_batches(
        stem=region.stem,
        affected_qids=affected_qids,
        checkpoint_store=checkpoint_store,
        build_batch=build_batch,
        emit=emit,
        scheduler_snapshot=scheduler_snapshot,
    )
    return _merge_repair_outputs(region, inputs, completed_batches), checkpoint_store


def _persist_repair_outputs(
    data_root: DataRoot,
    region: RegionAuditResult,
    inputs: _RepairInputs,
    outputs: _RepairOutputs,
    checkpoint_store: RecoveryCheckpointStore,
    *,
    transaction_root: Any,
    wikidata_client: WikidataClient,
    settings: Settings,
    before_commit: Callable[[], None] | None,
) -> RecoveryRepairResult:
    """Preserve facade-level audit and receipt monkeypatch seams."""
    return _persist_repair_outputs_impl(
        data_root,
        region,
        inputs,
        outputs,
        checkpoint_store,
        transaction_root=transaction_root,
        wikidata_client=wikidata_client,
        settings=settings,
        before_commit=before_commit,
        audit_fn=audit_wikidata_integrity,
        record_receipt_fn=record_region_recovery_receipt,
    )


def repair_wikidata_region(
    data_root: DataRoot,
    region: RegionAuditResult,
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    augmentation_client: AugmentationClient,
    settings: Settings,
    before_commit: Callable[[], None] | None = None,
    log: Callable[[str], None] | None = None,
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None = None,
) -> RecoveryRepairResult:
    """Repair only the affected QID relationships in one finalized shard."""
    if region.blocked_reason:
        raise RecoveryRepairError(region.blocked_reason)
    if not region.requires_repair:
        return RecoveryRepairResult(region.stem, False, (), 0, (), False)
    transaction_root = data_root.cache / "wikidata_recovery" / "transactions"
    recover_interrupted_transactions(transaction_root)
    emit = log or (lambda _message: None)
    inputs = _load_repair_inputs(data_root, region)
    outputs, checkpoint_store = _build_repair_outputs(
        data_root,
        region,
        inputs,
        wikidata_client=wikidata_client,
        wikipedia_client=wikipedia_client,
        augmentation_client=augmentation_client,
        settings=settings,
        emit=emit,
        scheduler_snapshot=scheduler_snapshot,
    )
    return _persist_repair_outputs(
        data_root,
        region,
        inputs,
        outputs,
        checkpoint_store,
        transaction_root=transaction_root,
        wikidata_client=wikidata_client,
        settings=settings,
        before_commit=before_commit,
    )


__all__ = ["RecoveryRepairError", "RecoveryRepairResult", "repair_wikidata_region"]
