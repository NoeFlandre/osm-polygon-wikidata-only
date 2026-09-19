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
    _build_and_checkpoint,
    _build_batch_artifacts,
    _collect_recovery_futures,
    _recovery_qid_batches,
    _run_missing_recovery_batches,
)
from .repair_artifacts import _execute_recovery_batches as _execute_recovery_batches_impl
from .repair_fetch import (
    RECOVERY_NETWORK_WORKERS,
    _advance_missing_document,
    _document_from_recovery_article,
    _eligible_sitelinks,
    _fetch_missing_documents,
    _fetch_missing_documents_parallel,
    _fetch_qid_documents,
    _fetch_recovery_article,
    _fetch_recovery_document,
    _language_sitelinks,
    _limit_sitelinks,
    _missing_recovery_article,
    _parse_recovery_document,
    _parse_recovery_documents,
    _resolve_entities,
    _resolved_entity_map,
    _sections_for_new_documents,
    _select_new_documents,
    _validate_recovery_fetch,
)
from .repair_fields import (
    _apply_best_language_links,
    _has_article_text,
    _preferred_language,
    _recompute_affected_polygon_fields,
    _recompute_polygon_row,
    _recompute_polygon_rows,
    _summarize_polygon_links,
)
from .repair_inputs import (
    _load_repair_inputs,
    _load_repair_links,
    _orphan_article_ids,
    _repair_checkpoint_store,
    _retain_documents,
    _retain_facts,
    _retain_links,
    _retain_repair_rows,
    _retain_sections,
)
from .repair_merge import (
    _affected_polygon_ids,
    _append_merge_rows,
    _flatten_batch_rows,
    _flatten_recovery_batches,
    _merge_repair_outputs,
    _merge_repair_tables,
    _merge_rows,
    _persisted_repair_links,
    _removed_section_ids,
    _repair_change_flags,
    _sort_repair_tables,
    _terminal_classification,
    _terminal_classifications,
    _validate_merge_rows,
    _validate_merged_repair,
)
from .repair_outputs import (
    _polygon_qids,
    _processed_manifest_statistics,
    _stage_augmentation_manifest,
    _stage_manifests,
    _stage_processed_manifest,
    _stage_repair_tables,
    _staged_repair_paths,
)
from .repair_outputs import persist_repair_outputs as _persist_repair_outputs_impl
from .repair_types import _RepairInputs, _RepairOutputs
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
