"""Build recoverable Wikidata artifacts and durable recovery batches."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.steps import (
    AugmentationClient,
    build_wikidata_facts,
)
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot
from osm_polygon_wikidata_only.utils.retry import (
    _cancel_pending_retries,
    _reset_retry_cancellation,
)

from .checkpoints import (
    RECOVERY_QID_BATCH_SIZE,
    RecoveryBatchArtifacts,
    RecoveryCheckpointStore,
)
from .models import RecoveryRepairError
from .progress import RecoveryHeartbeat, RecoveryProgress
from .repair_fetch import (
    _fetch_missing_documents,
    _resolve_entities,
    _sections_for_new_documents,
)

RECOVERY_NETWORK_WORKERS = 8
RECOVERY_BATCH_WINDOW = 3


def _recovery_qid_batches(affected_qids: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Split affected QIDs into deterministic checkpoint-sized batches."""
    return [
        affected_qids[start : start + RECOVERY_QID_BATCH_SIZE]
        for start in range(0, len(affected_qids), RECOVERY_QID_BATCH_SIZE)
    ]


def _load_recovery_checkpoints(
    stem: str,
    batches: list[tuple[str, ...]],
    checkpoint_store: RecoveryCheckpointStore,
    emit: Callable[[str], None],
) -> tuple[dict[int, RecoveryBatchArtifacts], list[tuple[int, tuple[str, ...]]]]:
    """Load reusable recovery checkpoints and return missing batches."""
    completed: dict[int, RecoveryBatchArtifacts] = {}
    missing: list[tuple[int, tuple[str, ...]]] = []
    for index, batch_qids in enumerate(batches):
        artifacts = checkpoint_store.load(index, batch_qids)
        if artifacts is None:
            missing.append((index, batch_qids))
            continue
        completed[index] = artifacts
        emit(
            f"Wikidata recovery {stem}: batch {index + 1}/{len(batches)} "
            f"reused durable checkpoint ({len(batch_qids)} QIDs)"
        )
    return completed, missing


def _build_and_checkpoint(
    index: int,
    batch_qids: tuple[str, ...],
    *,
    stem: str,
    batch_total: int,
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None,
) -> tuple[int, RecoveryBatchArtifacts]:
    """Build one recovery batch and persist its durable checkpoint."""
    progress = RecoveryProgress(stem, batch_total, scheduler_snapshot=scheduler_snapshot)
    progress.start_batch(index + 1, batch_qids)
    with RecoveryHeartbeat(progress, emit):
        artifacts = build_batch(batch_qids, progress)
    checkpoint_store.save(index, artifacts)
    progress.checkpoint_saved(
        documents=len(artifacts.documents),
        sections=len(artifacts.sections),
        facts=len(artifacts.facts),
    )
    emit(progress.message())
    return index, artifacts


def _collect_recovery_futures(
    futures: list[Future[tuple[int, RecoveryBatchArtifacts]]],
    completed: dict[int, RecoveryBatchArtifacts],
    *,
    as_completed_fn: Callable[[object], Any] | None = None,
) -> None:
    """Collect completed recovery futures and cancel the remainder on failure."""
    completion = as_completed if as_completed_fn is None else as_completed_fn
    try:
        for future in completion(futures):
            index, artifacts = future.result()
            completed[index] = artifacts
    except BaseException:
        _cancel_pending_retries()
        for future in futures:
            future.cancel()
        raise


def _run_missing_recovery_batches(
    missing: list[tuple[int, tuple[str, ...]]],
    completed: dict[int, RecoveryBatchArtifacts],
    *,
    stem: str,
    batch_total: int,
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None,
    batch_window: int,
    as_completed_fn: Callable[[object], Any] | None = None,
) -> None:
    """Build missing batches concurrently while preserving retry cancellation."""
    _reset_retry_cancellation()
    try:
        with ThreadPoolExecutor(max_workers=min(batch_window, len(missing))) as executor:
            futures = [
                executor.submit(
                    _build_and_checkpoint,
                    index,
                    batch_qids,
                    stem=stem,
                    batch_total=batch_total,
                    checkpoint_store=checkpoint_store,
                    build_batch=build_batch,
                    emit=emit,
                    scheduler_snapshot=scheduler_snapshot,
                )
                for index, batch_qids in missing
            ]
            _collect_recovery_futures(
                futures,
                completed,
                as_completed_fn=as_completed_fn,
            )
    finally:
        _reset_retry_cancellation()


def _execute_recovery_batches(
    *,
    stem: str,
    affected_qids: tuple[str, ...],
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None = None,
    batch_window: int = RECOVERY_BATCH_WINDOW,
    as_completed_fn: Callable[[object], Any] | None = None,
) -> list[RecoveryBatchArtifacts]:
    """Build independent recovery batches concurrently and return input order."""
    if batch_window < 1:
        raise ValueError("batch_window must be at least 1")
    batches = _recovery_qid_batches(affected_qids)
    batch_total = len(batches)
    completed, missing = _load_recovery_checkpoints(stem, batches, checkpoint_store, emit)

    if missing:
        _run_missing_recovery_batches(
            missing,
            completed,
            stem=stem,
            batch_total=batch_total,
            checkpoint_store=checkpoint_store,
            build_batch=build_batch,
            emit=emit,
            scheduler_snapshot=scheduler_snapshot,
            batch_window=batch_window,
            as_completed_fn=as_completed_fn,
        )
    return [completed[index] for index in range(batch_total)]


def _build_batch_artifacts(
    qids: tuple[str, ...],
    *,
    existing_documents: list[dict[str, Any]],
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    augmentation_client: AugmentationClient,
    settings: Settings,
    progress: RecoveryProgress,
) -> RecoveryBatchArtifacts:
    progress.set_stage("Wikidata entities", total=len(qids))
    entities = _resolve_entities(wikidata_client, qids)
    progress.advance(len(qids))
    documents = _fetch_missing_documents(
        qids,
        entities=entities,
        existing_documents=existing_documents,
        wikipedia_client=wikipedia_client,
        settings=settings,
        progress=progress,
    )
    document_ids = {str(row["document_id"]) for row in documents}
    sections = _sections_for_new_documents(
        documents,
        document_ids,
        augmentation_client=augmentation_client,
        progress=progress,
    )
    progress.set_stage("Wikidata facts", total=len(qids))
    raw_entities = augmentation_client.entities(list(qids), props="sitelinks|claims")
    missing_raw = sorted(set(qids) - set(raw_entities))
    if missing_raw:
        raise RecoveryRepairError(f"Augmentation Wikidata response omitted QIDs: {missing_raw}")
    facts = [
        fact.to_dict()
        for fact in build_wikidata_facts(
            augmentation_client,
            entities={qid: raw_entities[qid] for qid in qids},
            progress=AugmentationProgress(),
        )
    ]
    progress.advance(len(qids), facts=len(facts))
    return RecoveryBatchArtifacts(
        qids=qids,
        documents=tuple(documents),
        sections=tuple(sections),
        facts=tuple(facts),
    )
