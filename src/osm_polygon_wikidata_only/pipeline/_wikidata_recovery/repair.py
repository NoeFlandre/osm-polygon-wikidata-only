"""Repair Wikidata enrichment artifacts for a region."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.models import document_from_article_row
from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.schema import (
    FACT_COLUMNS,
    SECTION_COLUMNS,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.sections import parse_sections
from osm_polygon_wikidata_only.augmentation.steps import (
    CONTRACT_VERSION,
    AugmentationClient,
    build_wikidata_facts,
    sha256_file,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WIKIPEDIA_DOCUMENT_COLUMNS,
    wikipedia_document_from_article_row,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import article_id
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    CANONICAL_COLUMNS,
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
    polygon_article_schema,
    polygon_schema,
)
from osm_polygon_wikidata_only.enrichment.article_linker import PREFERRED_LANGUAGES, LinkSummary
from osm_polygon_wikidata_only.enrichment.wikidata.models import (
    BatchWikidataClient,
    WikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    language_from_site,
    qids_from_osm_tag,
)
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.completeness import NON_FATAL_FETCH_STATUSES
from osm_polygon_wikidata_only.pipeline.row_construction import article_row
from osm_polygon_wikidata_only.utils.json import dumps
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot
from osm_polygon_wikidata_only.utils.retry import (
    _cancel_pending_retries,
    _reset_retry_cancellation,
)

from .audit import (
    RECOVERY_CONTRACT_VERSION,
    audit_wikidata_integrity,
    record_region_recovery_receipt,
)
from .checkpoints import (
    RECOVERY_QID_BATCH_SIZE,
    RecoveryBatchArtifacts,
    RecoveryCheckpointStore,
    recovery_plan_key,
)
from .link_rows import (
    canonical_wikipedia_links_to_legacy as _canonical_wikipedia_links_to_legacy,
)
from .link_rows import (
    legacy_wikipedia_links_to_canonical as _legacy_wikipedia_links_to_canonical,
)
from .link_rows import merge_links as _merge_links
from .models import (
    RecoveryClassification,
    RecoveryRepairError,
    RecoveryRepairResult,
    RegionAuditResult,
)
from .progress import RecoveryHeartbeat, RecoveryProgress
from .storage import read_table as _read_table
from .storage import region_paths as _region_paths
from .storage import write_table as _write_table
from .transaction import (
    commit_replacements,
    recover_interrupted_transactions,
    transaction_directory,
)
from .validation import (
    validate_existing_rows as _validate_existing_rows,
)
from .validation import (
    validate_preservation as _validate_preservation,
)

RECOVERY_NETWORK_WORKERS = 8
RECOVERY_BATCH_WINDOW = 3


@dataclass(frozen=True, slots=True)
class _RepairInputs:
    """Validated source tables and rows retained for a regional repair."""

    paths: dict[str, Path]
    polygons: list[dict[str, Any]]
    stored_links: list[dict[str, Any]]
    links: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    preserved_wikivoyage_links: list[dict[str, Any]]
    canonical_links: bool
    retained_facts: list[dict[str, Any]]
    retained_documents: list[dict[str, Any]]
    retained_sections: list[dict[str, Any]]
    retained_links: list[dict[str, Any]]
    orphan_document_ids: set[str]


@dataclass(frozen=True, slots=True)
class _RepairOutputs:
    """Merged and validated tables ready for transactional persistence."""

    updated_polygons: list[dict[str, Any]]
    persisted_links: list[dict[str, Any]]
    merged_documents: list[dict[str, Any]]
    merged_sections: list[dict[str, Any]]
    merged_facts: list[dict[str, Any]]
    terminal_classifications: dict[str, RecoveryClassification]
    affected_qids: tuple[str, ...]
    affected_polygon_ids: set[str]
    map_inputs_changed: bool
    changed: bool


def _load_repair_links(
    paths: dict[str, Path],
    *,
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Load either canonical or legacy links and preserve Wikivoyage rows."""
    links_schema = pq.read_schema(paths["links"])  # type: ignore[no-untyped-call]
    canonical_links = links_schema.equals(polygon_document_link_schema(), check_metadata=True)
    if canonical_links:
        stored_links = _read_table(paths["links"], polygon_document_link_schema())
        preserved = [dict(row) for row in stored_links if row["project"] == "wikivoyage"]
        links = _canonical_wikipedia_links_to_legacy(
            stored_links,
            documents,
            polygons,
            affected_qids=affected_qids,
        )
        return stored_links, links, preserved, True
    if links_schema.equals(polygon_article_schema(), check_metadata=True):
        stored_links = _read_table(paths["links"], polygon_article_schema())
        return stored_links, stored_links, [], False
    raise RecoveryRepairError(f"Recovery input schema mismatch: {paths['links']}")


def _retain_repair_rows(
    region: RegionAuditResult,
    *,
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]
]:
    """Drop planned orphan rows and validate that the plan matches its inputs."""
    orphan_fact_ids = set(region.orphan_fact_ids)
    orphan_document_ids = set(region.orphan_document_ids)
    retained_facts = _retain_facts(facts, orphan_fact_ids)
    if len(facts) - len(retained_facts) != len(orphan_fact_ids):
        raise RecoveryRepairError("Recovery plan contains stale or duplicate orphan fact IDs")
    orphan_article_ids = _orphan_article_ids(documents, orphan_document_ids)
    retained_documents = _retain_documents(documents, orphan_document_ids)
    if len(documents) - len(retained_documents) != len(orphan_document_ids):
        raise RecoveryRepairError("Recovery plan contains stale or duplicate orphan document IDs")
    retained_sections = _retain_sections(sections, orphan_document_ids)
    retained_links = _retain_links(links, orphan_article_ids)
    return (
        retained_facts,
        retained_documents,
        retained_sections,
        retained_links,
        orphan_document_ids,
    )


def _retain_facts(facts: list[dict[str, Any]], orphan_fact_ids: set[str]) -> list[dict[str, Any]]:
    """Keep facts not listed as planned orphans."""
    return [row for row in facts if str(row["fact_id"]) not in orphan_fact_ids]


def _orphan_article_ids(documents: list[dict[str, Any]], orphan_document_ids: set[str]) -> set[str]:
    """Find article identities belonging to orphan documents."""
    return {
        str(row["article_id"])
        for row in documents
        if str(row["document_id"]) in orphan_document_ids
    }


def _retain_documents(
    documents: list[dict[str, Any]], orphan_document_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep documents not listed as planned orphans."""
    return [row for row in documents if str(row["document_id"]) not in orphan_document_ids]


def _retain_sections(
    sections: list[dict[str, Any]], orphan_document_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep sections belonging to retained documents."""
    return [row for row in sections if str(row["document_id"]) not in orphan_document_ids]


def _retain_links(
    links: list[dict[str, Any]], orphan_article_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep links belonging to retained articles."""
    return [row for row in links if str(row["article_id"]) not in orphan_article_ids]


def _load_repair_inputs(data_root: DataRoot, region: RegionAuditResult) -> _RepairInputs:
    """Load, normalize, and validate all regional repair inputs."""
    paths = _region_paths(data_root, region.stem)
    polygons = _read_table(paths["polygons"], polygon_schema())
    documents = _read_table(paths["documents"], wikipedia_document_schema())
    sections = _read_table(paths["sections"], section_schema())
    facts = _read_table(paths["facts"], fact_schema())
    stored_links, links, preserved, canonical_links = _load_repair_links(
        paths,
        polygons=polygons,
        documents=documents,
        affected_qids=set(region.affected_qids),
    )
    retained_facts, retained_documents, retained_sections, retained_links, orphan_document_ids = (
        _retain_repair_rows(
            region,
            documents=documents,
            sections=sections,
            facts=facts,
            links=links,
        )
    )
    _validate_existing_rows(
        polygons,
        retained_links,
        retained_documents,
        retained_sections,
        retained_facts,
    )
    return _RepairInputs(
        paths=paths,
        polygons=polygons,
        stored_links=stored_links,
        links=links,
        documents=documents,
        sections=sections,
        facts=facts,
        preserved_wikivoyage_links=preserved,
        canonical_links=canonical_links,
        retained_facts=retained_facts,
        retained_documents=retained_documents,
        retained_sections=retained_sections,
        retained_links=retained_links,
        orphan_document_ids=orphan_document_ids,
    )


def _repair_checkpoint_store(
    data_root: DataRoot,
    region: RegionAuditResult,
    inputs: _RepairInputs,
    settings: Settings,
    affected_qids: tuple[str, ...],
) -> RecoveryCheckpointStore:
    """Create the checkpoint store keyed by all repair inputs and settings."""
    return RecoveryCheckpointStore(
        data_root.cache / "wikidata_recovery" / "checkpoints",
        region.stem,
        recovery_plan_key(
            fingerprints=region.fingerprints,
            affected_qids=affected_qids,
            sections_hash=sha256_file(inputs.paths["sections"]),
            settings_identity=(
                tuple(settings.languages) if settings.languages is not None else None,
                settings.max_articles_per_qid,
                settings.fetch_full_text,
            ),
        ),
    )


def _flatten_recovery_batches(
    completed_batches: list[RecoveryBatchArtifacts],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten checkpoint batches without changing their deterministic order."""
    return (
        _flatten_batch_rows(completed_batches, "documents"),
        _flatten_batch_rows(completed_batches, "sections"),
        _flatten_batch_rows(completed_batches, "facts"),
    )


def _flatten_batch_rows(
    completed_batches: list[RecoveryBatchArtifacts], attribute: str
) -> list[dict[str, Any]]:
    """Flatten one named row collection from recovery batches."""
    return [row for batch in completed_batches for row in getattr(batch, attribute)]


def _affected_polygon_ids(
    region: RegionAuditResult,
    affected_qids: tuple[str, ...],
) -> set[str]:
    """Collect polygon IDs whose QID relationships are being repaired."""
    return {
        polygon_id
        for qid, polygon_ids in region.affected_polygon_ids_by_qid
        if qid in affected_qids
        for polygon_id in polygon_ids
    }


def _sort_repair_tables(
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    """Apply the stable row ordering used by repaired artifacts."""
    documents.sort(key=lambda row: str(row["document_id"]))
    sections.sort(key=lambda row: (str(row["document_id"]), int(row["section_index"])))
    facts.sort(key=lambda row: str(row["fact_id"]))


def _removed_section_ids(sections: list[dict[str, Any]], orphan_document_ids: set[str]) -> set[str]:
    """Return section IDs cascaded from orphaned documents."""
    return {
        str(row["section_id"]) for row in sections if str(row["document_id"]) in orphan_document_ids
    }


def _merge_repair_tables(
    region: RegionAuditResult,
    inputs: _RepairInputs,
    completed_batches: list[RecoveryBatchArtifacts],
    affected_qids: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[str],
]:
    """Merge documents, links, sections, and facts from completed batches."""
    batch_documents, batch_sections, batch_facts = _flatten_recovery_batches(completed_batches)
    merged_documents, _ = _merge_rows(
        inputs.retained_documents,
        batch_documents,
        primary_key="document_id",
        label="document_id",
        secondary_key="article_id",
    )
    merged_links = _merge_links(
        inputs.polygons,
        inputs.retained_links,
        merged_documents,
        affected_qids=set(affected_qids),
    )
    updated_polygons, updated_links = _recompute_affected_polygon_fields(
        inputs.polygons,
        merged_links,
        merged_documents,
        affected_qids=set(affected_qids),
    )
    merged_sections, _ = _merge_rows(
        inputs.retained_sections,
        batch_sections,
        primary_key="section_id",
        label="section_id",
    )
    merged_facts, _ = _merge_rows(
        inputs.retained_facts,
        batch_facts,
        primary_key="fact_id",
        label="fact_id",
    )
    _sort_repair_tables(merged_documents, merged_sections, merged_facts)
    return (
        updated_polygons,
        updated_links,
        merged_documents,
        merged_sections,
        merged_facts,
        _affected_polygon_ids(region, affected_qids),
    )


def _validate_merged_repair(
    inputs: _RepairInputs,
    merged: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        set[str],
    ],
    affected_qids: tuple[str, ...],
) -> None:
    """Validate merged rows and preservation invariants before publication."""
    updated_polygons, updated_links, merged_documents, merged_sections, merged_facts, _ = merged
    _validate_existing_rows(
        updated_polygons,
        updated_links,
        merged_documents,
        merged_sections,
        merged_facts,
    )
    _validate_preservation(
        inputs.polygons,
        updated_polygons,
        inputs.documents,
        merged_documents,
        inputs.sections,
        merged_sections,
        inputs.retained_facts,
        merged_facts,
        affected_qids=set(affected_qids),
        removed_document_ids=inputs.orphan_document_ids,
        removed_section_ids=_removed_section_ids(inputs.sections, inputs.orphan_document_ids),
    )


def _persisted_repair_links(
    updated_links: list[dict[str, Any]],
    merged_documents: list[dict[str, Any]],
    inputs: _RepairInputs,
) -> list[dict[str, Any]]:
    """Restore canonical link rows when the input artifact uses that schema."""
    if not inputs.canonical_links:
        return updated_links
    return _legacy_wikipedia_links_to_canonical(
        updated_links,
        merged_documents,
        inputs.preserved_wikivoyage_links,
    )


def _repair_change_flags(
    inputs: _RepairInputs,
    updated_polygons: list[dict[str, Any]],
    persisted_links: list[dict[str, Any]],
    merged_documents: list[dict[str, Any]],
    merged_sections: list[dict[str, Any]],
    merged_facts: list[dict[str, Any]],
) -> tuple[bool, bool]:
    """Return total-change and map-input-change flags for repaired tables."""
    changed = any(
        before != after
        for before, after in (
            (inputs.polygons, updated_polygons),
            (inputs.stored_links, persisted_links),
            (inputs.documents, merged_documents),
            (inputs.sections, merged_sections),
            (inputs.facts, merged_facts),
        )
    )
    map_inputs_changed = any(
        before != after
        for before, after in (
            (inputs.polygons, updated_polygons),
            (inputs.stored_links, persisted_links),
            (inputs.documents, merged_documents),
        )
    )
    return changed, map_inputs_changed


def _merge_repair_outputs(
    region: RegionAuditResult,
    inputs: _RepairInputs,
    completed_batches: list[RecoveryBatchArtifacts],
) -> _RepairOutputs:
    """Merge recovered rows, recompute affected fields, and validate preservation."""
    affected_qids = tuple(sorted(region.affected_qids))
    merged = _merge_repair_tables(region, inputs, completed_batches, affected_qids)
    _validate_merged_repair(inputs, merged, affected_qids)
    (
        updated_polygons,
        updated_links,
        merged_documents,
        merged_sections,
        merged_facts,
        affected_polygon_ids,
    ) = merged
    persisted_links = _persisted_repair_links(updated_links, merged_documents, inputs)
    changed, map_inputs_changed = _repair_change_flags(
        inputs,
        updated_polygons,
        persisted_links,
        merged_documents,
        merged_sections,
        merged_facts,
    )
    terminal_classifications = _terminal_classifications(region, persisted_links)
    return _RepairOutputs(
        updated_polygons=updated_polygons,
        persisted_links=persisted_links,
        merged_documents=merged_documents,
        merged_sections=merged_sections,
        merged_facts=merged_facts,
        terminal_classifications=terminal_classifications,
        affected_qids=affected_qids,
        affected_polygon_ids=affected_polygon_ids,
        map_inputs_changed=map_inputs_changed,
        changed=changed,
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
    """Build all repaired tables from durable recovery batches."""
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


def _staged_repair_paths(directory: Path) -> dict[str, Path]:
    """Return deterministic temporary paths for repaired regional artifacts."""
    return {
        "polygons": directory / "staged-polygons.parquet",
        "links": directory / "staged-polygon-articles.parquet",
        "documents": directory / "staged-wikipedia-documents.parquet",
        "sections": directory / "staged-wikipedia-sections.parquet",
        "facts": directory / "staged-wikidata-facts.parquet",
        "processed_manifest": directory / "staged-processed-manifest.json",
        "augmentation_manifest": directory / "staged-augmentation-manifest.json",
    }


def _stage_repair_tables(
    stem: str,
    inputs: _RepairInputs,
    outputs: _RepairOutputs,
    *,
    paths: dict[str, Path],
    directory: Path,
) -> dict[str, Path]:
    """Write all repaired tables and manifests into a transaction directory."""
    staged = _staged_repair_paths(directory)
    _write_table(staged["polygons"], outputs.updated_polygons, POLYGON_COLUMNS, polygon_schema())
    if inputs.canonical_links:
        _write_table(
            staged["links"],
            outputs.persisted_links,
            CANONICAL_COLUMNS,
            polygon_document_link_schema(),
        )
    else:
        _write_table(
            staged["links"],
            outputs.persisted_links,
            POLYGON_ARTICLE_COLUMNS,
            polygon_article_schema(),
        )
    _write_table(
        staged["documents"],
        outputs.merged_documents,
        WIKIPEDIA_DOCUMENT_COLUMNS,
        wikipedia_document_schema(),
    )
    _write_table(staged["sections"], outputs.merged_sections, SECTION_COLUMNS, section_schema())
    _write_table(staged["facts"], outputs.merged_facts, FACT_COLUMNS, fact_schema())
    _stage_manifests(
        stem,
        paths=paths,
        staged=staged,
        polygons=outputs.updated_polygons,
        documents=outputs.merged_documents,
        sections=outputs.merged_sections,
        facts=outputs.merged_facts,
        affected_qids=outputs.affected_qids,
        affected_polygon_count=len(outputs.affected_polygon_ids),
    )
    return staged


def _persist_repair_outputs(
    data_root: DataRoot,
    region: RegionAuditResult,
    inputs: _RepairInputs,
    outputs: _RepairOutputs,
    checkpoint_store: RecoveryCheckpointStore,
    *,
    transaction_root: Path,
    wikidata_client: WikidataClient,
    settings: Settings,
    before_commit: Callable[[], None] | None,
) -> RecoveryRepairResult:
    """Persist changed repair outputs transactionally and verify convergence."""
    if not outputs.changed:
        record_region_recovery_receipt(data_root, region.stem, outputs.terminal_classifications)
        checkpoint_store.clear()
        return RecoveryRepairResult(
            region.stem,
            False,
            outputs.affected_qids,
            len(outputs.affected_polygon_ids),
            (),
            False,
        )
    directory = transaction_directory(transaction_root, region.stem)
    directory.mkdir(parents=True, exist_ok=False)
    staged = _stage_repair_tables(
        region.stem,
        inputs,
        outputs,
        paths=inputs.paths,
        directory=directory,
    )
    replacements = [(inputs.paths[key], staged[key]) for key in staged]
    commit_replacements(directory, region.stem, replacements, before_commit=before_commit)
    record_region_recovery_receipt(data_root, region.stem, outputs.terminal_classifications)
    post_audit = audit_wikidata_integrity(
        data_root,
        [region.stem],
        wikidata_client,
        batch_size=settings.enrichment_batch_size,
        languages=settings.languages,
        max_articles_per_qid=settings.max_articles_per_qid,
    )
    if post_audit.region(region.stem).affected_qids:
        raise RecoveryRepairError(f"Recovery did not converge for region {region.stem!r}")
    checkpoint_store.clear()
    repaired_paths = tuple(target for target, _ in replacements)
    return RecoveryRepairResult(
        region.stem,
        True,
        outputs.affected_qids,
        len(outputs.affected_polygon_ids),
        repaired_paths,
        outputs.map_inputs_changed,
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
) -> None:
    """Collect completed recovery futures and cancel the remainder on failure."""
    try:
        for future in as_completed(futures):
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
            _collect_recovery_futures(futures, completed)
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


def _resolved_entity_map(
    qids: tuple[str, ...],
    values: list[WikidataEntity | None],
) -> dict[str, WikidataEntity]:
    """Validate a Wikidata response and map each requested QID to its entity."""
    if len(values) != len(qids):
        raise RecoveryRepairError("Wikidata client returned the wrong result count")
    resolved: dict[str, WikidataEntity] = {}
    for qid, entity in zip(qids, values, strict=True):
        if entity is None:
            raise RecoveryRepairError(f"Affected QID became authoritatively missing: {qid}")
        resolved[qid] = entity
    return resolved


def _resolve_entities(
    client: WikidataClient,
    qids: tuple[str, ...],
) -> dict[str, WikidataEntity]:
    if isinstance(client, BatchWikidataClient):
        values = client.get_entities(qids)
    else:
        values = [client.get_entity(qid) for qid in qids]
    return _resolved_entity_map(qids, values)


def _eligible_sitelinks(entity: WikidataEntity, settings: Settings) -> list[tuple[str, str]]:
    allowed = set(settings.languages) if settings.languages is not None else None
    links = _language_sitelinks(entity, allowed)
    return _limit_sitelinks(links, settings.max_articles_per_qid)


def _language_sitelinks(
    entity: WikidataEntity,
    allowed: set[str] | None,
) -> list[tuple[str, str]]:
    return [
        (site, title)
        for site, title in sorted(entity.sitelinks.items())
        if allowed is None or language_from_site(site) in allowed
    ]


def _limit_sitelinks(
    links: list[tuple[str, str]],
    max_articles_per_qid: int | None,
) -> list[tuple[str, str]]:
    if max_articles_per_qid is None:
        return links
    return links[: max(0, max_articles_per_qid)]


def _fetch_recovery_document(
    qid: str,
    site: str,
    title: str,
    *,
    entity: WikidataEntity,
    summary: LinkSummary,
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None,
) -> dict[str, Any] | None:
    """Fetch and normalize one missing Wikipedia document."""
    language = language_from_site(site)
    result = _fetch_recovery_article(
        language,
        site,
        title,
        entity=entity,
        wikipedia_client=wikipedia_client,
        settings=settings,
    )
    summary.statuses[site] = result.status
    _validate_recovery_fetch(qid, site, result.status, result.error)
    if _missing_recovery_article(result):
        _advance_missing_document(progress)
        return None
    summary.articles.append(result.article)
    return _document_from_recovery_article(qid, language, result.article, summary, progress)


def _fetch_recovery_article(
    language: str,
    site: str,
    title: str,
    *,
    entity: WikidataEntity,
    wikipedia_client: WikipediaClient,
    settings: Settings,
) -> Any:
    """Call the Wikipedia client with the recovery entity context."""
    return wikipedia_client.fetch_article(
        language,
        site,
        title,
        wikidata_label=entity.labels.get(language) or entity.labels.get("en", ""),
        wikidata_description=entity.descriptions.get(language) or entity.descriptions.get("en", ""),
        wikidata_aliases=entity.aliases.get(language) or entity.aliases.get("en", []),
        fetch_full_text=settings.fetch_full_text,
    )


def _missing_recovery_article(result: Any) -> bool:
    """Return whether a fetch completed without an article payload."""
    return result.article is None or result.status == "article_not_found"


def _validate_recovery_fetch(qid: str, site: str, status: str, error: str) -> None:
    """Reject a terminal Wikipedia fetch status that cannot be preserved."""
    if status not in NON_FATAL_FETCH_STATUSES and status != "ok":
        raise RecoveryRepairError(
            f"Incomplete Wikipedia recovery for {qid}:{site} ({status}): {error}"
        )


def _advance_missing_document(progress: RecoveryProgress | None) -> None:
    """Advance progress for an absent but terminally handled document."""
    if progress is not None:
        progress.advance()


def _document_from_recovery_article(
    qid: str,
    language: str,
    article_result: Any,
    summary: LinkSummary,
    progress: RecoveryProgress | None,
) -> dict[str, Any]:
    """Convert a fetched article into the canonical Wikipedia document row."""
    identifier = article_id(qid, language, article_result.page_id, article_result.revision_id)
    article = article_row(identifier, qid, article_result, summary)
    document = wikipedia_document_from_article_row(article.__dict__)
    if progress is not None:
        progress.advance(documents=1)
    return document.to_dict()


def _fetch_qid_documents(
    qid: str,
    *,
    entity: WikidataEntity,
    existing: set[tuple[str, str, str]],
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Fetch all missing eligible documents for one QID."""
    summary = LinkSummary(qid=qid, entity=entity)
    documents: list[dict[str, Any]] = []
    for site, title in _eligible_sitelinks(entity, settings):
        if (qid, site, title) in existing:
            if progress is not None:
                progress.advance()
            continue
        document = _fetch_recovery_document(
            qid,
            site,
            title,
            entity=entity,
            summary=summary,
            wikipedia_client=wikipedia_client,
            settings=settings,
            progress=progress,
        )
        if document is not None:
            documents.append(document)
    return documents


def _fetch_missing_documents(
    affected_qids: tuple[str, ...],
    *,
    entities: dict[str, WikidataEntity],
    existing_documents: list[dict[str, Any]],
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None = None,
) -> list[dict[str, Any]]:
    existing = {
        (str(row["wikidata"]), str(row["site"]), str(row["title"])) for row in existing_documents
    }
    total_sites = sum(len(_eligible_sitelinks(entities[qid], settings)) for qid in affected_qids)
    if progress is not None:
        progress.set_stage("Wikipedia documents", total=total_sites)
    return _fetch_missing_documents_parallel(
        affected_qids,
        entities=entities,
        existing=existing,
        wikipedia_client=wikipedia_client,
        settings=settings,
        progress=progress,
    )


def _fetch_missing_documents_parallel(
    affected_qids: tuple[str, ...],
    *,
    entities: dict[str, WikidataEntity],
    existing: set[tuple[str, str, str]],
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Fetch affected QIDs concurrently while retaining input order."""
    if not affected_qids:
        return []
    with ThreadPoolExecutor(
        max_workers=min(RECOVERY_NETWORK_WORKERS, len(affected_qids))
    ) as executor:
        per_qid = executor.map(
            lambda qid: _fetch_qid_documents(
                qid,
                entity=entities[qid],
                existing=existing,
                wikipedia_client=wikipedia_client,
                settings=settings,
                progress=progress,
            ),
            affected_qids,
        )
        return [document for documents in per_qid for document in documents]


def _summarize_polygon_links(
    polygon_links: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return unique article IDs and languages in deterministic order."""
    return (
        sorted({str(link["article_id"]) for link in polygon_links}),
        sorted({str(link["language"]) for link in polygon_links}),
    )


def _preferred_language(languages: list[str]) -> str:
    """Choose the configured preferred language, falling back to sorted input."""
    best = next((language for language in PREFERRED_LANGUAGES if language in languages), "")
    if not best and languages:
        return languages[0]
    return best


def _has_article_text(
    article_ids: list[str],
    documents_by_article: dict[str, dict[str, Any]],
) -> bool:
    """Return whether at least one linked document has non-empty full text."""
    return any(
        bool(str(documents_by_article[article]["full_text"]).strip()) for article in article_ids
    )


def _recompute_polygon_row(
    original: dict[str, Any],
    polygon_links: list[dict[str, Any]],
    documents_by_article: dict[str, dict[str, Any]],
    affected_qids: set[str],
) -> tuple[dict[str, Any], str]:
    """Recompute one affected polygon's derived Wikipedia fields."""
    row = dict(original)
    if not set(qids_from_osm_tag(str(row["wikidata"]))) & affected_qids:
        return row, ""
    article_ids, languages = _summarize_polygon_links(polygon_links)
    best = _preferred_language(languages)
    row.update(
        {
            "has_wikipedia": bool(article_ids),
            "wikipedia_language_count": len(languages),
            "wikipedia_languages": dumps(languages),
            "wikipedia_article_count": len(article_ids),
            "has_english_wikipedia": "en" in languages,
            "has_french_wikipedia": "fr" in languages,
            "text_available": _has_article_text(article_ids, documents_by_article),
            "best_language": best,
        }
    )
    return row, best


def _recompute_polygon_rows(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Recompute derived fields for all affected polygons."""
    documents_by_article = {str(row["article_id"]): row for row in documents}
    links_by_polygon: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_polygon.setdefault(str(link["polygon_id"]), []).append(link)
    updated: list[dict[str, Any]] = []
    best_by_polygon: dict[str, str] = {}
    for original in polygons:
        polygon_id = str(original["polygon_id"])
        row, best = _recompute_polygon_row(
            original,
            links_by_polygon.get(polygon_id, []),
            documents_by_article,
            affected_qids,
        )
        updated.append(row)
        if best:
            best_by_polygon[polygon_id] = best
    return updated, best_by_polygon


def _apply_best_language_links(
    links: list[dict[str, Any]],
    best_by_polygon: dict[str, str],
) -> list[dict[str, Any]]:
    """Apply the recomputed best-language flag to Wikipedia links."""
    updated: list[dict[str, Any]] = []
    for original in links:
        row = dict(original)
        polygon_id = str(row["polygon_id"])
        if polygon_id in best_by_polygon:
            row["is_best_language"] = str(row["language"]) == best_by_polygon[polygon_id]
        updated.append(row)
    return updated


def _recompute_affected_polygon_fields(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated_polygons, best_by_polygon = _recompute_polygon_rows(
        polygons, links, documents, affected_qids
    )
    return updated_polygons, _apply_best_language_links(links, best_by_polygon)


def _sections_for_new_documents(
    documents: list[dict[str, Any]],
    new_document_ids: set[str],
    *,
    augmentation_client: AugmentationClient,
    progress: RecoveryProgress | None = None,
) -> list[dict[str, Any]]:
    selected = _select_new_documents(documents, new_document_ids)
    if progress is not None:
        progress.set_stage("Wikipedia sections", total=len(selected))
    return _parse_recovery_documents(
        selected, augmentation_client=augmentation_client, progress=progress
    )


def _select_new_documents(
    documents: list[dict[str, Any]], new_document_ids: set[str]
) -> list[dict[str, Any]]:
    """Select only documents that need section parsing."""
    return [row for row in documents if str(row["document_id"]) in new_document_ids]


def _parse_recovery_documents(
    selected: list[dict[str, Any]],
    *,
    augmentation_client: AugmentationClient,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Parse selected recovery documents concurrently in input order."""
    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=min(RECOVERY_NETWORK_WORKERS, len(selected))) as executor:
        per_document = executor.map(
            lambda document_row: _parse_recovery_document(
                document_row,
                augmentation_client=augmentation_client,
                progress=progress,
            ),
            selected,
        )
        return [section for sections in per_document for section in sections]


def _parse_recovery_document(
    document_row: dict[str, Any],
    *,
    augmentation_client: AugmentationClient,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Fetch and parse sections for one newly recovered document."""
    article = {column: document_row[column] for column in ARTICLE_COLUMNS}
    document = document_from_article_row(article)
    html = augmentation_client.parse_html(document.project, document.language, document.revision_id)
    parsed = [section.to_dict() for section in parse_sections(document, html)]
    if progress is not None:
        progress.advance(sections=len(parsed))
    return parsed


def _validate_merge_rows(
    existing: list[dict[str, Any]],
    *,
    primary_key: str,
    label: str,
    secondary_key: str | None,
) -> tuple[set[str], set[str]]:
    """Validate existing primary and optional secondary identities."""
    primary: set[str] = set()
    secondary: set[str] = set()
    for row in existing:
        identifier = str(row[primary_key])
        if identifier in primary:
            raise RecoveryRepairError(f"duplicate {label} {identifier!r}")
        primary.add(identifier)
        if secondary_key is not None:
            secondary_identifier = str(row[secondary_key])
            if secondary_identifier in secondary:
                raise RecoveryRepairError(f"duplicate {secondary_key} {secondary_identifier!r}")
            secondary.add(secondary_identifier)
    return primary, secondary


def _append_merge_rows(
    merged: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    *,
    primary: set[str],
    secondary: set[str],
    primary_key: str,
    label: str,
    secondary_key: str | None,
) -> set[str]:
    """Append unseen rows while enforcing merge identity constraints."""
    added: set[str] = set()
    for row in additions:
        identifier = str(row[primary_key])
        if identifier in primary:
            continue
        if secondary_key is not None:
            secondary_identifier = str(row[secondary_key])
            if secondary_identifier in secondary:
                raise RecoveryRepairError(f"duplicate {secondary_key} {secondary_identifier!r}")
            secondary.add(secondary_identifier)
        primary.add(identifier)
        added.add(identifier)
        merged.append(dict(row))
    return added


def _merge_rows(
    existing: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    *,
    primary_key: str,
    label: str,
    secondary_key: str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    merged = [dict(row) for row in existing]
    primary, secondary = _validate_merge_rows(
        existing,
        primary_key=primary_key,
        label=label,
        secondary_key=secondary_key,
    )
    added = _append_merge_rows(
        merged,
        additions,
        primary=primary,
        secondary=secondary,
        primary_key=primary_key,
        label=label,
        secondary_key=secondary_key,
    )
    return merged, added


def _terminal_classifications(
    region: RegionAuditResult,
    links: list[dict[str, Any]],
) -> dict[str, RecoveryClassification]:
    linked_polygon_qids = {
        (str(link["polygon_id"]), str(link["wikidata"]))
        for link in links
        if link.get("project", "wikipedia") == "wikipedia"
    }
    polygons_by_qid = dict(region.polygon_ids_by_qid)
    terminal: dict[str, RecoveryClassification] = {}
    for qid, state in region.classifications:
        terminal[qid] = _terminal_classification(
            qid,
            state,
            affected_qids=set(region.affected_qids),
            polygon_ids=polygons_by_qid.get(qid, ()),
            linked_polygon_qids=linked_polygon_qids,
        )
    return terminal


def _terminal_classification(
    qid: str,
    state: RecoveryClassification,
    *,
    affected_qids: set[str],
    polygon_ids: tuple[str, ...] | list[str],
    linked_polygon_qids: set[tuple[str, str]],
) -> RecoveryClassification:
    """Classify one repaired QID after checking all affected polygon links."""
    if qid not in affected_qids:
        return state
    if all((polygon_id, qid) in linked_polygon_qids for polygon_id in polygon_ids):
        return RecoveryClassification.CURRENT
    return RecoveryClassification.AUTHORITATIVE_NO_ARTICLE


def _stage_manifests(
    stem: str,
    *,
    paths: dict[str, Path],
    staged: dict[str, Path],
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    affected_qids: tuple[str, ...],
    affected_polygon_count: int,
) -> None:
    """Stage both manifests from the same repaired artifact snapshot."""
    _stage_processed_manifest(
        stem,
        paths=paths,
        staged=staged["processed_manifest"],
        polygons=polygons,
        documents=documents,
        affected_qids=affected_qids,
        affected_polygon_count=affected_polygon_count,
    )
    _stage_augmentation_manifest(
        stem,
        paths=paths,
        staged=staged["augmentation_manifest"],
        staged_polygons=staged["polygons"],
        staged_documents=staged["documents"],
        documents=documents,
        sections=sections,
        facts=facts,
    )


def _stage_processed_manifest(
    stem: str,
    *,
    paths: dict[str, Path],
    staged: Path,
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: tuple[str, ...],
    affected_polygon_count: int,
) -> None:
    manifest = load_manifest(paths["processed_manifest"])
    manifest_key = f"{stem}.osm.pbf"
    if manifest_key not in manifest:
        raise RecoveryRepairError(f"Processed manifest is missing {manifest_key!r}")
    entry = dict(manifest[manifest_key])
    entry.update(
        _processed_manifest_statistics(
            polygons=polygons,
            documents=documents,
            affected_qids=affected_qids,
            affected_polygon_count=affected_polygon_count,
        )
    )
    manifest[manifest_key] = entry
    atomic_write_text(staged, dumps(manifest) + "\n")


def _processed_manifest_statistics(
    *,
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: tuple[str, ...],
    affected_polygon_count: int,
) -> dict[str, object]:
    languages = sorted({str(row["language"]) for row in documents})
    return {
        "polygon_count": len(polygons),
        "unique_wikidata_count": len(_polygon_qids(polygons)),
        "article_count": len(documents),
        "language_count": len(languages),
        "languages": languages,
        "rows_with_wikipedia": sum(bool(row["has_wikipedia"]) for row in polygons),
        "rows_with_full_text": sum(bool(row["text_available"]) for row in polygons),
        "total_full_text_chars": sum(len(str(row["full_text"])) for row in documents),
        "wikidata_recovery": {
            "contract_version": RECOVERY_CONTRACT_VERSION,
            "affected_qids": list(affected_qids),
            "affected_polygon_count": affected_polygon_count,
        },
    }


def _polygon_qids(polygons: list[dict[str, Any]]) -> set[str]:
    return {qid for row in polygons for qid in qids_from_osm_tag(str(row["wikidata"]))}


def _stage_augmentation_manifest(
    stem: str,
    *,
    paths: dict[str, Path],
    staged: Path,
    staged_polygons: Path,
    staged_documents: Path,
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    try:
        augmentation: object = json.loads(
            paths["augmentation_manifest"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryRepairError(f"Augmentation manifest is unreadable: {error}") from error
    if not isinstance(augmentation, dict) or not isinstance(augmentation.get(stem), dict):
        raise RecoveryRepairError(f"Augmentation manifest is missing region {stem!r}")
    augmentation_mapping = cast(dict[str, object], augmentation)
    augmentation_entry = dict(cast(dict[str, object], augmentation_mapping[stem]))
    counts = augmentation_entry.get("counts")
    if not isinstance(counts, dict):
        raise RecoveryRepairError(f"Augmentation manifest counts are invalid for {stem!r}")
    updated_counts = dict(cast(dict[str, object], counts))
    updated_counts.update(
        {
            "wikipedia_documents": len(documents),
            "wikipedia_sections": len(sections),
            "wikidata_facts": len(facts),
        }
    )
    augmentation_entry.update(
        {
            "contract_version": CONTRACT_VERSION,
            "core_hashes": {
                str(paths["polygons"]): sha256_file(staged_polygons),
                str(paths["documents"]): sha256_file(staged_documents),
            },
            "counts": updated_counts,
        }
    )
    augmentation_mapping[stem] = augmentation_entry
    atomic_write_text(staged, dumps(augmentation_mapping) + "\n")


__all__ = ["RecoveryRepairError", "RecoveryRepairResult", "repair_wikidata_region"]
