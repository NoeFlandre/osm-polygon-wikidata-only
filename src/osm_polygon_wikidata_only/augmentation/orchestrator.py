"""Orchestration policy for the augmentation pipeline.

This module owns the orchestration policy: phase ordering, progress
transitions, sidecar paths, the core-hash drift check, the manifest
write (the orchestrator constructs the counts dict and decides *when*
to merge the entry, then :func:`augmentation.steps.update_augmentation_manifest`
performs the actual atomic merge), and the thread-pool lifecycle
(selection of worker counts, opening and closing both pools).

The actual side-effectful mechanics -- reading core inputs, fetching
entities, fetching Wikivoyage documents, parsing article HTML, building
Wikidata facts, serializing sidecars, merging the manifest entry --
live as focused helpers in :mod:`augmentation.steps`. The
``sha256_file`` helper from that module is reused by the drift and
resumability checks here so a single implementation backs every
content-addressed hash in this package.

``augment_region`` is the stable facade; its signature, return type, canonical
outputs, and manifest semantics remain unchanged. Durable derived-work
checkpoints live only below the configured data-root cache.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.augmentation.checkpoints import (
    SECTION_CHECKPOINT_BATCH_SIZE,
    AugmentationCheckpointStore,
    augmentation_plan_key,
    document_identities,
    entities_digest,
)
from osm_polygon_wikidata_only.augmentation.integrity import (
    INTEGRITY_CONTRACT_VERSION,
    WikivoyageIntegrityResult,
    enforce_wikivoyage_integrity,
)
from osm_polygon_wikidata_only.augmentation.models import Document, Section, WikidataFact
from osm_polygon_wikidata_only.augmentation.wikimedia import discover_wikivoyage_sitelinks
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .existing_results import (
    augmentation_is_current,
    augmentation_manifest_entry,
    completed_region_stems,
    read_augmentation_manifest,
    sidecar_paths,
    validate_augmentation_entry,
    validate_sidecars,
)
from .progress import AugmentationProgress
from .steps import (
    AugmentationClient,
    CoreInputs,
    build_wikidata_facts,
    fetch_document_sections_batch,
    fetch_wikivoyage_documents,
    load_core_inputs,
    resolve_entities,
    sha256_file,
    update_augmentation_manifest,
    write_sidecars,
)

VOYAGE_WORKERS = 8
ARTICLE_WORKERS = 8
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    wikipedia_documents_path: Path
    wikipedia_sections_path: Path
    wikivoyage_documents_path: Path
    wikivoyage_sections_path: Path
    wikidata_facts_path: Path
    manifest_path: Path
    counts: dict[str, int]
    wikivoyage_integrity: WikivoyageIntegrityResult | None = None
    polygon_document_links_path: Path | None = None


def _load_or_resolve_entities(
    checkpoint_store: AugmentationCheckpointStore,
    client: AugmentationClient,
    qids: list[str],
    core_inputs: CoreInputs,
    progress: AugmentationProgress,
    stem: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load entity work from a checkpoint or resolve and save it."""
    entities = checkpoint_store.load_entities(core_inputs.qids)
    if entities is None:
        entities = resolve_entities(client, qids, progress=progress)
        checkpoint_store.save_entities(core_inputs.qids, entities)
    else:
        LOGGER.info("Augmentation checkpoint %s: reused Wikidata entities", stem)
        progress.start("Wikidata entities", total=len(qids))
        progress.complete()
    return entities, entities_digest(entities)


def _load_or_fetch_voyage_documents(
    checkpoint_store: AugmentationCheckpointStore,
    client: AugmentationClient,
    entities: dict[str, dict[str, Any]],
    entity_digest: str,
    progress: AugmentationProgress,
    stem: str,
) -> list[Document]:
    """Load Wikivoyage documents from a checkpoint or fetch and save them."""
    voyage_documents = checkpoint_store.load_voyage_documents(entity_digest)
    with ThreadPoolExecutor(max_workers=VOYAGE_WORKERS) as voyage_executor:
        if voyage_documents is None:
            voyage_documents = fetch_wikivoyage_documents(
                client,
                entities=entities,
                progress=progress,
                executor=voyage_executor,
            )
            checkpoint_store.save_voyage_documents(entity_digest, voyage_documents)
        else:
            LOGGER.info("Augmentation checkpoint %s: reused Wikivoyage documents", stem)
            voyage_total = sum(
                len(discover_wikivoyage_sitelinks(entity)) for entity in entities.values()
            )
            progress.start("Wikivoyage documents", total=voyage_total)
            progress.complete()
    return voyage_documents


def _load_or_build_facts(
    checkpoint_store: AugmentationCheckpointStore,
    client: AugmentationClient,
    entities: dict[str, dict[str, Any]],
    entity_digest: str,
    progress: AugmentationProgress,
    stem: str,
) -> list[WikidataFact]:
    """Load Wikidata facts from a checkpoint or build and save them."""
    facts = checkpoint_store.load_facts(entity_digest)
    if facts is None:
        facts = build_wikidata_facts(client, entities=entities, progress=progress)
        checkpoint_store.save_facts(entity_digest, facts)
    else:
        LOGGER.info("Augmentation checkpoint %s: reused Wikidata facts", stem)
        progress.start("Wikidata facts", total=len(entities))
        progress.complete()
    return facts


def _core_artifacts_unchanged(core_inputs: CoreInputs) -> bool:
    """Return whether core files still match the pre-augmentation hashes."""
    current = {str(path): sha256_file(path) for path in core_inputs.core_paths}
    return core_inputs.core_hashes == current


def _integrity_rejections_payload(
    integrity: WikivoyageIntegrityResult,
) -> dict[str, Any] | None:
    """Serialize rejection details only when integrity removed documents."""
    if integrity.rejected_document_count <= 0:
        return None
    return {
        "contract_version": INTEGRITY_CONTRACT_VERSION,
        "shard": integrity.shard,
        "original_document_count": integrity.original_document_count,
        "retained_document_count": integrity.retained_document_count,
        "rejected_document_count": integrity.rejected_document_count,
        "original_section_count": integrity.original_section_count,
        "retained_section_count": integrity.retained_section_count,
        "cascaded_section_count": integrity.cascaded_section_count,
        "rewritten_documents": integrity.rewritten_documents,
        "rewritten_sections": integrity.rewritten_sections,
        "rejections": [rejection.to_dict() for rejection in integrity.rejections],
    }


def augment_region(
    data_root: DataRoot,
    stem: str,
    client: AugmentationClient,
    *,
    progress: AugmentationProgress | None = None,
) -> AugmentationResult:
    progress = progress or AugmentationProgress()
    paths = sidecar_paths(data_root, stem)

    core_inputs = load_core_inputs(data_root, stem)
    wikipedia_documents = list(core_inputs.wikipedia_documents)
    qids = list(core_inputs.qids)

    checkpoint_store = AugmentationCheckpointStore(
        data_root.cache / "augmentation_checkpoints",
        stem,
        augmentation_plan_key(
            core_hashes=core_inputs.core_hashes,
            qids=core_inputs.qids,
            document_identities=document_identities(core_inputs.wikipedia_documents),
        ),
    )

    entities, entity_digest = _load_or_resolve_entities(
        checkpoint_store, client, qids, core_inputs, progress, stem
    )
    voyage_documents = _load_or_fetch_voyage_documents(
        checkpoint_store, client, entities, entity_digest, progress, stem
    )

    all_documents = wikipedia_documents + voyage_documents
    sections_by_project = _checkpointed_document_sections(
        checkpoint_store,
        client,
        all_documents,
        progress,
        stem=stem,
    )

    facts = _load_or_build_facts(checkpoint_store, client, entities, entity_digest, progress, stem)

    if not _core_artifacts_unchanged(core_inputs):
        raise RuntimeError("Core artifacts changed during augmentation")

    write_sidecars(
        paths,
        wikipedia_documents=wikipedia_documents,
        wikivoyage_documents=voyage_documents,
        sections_by_project=sections_by_project,
        facts=facts,
        progress=progress,
        articles_path=core_inputs.core_paths[0],
    )

    if not _core_artifacts_unchanged(core_inputs):
        raise RuntimeError("Core artifacts changed during augmentation")

    # Deterministic join-integrity enforcement: reject every wikivoyage
    # document whose wikidata is absent from the shard's polygons and
    # cascade to its sections. Path A only -- never rewrite QIDs.
    wikivoyage_integrity = enforce_wikivoyage_integrity(data_root, stem)

    counts = {
        "wikipedia_documents": len(wikipedia_documents),
        "wikipedia_sections": len(sections_by_project["wikipedia"]),
        "wikivoyage_documents": len(voyage_documents),
        "wikivoyage_sections": len(sections_by_project["wikivoyage"]),
        "wikidata_facts": len(facts),
    }

    rejections_payload = _integrity_rejections_payload(wikivoyage_integrity)

    manifest_path = update_augmentation_manifest(
        data_root,
        stem=stem,
        paths=paths,
        core_hashes=core_inputs.core_hashes,
        counts=counts,
        completed_at=utc_now_iso(),
        rejections=rejections_payload,
    )
    checkpoint_store.clear()
    return AugmentationResult(
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        paths[4],
        manifest_path,
        counts,
        wikivoyage_integrity=wikivoyage_integrity,
        polygon_document_links_path=None,
    )


def _checkpointed_document_sections(
    checkpoint_store: AugmentationCheckpointStore,
    client: AugmentationClient,
    documents: list[Document],
    progress: AugmentationProgress,
    *,
    stem: str,
) -> dict[str, list[Section]]:
    """Build and durably checkpoint deterministic document batches."""
    sections_by_project: dict[str, list[Section]] = {"wikipedia": [], "wikivoyage": []}
    reused_batches = 0
    total_batches = (len(documents) + SECTION_CHECKPOINT_BATCH_SIZE - 1) // (
        SECTION_CHECKPOINT_BATCH_SIZE
    )
    progress.start("Article sections", total=len(documents))
    with ThreadPoolExecutor(max_workers=ARTICLE_WORKERS) as article_executor:
        for index, start in enumerate(range(0, len(documents), SECTION_CHECKPOINT_BATCH_SIZE)):
            batch = documents[start : start + SECTION_CHECKPOINT_BATCH_SIZE]
            identities = document_identities(batch)
            rows = checkpoint_store.load_section_batch(index, identities)
            if rows is None:
                fresh = fetch_document_sections_batch(
                    client,
                    documents=batch,
                    progress=progress,
                    executor=article_executor,
                )
                rows = [*fresh["wikipedia"], *fresh["wikivoyage"]]
                checkpoint_store.save_section_batch(index, identities, rows)
            else:
                reused_batches += 1
                progress.advance(len(batch))
            for section in rows:
                sections_by_project[section.project].append(section)
    for rows in sections_by_project.values():
        rows.sort(key=lambda row: (row.document_id, row.section_index))
    _log_reused_section_batches(stem, reused_batches, total_batches)
    return sections_by_project


def _log_reused_section_batches(stem: str, reused_batches: int, total_batches: int) -> None:
    """Log checkpoint reuse only when at least one batch was reused."""
    if reused_batches:
        LOGGER.info(
            "Augmentation checkpoint %s: reused %d/%d article-section batches",
            stem,
            reused_batches,
            total_batches,
        )


def load_existing_augmentation_result(data_root: DataRoot, stem: str) -> AugmentationResult:
    """Load and validate an existing augmentation result without remote work."""
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    manifest = read_augmentation_manifest(manifest_path)
    entry = augmentation_manifest_entry(manifest, stem)
    counts = cast(dict[str, int], validate_augmentation_entry(entry, stem))

    paths = sidecar_paths(data_root, stem)
    validate_sidecars(paths)

    return AugmentationResult(
        wikipedia_documents_path=paths[0],
        wikipedia_sections_path=paths[1],
        wikivoyage_documents_path=paths[2],
        wikivoyage_sections_path=paths[3],
        wikidata_facts_path=paths[4],
        manifest_path=manifest_path,
        counts=counts,
        polygon_document_links_path=data_root.processed_links / f"{stem}.parquet",
    )


__all__ = [
    "AugmentationResult",
    "augment_region",
    "augmentation_is_current",
    "completed_region_stems",
    "load_existing_augmentation_result",
    "sidecar_paths",
]
