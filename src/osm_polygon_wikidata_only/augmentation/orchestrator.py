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

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

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
from osm_polygon_wikidata_only.augmentation.models import Document, Section
from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.wikimedia import discover_wikivoyage_sitelinks
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .progress import AugmentationProgress
from .steps import (
    CONTRACT_VERSION,
    AugmentationClient,
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


def sidecar_paths(data_root: DataRoot, stem: str) -> tuple[Path, Path, Path, Path, Path]:
    root = data_root.processed
    return (
        root / "wikipedia" / "documents" / f"{stem}.parquet",
        root / "wikipedia" / "sections" / f"{stem}.parquet",
        root / "wikivoyage" / "documents" / f"{stem}.parquet",
        root / "wikivoyage" / "sections" / f"{stem}.parquet",
        root / "wikidata" / "facts" / f"{stem}.parquet",
    )


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

    entities = checkpoint_store.load_entities(core_inputs.qids)
    if entities is None:
        entities = resolve_entities(client, qids, progress=progress)
        checkpoint_store.save_entities(core_inputs.qids, entities)
    else:
        LOGGER.info("Augmentation checkpoint %s: reused Wikidata entities", stem)
        progress.start("Wikidata entities", total=len(qids))
        progress.complete()
    entity_digest = entities_digest(entities)

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

    all_documents = wikipedia_documents + voyage_documents
    sections_by_project = _checkpointed_document_sections(
        checkpoint_store,
        client,
        all_documents,
        progress,
        stem=stem,
    )

    facts = checkpoint_store.load_facts(entity_digest)
    if facts is None:
        facts = build_wikidata_facts(client, entities=entities, progress=progress)
        checkpoint_store.save_facts(entity_digest, facts)
    else:
        LOGGER.info("Augmentation checkpoint %s: reused Wikidata facts", stem)
        progress.start("Wikidata facts", total=len(entities))
        progress.complete()

    if core_inputs.core_hashes != {str(path): sha256_file(path) for path in core_inputs.core_paths}:
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

    if core_inputs.core_hashes != {str(path): sha256_file(path) for path in core_inputs.core_paths}:
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

    rejections_payload: dict[str, Any] | None = None
    if wikivoyage_integrity.rejected_document_count > 0:
        rejections_payload = {
            "contract_version": INTEGRITY_CONTRACT_VERSION,
            "shard": wikivoyage_integrity.shard,
            "original_document_count": wikivoyage_integrity.original_document_count,
            "retained_document_count": wikivoyage_integrity.retained_document_count,
            "rejected_document_count": wikivoyage_integrity.rejected_document_count,
            "original_section_count": wikivoyage_integrity.original_section_count,
            "retained_section_count": wikivoyage_integrity.retained_section_count,
            "cascaded_section_count": wikivoyage_integrity.cascaded_section_count,
            "rewritten_documents": wikivoyage_integrity.rewritten_documents,
            "rewritten_sections": wikivoyage_integrity.rewritten_sections,
            "rejections": [r.to_dict() for r in wikivoyage_integrity.rejections],
        }

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
    if reused_batches:
        LOGGER.info(
            "Augmentation checkpoint %s: reused %d/%d article-section batches",
            stem,
            reused_batches,
            total_batches,
        )
    return sections_by_project


def completed_region_stems(data_root: DataRoot) -> list[str]:
    article_stems = {path.stem for path in data_root.processed_articles.glob("*.parquet")}
    article_stems.update(
        path.stem for path in (data_root.processed / "wikipedia" / "documents").glob("*.parquet")
    )
    polygon_stems = {path.stem for path in data_root.processed_polygons.glob("*.parquet")}
    return sorted(article_stems & polygon_stems)


def augmentation_is_current(data_root: DataRoot, stem: str) -> bool:
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if not manifest_path.exists() or not all(
        path.exists() for path in sidecar_paths(data_root, stem)
    ):
        return False
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get(stem, {})
    if entry.get("contract_version") != CONTRACT_VERSION:
        return False
    if "link_schema_version" in entry or "link_artifact_sha256" in entry:
        links_path = data_root.processed_links / f"{stem}.parquet"
        if (
            entry.get("link_schema_version") != "polygon-document-links-v1"
            or not links_path.is_file()
            or entry.get("link_artifact_sha256") != sha256_file(links_path)
        ):
            return False
        processed_manifest_path = data_root.processed_manifests / "processed_pbfs.json"
        try:
            processed_entry = json.loads(processed_manifest_path.read_text(encoding="utf-8"))[
                f"{stem}.osm.pbf"
            ]
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
            return False
        if (
            processed_entry.get("link_schema_version") != "polygon-document-links-v1"
            or processed_entry.get("link_count") != pq.read_metadata(links_path).num_rows  # type: ignore[no-untyped-call]
        ):
            return False
    expected_hashes = entry.get("core_hashes")
    if not _is_valid_core_hashes(expected_hashes, data_root, stem):
        return False
    # Validate exactly the source paths named by the manifest. After
    # ``prepare_local_retirement`` repoints the manifest to the canonical
    # document, the legacy staging file may still exist but must NOT be
    # chosen over the canonical source the manifest records.
    expected_paths = [Path(key) for key in expected_hashes]
    if not all(path.is_file() for path in expected_paths):
        return False
    current = {str(path): sha256_file(path) for path in expected_paths}
    return bool(expected_hashes == current)


_SHA256_HEX_LENGTH = 64


def _is_valid_core_hashes(value: object, data_root: DataRoot, stem: str) -> bool:
    """Return True iff *value* is the exact two-entry hash dict we accept.

    Accepts only:

    * ``processed/polygons/<stem>.parquet`` (the polygon table), and
    * exactly one of:
      - ``processed/articles/<stem>.parquet``, or
      - ``processed/wikipedia/documents/<stem>.parquet``.

    Rejects missing entries, extra entries, duplicate keys, malformed
    paths (wrong stem, directory traversal, non-string keys), malformed
    hashes (non-string values, wrong length, non-hex), and paths outside
    the resolved ``data_root.processed`` subtree. The polygon path is
    never used to select between legacy and canonical; both layout
    variants share it.
    """
    if not isinstance(value, dict) or not value:
        return False
    if len(value) != 2:
        return False

    polygon_path = data_root.processed_polygons / f"{stem}.parquet"
    legacy_path = data_root.processed_articles / f"{stem}.parquet"
    canonical_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    processed_root = data_root.processed.resolve()

    polygon_key = str(polygon_path)
    legacy_key = str(legacy_path)
    canonical_key = str(canonical_path)

    if polygon_key not in value:
        return False
    sources_present = sum(1 for key in (legacy_key, canonical_key) if key in value)
    if sources_present != 1:
        return False

    for key, hash_value in value.items():
        if not isinstance(key, str):
            return False
        if not isinstance(hash_value, str):
            return False
        if len(hash_value) != _SHA256_HEX_LENGTH:
            return False
        if not all(ch in "0123456789abcdef" for ch in hash_value):
            return False
        # Only the two allowed paths are accepted.
        if key not in {polygon_key, legacy_key, canonical_key}:
            return False
        # Traversal and out-of-tree paths are rejected even if they
        # coincidentally match the allowed layout.
        try:
            resolved = Path(key).resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        try:
            resolved.relative_to(processed_root)
        except ValueError:
            return False

    return True


def load_existing_augmentation_result(data_root: DataRoot, stem: str) -> AugmentationResult:
    """Load and validate an existing augmentation result without remote work."""
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Augmentation manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Augmentation manifest is not valid JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise TypeError("Augmentation manifest must be a JSON object")

    if stem not in manifest:
        raise KeyError(f"Stem {stem!r} not found in augmentation manifest")

    entry = manifest[stem]
    if not isinstance(entry, dict):
        raise TypeError(f"Manifest entry for {stem!r} must be a JSON object")

    if entry.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"Invalid contract version for {stem!r} in manifest: expected {CONTRACT_VERSION!r}, got {entry.get('contract_version')!r}"
        )

    counts = entry.get("counts")
    if not isinstance(counts, dict):
        raise TypeError(f"Manifest entry counts for {stem!r} must be a JSON object")

    expected_keys = {
        "wikipedia_documents",
        "wikipedia_sections",
        "wikivoyage_documents",
        "wikivoyage_sections",
        "wikidata_facts",
    }
    if not expected_keys.issubset(counts.keys()):
        raise ValueError(
            f"Manifest entry counts for {stem!r} is missing required fields: expected {expected_keys}, got {set(counts.keys())}"
        )

    for k in expected_keys:
        if not isinstance(counts[k], int) or counts[k] < 0:
            raise TypeError(f"Manifest count for {k!r} must be a non-negative integer")

    # Validate sidecar files
    paths = sidecar_paths(data_root, stem)

    expected_schemas = (
        wikipedia_document_schema(),
        section_schema(),
        document_schema(),
        section_schema(),
        fact_schema(),
    )

    for path, expected_schema in zip(paths, expected_schemas, strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"Sidecar file is missing: {path}")
        try:
            actual_schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
        except (OSError, ValueError) as exc:
            raise ValueError(f"Sidecar file is unreadable: {path} ({exc})") from exc
        if not actual_schema.equals(expected_schema, check_metadata=True):
            raise ValueError(f"Sidecar schema mismatch for {path}")

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
