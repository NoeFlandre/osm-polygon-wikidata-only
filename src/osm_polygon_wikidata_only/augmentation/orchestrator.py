"""Orchestration policy for the augmentation pipeline.

This module owns the orchestration policy: phase ordering, progress
transitions, sidecar paths, the core-hash drift check, the manifest
*write ordering*, and the thread-pool lifecycle (selection of worker
counts, opening and closing both pools).

The actual side-effectful mechanics -- reading core inputs, fetching
entities, fetching Wikivoyage documents, parsing article HTML, building
Wikidata facts, serializing sidecars, merging the manifest entry --
live as focused helpers in :mod:`augmentation.steps`. The
``sha256_file`` helper from that module is reused by the drift and
resumability checks here so a single implementation backs every
content-addressed hash in this package.

``augment_region`` is the stable facade; its signature, return type,
and side effects on disk are unchanged across the decomposition.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .progress import AugmentationProgress
from .steps import (
    CONTRACT_VERSION,
    AugmentationClient,
    build_wikidata_facts,
    fetch_document_sections,
    fetch_wikivoyage_documents,
    load_core_inputs,
    resolve_entities,
    sha256_file,
    update_augmentation_manifest,
    write_sidecars,
)

VOYAGE_WORKERS = 8
ARTICLE_WORKERS = 8


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    wikipedia_documents_path: Path
    wikipedia_sections_path: Path
    wikivoyage_documents_path: Path
    wikivoyage_sections_path: Path
    wikidata_facts_path: Path
    manifest_path: Path
    counts: dict[str, int]


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

    entities = resolve_entities(client, qids, progress=progress)

    with ThreadPoolExecutor(max_workers=VOYAGE_WORKERS) as voyage_executor:
        voyage_documents = fetch_wikivoyage_documents(
            client,
            entities=entities,
            progress=progress,
            executor=voyage_executor,
        )

    all_documents = wikipedia_documents + voyage_documents
    with ThreadPoolExecutor(max_workers=ARTICLE_WORKERS) as article_executor:
        sections_by_project = fetch_document_sections(
            client,
            documents=all_documents,
            progress=progress,
            executor=article_executor,
        )

    facts = build_wikidata_facts(client, entities=entities, progress=progress)

    write_sidecars(
        paths,
        wikipedia_documents=wikipedia_documents,
        wikivoyage_documents=voyage_documents,
        sections_by_project=sections_by_project,
        facts=facts,
        progress=progress,
    )

    if core_inputs.core_hashes != {str(path): sha256_file(path) for path in core_inputs.core_paths}:
        raise RuntimeError("Core artifacts changed during augmentation")

    counts = {
        "wikipedia_documents": len(wikipedia_documents),
        "wikipedia_sections": len(sections_by_project["wikipedia"]),
        "wikivoyage_documents": len(voyage_documents),
        "wikivoyage_sections": len(sections_by_project["wikivoyage"]),
        "wikidata_facts": len(facts),
    }
    manifest_path = update_augmentation_manifest(
        data_root,
        stem=stem,
        paths=paths,
        core_hashes=core_inputs.core_hashes,
        counts=counts,
        completed_at=utc_now_iso(),
    )
    return AugmentationResult(
        paths[0], paths[1], paths[2], paths[3], paths[4], manifest_path, counts
    )


def completed_region_stems(data_root: DataRoot) -> list[str]:
    article_stems = {path.stem for path in data_root.processed_articles.glob("*.parquet")}
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
    core_paths = (
        data_root.processed_articles / f"{stem}.parquet",
        data_root.processed_polygons / f"{stem}.parquet",
    )
    if not all(path.exists() for path in core_paths):
        return False
    expected = entry.get("core_hashes")
    current = {str(path): sha256_file(path) for path in core_paths}
    return bool(expected == current)


__all__ = [
    "AugmentationResult",
    "augment_region",
    "augmentation_is_current",
    "completed_region_stems",
    "sidecar_paths",
]
