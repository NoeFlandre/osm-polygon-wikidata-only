"""Validation and loading of local artifacts used for publication."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

from .models import CorePublicationArtifacts, PublicationValidationError


def validate_core_artifacts(core: ProcessResult | CorePublicationArtifacts) -> None:
    """Fail before snapshot work when required core files are absent."""
    paths: Sequence[Path]
    if isinstance(core, CorePublicationArtifacts):
        paths = (core.polygons_path, core.polygon_articles_path, core.manifest_path)
    else:
        paths = (
            core.polygons_path,
            core.articles_path,
            core.polygon_articles_path,
            core.manifest_path,
        )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Core artifact missing before upload: {path}")


def validate_augmentation_artifacts(augmentation: AugmentationResult) -> None:
    """Fail before snapshot work when required sidecars are absent."""
    paths: Sequence[Path] = (
        augmentation.wikipedia_documents_path,
        augmentation.wikipedia_sections_path,
        augmentation.wikivoyage_documents_path,
        augmentation.wikivoyage_sections_path,
        augmentation.wikidata_facts_path,
        augmentation.manifest_path,
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Augmentation artifact missing before upload: {path}")
    links = augmentation.polygon_document_links_path
    if links is not None and not links.is_file():
        raise FileNotFoundError(
            f"Unified polygon-document links artifact missing before upload: {links}"
        )


def load_existing_core_artifacts(data_root: DataRoot, stem: str) -> CorePublicationArtifacts:
    """Load and validate a finalized local core shard for republishing."""
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    links_path = data_root.processed_links / f"{stem}.parquet"
    manifest_path = data_root.processed_manifests / "processed_pbfs.json"
    _require_file(polygons_path, "Polygons file")
    _require_file(links_path, "Polygon articles links file")
    _require_file(manifest_path, "Processed manifest")

    manifest_key = f"{stem}.osm.pbf"
    manifest = load_manifest(manifest_path)
    if manifest_key not in manifest:
        raise KeyError(f"Stem {stem} not found in processed manifests")
    entry = manifest[manifest_key]
    _require_manifest_value(entry, "source_pbf", manifest_key, manifest_key)
    _require_manifest_value(
        entry,
        "polygons_path",
        f"polygons/{stem}.parquet",
        manifest_key,
    )
    _require_manifest_value(
        entry,
        "polygon_articles_path",
        f"polygon_articles/{stem}.parquet",
        manifest_key,
    )
    _require_schema(polygons_path, polygon_schema(), "polygons parquet")
    links_schema = _read_schema(links_path)
    if not (
        links_schema.equals(polygon_document_link_schema(), check_metadata=True)
        or links_schema.equals(polygon_article_schema(), check_metadata=True)
    ):
        raise PublicationValidationError(f"Schema mismatch for polygon links parquet: {links_path}")

    documents_path = _optional_wikipedia_documents(data_root, stem, entry, manifest_key)
    return CorePublicationArtifacts(
        polygons_path=polygons_path,
        polygon_articles_path=links_path,
        wikipedia_documents_path=documents_path,
        manifest_path=manifest_path,
        stem=stem,
        manifest_entry=entry,
    )


def _optional_wikipedia_documents(
    data_root: DataRoot,
    stem: str,
    entry: dict[str, object],
    manifest_key: str,
) -> Path | None:
    path = data_root.processed / "wikipedia/documents" / f"{stem}.parquet"
    expected = f"wikipedia/documents/{stem}.parquet"
    if path.is_file():
        configured = entry.get("wikipedia_documents_path")
        if configured and configured != expected:
            raise PublicationValidationError(
                f"Manifest entry wikipedia_documents_path mismatch for key '{manifest_key}': "
                f"expected '{expected}', got '{configured}'"
            )
        _require_schema(path, wikipedia_document_schema(), "wikipedia documents parquet")
        return path

    augmentation_manifest = (
        data_root.processed / "augmentation/manifests/augmentation_manifest.json"
    )
    if augmentation_manifest.is_file():
        try:
            augmentation = json.loads(augmentation_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise PublicationValidationError(
                f"Malformed augmentation manifest JSON: {error}"
            ) from error
        if stem in augmentation:
            raise PublicationValidationError(
                f"Wikipedia documents file missing for augmented region {stem}: {path}"
            )
    return None


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def _require_manifest_value(
    entry: dict[str, object],
    field: str,
    expected: str,
    manifest_key: str,
) -> None:
    actual = entry.get(field)
    if actual != expected:
        raise PublicationValidationError(
            f"Manifest entry {field} mismatch for key '{manifest_key}': "
            f"expected '{expected}', got '{actual}'"
        )


def _read_schema(path: Path) -> pa.Schema:
    try:
        return pq.read_schema(path)
    except Exception as error:
        raise PublicationValidationError(f"Could not read schema for {path}: {error}") from error


def _require_schema(path: Path, expected: pa.Schema, label: str) -> None:
    if not _read_schema(path).equals(expected, check_metadata=True):
        raise PublicationValidationError(f"Schema mismatch for {label}: {path}")


__all__ = [
    "load_existing_core_artifacts",
    "validate_augmentation_artifacts",
    "validate_core_artifacts",
]
