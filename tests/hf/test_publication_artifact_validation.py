from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._publication.artifacts import (
    _load_augmentation_manifest,
    _reject_missing_augmented_documents,
    validate_augmentation_artifacts,
    validate_core_artifacts,
)
from osm_polygon_wikidata_only.hf._publication.models import (
    CorePublicationArtifacts,
    PublicationValidationError,
)
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult


def _touch_all(paths: tuple[Path, ...]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")


def _augmentation(tmp_path: Path, *, links: Path | None = None) -> AugmentationResult:
    paths = (
        *(tmp_path / name for name in ("wikipedia.parquet", "sections.parquet")),
        tmp_path / "voyage.parquet",
        tmp_path / "voyage-sections.parquet",
        tmp_path / "facts.parquet",
        tmp_path / "manifest.json",
    )
    _touch_all(paths)
    return AugmentationResult(*paths, counts={}, polygon_document_links_path=links)


def test_validate_augmentation_artifacts_accepts_complete_outputs(tmp_path: Path) -> None:
    validate_augmentation_artifacts(_augmentation(tmp_path))


def test_validate_augmentation_artifacts_rejects_missing_sidecar(tmp_path: Path) -> None:
    augmentation = _augmentation(tmp_path)
    augmentation.wikipedia_sections_path.unlink()

    with pytest.raises(FileNotFoundError, match="Augmentation artifact"):
        validate_augmentation_artifacts(augmentation)


def test_validate_augmentation_artifacts_rejects_missing_links(tmp_path: Path) -> None:
    links = tmp_path / "links.parquet"
    augmentation = _augmentation(tmp_path, links=links)

    with pytest.raises(FileNotFoundError, match="links artifact"):
        validate_augmentation_artifacts(augmentation)


def test_validate_core_artifacts_accepts_existing_core_artifacts(tmp_path: Path) -> None:
    paths = tuple(tmp_path / name for name in ("polygons", "links", "manifest"))
    _touch_all(paths)
    core = CorePublicationArtifacts(
        polygons_path=paths[0],
        polygon_articles_path=paths[1],
        wikipedia_documents_path=None,
        manifest_path=paths[2],
        stem="demo-latest",
        manifest_entry={},
    )

    validate_core_artifacts(core)


def test_validate_core_artifacts_checks_process_articles_path(tmp_path: Path) -> None:
    paths = tuple(tmp_path / name for name in ("polygons", "articles", "links", "manifest"))
    _touch_all(paths)
    result = ProcessResult(
        polygons_path=paths[0],
        articles_path=paths[1],
        polygon_articles_path=paths[2],
        manifest_path=paths[3],
        polygon_count=0,
        article_count=0,
        link_count=0,
        manifest_entry={},
        stage_timings_s={},
    )

    validate_core_artifacts(result)


def test_validate_core_artifacts_rejects_missing_required_file(tmp_path: Path) -> None:
    paths = tuple(tmp_path / name for name in ("polygons", "links", "manifest"))
    _touch_all(paths[:2])
    core = CorePublicationArtifacts(
        polygons_path=paths[0],
        polygon_articles_path=paths[1],
        wikipedia_documents_path=None,
        manifest_path=paths[2],
        stem="demo-latest",
        manifest_entry={},
    )

    with pytest.raises(FileNotFoundError, match="Core artifact"):
        validate_core_artifacts(core)


def test_reject_missing_augmented_documents_checks_manifest_entry(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    manifest = data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"demo-latest": {}}), encoding="utf-8")

    with pytest.raises(PublicationValidationError, match="missing for augmented region"):
        _reject_missing_augmented_documents(
            data_root, "demo-latest", tmp_path / "documents.parquet"
        )


def test_reject_missing_augmented_documents_ignores_unlisted_region(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    manifest = data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"other-latest": {}}), encoding="utf-8")

    _reject_missing_augmented_documents(data_root, "demo-latest", tmp_path / "documents.parquet")


def test_load_augmentation_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    manifest = tmp_path / "augmentation_manifest.json"
    manifest.write_text("not-json", encoding="utf-8")

    with pytest.raises(PublicationValidationError, match="Malformed augmentation manifest"):
        _load_augmentation_manifest(manifest)


def test_load_augmentation_manifest_requires_object(tmp_path: Path) -> None:
    manifest = tmp_path / "augmentation_manifest.json"
    manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(PublicationValidationError, match="expected an object"):
        _load_augmentation_manifest(manifest)
