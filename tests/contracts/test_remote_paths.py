"""Freeze the deterministic Hugging Face remote paths."""

from __future__ import annotations

from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
    REMOTE_ARTICLES_DIR,
    REMOTE_AUGMENTATION_MANIFEST_FILE,
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
    remote_dataset_card_path,
    remote_parquet_path,
)


def test_directory_constants_are_stable() -> None:
    assert REMOTE_POLYGONS_DIR == "polygons"
    assert REMOTE_ARTICLES_DIR == "articles"
    assert REMOTE_LINKS_DIR == "polygon_articles"
    assert REMOTE_MANIFEST_FILE == "manifests/processed_pbfs.json"


def test_asset_paths_are_stable() -> None:
    assert REMOTE_COVERAGE_MAP_FILE == "coverage_map.png"
    assert REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE == ("assets/geographic_wikipedia_text_coverage.png")
    assert REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE == "assets/geographic_polygon_count.png"


def test_remote_dataset_card_path() -> None:
    assert remote_dataset_card_path() == "README.md"


def test_remote_parquet_path() -> None:
    assert remote_parquet_path("polygons", "monaco-latest") == "polygons/monaco-latest.parquet"
    assert remote_parquet_path("articles", "andorra-latest") == "articles/andorra-latest.parquet"


def test_canonical_augmentation_manifest_path() -> None:
    """The canonical remote augmentation manifest lives under ``manifests/``."""
    assert REMOTE_AUGMENTATION_MANIFEST_FILE == "manifests/augmentation_manifest.json"


def test_legacy_augmentation_manifest_path_named_explicitly() -> None:
    """The legacy remote augmentation manifest path is preserved under a
    named constant so the migration commit knows what to delete.

    This is the ONLY place the legacy string lives; production code
    reads it exclusively through the named constant.
    """
    assert (
        LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE
        == "augmentation/manifests/augmentation_manifest.json"
    )
