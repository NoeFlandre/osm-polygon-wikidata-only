"""Freeze the deterministic Hugging Face remote paths."""

from __future__ import annotations

from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_ARTICLES_DIR,
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
