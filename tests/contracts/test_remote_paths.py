"""Freeze the deterministic Hugging Face remote paths."""

from __future__ import annotations

from osm_polygon_wikidata_only.hf.repo_layout import (
    LOCAL_DATASET_HERO_FILE,
    remote_dataset_card_path,
    remote_parquet_path,
)


def test_dataset_hero_asset_exists() -> None:
    assert LOCAL_DATASET_HERO_FILE.is_file()


def test_remote_dataset_card_path() -> None:
    assert remote_dataset_card_path() == "README.md"


def test_remote_parquet_path() -> None:
    assert remote_parquet_path("polygons", "monaco-latest") == "polygons/monaco-latest.parquet"
    assert remote_parquet_path("articles", "andorra-latest") == "articles/andorra-latest.parquet"
