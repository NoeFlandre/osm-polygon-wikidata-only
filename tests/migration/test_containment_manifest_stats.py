"""Contracts for containment manifest statistics aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.pipeline.containment_migration import (
    _document_manifest_stats,
    _polygon_manifest_stats,
    load_retired_parent_children,
)


def test_polygon_manifest_stats_count_values_and_ignore_bad_tag_json() -> None:
    rows = [
        {
            "area_bucket": "small",
            "tag_keys": '["wikidata", "name"]',
            "wikidata": "Q42",
            "has_wikipedia": True,
            "text_available": True,
        },
        {
            "area_bucket": "large",
            "tag_keys": "not-json",
            "wikidata": "",
            "has_wikipedia": False,
            "text_available": False,
        },
        {
            "area_bucket": "small",
            "tag_keys": '["name"]',
            "wikidata": "Q42",
            "has_wikipedia": True,
            "text_available": False,
        },
    ]

    assert _polygon_manifest_stats(rows) == {
        "polygon_count": 3,
        "unique_wikidata_count": 1,
        "rows_with_wikipedia": 2,
        "rows_with_full_text": 1,
        "area_bucket_counts": {"small": 2, "large": 1},
        "top_tag_keys": {"wikidata": 1, "name": 2},
    }


def test_document_manifest_stats_sort_languages_and_sum_characters() -> None:
    rows = [
        {"language": "fr", "article_length_chars": 12},
        {"language": "en", "article_length_chars": 30},
        {"language": "fr", "article_length_chars": 8},
    ]

    assert _document_manifest_stats(rows) == {
        "article_count": 3,
        "language_count": 2,
        "languages": ["en", "fr"],
        "total_full_text_chars": 50,
    }


def test_retired_parent_children_groups_and_sorts_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "processed" / "manifests" / "containment_retirements.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "contract_version": "contained-region-v1",
                "retired": {
                    "child-b": {"parent": "parent"},
                    "child-a": {"parent": "parent"},
                    "other": {"parent": "another"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_retired_parent_children(tmp_path / "processed") == {
        "another": ("other",),
        "parent": ("child-a", "child-b"),
    }


def test_retired_parent_children_returns_empty_without_manifest(tmp_path: Path) -> None:
    assert load_retired_parent_children(tmp_path / "processed") == {}


def test_retired_parent_children_rejects_malformed_parent(tmp_path: Path) -> None:
    manifest = tmp_path / "processed" / "manifests" / "containment_retirements.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "contract_version": "contained-region-v1",
                "retired": {"child": {"parent": None}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed containment retirement entry"):
        load_retired_parent_children(tmp_path / "processed")
