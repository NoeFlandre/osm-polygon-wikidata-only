"""Tests for the augmentation statistics layer.

These tests verify that ``compute_augmentation_stats`` produces
factual counts derived directly from the local finalized Parquet
files, and that ``render_stats_section`` honors the augmentation
extension for backwards-compatible callers. The tests also pin
the cache contract: second refresh performs zero Parquet table
reads, a single changed file is rescanned, deleted files are
removed.

The fixtures used here are small, purpose-built, and never touch
Wikimedia or Hugging Face.
"""

from __future__ import annotations

# ruff: noqa: F401

import json
import logging
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmentation_module
from osm_polygon_wikidata_only.hf._dataset_stats.augmentation import (
    _augmentation_bytes,
    _has_readable_summary,
    _load_or_scan_summary,
    compute_augmentation_stats,
)
from osm_polygon_wikidata_only.hf._dataset_stats.models import (
    AugmentationStats,
    CombinedLanguageStats,
    PerFileSummary,
    ProjectTextStats,
    WikidataFactStats,
)
from osm_polygon_wikidata_only.hf._dataset_stats.rendering import (
    _article_tail_counts,
    _fmt_int,
    _fmt_size,
    _render_unreadable_note,
)
from osm_polygon_wikidata_only.hf.dataset_stats import (
    DatasetStats,
    render_stats_section,
)

# --- helpers ------------------------------------------------------------

def _write_parquet(path: Path, columns: list[str], rows: list[dict]) -> Path:
    """Write a tiny parquet file with the requested columns."""
    data: dict[str, list] = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            data[c].append(row.get(c))
    table = pa.table(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path

def _write_documents(
    path: Path,
    rows: list[dict],
) -> Path:
    return _write_parquet(
        path,
        [
            "document_id",
            "wikidata",
            "project",
            "language",
            "full_text",
            "article_length_chars",
            "article_length_words",
            "article_length_tokens_estimate",
        ],
        rows,
    )

def _write_sections(
    path: Path,
    rows: list[dict],
) -> Path:
    return _write_parquet(
        path,
        [
            "section_id",
            "document_id",
            "wikidata",
            "project",
            "language",
            "text",
            "text_length_chars",
            "text_length_words",
            "text_length_tokens_estimate",
        ],
        rows,
    )

def _write_facts(path: Path, rows: list[dict]) -> Path:
    return _write_parquet(
        path,
        [
            "fact_id",
            "wikidata",
            "property_id",
            "property_label_en",
            "property_labels",
            "value_type",
            "value_entity_id",
            "value_label_en",
            "value_labels",
            "value_text",
            "qualifiers",
            "references",
        ],
        rows,
    )

def _setup_processed_dir(tmp_path: Path) -> Path:
    """Create the canonical processed/ sub-directory layout used by the
    pipeline and the stats scanner."""
    processed = tmp_path / "processed"
    (processed / "polygons").mkdir(parents=True)
    (processed / "articles").mkdir(parents=True)
    (processed / "polygon_articles").mkdir(parents=True)
    (processed / "wikipedia" / "documents").mkdir(parents=True)
    (processed / "wikipedia" / "sections").mkdir(parents=True)
    (processed / "wikivoyage" / "documents").mkdir(parents=True)
    (processed / "wikivoyage" / "sections").mkdir(parents=True)
    (processed / "wikidata" / "facts").mkdir(parents=True)
    return processed

def _empty_dataset_stats() -> DatasetStats:
    return DatasetStats(
        polygon_count=0,
        unique_wikidata_count=0,
        article_count=0,
        link_count=0,
        language_count=0,
        region_count=0,
        total_words=0,
        total_tokens_estimate=0,
        dataset_size_bytes=0,
        polygons_with_wikipedia=0,
        polygons_with_text=0,
        polygons_with_english=0,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=0,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={},
        polygons_per_language={},
    )

def _stats(processed: Path, tmp_path: Path) -> AugmentationStats:
    return compute_augmentation_stats(processed, cache_index_dir=tmp_path / "cache")

def _setup_processed_dir_with_zero_row_parquets(base: Path) -> Path:
    processed = _setup_processed_dir(base)
    _write_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])
    return processed

def _setup_missing_processed(tmp_path: Path) -> Path:
    import shutil

    processed = _setup_processed_dir(tmp_path)
    for d in ("wikipedia", "wikivoyage", "wikidata"):
        shutil.rmtree(processed / d)
    return processed

def _sample_augmentation_stats() -> AugmentationStats:
    """Deterministic augmentation snapshot for wording-contract tests.

    Word totals are intentionally non-zero for both document sets and
    both section sets so the exclusion assertions are meaningful.
    """
    return AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(rows=433201, total_words=164952567),
        wikipedia_sections=ProjectTextStats(rows=2318909, total_words=230802671),
        wikivoyage_documents=ProjectTextStats(rows=3876, total_words=4896213),
        wikivoyage_sections=ProjectTextStats(rows=79889, total_words=1234567),
        wikidata_facts=WikidataFactStats(rows=1018033),
        core_parquet_bytes=3140525690,
        augmentation_parquet_bytes=3009776614,
        total_parquet_bytes=6150302304,
        unreadable_file_count=0,
    )

__all__ = [name for name in globals() if not name.startswith("__")]

