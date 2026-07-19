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

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf._dataset_stats.augmentation import (
    compute_augmentation_stats,
)
from osm_polygon_wikidata_only.hf._dataset_stats.models import (
    AugmentationStats,
    ProjectTextStats,
    WikidataFactStats,
)
from osm_polygon_wikidata_only.hf._dataset_stats.rendering import _fmt_int
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


def _write_wikipedia_documents(
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


def _write_wikipedia_sections(
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


def _write_wikivoyage_documents(
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


def _write_wikivoyage_sections(
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


def _cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


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
    return compute_augmentation_stats(processed, cache_index_dir=_cache_dir(tmp_path))


# --- coverage classification ------------------------------------------


def test_fully_augmented_classified_when_all_five_sidecars_present(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])

    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 1
    assert stats.fully_augmented_count == 1
    assert stats.partial_augmented_count == 0
    assert stats.not_augmented_count == 0
    assert stats.orphan_sidecar_stems == ()


def test_partial_augmented_classified_when_some_sidecars_present(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 1
    assert stats.fully_augmented_count == 0
    assert stats.partial_augmented_count == 1
    assert stats.not_augmented_count == 0


def test_not_augmented_classified_when_no_sidecars_present(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 1
    assert stats.fully_augmented_count == 0
    assert stats.partial_augmented_count == 0
    assert stats.not_augmented_count == 1


def test_orphan_sidecar_stems_classified_when_no_core_polygon(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "ghost-latest.parquet",
        [],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 0
    assert stats.orphan_sidecar_stems == ("ghost-latest",)


def test_augmented_region_can_be_empty_rows_but_still_structural(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])
    stats = _stats(processed, tmp_path)
    assert stats.fully_augmented_count == 1


# --- wikipedia documents ------------------------------------------------


def test_wikipedia_documents_basic_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello world.",
                "article_length_chars": 12,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
            {
                "document_id": "d2",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "fr",
                "full_text": "Bonjour le monde.",
                "article_length_chars": 16,
                "article_length_words": 3,
                "article_length_tokens_estimate": 4,
            },
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "duplicate row",
                "article_length_chars": 13,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikipedia_documents.rows == 3
    assert stats.wikipedia_documents.unique_documents == 2
    assert stats.wikipedia_documents.unique_qids == 1
    assert stats.wikipedia_documents.language_count == 2
    assert stats.wikipedia_documents.region_count == 1


def test_wikipedia_documents_non_empty_and_empty_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello world.",
                "article_length_chars": 12,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
            {
                "document_id": "d2",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "fr",
                "full_text": "",
                "article_length_chars": 0,
                "article_length_words": 0,
                "article_length_tokens_estimate": 0,
            },
            {
                "document_id": "d3",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "de",
                "full_text": None,
                "article_length_chars": 0,
                "article_length_words": 0,
                "article_length_tokens_estimate": 0,
            },
            {
                "document_id": "d4",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "es",
                "full_text": "   ",
                "article_length_chars": 3,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikipedia_documents.rows == 4
    assert stats.wikipedia_documents.non_empty == 1
    assert stats.wikipedia_documents.empty_or_null == 3
    # non_empty_rate is rows=4 + non_empty=1 → 25.0%.
    assert abs(stats.wikipedia_documents.non_empty_rate - 0.25) < 1e-9


def test_wikipedia_documents_top_languages_deterministic_ties(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    rows = []
    for lang, count in (("aa", 1), ("bb", 1), ("cc", 2)):
        for idx in range(count):
            rows.append(
                {
                    "document_id": f"d-{lang}-{idx}",
                    "wikidata": "Q1",
                    "project": "wikipedia",
                    "language": lang,
                    "full_text": "x",
                    "article_length_chars": 1,
                    "article_length_words": 1,
                    "article_length_tokens_estimate": 1,
                }
            )
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        rows,
    )
    stats = _stats(processed, tmp_path)
    languages = [lang for lang, _ in stats.wikipedia_documents.top_languages]
    assert languages[0] == "cc"
    assert languages.index("aa") < languages.index("bb")


def test_wikipedia_documents_total_words_and_tokens(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 100,
                "article_length_tokens_estimate": 25,
            },
            {
                "document_id": "d2",
                "wikidata": "Q2",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 200,
                "article_length_tokens_estimate": 50,
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikipedia_documents.total_words == 300
    assert stats.wikipedia_documents.total_tokens_estimate == 75
    assert stats.wikipedia_documents.total_chars == 2


# --- wikipedia sections -------------------------------------------------


def test_wikipedia_sections_basic_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "s1",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "Intro",
                "text_length_chars": 5,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            },
            {
                "section_id": "s2",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "Body",
                "text_length_chars": 4,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            },
            {
                "section_id": "s3",
                "document_id": "d2",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "fr",
                "text": "Intro FR",
                "text_length_chars": 8,
                "text_length_words": 2,
                "text_length_tokens_estimate": 2,
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikipedia_sections.rows == 3
    # Unique sections is the count of distinct `section_id` values.
    assert stats.wikipedia_sections.unique_section_ids == 3
    # Unique documents is the count of distinct `document_id` values.
    assert stats.wikipedia_sections.unique_documents == 2
    assert stats.wikipedia_sections.unique_qids == 1
    assert stats.wikipedia_sections.total_words == 4
    assert stats.wikipedia_sections.total_tokens_estimate == 4


def test_wikipedia_sections_avg_per_represented_doc(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    rows = [
        {
            "section_id": f"s{i}",
            "document_id": "d1" if i < 3 else "d2",
            "wikidata": "Q1",
            "project": "wikipedia",
            "language": "en",
            "text": "x",
            "text_length_chars": 1,
            "text_length_words": 1,
            "text_length_tokens_estimate": 1,
        }
        for i in range(4)
    ]
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        rows,
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikipedia_sections.avg_sections_per_doc == 2.0
    # unique_section_ids counts distinct section IDs even though we
    # re-used the same one for d1 (4 distinct: s0..s3).
    assert stats.wikipedia_sections.unique_section_ids == 4
    # Two distinct documents.
    assert stats.wikipedia_sections.unique_documents == 2


# --- wikivoyage --------------------------------------------------------


def test_wikivoyage_documents_basic_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "v1",
                "wikidata": "Q1",
                "project": "wikivoyage",
                "language": "en",
                "full_text": "Visit Monaco.",
                "article_length_chars": 12,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
            {
                "document_id": "v2",
                "wikidata": "Q1",
                "project": "wikivoyage",
                "language": "en",
                "full_text": "",
                "article_length_chars": 0,
                "article_length_words": 0,
                "article_length_tokens_estimate": 0,
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikivoyage_documents.rows == 2
    assert stats.wikivoyage_documents.unique_documents == 2
    assert stats.wikivoyage_documents.non_empty == 1
    assert stats.wikivoyage_documents.empty_or_null == 1
    assert stats.wikivoyage_documents.total_words == 2


def test_wikivoyage_sections_basic_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_wikivoyage_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "sv1",
                "document_id": "v1",
                "wikidata": "Q1",
                "project": "wikivoyage",
                "language": "en",
                "text": "x",
                "text_length_chars": 1,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            }
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikivoyage_sections.rows == 1
    assert stats.wikivoyage_sections.unique_section_ids == 1
    assert stats.wikivoyage_sections.unique_documents == 1


def test_wikivoyage_language_distribution(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    rows = []
    for lang, n in (("en", 3), ("fr", 2), ("de", 1)):
        for i in range(n):
            rows.append(
                {
                    "document_id": f"voy-{lang}-{i}",
                    "wikidata": "Q1",
                    "project": "wikivoyage",
                    "language": lang,
                    "full_text": "x",
                    "article_length_chars": 1,
                    "article_length_words": 1,
                    "article_length_tokens_estimate": 1,
                }
            )
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        rows,
    )
    stats = _stats(processed, tmp_path)
    assert [lang for lang, _ in stats.wikivoyage_documents.top_languages] == ["en", "fr", "de"]


# --- wikidata facts -----------------------------------------------------


def test_wikidata_facts_unique_subjects_and_properties(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "{}",
                "references": "[]",
            },
            {
                "fact_id": "f2",
                "wikidata": "Q1",
                "property_id": "P31",
                "property_label_en": "instance of",
                "property_labels": '{"en": "instance of"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q6256",
                "value_label_en": "country",
                "value_labels": '{"en": "country"}',
                "value_text": "Q6256",
                "qualifiers": "{}",
                "references": "[]",
            },
            {
                "fact_id": "f3",
                "wikidata": "Q235",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "string",
                "value_entity_id": "",
                "value_label_en": "",
                "value_labels": "{}",
                "value_text": "France",
                "qualifiers": "{}",
                "references": "[]",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.rows == 3
    assert stats.wikidata_facts.unique_facts == 3
    assert stats.wikidata_facts.unique_subjects == 2
    assert stats.wikidata_facts.distinct_property_ids == 2


def test_wikidata_facts_english_label_coverage(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "{}",
                "references": "[]",
            },
            {
                "fact_id": "f2",
                "wikidata": "Q1",
                "property_id": "P9999",
                "property_label_en": "",
                "property_labels": "{}",
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q1",
                "value_label_en": "",
                "value_labels": "{}",
                "value_text": "Q1",
                "qualifiers": "{}",
                "references": "[]",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.with_property_en_label == 1
    assert stats.wikidata_facts.with_value_en_label == 1
    assert stats.wikidata_facts.with_qualifiers == 0
    assert stats.wikidata_facts.with_references == 0


def test_wikidata_facts_qualifiers_and_references_detected(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": json.dumps({"P580": "2000-01-01"}),
                "references": json.dumps([{"snaks": {"P248": ["Q5"]}}]),
            },
            {
                "fact_id": "f2",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "null",
                "references": "[]",
            },
            {
                "fact_id": "f3",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "",
                "references": "   ",
            },
            {
                "fact_id": "f4",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "{}",
                "references": "{}",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.with_qualifiers == 1
    assert stats.wikidata_facts.with_references == 1


def test_wikidata_facts_top_properties_deterministic(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    rows: list[dict] = []
    facts = [("P17", "country"), ("P31", "instance of"), ("P131", "located in")]
    for property_id, label in facts:
        for _ in range(2):
            rows.append(
                {
                    "fact_id": f"{property_id}-{label}",
                    "wikidata": "Q1",
                    "property_id": property_id,
                    "property_label_en": label,
                    "property_labels": json.dumps({"en": label}),
                    "value_type": "wikibase-entityid",
                    "value_entity_id": "Q2",
                    "value_label_en": "x",
                    "value_labels": '{"en": "x"}',
                    "value_text": "Q2",
                    "qualifiers": "{}",
                    "references": "[]",
                }
            )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", rows)
    stats = _stats(processed, tmp_path)
    top = stats.wikidata_facts.top_properties
    assert [prop for prop, _, _ in top] == ["P131", "P17", "P31"]


def test_wikidata_facts_top_properties_falls_back_to_property_id(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P9999",
                "property_label_en": "",
                "property_labels": "{}",
                "value_type": "string",
                "value_entity_id": "",
                "value_label_en": "",
                "value_labels": "{}",
                "value_text": "n/a",
                "qualifiers": "{}",
                "references": "[]",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.top_properties[0][0] == "P9999"


def test_wikidata_malformed_qualifiers_count_as_unavailable(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q2",
                "value_label_en": "x",
                "value_labels": '{"en": "x"}',
                "value_text": "Q2",
                "qualifiers": "{this is not valid json",
                "references": "[]",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.with_qualifiers == 0
    assert stats.wikidata_facts.unavailable_qualifiers == 1


# --- storage bytes -----------------------------------------------------


def test_storage_bytes_separate_core_augmentation_total(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    polygons_path = _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    wiki_doc_path = _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )
    stats = _stats(processed, tmp_path)
    expected_core = polygons_path.stat().st_size
    expected_aug = wiki_doc_path.stat().st_size
    assert stats.core_parquet_bytes == expected_core
    assert stats.augmentation_parquet_bytes == expected_aug
    assert stats.total_parquet_bytes == expected_core + expected_aug


# --- missing / partial / unreadable handling --------------------------


def test_compute_augmentation_stats_handles_missing_directories(tmp_path: Path) -> None:
    """Missing sidecar sub-directories surface "No data exists yet".

    The runtime path produces a present-but-zero ProjectTextStats for
    the four document/section kinds and an empty WikidataFactStats.
    """
    import shutil

    processed = _setup_processed_dir(tmp_path)
    shutil.rmtree(processed / "wikipedia")
    shutil.rmtree(processed / "wikivoyage")
    shutil.rmtree(processed / "wikidata")
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 1
    assert stats.not_augmented_count == 1
    assert stats.fully_augmented_count == 0
    # All four text aggregations and the facts aggregation are present
    # but zeroed.
    assert stats.wikidata_facts.rows == 0
    assert stats.wikipedia_documents.rows == 0
    assert stats.wikipedia_documents.region_count == 0
    assert stats.wikivoyage_sections.rows == 0


def test_compute_augmentation_stats_handles_empty_sidecar_dirs(tmp_path: Path) -> None:
    """Sidecar dirs that exist but contain no parquet must not crash."""
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikipedia_documents.rows == 0
    assert stats.wikidata_facts.rows == 0


def test_compute_augmentation_stats_skips_unreadable_sidecar(
    tmp_path: Path,
    caplog: logging.LogCaptureFixture,
) -> None:
    """An unreadable Parquet file is counted as unreadable and skipped.

    The bytes still count toward ``augmentation_parquet_bytes`` so the
    storage accounting invariant
    ``core + augmentation == total`` keeps holding.
    """
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    # Plant a corrupt parquet by writing garbage.
    bad_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    bad_path.write_bytes(b"not a parquet at all")
    caplog.set_level(logging.WARNING)
    stats = _stats(processed, tmp_path)
    # We don't read the table a second time; the unreadable count is
    # collected during the primary scan.
    assert stats.unreadable_file_count == 1
    assert any("Skipping" in r.getMessage() for r in caplog.records)
    # Storage bytes still include the corrupt file.
    assert stats.augmentation_parquet_bytes == bad_path.stat().st_size
    # But the rows are not aggregated.
    assert stats.wikipedia_documents.rows == 0


def test_compute_augmentation_stats_records_one_region_per_core_stem(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_parquet(
        processed / "polygons" / "albania-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q2"}],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 2
    assert stats.not_augmented_count == 2


# --- ProjectTextStats / dataclass contract ----------------------------


def test_augmentation_stats_private_models_are_unfrozen_with_slots() -> None:
    """The augmentation models are private; verify the frozen/slot shape."""
    aug = AugmentationStats(
        core_region_count=0,
        fully_augmented_count=0,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=0,
        augmentation_parquet_bytes=0,
        total_parquet_bytes=0,
        unreadable_file_count=0,
    )
    assert aug.core_region_count == 0


# --- render: backwards-compatible callers ------------------------------


def test_render_stats_section_without_augmentation_stats_unchanged() -> None:
    """render_stats_section(stats) (no augmentation kwarg) must produce
    the exact previous output."""
    stats = _empty_dataset_stats()
    md = render_stats_section(stats)
    assert "## Dataset snapshot" in md
    assert "## Wikipedia coverage funnel" in md
    assert "## Language distribution" in md
    assert "## Wikipedia text corpus" not in md
    assert "## Wikidata facts" not in md
    assert "## Augmentation coverage" not in md


def test_render_stats_section_with_augmentation_adds_new_sections() -> None:
    """render_stats_section(stats, augmentation_stats=...) MUST append
    the new sections in the documented order."""
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=10,
        augmentation_parquet_bytes=20,
        total_parquet_bytes=30,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "## Augmentation coverage" not in md
    assert "## Storage accounting" in md
    assert "## Wikipedia text corpus" in md
    assert "## Wikivoyage text corpus" in md
    assert "## Wikidata facts" in md
    assert (
        md.index("## Storage accounting")
        < md.index("## Wikipedia text corpus")
        < md.index("## Wikivoyage text corpus")
        < md.index("## Wikidata facts")
    )


def test_render_stats_section_legacy_three_sections_byte_identical() -> None:
    """The legacy three sections must remain byte-identical when
    augmentation is provided.

    Equality is asserted directly on the rendered string, not via a
    substring prefix; only the headline table grows when augmentation
    is supplied, so the legacy-only output plus the first part of the
    with-augmentation output share exactly the same byte sequence up
    to the augmentation-specific rows of the headline table.
    """
    stats = DatasetStats(
        polygon_count=2,
        unique_wikidata_count=1,
        article_count=3,
        link_count=3,
        language_count=2,
        region_count=1,
        total_words=200,
        total_tokens_estimate=50,
        dataset_size_bytes=4096,
        polygons_with_wikipedia=2,
        polygons_with_text=2,
        polygons_with_english=2,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=2,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={"en": 2, "fr": 1},
        polygons_per_language={"en": 2, "fr": 1},
    )
    no_aug = render_stats_section(stats)
    with_aug = render_stats_section(
        stats,
        augmentation_stats=AugmentationStats(
            core_region_count=1,
            fully_augmented_count=1,
            partial_augmented_count=0,
            not_augmented_count=0,
            orphan_sidecar_stems=[],
            wikipedia_documents=ProjectTextStats(),
            wikipedia_sections=ProjectTextStats(),
            wikivoyage_documents=ProjectTextStats(),
            wikivoyage_sections=ProjectTextStats(),
            wikidata_facts=WikidataFactStats(),
            core_parquet_bytes=10,
            augmentation_parquet_bytes=20,
            total_parquet_bytes=30,
            unreadable_file_count=0,
        ),
    )
    # The legacy last headline row is "Dataset size on disk | 4.0 KB
    # |". The augmentation-aware render renames the label to "Core
    # tables size". The numeric suffix ("| 4.0 KB |") is unchanged.
    legacy_label_row = "| Dataset size on disk |"
    aug_label_row = "| Polygon and link tables size |"
    legacy_offset = no_aug.index(legacy_label_row) + len(legacy_label_row)
    aug_offset = with_aug.index(aug_label_row) + len(aug_label_row)
    # The legacy render keeps the redundant "Wikipedia articles" and
    # "Total words" rows; the augmentation render drops them. Both
    # share the same leading rows (Polygons, Unique Wikidata entities)
    # and the same language-distribution section content; only the
    # headline's middle rows diverge by design.
    assert "| Wikipedia articles |" in no_aug
    assert "| Wikipedia articles |" not in with_aug
    assert "| Total words |" in no_aug
    assert "| Total words |" not in with_aug
    # The numeric tail (" 4.0 KB |") is identical in both versions.
    legacy_suffix = no_aug[legacy_offset : legacy_offset + len(" 4.0 KB |")]
    aug_suffix = with_aug[aug_offset : aug_offset + len(" 4.0 KB |")]
    assert legacy_suffix == aug_suffix == " 4.0 KB |"
    # The legacy label must NOT appear in the augmentation render.
    assert legacy_label_row not in with_aug


def test_render_stats_section_storage_bytes_labels() -> None:
    """The rendered sections must label storage bytes with the new wording
    pinned by the task."""
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=0,
        partial_augmented_count=1,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=4096,
        augmentation_parquet_bytes=2048,
        total_parquet_bytes=6144,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "Wikipedia, Wikivoyage, and Wikidata tables size" in md
    assert "Total Parquet size" in md
    # The public polygon/link size label is used in the storage block
    # table (not the headline row). It must appear there.
    assert "Polygon and link tables size |" in md


def test_render_stats_section_legacy_storage_bytes_label() -> None:
    """Without augmentation, the storage-bytes wording is NOT rendered.

    The legacy sections do not contain any storage accounting at all.
    """
    stats = _empty_dataset_stats()
    md = render_stats_section(stats)
    assert "Core tables size" not in md
    assert "Augmentation tables size" not in md
    assert "Total Parquet size" not in md


# --- distinct rendering states (missing vs empty) ---------------------


def test_render_stats_section_distinguishes_missing_vs_present_empty() -> None:
    """A sub-directory that does not exist OR exists but holds zero
    Parquet files renders "No data exists yet."

    A sub-directory with at least one readable, valid zero-row Parquet
    sidecar renders "This sidecar is present but empty."
    """
    import tempfile

    stats = _empty_dataset_stats()
    with tempfile.TemporaryDirectory() as tmp:
        missing_path = Path(tmp) / "missing"
        empty_path = Path(tmp) / "empty"
        aug_missing = compute_augmentation_stats(
            _setup_missing_processed(missing_path),
            cache_index_dir=missing_path / "cache",
        )
        aug_empty = compute_augmentation_stats(
            _setup_empty_processed(empty_path),
            cache_index_dir=empty_path / "cache",
        )
    md_missing = render_stats_section(stats, augmentation_stats=aug_missing)
    md_empty = render_stats_section(stats, augmentation_stats=aug_empty)

    # Both fixtures have NO readable Parquet anywhere inside the
    # augmentation sub-directories, so both are "No data exists yet."
    assert md_missing.count("No data exists yet.") >= 5
    assert md_empty.count("No data exists yet.") >= 5
    assert md_empty.count("This sidecar is present but empty.") == 0
    # Headline rows still rendered for the empty case (zero rows are
    # a valid metric).
    assert "| Wikipedia documents | 0 |" in md_empty
    assert "| Wikipedia sections | 0 |" in md_empty


def test_render_stats_section_present_zero_row_sidecar_distinct(tmp_path: Path) -> None:
    """A readable, valid zero-row parquet sidecar renders "present but empty".

    The presence of a real (zero-row) parquet flips the renderer from
    "No data exists yet." to "This sidecar is present but empty."
    while leaving headline rows unchanged.
    """
    processed_dir = _setup_zero_row_present(tmp_path)
    aug = compute_augmentation_stats(
        processed_dir,
        cache_index_dir=tmp_path / "cache",
    )
    md = render_stats_section(_empty_dataset_stats(), augmentation_stats=aug)
    assert md.count("This sidecar is present but empty.") >= 5
    assert md.count("No data exists yet.") == 0


def _setup_processed_dir_with_zero_row_parquets(base: Path) -> Path:
    processed = _setup_processed_dir(base)
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])
    return processed


def _setup_zero_row_present(base: Path) -> Path:
    return _setup_processed_dir_with_zero_row_parquets(base)


def _setup_missing_processed(tmp_path: Path) -> Path:
    import shutil

    processed = _setup_processed_dir(tmp_path)
    for d in ("wikipedia", "wikivoyage", "wikidata"):
        shutil.rmtree(processed / d)
    return processed


def _setup_empty_processed(tmp_path: Path) -> Path:
    processed = _setup_processed_dir(tmp_path)
    # Sub-directories are already present but contain no files.
    return processed


# --- exact markdown rows ----------------------------------------------


def test_render_stats_section_sections_table_row_layout() -> None:
    """The Sections metrics row closes its markdown row with a final `|`.

    Also verify the exact byte layout for the four section-only rows
    to catch any malformed markdown row regressions.
    """
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(
            subdir_present=True,
            rows=4,
            unique_section_ids=4,
            unique_documents=2,
            region_count=2,
            avg_sections_per_doc=2.0,
            non_empty=3,
            empty_or_null=1,
            non_empty_rate=0.75,
            total_words=8,
        ),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=10,
        augmentation_parquet_bytes=20,
        total_parquet_bytes=30,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    # The 'Avg sections per represented document' row must close
    # with a final '|'.
    avg_line = "| Avg sections per represented document | 2.00 |"
    assert avg_line in md, f"Malformed markdown row, missing final '|': {avg_line!r}"
    # The non-empty section rate row must close with a final '|'.
    rate_line = "| Non-empty section rate | 75.0% |"
    assert rate_line in md, f"Malformed markdown row, missing final '|': {rate_line!r}"
    # Unique sections and Documents represented are distinct rows.
    assert "| Unique sections | 4 |" in md
    assert "| Documents represented | 2 |" in md


# --- headline extension -----------------------------------------------


def test_render_stats_section_headline_includes_augmentation_totals() -> None:
    """The Dataset snapshot headline shows the augmentation totals."""
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(rows=433201, total_words=164952567),
        wikipedia_sections=ProjectTextStats(rows=2318909, total_words=230802671),
        wikivoyage_documents=ProjectTextStats(rows=3876, total_words=4896213),
        wikivoyage_sections=ProjectTextStats(rows=79889),
        wikidata_facts=WikidataFactStats(rows=1018033),
        core_parquet_bytes=3140525690,
        augmentation_parquet_bytes=3009776614,
        total_parquet_bytes=6150302304,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    # Headline augmentation totals appear (no change to legacy rows).
    for label in (
        "Wikipedia documents",
        "Wikipedia sections",
        "Wikivoyage documents",
        "Wikivoyage sections",
        "Wikidata facts",
        "Wikipedia + Wikivoyage document words",
        "Total Parquet size",
    ):
        assert label in md, f"headline missing {label!r}"


# --- automated dataset-card wording contract (red → green → refactor) --


def test_render_stats_headline_drops_redundant_wikipedia_articles_row() -> None:
    """When augmentation stats are present the legacy ``Wikipedia
    articles`` headline row is dropped because it counts the same
    canonical Wikipedia document rows as ``Wikipedia documents`` and
    is therefore redundant.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "| Wikipedia articles |" not in md


def test_render_stats_headline_drops_ambiguous_total_words_row() -> None:
    """When augmentation stats are present the legacy ``Total words``
    headline row is dropped: it is ambiguous once the augmentation
    word totals (Wikipedia + Wikivoyage documents) are shown.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "| Total words |" not in md


def test_render_stats_headline_renames_document_corpus_words() -> None:
    """The combined word total is renamed to the explicit
    ``Wikipedia + Wikivoyage document words`` label.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "| Document corpus words |" not in md
    assert "| Wikipedia + Wikivoyage document words |" in md
    wiki = aug.wikipedia_documents.total_words
    voy = aug.wikivoyage_documents.total_words
    assert f"| Wikipedia + Wikivoyage document words | {_fmt_int(wiki + voy)} |" in md, (
        f"combined word value wrong; got snippet:\n{md[:600]!r}"
    )


def test_render_stats_headline_section_words_excluded_from_corpus_total() -> None:
    """The combined document-word total must equal Wikipedia document
    words plus Wikivoyage document words, and must NOT include either
    project's section words (sections duplicate document text).
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    wiki = aug.wikipedia_documents.total_words
    voy = aug.wikivoyage_documents.total_words
    sec = aug.wikipedia_sections.total_words + aug.wikivoyage_sections.total_words
    combined = wiki + voy
    assert sec > 0, "fixture must include non-zero section words to prove exclusion"
    assert combined != wiki + voy + sec
    assert f"| Wikipedia + Wikivoyage document words | {_fmt_int(combined)} |" in md


def test_render_stats_headline_includes_exclusion_sentence() -> None:
    """A one-line explanation immediately below the headline table
    states that the total sums full Wikipedia and Wikivoyage documents
    and excludes section rows because sections duplicate document text.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    snippet = "sums the full Wikipedia and Wikivoyage documents and excludes section rows"
    assert snippet in md, f"explanatory sentence missing; snippet:\n{md[:600]!r}"


def test_render_stats_headline_counts_from_supplied_snapshot() -> None:
    """Every displayed count in the augmentation-aware headline comes
    from the supplied/computed snapshot, not from hardcoded values.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    # Wikipedia documents count from the snapshot
    assert f"| Wikipedia documents | {_fmt_int(aug.wikipedia_documents.rows)} |" in md
    # Wikivoyage documents count from the snapshot
    assert f"| Wikivoyage documents | {_fmt_int(aug.wikivoyage_documents.rows)} |" in md
    assert "Fully augmented regions" not in md
    assert "Augmentation tables size" not in md


def test_render_stats_headline_deterministic() -> None:
    """Rendering the same stats + augmentation snapshot twice yields
    byte-identical output (no timestamp, UUID, or clock dependence).
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    first = render_stats_section(stats, augmentation_stats=aug)
    second = render_stats_section(stats, augmentation_stats=aug)
    assert first == second


def test_render_stats_language_section_uses_wikipedia_documents_terminology() -> None:
    """The public-facing language-distribution terminology describes
    the canonical rows as "Wikipedia documents" rather than "articles".

    This only applies to the augmentation-aware render, which is the
    surface that surfaces canonical Wikipedia document counts.
    """
    stats = DatasetStats(
        polygon_count=2,
        unique_wikidata_count=1,
        article_count=3,
        link_count=3,
        language_count=2,
        region_count=1,
        total_words=200,
        total_tokens_estimate=50,
        dataset_size_bytes=4096,
        polygons_with_wikipedia=2,
        polygons_with_text=2,
        polygons_with_english=2,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=2,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={"en": 2, "fr": 1},
        polygons_per_language={"en": 2, "fr": 1},
    )
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    # The explanatory notion "of all articles" is replaced.
    assert "of all articles" not in md
    assert "of all Wikipedia documents" in md


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


# --- cache contract ----------------------------------------------------


def test_second_refresh_reuses_cache_zero_parquet_reads(tmp_path: Path) -> None:
    """The per-file cache makes the second refresh a no-op for stable files.

    We assert that no Parquet table is read during the second call by
    spying on :func:`safe_table` calls. A reuse must, by definition, not
    touch any PyArrow table IO.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "s1",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "x",
                "text_length_chars": 1,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            }
        ],
    )
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])

    cache_dir = _cache_dir(tmp_path)

    # Cold refresh: many safe_table calls.
    real_safe_table = augmod.safe_table
    call_log: list[Path] = []

    def spy_safe_table(path, columns):
        call_log.append(Path(path))
        return real_safe_table(path, columns)

    augmod.safe_table = spy_safe_table  # type: ignore[assignment]
    try:
        first = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]
    cold_calls = len(call_log)
    assert cold_calls > 0
    assert first.fully_augmented_count == 1

    # Warm refresh: no new safe_table calls; the cache satisfies every
    # lookup.
    call_log.clear()
    augmod.safe_table = spy_safe_table  # type: ignore[assignment]
    try:
        second = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]
    warm_calls = len(call_log)
    assert warm_calls == 0
    # Same numbers across the two refreshes.
    assert second == first


def test_one_changed_file_rescans_only_that_file(tmp_path: Path) -> None:
    """A fingerprint change in one Parquet forces a rescan of that
    file (and its fingerprint change), and only that file's
    :func:`safe_table` is invoked.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    docs_path = _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "s1",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "x",
                "text_length_chars": 1,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            }
        ],
    )
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_wikivoyage_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])

    cache_dir = _cache_dir(tmp_path)
    # Cold then warm refresh.
    compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    compute_augmentation_stats(processed, cache_index_dir=cache_dir)

    # Force a fingerprint change on docs_path only.
    import time

    time.sleep(0.01)
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello world",
                "article_length_chars": 11,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
            {
                "document_id": "d2",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "And again",
                "article_length_chars": 9,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
        ],
    )
    assert docs_path.stat().st_size > 0  # touched.

    real_safe_table = augmod.safe_table
    call_log: list[Path] = []

    def spy_safe_table(path, columns):
        call_log.append(Path(path))
        return real_safe_table(path, columns)

    augmod.safe_table = spy_safe_table  # type: ignore[assignment]
    try:
        third = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]
    # Only the changed file's table is read.
    assert len(call_log) == 1
    assert call_log[0] == docs_path
    assert third.wikipedia_documents.rows == 2


def test_deleted_files_removed_from_aggregates(tmp_path: Path) -> None:
    """A sidecar removed from disk disappears from the next refresh's
    aggregates; the cache key for the missing file is dropped.
    """
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )
    _write_wikipedia_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "s1",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "x",
                "text_length_chars": 1,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            }
        ],
    )
    _write_wikivoyage_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet", []
    )
    _write_wikivoyage_sections(processed / "wikivoyage" / "sections" / "monaco-latest.parquet", [])
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    cache_dir = _cache_dir(tmp_path)

    first = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert first.wikipedia_documents.rows == 1
    assert first.fully_augmented_count == 1

    docs_path.unlink()
    second = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert second.wikipedia_documents.rows == 0
    assert second.wikipedia_documents.region_count == 0
    # Coverage drops from fully to partial after the deletion.
    assert second.fully_augmented_count == 0
    assert second.partial_augmented_count == 1


def test_publication_uses_external_cache_dir(tmp_path: Path) -> None:
    """The publication layer points compute_augmentation_stats at
    data_root.cache (an external directory). Sanity-check that path.
    """
    data_root_path = tmp_path / "data"
    cache_path = data_root_path / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    # Touching the cache index file proves compute_augmentation_stats
    # writes under data_root.cache.
    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.hf.publication import write_readme_snapshot

    data_root = DataRoot(data_root_path)
    data_root.ensure()
    # No core or augmentation artifacts; write_readme_snapshot still
    # writes the cache index.
    write_readme_snapshot(
        data_root,
        "test/repo",
        tmp_path / "out.md",
    )
    assert (cache_path / "stats_cache" / "index.json").exists()


# --- Legacy rendering byte-for-byte ----------------------------------


def test_render_stats_section_legacy_headline_label_unchanged() -> None:
    """Without augmentation, the headline's last label MUST stay
    ``Dataset size on disk`` -- the exact wording used before the
    augmentation extension was introduced.
    """
    stats = DatasetStats(
        polygon_count=2,
        unique_wikidata_count=1,
        article_count=3,
        link_count=3,
        language_count=2,
        region_count=1,
        total_words=200,
        total_tokens_estimate=50,
        dataset_size_bytes=4096,
        polygons_with_wikipedia=2,
        polygons_with_text=2,
        polygons_with_english=2,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=2,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={"en": 2, "fr": 1},
        polygons_per_language={"en": 2, "fr": 1},
    )
    md = render_stats_section(stats)
    # Locate the legacy last headline row by its unique label.
    assert "| Dataset size on disk | 4.0 KB |" in md, (
        f"legacy headline last row drifted; snippet: {md[:500]!r}"
    )
    # The legacy-only render must not include the augmentation
    # headline row "Wikipedia documents | <n> |" (a value-bearing
    # table row). The language-distribution section legitimately
    # uses the "Wikipedia documents" column header, so we anchor on
    # the headline table's value row shape.
    assert "| Core tables size |" not in md
    import re as _re

    headline_document_row = _re.compile(r"\| Wikipedia documents \| \d")
    assert not headline_document_row.search(md)
    assert "Total sidecar words" not in md
    # The new headline label (augmentation-aware only).
    assert "| Wikipedia + Wikivoyage document words |" not in md
    # The public-facing language section uses canonical terminology.
    assert "| Language | Wikipedia documents | % of total | Polygons |" in md


# --- Atomic cache writes ----------------------------------------------


def test_cache_index_written_atomically(tmp_path: Path, monkeypatch) -> None:
    """``write_cache_index`` must use the documented atomic_write_text.

    A failed write MUST clean up its sibling temp file. We simulate
    the failure by monkey-patching ``atomic_write_text`` to raise.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod
    from osm_polygon_wikidata_only.io.atomic import atomic_write_text

    cachemod.cache_dir(tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    index_path = cachemod.index_path(tmp_path / "cache")

    # Recreate: succeed path → atomic write creates the final file.
    payload = {"a": {"v": 1}}
    cachemod.write_cache_index(tmp_path / "cache", payload)
    assert index_path.exists()
    cached = cachemod.load_cache_index(tmp_path / "cache")
    assert cached == payload

    # The write must use atomic_write_text, not direct .write_text.
    calls: list[tuple[Path, str]] = []
    real = atomic_write_text

    def tracker(path: Path, text: str, *, encoding: str = "utf-8") -> None:
        calls.append((path, text))
        return real(path, text, encoding=encoding)

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf._dataset_stats.cache.atomic_write_text",
        tracker,
    )
    cachemod.write_cache_index(tmp_path / "cache", {"b": {"v": 2}})
    assert any(call[0] == index_path for call in calls), (
        "write_cache_index must call atomic_write_text on the index path"
    )


def test_cache_write_cleanups_sibling_on_interruption(tmp_path: Path, monkeypatch) -> None:
    """If the atomic write raises mid-flight, no stale temp file remains.

    We simulate a write failure: ``atomic_write_text`` raises, but the
    ``Path.unlink`` on the temporary file runs in the ``except`` branch.
    This test asserts the file system shows no leftover ``.tmp`` siblings.
    """
    import pytest as _pytest

    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    cachemod.cache_dir(tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    index_path = cachemod.index_path(tmp_path / "cache")

    def explode(path: Path, text: str, *, encoding: str = "utf-8") -> None:
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf._dataset_stats.cache.atomic_write_text",
        explode,
    )
    with _pytest.raises(RuntimeError, match="simulated crash mid-write"):
        cachemod.write_cache_index(tmp_path / "cache", {"x": 1})
    leftovers = [p for p in (tmp_path / "cache" / cachemod.CACHE_SUBDIR).glob("*.tmp")]
    assert leftovers == []
    assert not index_path.exists()


# --- Cache contract version ------------------------------------------


def test_cache_index_has_contract_version(tmp_path: Path) -> None:
    """The cache index must declare an explicit contract version.

    Future changes to summary fields or counting rules bump the
    version. Loading a stale or missing version REBUILDS the cache
    instead of producing wrong numbers.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    cachemod.cache_dir(tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    cachemod.write_cache_index(tmp_path / "cache", {"a": {"v": 1}})
    raw = json.loads(cachemod.index_path(tmp_path / "cache").read_text())
    assert "__contract_version__" in raw


def test_cache_load_rejects_missing_version_and_rebuilds(tmp_path: Path) -> None:
    """An index whose contract version is unknown must trigger a full
    rebuild.

    We construct an index that LOOKS fingerprint-compatible with the
    live file but whose ``__contract_version__`` is missing. A correct
    implementation MUST treat the cache as incompatible and rescan
    from scratch.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    processed = _setup_processed_dir(tmp_path)
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wikipedia_documents(
        docs_path,
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )

    cache_dir = _cache_dir(tmp_path)
    cachemod.cache_dir(cache_dir).mkdir(parents=True, exist_ok=True)
    # Plant a fingerprint-matching, version-MISSING entry. If the
    # loader trusts this, it returns rows=999 instead of 1.
    live_fp = cachemod._file_fingerprint(docs_path)
    cachemod.write_cache_index(
        cache_dir,
        {
            "wikipedia/documents/monaco-latest.parquet": {
                "relative_path": "wikipedia/documents/monaco-latest.parquet",
                "fingerprint": live_fp,
                "file_size_bytes": docs_path.stat().st_size,
                "kind": "documents",
                "scan_failed": False,
                "rows": 999,
                "non_empty": 999,
                "empty_or_null": 0,
                "total_chars": 11,
                "total_words": 2,
                "total_tokens_estimate": 3,
                "document_ids": ["d1"],
                "section_ids": [],
                "qids": ["Q1"],
                "languages": {"en": 1},
                "fact_rows": 0,
                "fact_ids": [],
                "subject_qids": [],
                "property_ids": [],
                "property_labels": {},
                "property_counts": {},
                "with_property_en_label": 0,
                "with_value_en_label": 0,
                "with_qualifiers": 0,
                "with_references": 0,
                "unavailable_qualifiers": 0,
                "unavailable_references": 0,
                "value_type_counts": {},
            }
        },
    )
    # Remove the contract version: write directly to the file.
    raw = json.loads(cachemod.index_path(cache_dir).read_text())
    raw.pop("__contract_version__", None)
    cachemod.index_path(cache_dir).write_text(json.dumps(raw), encoding="utf-8")

    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    real_safe_table = augmod.safe_table
    calls: list[Path] = []

    def spy(path, cols):
        calls.append(Path(path))
        return real_safe_table(path, cols)

    augmod.safe_table = spy  # type: ignore[assignment]
    try:
        stats = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]

    assert stats.wikipedia_documents.rows == 1
    assert any(c == docs_path for c in calls), (
        "Missing version must trigger a full rebuild; live fingerprint alone is not enough"
    )


# --- Failed-scan caching ----------------------------------------------


def test_scan_failed_entry_is_retried_on_next_refresh(tmp_path: Path) -> None:
    """A cached ``scan_failed=True`` entry must be retried on the next
    refresh. The current code reuses it forever while the fingerprint
    is unchanged, contradicting its docstring and preventing recovery.
    Once a sidecar becomes readable, the next refresh must surface its
    rows and remove the failure flag.

    To force the cache HIT path (which is where the bug lives), the
    test plants a valid sidecar whose fingerprint matches the cached
    one for that file but whose actual content was rewritten via the
    direct cache to look like ``scan_failed=True``. The recovery must
    come from the ``scan_failed`` retry rule, not from a fingerprint
    change.
    """
    processed = _setup_processed_dir(tmp_path)
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wikipedia_documents(
        docs_path,
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello",
                "article_length_chars": 5,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )

    cache_dir = _cache_dir(tmp_path)
    first = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert first.wikipedia_documents.rows == 1
    assert first.unreadable_file_count == 0

    # Inject a CACHED entry whose fingerprint matches but whose
    # ``scan_failed=True`` would otherwise mask the readable file.
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    cachemod.cache_dir(cache_dir).mkdir(parents=True, exist_ok=True)
    cached_index = cachemod.load_cache_index(cache_dir)
    target_key = "wikipedia/documents/monaco-latest.parquet"
    real_blob = cached_index[target_key]
    sabotaged = dict(real_blob)
    sabotaged["scan_failed"] = True
    sabotaged["rows"] = 0
    cached_index[target_key] = sabotaged
    cachemod.write_cache_index(cache_dir, cached_index)

    # Call again with the SAME file on disk + sabotaged cache. The
    # correct behaviour: ``scan_failed=True`` triggers a rescan, so
    # the rows become 1 again and the unreadable count stays 0.
    second = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert second.wikipedia_documents.rows == 1, (
        "Failed-scan entries must be retried on the next refresh"
    )
    assert second.unreadable_file_count == 0, (
        "A previously-failed sidecar must no longer be counted unreadable"
    )


# --- Stronger file invalidation ---------------------------------------


def test_fingerprint_detects_same_size_replacement_preserving_mtime(tmp_path: Path) -> None:
    """The fingerprint must catch a same-size replacement that preserves
    mtime. Size + mtime_ns alone is insufficient. We add inode + ctime_ns
    to the fingerprint and assert that a swap of content with identical
    bytes-of-length and identical mtime still triggers a rescan.

    To truly exercise the bug, we plant an ``old_fingerprint`` in the
    cache that pinpoints the regression: a string with the SAME
    ``size`` and ``mtime_ns`` as the current file. A correct
    implementation must add more keys to the fingerprint so that this
    fabricated stale entry no longer matches the live file.
    """
    import os

    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    processed = _setup_processed_dir(tmp_path)
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wikipedia_documents(
        docs_path,
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello world",
                "article_length_chars": 11,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            }
        ],
    )

    cache_dir = _cache_dir(tmp_path)
    cachemod.cache_dir(cache_dir).mkdir(parents=True, exist_ok=True)
    stat = docs_path.stat()
    raw_size_mtime_fp = f"{stat.st_size}:{stat.st_mtime_ns}"
    # Plant a stale cache entry whose fingerprint is exactly the
    # legacy ``size:mtime_ns`` form. A correct fingerprint includes
    # inode/ctime_ns, so the live fingerprint must NOT equal the
    # planted one.
    cachemod.write_cache_index(
        cache_dir,
        {
            "wikipedia/documents/monaco-latest.parquet": {
                "relative_path": "wikipedia/documents/monaco-latest.parquet",
                "fingerprint": raw_size_mtime_fp,
                "file_size_bytes": stat.st_size,
                "kind": "documents",
                "scan_failed": False,
                "rows": 999,
                "non_empty": 999,
                "empty_or_null": 0,
                "total_chars": 11,
                "total_words": 2,
                "total_tokens_estimate": 3,
                "document_ids": ["d1"],
                "section_ids": [],
                "qids": ["Q1"],
                "languages": {"en": 1},
                "fact_rows": 0,
                "fact_ids": [],
                "subject_qids": [],
                "property_ids": [],
                "property_labels": {},
                "property_counts": {},
                "with_property_en_label": 0,
                "with_value_en_label": 0,
                "with_qualifiers": 0,
                "with_references": 0,
                "unavailable_qualifiers": 0,
                "unavailable_references": 0,
                "value_type_counts": {},
            }
        },
    )

    # Live fingerprint must differ from the planted stale one.
    live_fp = cachemod._file_fingerprint(docs_path)
    assert live_fp != raw_size_mtime_fp, (
        "Fingerprint must encode more than size:mtime_ns; same-size "
        "same-mtime replacements must invalidate the cache"
    )

    # Warm refresh detects the change and re-scans, returning the
    # real row count (1, not the planted 999).
    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    real = augmod.safe_table
    calls: list[Path] = []

    def spy(path, cols):
        calls.append(Path(path))
        return real(path, cols)

    augmod.safe_table = spy  # type: ignore[assignment]
    try:
        stats = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real  # type: ignore[assignment]
    assert stats.wikipedia_documents.rows == 1
    assert any(c == docs_path for c in calls), (
        "Warm refresh must rescan the file even when size+mtime match"
    )
    # Suppress unused import lint.
    _ = os.path


# --- Empty-data classification ----------------------------------------


def test_subdir_present_requires_at_least_one_readable_parquet(tmp_path: Path) -> None:
    """``subdir_present`` must reflect whether a readable valid Parquet
    sidecar exists, not just whether the directory is on disk.

    * Directory exists but contains no parquets ⇒ "No data exists
      yet." (subdir_present ``False``).
    * Directory exists with at least one readable, valid zero-row
      Parquet ⇒ "This sidecar is present but empty." (subdir_present
      ``True``).
    """
    processed = _setup_processed_dir(tmp_path)
    stats = compute_augmentation_stats(processed, cache_index_dir=_cache_dir(tmp_path))
    assert stats.wikipedia_documents.subdir_present is False
    assert stats.wikivoyage_documents.subdir_present is False

    _write_wikipedia_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [],
    )
    stats = compute_augmentation_stats(processed, cache_index_dir=_cache_dir(tmp_path))
    assert stats.wikipedia_documents.subdir_present is True
    assert stats.wikipedia_sections.subdir_present is False
    assert stats.wikivoyage_documents.subdir_present is False
    assert stats.wikidata_facts.subdir_present is False


# --- Document corpus words --------------------------------------------


def test_headline_document_corpus_words_excludes_sections(tmp_path: Path) -> None:
    """The headline ``Wikipedia + Wikivoyage document words`` must
    aggregate document-only word totals (Wikipedia + Wikivoyage
    documents), not sections. Section word totals stay in their
    individual sections.
    """
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(rows=1, total_words=164_000_000),
        wikipedia_sections=ProjectTextStats(rows=100, total_words=230_000_000),
        wikivoyage_documents=ProjectTextStats(rows=1, total_words=4_900_000),
        wikivoyage_sections=ProjectTextStats(rows=10, total_words=50_000_000),
        wikidata_facts=WikidataFactStats(rows=1),
        core_parquet_bytes=10,
        augmentation_parquet_bytes=20,
        total_parquet_bytes=30,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    head_block = md.split("## Storage accounting", 1)[0]
    row = next(
        line
        for line in head_block.splitlines()
        if line.startswith("| Wikipedia + Wikivoyage document words ")
    )
    assert "168,900,000" in row, f"document corpus words must exclude sections, got {row!r}"


# --- Module surface cleanup -------------------------------------------


def test_cache_module_does_not_export_make_cache_key() -> None:
    """The unused ``make_cache_key`` helper must be removed from the
    cache module so the surface reflects reality.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    assert "make_cache_key" not in getattr(cachemod, "__all__", ())
    assert not hasattr(cachemod, "make_cache_key")


def test_avg_sections_docstring_does_not_claim_trailing_pipe() -> None:
    """``_avg_sections`` returns the float string; the caller adds the
    trailing ``|``. Its docstring must not claim it returns the pipe.
    """
    import inspect

    from osm_polygon_wikidata_only.hf._dataset_stats import rendering as rendering_mod

    doc = inspect.getdoc(rendering_mod._avg_sections) or ""
    assert "trailing" not in doc.lower(), (
        f"_avg_sections docstring must not claim it returns the trailing '|': {doc!r}"
    )


def test_cache_module_docstring_matches_storage_path() -> None:
    """The cache module docstring must reflect the actual storage path
    ``<data_root>/cache/stats_cache/index.json`` and the
    (relative_path, fingerprint) lookup key.
    """
    import inspect

    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    doc = inspect.getdoc(cachemod) or ""
    assert "stats_cache/index.json" in doc, (
        f"Cache docstring must mention storage path stats_cache/index.json: {doc!r}"
    )
    assert "(relative_path, fingerprint)" in doc or (
        "relative_path" in doc and "fingerprint" in doc
    ), f"Cache docstring must explain the cache key: {doc!r}"
    assert "stats_cache/index``" not in doc and "stats_cache/index " not in doc
