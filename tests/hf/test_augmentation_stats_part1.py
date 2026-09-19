"""Split coverage tests (part 1)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.hf.augmentation_stats_support import *


def test_fully_augmented_classified_when_all_five_sidecars_present(tmp_path: Path) -> None:
    """Readable sidecars count as present even when they contain zero rows."""
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
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
    _write_documents(
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
    _write_documents(
        processed / "wikipedia" / "documents" / "ghost-latest.parquet",
        [],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 0
    assert stats.orphan_sidecar_stems == ("ghost-latest",)

def test_wikipedia_documents_basic_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_documents(
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
    _write_documents(
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
    _write_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        rows,
    )
    stats = _stats(processed, tmp_path)
    languages = [lang for lang, _ in stats.wikipedia_documents.top_languages]
    assert languages[0] == "cc"
    assert languages.index("aa") < languages.index("bb")

def test_wikipedia_documents_total_words_and_tokens(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_documents(
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

def test_wikipedia_sections_basic_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_sections(
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
    _write_sections(
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

def test_wikivoyage_documents_basic_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_documents(
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
    _write_sections(
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
    _write_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        rows,
    )
    stats = _stats(processed, tmp_path)
    assert [lang for lang, _ in stats.wikivoyage_documents.top_languages] == ["en", "fr", "de"]

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
