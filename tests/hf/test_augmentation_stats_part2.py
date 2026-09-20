"""Split coverage tests (part 2)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.hf.augmentation_stats_support import *


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


def test_storage_bytes_separate_core_augmentation_total(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    polygons_path = _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    wiki_doc_path = _write_documents(
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
