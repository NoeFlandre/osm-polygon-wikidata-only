"""Focused contracts for section-sidecar row accounting."""

from __future__ import annotations

from collections import Counter
from typing import Any

from osm_polygon_wikidata_only.hf._dataset_stats import augmentation


def test_section_row_metrics_count_non_empty_text_and_lengths() -> None:
    assert augmentation._section_row_metrics(" text ", 4, "5", 6) == (1, 0, 4, 5, 6)


def test_section_row_metrics_count_empty_text_and_missing_lengths() -> None:
    assert augmentation._section_row_metrics(None, None, None, None) == (0, 1, 0, 0, 0)


def test_section_row_metrics_treats_non_string_text_as_empty() -> None:
    value: Any = {"text": "not text"}
    assert augmentation._section_row_metrics(value, 1, 2, 3) == (0, 1, 1, 2, 3)


def test_optional_string_returns_empty_for_missing_values() -> None:
    assert augmentation._optional_string(None) == ()
    assert augmentation._optional_string(0) == ()


def test_section_row_identity_normalizes_present_values() -> None:
    arrays = {
        "section_id": ["section"],
        "document_id": ["document"],
        "wikidata": ["Q1"],
        "language": ["en"],
    }
    assert augmentation._section_row_identity(arrays, 0) == (
        ("section",),
        ("document",),
        ("Q1",),
        ("en",),
    )


def test_document_row_identity_normalizes_present_values() -> None:
    arrays = {
        "document_id": ["document"],
        "wikidata": ["Q1"],
        "language": ["en"],
    }
    assert augmentation._document_row_identity(arrays, 0) == (
        ("document",),
        ("Q1",),
        ("en",),
    )


def test_json_field_counts_distinguish_content_and_unavailable_text() -> None:
    assert augmentation._json_field_counts({"source": "x"}) == (1, 0)
    assert augmentation._json_field_counts("not-json") == (0, 1)
    assert augmentation._json_field_counts("") == (0, 0)


def test_fact_row_identity_and_property_label_updates() -> None:
    arrays = {
        "fact_id": ["fact"],
        "wikidata": ["Q1"],
        "property_id": ["P31"],
    }
    assert augmentation._fact_row_identity(arrays, 0) == (
        ("fact",),
        ("Q1",),
        ("P31",),
    )
    assert augmentation._property_label_update("P31", " Instance ") == (("P31", "Instance"),)
    assert augmentation._property_label_update(None, " Instance ") == ()


def test_nonempty_text_flag_requires_a_nonblank_string() -> None:
    assert augmentation._nonempty_text(" value ") == 1
    assert augmentation._nonempty_text(" ") == 0
    assert augmentation._nonempty_text(3) == 0


def test_present_string_preserves_nonempty_string_values_only() -> None:
    assert augmentation._present_string(" ") == (" ",)
    assert augmentation._present_string(0) == ()
    assert augmentation._present_string("") == ()


def test_fact_scanner_marks_unreadable_parquet(tmp_path) -> None:
    parquet = tmp_path / "facts.parquet"
    parquet.write_text("not parquet", encoding="utf-8")
    summary = augmentation._scan_facts_file(tmp_path, parquet)
    assert summary.scan_failed is True
    assert summary.kind == augmentation.KIND_FACT


def test_record_property_updates_keeps_first_label() -> None:
    counts: Counter[str] = Counter()
    labels = {"P31": "Original"}
    augmentation._record_property_updates(("P31",), "Replacement", counts, labels)
    assert counts == {"P31": 1}
    assert labels == {"P31": "Original"}
