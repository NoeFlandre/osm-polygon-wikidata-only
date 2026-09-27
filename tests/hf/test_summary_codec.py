"""Exact round-trip and decoding-boundary tests for the summary cache codec."""

from __future__ import annotations

import json

import pytest

from osm_polygon_wikidata_only.hf._dataset_stats.models import PerFileSummary
from osm_polygon_wikidata_only.hf._dataset_stats.summary_codec import (
    summary_from_json,
    summary_to_json,
)


def _full_summary() -> PerFileSummary:
    # Every field carries a distinct non-default value so a swapped or
    # dropped field changes the encoded payload.
    return PerFileSummary(
        relative_path="wikipedia/en/part-0.parquet",
        fingerprint="fp-123",
        file_size_bytes=4096,
        kind="documents",
        scan_failed=True,
        rows=11,
        non_empty=12,
        empty_or_null=13,
        total_chars=14,
        total_words=15,
        total_tokens_estimate=16,
        document_ids=frozenset({"d2", "d1"}),
        section_ids=frozenset({"s2", "s1"}),
        qids=frozenset({"Q2", "Q1"}),
        languages={"fr": 2, "en": 1},
        fact_rows=17,
        fact_ids=frozenset({"f2", "f1"}),
        subject_qids=frozenset({"Q20", "Q10"}),
        property_ids=frozenset({"P2", "P1"}),
        property_labels={"P2": "two", "P1": "one"},
        property_counts={"P2": 4, "P1": 3},
        with_property_en_label=18,
        with_value_en_label=19,
        with_qualifiers=20,
        with_references=21,
        unavailable_qualifiers=22,
        unavailable_references=23,
        value_type_counts={"string": 6, "item": 5},
    )


EXPECTED_JSON = {
    "relative_path": "wikipedia/en/part-0.parquet",
    "fingerprint": "fp-123",
    "file_size_bytes": 4096,
    "kind": "documents",
    "scan_failed": True,
    "rows": 11,
    "non_empty": 12,
    "empty_or_null": 13,
    "total_chars": 14,
    "total_words": 15,
    "total_tokens_estimate": 16,
    "document_ids": ["d1", "d2"],
    "section_ids": ["s1", "s2"],
    "qids": ["Q1", "Q2"],
    "languages": {"en": 1, "fr": 2},
    "fact_rows": 17,
    "fact_ids": ["f1", "f2"],
    "subject_qids": ["Q10", "Q20"],
    "property_ids": ["P1", "P2"],
    "property_labels": {"P1": "one", "P2": "two"},
    "property_counts": {"P1": 3, "P2": 4},
    "with_property_en_label": 18,
    "with_value_en_label": 19,
    "with_qualifiers": 20,
    "with_references": 21,
    "unavailable_qualifiers": 22,
    "unavailable_references": 23,
    "value_type_counts": {"item": 5, "string": 6},
}


def test_summary_to_json_is_exact_and_sorted() -> None:
    encoded = summary_to_json(_full_summary())

    assert encoded == EXPECTED_JSON
    assert json.dumps(encoded) == json.dumps(EXPECTED_JSON)


def test_summary_round_trips_through_json_text() -> None:
    summary = _full_summary()

    decoded = summary_from_json(json.loads(json.dumps(summary_to_json(summary))))

    assert decoded == summary


def test_minimal_entry_decodes_to_defaults() -> None:
    decoded = summary_from_json(
        {"relative_path": "a", "fingerprint": "b", "file_size_bytes": 1, "kind": "facts"}
    )

    assert decoded == PerFileSummary(
        relative_path="a", fingerprint="b", file_size_bytes=1, kind="facts"
    )
    assert decoded is not None
    assert decoded.scan_failed is False


@pytest.mark.parametrize("missing", ["relative_path", "fingerprint", "file_size_bytes", "kind"])
def test_entry_missing_required_key_is_incompatible(missing: str) -> None:
    blob = {"relative_path": "a", "fingerprint": "b", "file_size_bytes": 1, "kind": "k"}
    del blob[missing]

    assert summary_from_json(blob) is None


def test_malformed_values_fall_back_to_safe_defaults() -> None:
    decoded = summary_from_json(
        {
            "relative_path": None,
            "fingerprint": 7,
            "file_size_bytes": [1],
            "kind": None,
            "scan_failed": None,
            "rows": 3.9,
            "non_empty": "5",
            "empty_or_null": b"6",
            "total_chars": None,
            "document_ids": "not-a-list",
            "section_ids": [1, 2],
            "languages": ["en"],
            "property_labels": {1: 2},
            "property_counts": {5: "7"},
            "value_type_counts": "x",
        }
    )

    assert decoded == PerFileSummary(
        relative_path="",
        fingerprint="7",
        file_size_bytes=0,
        kind="",
        scan_failed=False,
        rows=3,
        non_empty=5,
        empty_or_null=6,
        total_chars=0,
        document_ids=frozenset(),
        section_ids=frozenset({"1", "2"}),
        languages={},
        property_labels={"1": "2"},
        property_counts={"5": 7},
        value_type_counts={},
    )


def test_truthy_non_boolean_scan_flag_decodes_true_and_falsy_false() -> None:
    base = {"relative_path": "a", "fingerprint": "b", "file_size_bytes": 1, "kind": "k"}

    truthy = summary_from_json({**base, "scan_failed": 1})
    falsy = summary_from_json({**base, "scan_failed": 0})

    assert truthy is not None and truthy.scan_failed is True
    assert falsy is not None and falsy.scan_failed is False
