"""TDD contracts for automatic geographic NER silver validation."""

from __future__ import annotations

from osm_polygon_wikidata_only.ner.silver_validation import audit_entities, normalize_name


def test_audit_separates_reference_matches_artifacts_and_model_agreement() -> None:
    primary = [
        {"text": "Paris", "label": "named geographic location", "start": 0, "end": 5},
        {"text": "Q123", "label": "named geographic location", "start": 6, "end": 10},
        {"text": "London", "label": "named geographic location", "start": 11, "end": 17},
    ]
    secondary = [
        {"text": "Paris", "label": "named geographic location", "start": 0, "end": 5},
        {"text": "London", "label": "named geographic location", "start": 11, "end": 17},
    ]

    assert audit_entities(primary, secondary, ("Paris", "Berlin")) == {
        "primary_entities": 3,
        "primary_artifacts": 1,
        "primary_known_name_matches": 1,
        "primary_unmatched": 1,
        "secondary_entities": 2,
        "model_agreements": 2,
    }


def test_audit_normalizes_unicode_whitespace_and_case() -> None:
    primary = [{"text": "  MÜNCHEN  ", "start": 0, "end": 10}]
    secondary = [{"text": "München", "start": 0, "end": 10}]

    result = audit_entities(primary, secondary, ("münchen",))

    assert result["primary_known_name_matches"] == 1
    assert result["primary_unmatched"] == 0
    assert result["model_agreements"] == 1


def test_audit_does_not_count_numbers_or_urls_as_geographic_entities() -> None:
    primary = [
        {"text": "12345", "start": 0, "end": 5},
        {"text": "https://example.com", "start": 6, "end": 25},
    ]

    result = audit_entities(primary, [], ("12345", "https://example.com"))

    assert result["primary_artifacts"] == 2
    assert result["primary_known_name_matches"] == 0
    assert result["primary_unmatched"] == 0


def test_audit_handles_entities_without_a_text_surface() -> None:
    result = audit_entities([{}], [], ())

    assert result["primary_entities"] == 1
    assert result["primary_artifacts"] == 1
    assert result["primary_unmatched"] == 0


def test_normalization_preserves_word_boundaries() -> None:
    assert normalize_name("  New   York  ") == "new york"
