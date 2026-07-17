"""Contracts for canonical whole-file containment policy."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.pipeline.containment_policy import (
    CONTAINMENT_RULES,
    TABLE_CONTRACTS,
    child_stems,
    validate_stem,
)


def test_policy_is_deterministic_and_complete() -> None:
    assert [(rule.parent, rule.children) for rule in CONTAINMENT_RULES] == [
        ("brandenburg-latest", ("berlin-latest",)),
        ("china-guangdong-latest", ("china-hong-kong-latest", "china-macau-latest")),
        ("china-hebei-latest", ("china-beijing-latest", "china-tianjin-latest")),
        ("indonesia-nusa-tenggara-latest", ("east-timor-latest",)),
        ("italy-latest", ("centro-latest", "nord-est-latest")),
        ("morocco-latest", ("ceuta-latest", "melilla-latest")),
        ("niedersachsen-latest", ("bremen-latest",)),
    ]
    assert child_stems() == (
        "berlin-latest",
        "bremen-latest",
        "centro-latest",
        "ceuta-latest",
        "china-beijing-latest",
        "china-hong-kong-latest",
        "china-macau-latest",
        "china-tianjin-latest",
        "east-timor-latest",
        "melilla-latest",
        "nord-est-latest",
    )


def test_table_contracts_cover_all_canonical_region_artifacts() -> None:
    assert [(contract.subdir, contract.identity_columns) for contract in TABLE_CONTRACTS] == [
        ("polygons", ("osm_type", "osm_id")),
        ("polygon_articles", ("osm_type", "osm_id", "article_id")),
        ("wikipedia/documents", ("document_id",)),
        ("wikipedia/sections", ("section_id",)),
        ("wikivoyage/documents", ("document_id",)),
        ("wikivoyage/sections", ("section_id",)),
        ("wikidata/facts", ("fact_id",)),
    ]


@pytest.mark.parametrize("stem", ["", ".", "..", "a/b", "a\\b", "italy", "-latest"])
def test_invalid_stems_are_rejected(stem: str) -> None:
    with pytest.raises(ValueError, match="Invalid containment stem"):
        validate_stem(stem)


def test_valid_stem_is_returned_unchanged() -> None:
    assert validate_stem("italy-latest") == "italy-latest"
