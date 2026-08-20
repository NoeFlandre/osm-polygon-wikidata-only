"""Behavioral contracts for Wikidata fact normalization."""

from __future__ import annotations

import json

import pytest

from osm_polygon_wikidata_only.augmentation.wikimedia import normalize_facts


def _claim(datatype: str, value: object, *, snaktype: str = "value") -> dict[str, object]:
    return {
        "rank": "preferred",
        "mainsnak": {
            "snaktype": snaktype,
            "datatype": datatype,
            "datavalue": {"value": value},
        },
        "qualifiers": {"P580": [{"value": "start"}]},
        "references": [{"hash": "ref"}],
    }


@pytest.mark.parametrize(
    ("datatype", "value", "expected_text", "expected_numeric", "expected_unit"),
    [
        (
            "quantity",
            {"amount": "+12.5", "unit": "http://www.wikidata.org/entity/Q123"},
            "+12.5",
            12.5,
            "Q123",
        ),
        ("time", {"time": "+2020-01-01T00:00:00Z"}, "+2020-01-01T00:00:00Z", None, ""),
        ("string", "literal", "literal", None, ""),
    ],
)
def test_normalize_facts_preserves_supported_value_shapes(
    datatype: str,
    value: object,
    expected_text: str,
    expected_numeric: float | None,
    expected_unit: str,
) -> None:
    facts = normalize_facts(
        {"id": "Q1", "claims": {"P2044": [_claim(datatype, value)]}},
        {"P2044": {"en": "elevation"}},
    )

    assert len(facts) == 1
    assert facts[0].value_text == expected_text
    assert facts[0].numeric_value == expected_numeric
    assert facts[0].unit_entity_id == expected_unit
    assert json.loads(facts[0].qualifiers) == {"P580": [{"value": "start"}]}
    assert json.loads(facts[0].references) == [{"hash": "ref"}]


def test_normalize_facts_resolves_entity_labels_and_falls_back_to_qid() -> None:
    entity = {
        "id": "Q1",
        "claims": {
            "P31": [_claim("wikibase-item", {"id": "Q2"})],
            "P131": [_claim("wikibase-item", {"id": "Q3"})],
        },
    }

    facts = normalize_facts(entity, {"Q2": {"en": "country"}})

    by_property = {fact.property_id: fact for fact in facts}
    assert by_property["P31"].value_text == "country"
    assert by_property["P131"].value_text == "Q3"
    assert by_property["P31"].value_label_en == "country"
    assert by_property["P131"].value_label_en == "Q3"


def test_normalize_facts_skips_non_values_unsupported_properties_and_raw_values() -> None:
    entity = {
        "id": "Q1",
        "claims": {
            "P17": [_claim("string", {"unexpected": True})],
            "P31": [_claim("string", "ok", snaktype="somevalue")],
            "P999": [_claim("string", "ignored")],
        },
    }

    assert normalize_facts(entity, {}) == []
