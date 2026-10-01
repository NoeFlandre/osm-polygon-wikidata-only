from __future__ import annotations

from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataEntity
from osm_polygon_wikidata_only.enrichment.wikipedia.entity_context import (
    entity_context_for_language,
)


def test_entity_context_uses_localized_values_then_english_fallback() -> None:
    entity = WikidataEntity(
        qid="Q1",
        labels={"fr": "", "en": "English label"},
        descriptions={"fr": "Description française", "en": "English description"},
        aliases={"fr": [], "en": ["English alias"]},
    )

    context = entity_context_for_language(entity, "fr")

    assert context.label == "English label"
    assert context.description == "Description française"
    assert context.aliases == ["English alias"]


def test_entity_context_defaults_missing_metadata_to_empty_values() -> None:
    context = entity_context_for_language(WikidataEntity(qid="Q2"), "de")

    assert context.label == ""
    assert context.description == ""
    assert context.aliases == []
