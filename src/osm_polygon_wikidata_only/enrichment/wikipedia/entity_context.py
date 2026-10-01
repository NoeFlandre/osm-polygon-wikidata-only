"""Wikidata display metadata passed with Wikipedia article requests."""

from __future__ import annotations

from dataclasses import dataclass

from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataEntity


@dataclass(frozen=True, slots=True)
class WikipediaEntityContext:
    """Localized label, description, and aliases for one article fetch."""

    label: str
    description: str
    aliases: list[str]


def entity_context_for_language(
    entity: WikidataEntity,
    language: str,
) -> WikipediaEntityContext:
    """Choose localized metadata, falling back to English when it is empty."""
    return WikipediaEntityContext(
        label=entity.labels.get(language) or entity.labels.get("en", ""),
        description=entity.descriptions.get(language) or entity.descriptions.get("en", ""),
        aliases=entity.aliases.get(language) or entity.aliases.get("en", []),
    )


__all__ = ["WikipediaEntityContext", "entity_context_for_language"]
