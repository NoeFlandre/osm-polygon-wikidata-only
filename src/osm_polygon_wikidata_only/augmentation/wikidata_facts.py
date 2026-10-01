"""Resolve labels and normalize Wikidata claims into fact records."""

from __future__ import annotations

from typing import Any, Protocol

from osm_polygon_wikidata_only.augmentation.models import WikidataFact
from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.wikimedia import FACT_PROPERTIES, normalize_facts


class EntityResolver(Protocol):
    """Client interface needed to resolve Wikidata entity labels."""

    def entities(self, qids: list[str] | set[str], *, props: str) -> dict[str, dict[str, Any]]: ...


def build_wikidata_facts(
    client: EntityResolver,
    *,
    entities: dict[str, dict[str, Any]],
    progress: AugmentationProgress,
) -> list[WikidataFact]:
    """Collect supported claims and their entity labels into sorted facts."""
    label_ids = _fact_label_ids(entities)
    labels = _label_maps(client.entities(label_ids, props="labels"))
    progress.start("Wikidata facts", total=len(entities))
    facts = _facts_for_entities(entities, labels, progress)
    facts.sort(key=lambda row: row.fact_id)
    return facts


def _fact_label_ids(entities: dict[str, dict[str, Any]]) -> set[str]:
    """Collect fact property and entity IDs needed for label resolution."""
    label_ids = set(FACT_PROPERTIES)
    for entity in entities.values():
        label_ids.update(_fact_entity_label_ids(entity))
    return label_ids


def _fact_entity_label_ids(entity: dict[str, Any]) -> set[str]:
    """Collect entity-valued IDs from one entity's supported claims."""
    ids: set[str] = set()
    for property_id, claims in (entity.get("claims") or {}).items():
        if property_id in FACT_PROPERTIES:
            ids.update(_fact_claim_label_ids(claims))
    return ids


def _fact_claim_label_ids(claims: list[dict[str, Any]]) -> set[str]:
    """Collect entity IDs from a claim list."""
    return {
        entity_id for claim in claims if (entity_id := _fact_claim_entity_id(claim)) is not None
    }


def _fact_claim_entity_id(claim: dict[str, Any]) -> str | None:
    """Return the entity ID carried by a claim, when present."""
    value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    return None


def _label_maps(entities: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {qid: _label_map(entity) for qid, entity in entities.items()}


def _label_map(entity: dict[str, Any]) -> dict[str, str]:
    """Normalize one Wikidata entity's language labels."""
    labels = entity.get("labels") or {}
    return {
        str(language): str(value.get("value", ""))
        for language, value in labels.items()
        if isinstance(value, dict) and value.get("value")
    }


def _facts_for_entities(
    entities: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, str]],
    progress: AugmentationProgress,
) -> list[WikidataFact]:
    """Normalize facts for each entity while advancing progress."""
    facts: list[WikidataFact] = []
    for entity in entities.values():
        facts.extend(normalize_facts(entity, labels))
        progress.advance()
    return facts


__all__ = ["FACT_PROPERTIES", "EntityResolver", "build_wikidata_facts"]
