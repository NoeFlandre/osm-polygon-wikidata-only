"""Pure discovery and fact-normalization helpers for augmentation."""

from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.utils.json import dumps
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import WikidataFact, stable_id

FACT_PROPERTIES = frozenset({"P17", "P31", "P131", "P279", "P361", "P571", "P1435", "P2044"})


def discover_wikivoyage_sitelinks(entity: dict[str, Any]) -> list[tuple[str, str, str]]:
    links: list[tuple[str, str, str]] = []
    for site, value in (entity.get("sitelinks") or {}).items():
        link = _voyage_sitelink(site, value)
        if link is not None:
            links.append(link)
    return sorted(links)


def _voyage_sitelink(site: str, value: Any) -> tuple[str, str, str] | None:
    """Normalize one Wikivoyage sitelink when it has a usable title."""
    if not site.endswith("wikivoyage"):
        return None
    if not isinstance(value, dict):
        return None
    title = value.get("title")
    if not title:
        return None
    language = site[: -len("wikivoyage")].replace("_", "-")
    return language, site, str(title)


def normalize_facts(
    entity: dict[str, Any], labels: dict[str, dict[str, str]]
) -> list[WikidataFact]:
    qid = str(entity.get("id", ""))
    out: list[WikidataFact] = []
    now = utc_now_iso()
    for property_id in sorted(entity.get("claims") or {}):
        if property_id not in FACT_PROPERTIES:
            continue
        out.extend(
            _normalize_property_claims(qid, property_id, entity["claims"][property_id], labels, now)
        )
    return out


FactValue = tuple[str, float | None, str, str]


def _normalize_property_claims(
    qid: str,
    property_id: str,
    claims: list[dict[str, Any]],
    labels: dict[str, dict[str, str]],
    retrieved_at: str,
) -> list[WikidataFact]:
    """Normalize all accepted claims for one Wikidata property."""
    facts: list[WikidataFact] = []
    for ordinal, claim in enumerate(claims):
        fact = _normalize_claim(qid, property_id, claim, labels, ordinal, retrieved_at)
        if fact is not None:
            facts.append(fact)
    return facts


def _normalize_claim(
    qid: str,
    property_id: str,
    claim: dict[str, Any],
    labels: dict[str, dict[str, str]],
    ordinal: int,
    retrieved_at: str,
) -> WikidataFact | None:
    snak = claim.get("mainsnak", {})
    if snak.get("snaktype") != "value":
        return None
    value = _fact_value_fields(snak, labels)
    if value is None:
        return None
    entity_id, numeric, unit, value_text = value
    property_labels = labels.get(property_id, {})
    value_labels = labels.get(entity_id, {}) if entity_id else {}
    return WikidataFact(
        stable_id(qid, property_id, entity_id or value_text, ordinal),
        qid,
        property_id,
        property_labels.get("en", property_id),
        dumps(property_labels),
        str(snak.get("datatype", "")),
        entity_id,
        value_labels.get("en", entity_id),
        dumps(value_labels),
        value_text,
        numeric,
        unit,
        str(claim.get("rank", "normal")),
        dumps(claim.get("qualifiers", {})),
        dumps(claim.get("references", [])),
        retrieved_at,
        "wikidata_action_api",
    )


def _fact_value_fields(snak: dict[str, Any], labels: dict[str, dict[str, str]]) -> FactValue | None:
    raw = (snak.get("datavalue") or {}).get("value")
    datatype = str(snak.get("datatype", ""))
    quantity = _quantity_fields(datatype, raw)
    if quantity is not None:
        return quantity
    entity_id = _entity_id(raw)
    if entity_id:
        return entity_id, None, "", labels.get(entity_id, {}).get("en", entity_id)
    time_value = _time_fields(raw)
    if time_value is not None:
        return "", None, "", time_value
    return _primitive_fields(raw)


def _quantity_fields(datatype: str, raw: Any) -> FactValue | None:
    if datatype != "quantity" or not isinstance(raw, dict):
        return None
    amount = raw.get("amount", 0)
    unit = str(raw.get("unit", "")).rsplit("/", 1)[-1] if raw.get("unit") else ""
    return "", float(amount), unit, str(raw.get("amount", ""))


def _entity_id(raw: Any) -> str:
    return str(raw.get("id", "")) if isinstance(raw, dict) else ""


def _time_fields(raw: Any) -> str | None:
    if isinstance(raw, dict) and "time" in raw:
        return str(raw["time"])
    return None


def _primitive_fields(raw: Any) -> FactValue | None:
    if isinstance(raw, (str, int, float)):
        return "", None, "", str(raw)
    return None


__all__ = ["FACT_PROPERTIES", "discover_wikivoyage_sitelinks", "normalize_facts"]
