"""Pure discovery and fact-normalization helpers for augmentation."""

from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.utils.json import dumps
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import WikidataFact, stable_id

FACT_PROPERTIES = frozenset({"P17", "P31", "P131", "P279", "P361", "P571", "P1435", "P2044"})


def discover_wikivoyage_sitelinks(entity: dict[str, Any]) -> list[tuple[str, str, str]]:
    links = []
    for site, value in (entity.get("sitelinks") or {}).items():
        if site.endswith("wikivoyage") and isinstance(value, dict) and value.get("title"):
            links.append((site[: -len("wikivoyage")].replace("_", "-"), site, str(value["title"])))
    return sorted(links)


def normalize_facts(
    entity: dict[str, Any], labels: dict[str, dict[str, str]]
) -> list[WikidataFact]:
    qid = str(entity.get("id", ""))
    out: list[WikidataFact] = []
    now = utc_now_iso()
    for property_id in sorted(entity.get("claims") or {}):
        if property_id not in FACT_PROPERTIES:
            continue
        for ordinal, claim in enumerate(entity["claims"][property_id]):
            snak = claim.get("mainsnak", {})
            if snak.get("snaktype") != "value":
                continue
            raw = (snak.get("datavalue") or {}).get("value")
            datatype = str(snak.get("datatype", ""))
            entity_id = str(raw.get("id", "")) if isinstance(raw, dict) else ""
            numeric: float | None = None
            unit = ""
            value_text = ""
            if datatype == "quantity" and isinstance(raw, dict):
                numeric = float(raw.get("amount", 0))
                unit = str(raw.get("unit", "")).rsplit("/", 1)[-1] if raw.get("unit") else ""
                value_text = str(raw.get("amount", ""))
            elif entity_id:
                value_text = labels.get(entity_id, {}).get("en", entity_id)
            elif isinstance(raw, dict) and "time" in raw:
                value_text = str(raw["time"])
            elif isinstance(raw, (str, int, float)):
                value_text = str(raw)
            else:
                continue
            property_labels = labels.get(property_id, {})
            value_labels = labels.get(entity_id, {}) if entity_id else {}
            out.append(
                WikidataFact(
                    stable_id(qid, property_id, entity_id or value_text, ordinal),
                    qid,
                    property_id,
                    property_labels.get("en", property_id),
                    dumps(property_labels),
                    datatype,
                    entity_id,
                    value_labels.get("en", entity_id),
                    dumps(value_labels),
                    value_text,
                    numeric,
                    unit,
                    str(claim.get("rank", "normal")),
                    dumps(claim.get("qualifiers", {})),
                    dumps(claim.get("references", [])),
                    now,
                    "wikidata_action_api",
                )
            )
    return out


__all__ = ["FACT_PROPERTIES", "discover_wikivoyage_sitelinks", "normalize_facts"]
