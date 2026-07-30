"""Referential-integrity and preservation checks for recovery writes."""

from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag

from .models import RecoveryRepairError


def validate_existing_rows(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    """Validate all existing primary keys and cross-table references."""
    polygon_tags = _unique_mapping(polygons, "polygon_id", "wikidata", "polygon_id")
    polygon_qids = {
        polygon_id: qids_from_osm_tag(raw_tag) for polygon_id, raw_tag in polygon_tags.items()
    }
    invalid = next(
        (raw_tag for raw_tag in polygon_tags.values() if not qids_from_osm_tag(raw_tag)),
        None,
    )
    if invalid is not None:
        raise RecoveryRepairError(f"polygon contains invalid Wikidata identifier {invalid!r}")
    documents_by_article = _unique_rows(documents, "article_id", "article_id")
    _unique_rows(documents, "document_id", "document_id")
    document_ids = {str(row["document_id"]) for row in documents}
    _unique_rows(sections, "section_id", "section_id")
    _unique_rows(facts, "fact_id", "fact_id")

    link_ids: set[tuple[str, str]] = set()
    for link in links:
        polygon_id = str(link["polygon_id"])
        article_identifier = str(link["article_id"])
        identity = (polygon_id, article_identifier)
        if identity in link_ids:
            raise RecoveryRepairError(f"duplicate polygon-article identity {identity!r}")
        link_ids.add(identity)
        if polygon_id not in polygon_qids:
            raise RecoveryRepairError(f"link references missing polygon {polygon_id!r}")
        document = documents_by_article.get(article_identifier)
        if document is None:
            raise RecoveryRepairError(f"link references missing document {article_identifier!r}")
        qid = str(link["wikidata"])
        if qid not in polygon_qids[polygon_id] or str(document["wikidata"]) != qid:
            raise RecoveryRepairError(f"link QID mismatch for {identity!r}")
    for section in sections:
        if str(section["document_id"]) not in document_ids:
            raise RecoveryRepairError(
                f"section references missing document {section['document_id']!r}"
            )
    valid_qids = {qid for values in polygon_qids.values() for qid in values}
    for fact in facts:
        if str(fact["wikidata"]) not in valid_qids:
            raise RecoveryRepairError(f"fact references absent QID {fact['wikidata']!r}")


def validate_preservation(
    original_polygons: list[dict[str, Any]],
    updated_polygons: list[dict[str, Any]],
    original_documents: list[dict[str, Any]],
    updated_documents: list[dict[str, Any]],
    original_sections: list[dict[str, Any]],
    updated_sections: list[dict[str, Any]],
    original_facts: list[dict[str, Any]],
    updated_facts: list[dict[str, Any]],
    *,
    affected_qids: set[str],
    removed_document_ids: set[str],
    removed_section_ids: set[str],
) -> None:
    """Prove healthy rows remain byte-equivalent at the row level."""
    updated_polygon_map = {str(row["polygon_id"]): row for row in updated_polygons}
    for row in original_polygons:
        if (
            not (set(qids_from_osm_tag(str(row["wikidata"]))) & affected_qids)
            and updated_polygon_map[str(row["polygon_id"])] != row
        ):
            raise RecoveryRepairError(f"healthy polygon changed: {row['polygon_id']}")
    for original, updated, key, label in (
        (original_documents, updated_documents, "document_id", "document"),
        (original_sections, updated_sections, "section_id", "section"),
        (original_facts, updated_facts, "fact_id", "fact"),
    ):
        updated_map = {str(row[key]): row for row in updated}
        for row in original:
            if label == "document" and str(row[key]) in removed_document_ids:
                continue
            if label == "section" and str(row[key]) in removed_section_ids:
                continue
            if updated_map.get(str(row[key])) != row:
                raise RecoveryRepairError(f"existing {label} changed: {row[key]}")


def _unique_mapping(
    rows: list[dict[str, Any]],
    key: str,
    value: str,
    label: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        identifier = str(row[key])
        if identifier in result:
            raise RecoveryRepairError(f"duplicate {label} {identifier!r}")
        result[identifier] = str(row[value])
    return result


def _unique_rows(
    rows: list[dict[str, Any]],
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row[key])
        if identifier in result:
            raise RecoveryRepairError(f"duplicate {label} {identifier!r}")
        result[identifier] = row
    return result


__all__ = ["validate_existing_rows", "validate_preservation"]
