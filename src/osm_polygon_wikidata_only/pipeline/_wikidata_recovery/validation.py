"""Referential-integrity and preservation checks for recovery writes."""

from __future__ import annotations

from collections.abc import Iterable
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
    polygon_qids = _validate_polygons(polygons)
    documents_by_article, document_ids = _validate_documents(documents)
    _validate_links(links, polygon_qids, documents_by_article)
    _validate_sections(sections, document_ids)
    _validate_facts(facts, polygon_qids)


def _validate_polygons(polygons: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    polygon_tags = _unique_mapping(polygons, "polygon_id", "wikidata", "polygon_id")
    polygon_qids = {
        polygon_id: qids_from_osm_tag(raw_tag) for polygon_id, raw_tag in polygon_tags.items()
    }
    invalid = _first_invalid_polygon_tag(polygon_tags.values())
    if invalid is not None:
        raise RecoveryRepairError(f"polygon contains invalid Wikidata identifier {invalid!r}")
    return polygon_qids


def _first_invalid_polygon_tag(tags: Iterable[str]) -> str | None:
    return next((raw_tag for raw_tag in tags if not qids_from_osm_tag(raw_tag)), None)


def _validate_documents(
    documents: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    documents_by_article = _unique_rows(documents, "article_id", "article_id")
    _unique_rows(documents, "document_id", "document_id")
    return documents_by_article, {str(row["document_id"]) for row in documents}


def _validate_links(
    links: list[dict[str, Any]],
    polygon_qids: dict[str, tuple[str, ...]],
    documents_by_article: dict[str, dict[str, Any]],
) -> None:
    link_ids: set[tuple[str, str]] = set()
    for link in links:
        _validate_link(link, polygon_qids, documents_by_article, link_ids)


def _validate_link(
    link: dict[str, Any],
    polygon_qids: dict[str, tuple[str, ...]],
    documents_by_article: dict[str, dict[str, Any]],
    link_ids: set[tuple[str, str]],
) -> None:
    polygon_id = str(link["polygon_id"])
    article_identifier = str(link["article_id"])
    identity = (polygon_id, article_identifier)
    _record_link_identity(identity, link_ids)
    _validate_link_reference(
        link, polygon_id, article_identifier, identity, polygon_qids, documents_by_article
    )


def _record_link_identity(identity: tuple[str, str], link_ids: set[tuple[str, str]]) -> None:
    if identity in link_ids:
        raise RecoveryRepairError(f"duplicate polygon-article identity {identity!r}")
    link_ids.add(identity)


def _validate_link_reference(
    link: dict[str, Any],
    polygon_id: str,
    article_identifier: str,
    identity: tuple[str, str],
    polygon_qids: dict[str, tuple[str, ...]],
    documents_by_article: dict[str, dict[str, Any]],
) -> None:
    if polygon_id not in polygon_qids:
        raise RecoveryRepairError(f"link references missing polygon {polygon_id!r}")
    document = documents_by_article.get(article_identifier)
    if document is None:
        raise RecoveryRepairError(f"link references missing document {article_identifier!r}")
    qid = str(link["wikidata"])
    if qid not in polygon_qids[polygon_id] or str(document["wikidata"]) != qid:
        raise RecoveryRepairError(f"link QID mismatch for {identity!r}")


def _validate_sections(sections: list[dict[str, Any]], document_ids: set[str]) -> None:
    _unique_rows(sections, "section_id", "section_id")
    for section in sections:
        _validate_section(section, document_ids)


def _validate_section(section: dict[str, Any], document_ids: set[str]) -> None:
    if str(section["document_id"]) not in document_ids:
        raise RecoveryRepairError(f"section references missing document {section['document_id']!r}")


def _validate_facts(facts: list[dict[str, Any]], polygon_qids: dict[str, tuple[str, ...]]) -> None:
    _unique_rows(facts, "fact_id", "fact_id")
    valid_qids = {qid for values in polygon_qids.values() for qid in values}
    for fact in facts:
        _validate_fact(fact, valid_qids)


def _validate_fact(fact: dict[str, Any], valid_qids: set[str]) -> None:
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
    _validate_healthy_polygon_preservation(
        original_polygons,
        updated_polygons,
        affected_qids=affected_qids,
    )
    for original, updated, key, label, removed_ids in (
        (original_documents, updated_documents, "document_id", "document", removed_document_ids),
        (original_sections, updated_sections, "section_id", "section", removed_section_ids),
        (original_facts, updated_facts, "fact_id", "fact", set()),
    ):
        _validate_rows_preserved(
            original,
            updated,
            key=key,
            label=label,
            removed_ids=removed_ids,
        )


def _validate_healthy_polygon_preservation(
    original_polygons: list[dict[str, Any]],
    updated_polygons: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> None:
    updated_polygon_map = {str(row["polygon_id"]): row for row in updated_polygons}
    for row in original_polygons:
        if set(qids_from_osm_tag(str(row["wikidata"]))) & affected_qids:
            continue
        if updated_polygon_map[str(row["polygon_id"])] != row:
            raise RecoveryRepairError(f"healthy polygon changed: {row['polygon_id']}")


def _validate_rows_preserved(
    original: list[dict[str, Any]],
    updated: list[dict[str, Any]],
    *,
    key: str,
    label: str,
    removed_ids: set[str],
) -> None:
    updated_map = {str(row[key]): row for row in updated}
    for row in original:
        identifier = str(row[key])
        if identifier in removed_ids:
            continue
        if updated_map.get(identifier) != row:
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
