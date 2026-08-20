"""Pure legacy-to-canonical polygon-document link conversion."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from osm_polygon_wikidata_only.domain.polygon_document_links import (
    validate_polygon_document_links,
)
from osm_polygon_wikidata_only.domain.schema import POLYGON_ARTICLE_COLUMNS
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag


def build_canonical_rows(
    stem: str,
    legacy_table: pa.Table,
    polygons_table: pa.Table,
    docs_table: pa.Table,
) -> list[dict[str, Any]]:
    """Convert distinct legacy links without inventing relationships."""
    polygons_by_id, docs_by_article_id, polygon_qids = _conversion_indexes(
        polygons_table, docs_table
    )
    canonical_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    legacy_by_identity: dict[tuple[str, str], dict[str, Any]] = {}

    for legacy_row in legacy_table.to_pylist():
        identity, normalized = _legacy_identity(legacy_row)
        if identity in legacy_by_identity:
            _validate_duplicate_legacy_row(legacy_by_identity[identity], normalized, identity)
            continue
        legacy_by_identity[identity] = normalized
        polygon_id, article_id = identity
        polygon = _polygon_for_legacy_row(polygons_by_id, polygon_id, stem)
        document = _document_for_legacy_row(docs_by_article_id, article_id, polygon_id, stem)
        document_qid = str(document["wikidata"])
        if document_qid and document_qid not in polygon_qids.get(polygon_id, set()):
            continue

        _validate_legacy_identity(legacy_row, document, polygon_id, article_id)
        canonical_by_identity[identity] = _canonical_row(
            polygon_id, polygon, document, document_qid
        )
    return validate_polygon_document_links(canonical_by_identity.values())


def _conversion_indexes(
    polygons_table: pa.Table,
    docs_table: pa.Table,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, set[str]],
]:
    """Build bounded lookup indexes for the conversion."""
    polygons_by_id = {str(row["polygon_id"]): row for row in polygons_table.to_pylist()}
    docs_by_article_id: dict[str, list[dict[str, Any]]] = {}
    for document in docs_table.to_pylist():
        docs_by_article_id.setdefault(str(document["article_id"]), []).append(document)
    polygon_qids = {
        polygon_id: set(qids_from_osm_tag(str(row.get("wikidata", ""))))
        for polygon_id, row in polygons_by_id.items()
    }
    return polygons_by_id, docs_by_article_id, polygon_qids


def _legacy_identity(
    legacy_row: dict[str, Any],
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Return the stable identity and normalized legacy values."""
    identity = (str(legacy_row.get("polygon_id", "")), str(legacy_row.get("article_id", "")))
    normalized = {column: legacy_row.get(column) for column in POLYGON_ARTICLE_COLUMNS}
    return identity, normalized


def _validate_duplicate_legacy_row(
    original: dict[str, Any], normalized: dict[str, Any], identity: tuple[str, str]
) -> None:
    """Allow identical duplicate legacy rows but reject conflicting values."""
    if original != normalized:
        raise ValueError(
            f"conflicting duplicate legacy rows for (polygon_id={identity[0]!r}, "
            f"article_id={identity[1]!r}); cannot collapse"
        )


def _polygon_for_legacy_row(
    polygons_by_id: dict[str, dict[str, Any]], polygon_id: str, stem: str
) -> dict[str, Any]:
    """Resolve the polygon source row for one legacy link."""
    polygon = polygons_by_id.get(polygon_id)
    if polygon is None:
        raise ValueError(
            f"legacy polygon_id={polygon_id!r} is not present in polygons/{stem}.parquet"
        )
    return polygon


def _document_for_legacy_row(
    docs_by_article_id: dict[str, list[dict[str, Any]]],
    article_id: str,
    polygon_id: str,
    stem: str,
) -> dict[str, Any]:
    """Resolve the unique document source row for one legacy link."""
    matching_documents = docs_by_article_id.get(article_id)
    if not matching_documents:
        raise ValueError(
            f"legacy article_id={article_id!r} for polygon_id={polygon_id!r} "
            f"has no matching wikipedia/documents/{stem}.parquet row"
        )
    if len(matching_documents) > 1:
        raise ValueError(
            f"legacy article_id={article_id!r} for polygon_id={polygon_id!r} "
            f"is ambiguous: {len(matching_documents)} matching documents"
        )
    return matching_documents[0]


def _canonical_row(
    polygon_id: str,
    polygon: dict[str, Any],
    document: dict[str, Any],
    document_qid: str,
) -> dict[str, Any]:
    """Build one canonical polygon-document row."""
    return {
        "polygon_id": polygon_id,
        "document_id": str(document["document_id"]),
        "project": "wikipedia",
        "wikidata": document_qid,
        "language": str(document.get("language", "")),
        "source_pbf": str(polygon.get("source_pbf", "")),
        "region": str(polygon.get("region", "")),
        "osm_type": str(polygon.get("osm_type", "")),
        "osm_id": int(polygon.get("osm_id", 0)),
        "page_id": int(document.get("page_id", 0)),
        "revision_id": int(document.get("revision_id", 0)),
    }


def _validate_legacy_identity(
    legacy: dict[str, Any],
    document: dict[str, Any],
    polygon_id: str,
    article_id: str,
) -> None:
    comparisons = (
        ("wikidata", str, ""),
        ("page_id", int, 0),
        ("revision_id", int, 0),
        ("language", str, ""),
    )
    for field, coerce, empty in comparisons:
        legacy_value = coerce(legacy.get(field, empty))
        document_value = coerce(document.get(field, empty))
        if legacy_value and legacy_value != document_value:
            raise ValueError(
                f"legacy row (polygon_id={polygon_id!r}, article_id={article_id!r}) "
                f"{field}={legacy_value!r} conflicts with document {field}={document_value!r}"
            )


__all__ = ["build_canonical_rows"]
