"""Wikipedia link-row conversion and merge helpers for recovery."""

from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.domain.polygon_document_links import (
    validate_polygon_document_links,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag

from .models import RecoveryRepairError


def merge_links(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> list[dict[str, Any]]:
    """Add only absent polygon-article identities for affected QIDs."""
    identities = _link_identities(links)
    merged = [dict(row) for row in links]
    documents_by_qid = _documents_by_qid(documents)
    for polygon in polygons:
        _append_polygon_documents(
            merged,
            identities,
            polygon,
            documents_by_qid,
            affected_qids,
        )
    return merged


def _link_identities(links: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Validate and collect existing polygon-article identities."""
    identities: set[tuple[str, str]] = set()
    for row in links:
        identity = (str(row["polygon_id"]), str(row["article_id"]))
        if identity in identities:
            raise RecoveryRepairError(f"duplicate polygon-article identity {identity!r}")
        identities.add(identity)
    return identities


def _documents_by_qid(documents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group documents by QID with stable document ordering."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        grouped.setdefault(str(document["wikidata"]), []).append(document)
    for values in grouped.values():
        values.sort(key=lambda row: str(row["document_id"]))
    return grouped


def _append_polygon_documents(
    merged: list[dict[str, Any]],
    identities: set[tuple[str, str]],
    polygon: dict[str, Any],
    documents_by_qid: dict[str, list[dict[str, Any]]],
    affected_qids: set[str],
) -> None:
    """Append missing document links for one polygon's affected QIDs."""
    for qid in qids_from_osm_tag(str(polygon["wikidata"])):
        if qid not in affected_qids:
            continue
        for document in documents_by_qid.get(qid, []):
            identity = (str(polygon["polygon_id"]), str(document["article_id"]))
            if identity in identities:
                continue
            merged.append(
                {
                    "polygon_id": polygon["polygon_id"],
                    "article_id": document["article_id"],
                    "wikidata": qid,
                    "language": document["language"],
                    "source_pbf": polygon["source_pbf"],
                    "region": polygon["region"],
                    "osm_type": polygon["osm_type"],
                    "osm_id": polygon["osm_id"],
                    "page_id": document["page_id"],
                    "revision_id": document["revision_id"],
                    "is_best_language": False,
                }
            )
            identities.add(identity)


def canonical_wikipedia_links_to_legacy(
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    polygons: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> list[dict[str, Any]]:
    """Adapt canonical Wikipedia links to the recovery engine's legacy shape."""
    documents_by_id = {str(row["document_id"]): row for row in documents}
    best_by_polygon = {
        str(row["polygon_id"]): str(row.get("best_language") or "") for row in polygons
    }
    return _convert_canonical_links(
        _wikipedia_links(links),
        documents_by_id=documents_by_id,
        best_by_polygon=best_by_polygon,
        affected_qids=affected_qids,
    )


def _wikipedia_links(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select canonical rows for the Wikipedia project."""
    return [link for link in links if link["project"] == "wikipedia"]


def _convert_canonical_links(
    links: list[dict[str, Any]],
    *,
    documents_by_id: dict[str, dict[str, Any]],
    best_by_polygon: dict[str, str],
    affected_qids: set[str],
) -> list[dict[str, Any]]:
    """Convert selected canonical links while handling affected orphans."""
    legacy: list[dict[str, Any]] = []
    for link in links:
        converted = _legacy_link_from_canonical(
            link,
            documents_by_id=documents_by_id,
            best_by_polygon=best_by_polygon,
            affected_qids=affected_qids,
        )
        if converted is not None:
            legacy.append(converted)
    return legacy


def _legacy_link_from_canonical(
    link: dict[str, Any],
    *,
    documents_by_id: dict[str, dict[str, Any]],
    best_by_polygon: dict[str, str],
    affected_qids: set[str],
) -> dict[str, Any] | None:
    """Convert one canonical Wikipedia link or skip an affected orphan."""
    document_id = str(link["document_id"])
    document = documents_by_id.get(document_id)
    if document is None:
        if str(link["wikidata"]) in affected_qids:
            return None
        raise RecoveryRepairError(
            f"canonical Wikipedia link references missing document {document_id!r}"
        )
    return {
        "polygon_id": link["polygon_id"],
        "article_id": document["article_id"],
        "wikidata": link["wikidata"],
        "language": link["language"],
        "source_pbf": link["source_pbf"],
        "region": link["region"],
        "osm_type": link["osm_type"],
        "osm_id": link["osm_id"],
        "page_id": link["page_id"],
        "revision_id": link["revision_id"],
        "is_best_language": str(link["language"])
        == best_by_polygon.get(str(link["polygon_id"]), ""),
    }


def legacy_wikipedia_links_to_canonical(
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    preserved_wikivoyage_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore Wikipedia rows while preserving Wikivoyage rows exactly."""
    documents_by_article = {str(row["article_id"]): row for row in documents}
    canonical = [dict(row) for row in preserved_wikivoyage_links]
    for link in links:
        canonical.append(_canonical_link_row(link, documents_by_article))
    try:
        return validate_polygon_document_links(canonical)
    except (TypeError, ValueError) as error:
        raise RecoveryRepairError(
            f"Recovered canonical polygon-document links are invalid: {error}"
        ) from error


def _canonical_link_row(
    link: dict[str, Any], documents_by_article: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Convert one legacy link to its canonical document identity row."""
    article_identifier = str(link["article_id"])
    document = documents_by_article.get(article_identifier)
    if document is None:
        raise RecoveryRepairError(
            f"recovered link references missing article {article_identifier!r}"
        )
    return {
        "polygon_id": link["polygon_id"],
        "document_id": document["document_id"],
        "project": "wikipedia",
        "wikidata": link["wikidata"],
        "language": link["language"],
        "source_pbf": link["source_pbf"],
        "region": link["region"],
        "osm_type": link["osm_type"],
        "osm_id": link["osm_id"],
        "page_id": link["page_id"],
        "revision_id": link["revision_id"],
    }


__all__ = [
    "canonical_wikipedia_links_to_legacy",
    "legacy_wikipedia_links_to_canonical",
    "merge_links",
]
