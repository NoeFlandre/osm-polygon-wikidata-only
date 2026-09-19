"""Recompute derived polygon and language fields after recovery."""

from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.enrichment.article_linker import PREFERRED_LANGUAGES
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.utils.json import dumps


def _summarize_polygon_links(
    polygon_links: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return unique article IDs and languages in deterministic order."""
    return (
        sorted({str(link["article_id"]) for link in polygon_links}),
        sorted({str(link["language"]) for link in polygon_links}),
    )


def _preferred_language(languages: list[str]) -> str:
    """Choose the configured preferred language, falling back to sorted input."""
    best = next((language for language in PREFERRED_LANGUAGES if language in languages), "")
    if not best and languages:
        return languages[0]
    return best


def _has_article_text(
    article_ids: list[str],
    documents_by_article: dict[str, dict[str, Any]],
) -> bool:
    """Return whether at least one linked document has non-empty full text."""
    return any(
        bool(str(documents_by_article[article]["full_text"]).strip()) for article in article_ids
    )


def _recompute_polygon_row(
    original: dict[str, Any],
    polygon_links: list[dict[str, Any]],
    documents_by_article: dict[str, dict[str, Any]],
    affected_qids: set[str],
) -> tuple[dict[str, Any], str]:
    """Recompute one affected polygon's derived Wikipedia fields."""
    row = dict(original)
    if not set(qids_from_osm_tag(str(row["wikidata"]))) & affected_qids:
        return row, ""
    article_ids, languages = _summarize_polygon_links(polygon_links)
    best = _preferred_language(languages)
    row.update(
        {
            "has_wikipedia": bool(article_ids),
            "wikipedia_language_count": len(languages),
            "wikipedia_languages": dumps(languages),
            "wikipedia_article_count": len(article_ids),
            "has_english_wikipedia": "en" in languages,
            "has_french_wikipedia": "fr" in languages,
            "text_available": _has_article_text(article_ids, documents_by_article),
            "best_language": best,
        }
    )
    return row, best


def _recompute_polygon_rows(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Recompute derived fields for all affected polygons."""
    documents_by_article = {str(row["article_id"]): row for row in documents}
    links_by_polygon: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_polygon.setdefault(str(link["polygon_id"]), []).append(link)
    updated: list[dict[str, Any]] = []
    best_by_polygon: dict[str, str] = {}
    for original in polygons:
        polygon_id = str(original["polygon_id"])
        row, best = _recompute_polygon_row(
            original,
            links_by_polygon.get(polygon_id, []),
            documents_by_article,
            affected_qids,
        )
        updated.append(row)
        if best:
            best_by_polygon[polygon_id] = best
    return updated, best_by_polygon


def _apply_best_language_links(
    links: list[dict[str, Any]],
    best_by_polygon: dict[str, str],
) -> list[dict[str, Any]]:
    """Apply the recomputed best-language flag to Wikipedia links."""
    updated: list[dict[str, Any]] = []
    for original in links:
        row = dict(original)
        polygon_id = str(row["polygon_id"])
        if polygon_id in best_by_polygon:
            row["is_best_language"] = str(row["language"]) == best_by_polygon[polygon_id]
        updated.append(row)
    return updated


def _recompute_affected_polygon_fields(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated_polygons, best_by_polygon = _recompute_polygon_rows(
        polygons, links, documents, affected_qids
    )
    return updated_polygons, _apply_best_language_links(links, best_by_polygon)


__all__ = [
    "_apply_best_language_links",
    "_has_article_text",
    "_preferred_language",
    "_recompute_affected_polygon_fields",
    "_recompute_polygon_row",
    "_recompute_polygon_rows",
    "_summarize_polygon_links",
]
