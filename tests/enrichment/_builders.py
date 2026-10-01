"""Shared constructible fixtures for enrichment-driven pipeline tests."""

from __future__ import annotations

from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaArticle


def wikipedia_article(language: str, body: str) -> WikipediaArticle:
    """Build a small article value with stable attribution and revision metadata."""
    return WikipediaArticle(
        language=language,
        site=f"{language}wiki",
        title="X",
        page_id=10,
        revision_id=100,
        revision_timestamp="2026-01-01T00:00:00Z",
        url=f"https://{language}.wikipedia.org/wiki/X",
        lead_text=body,
        extract=body,
        full_text=body,
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="CC BY-SA 4.0",
        attribution="Wikipedia",
        source_api="mediawiki_action_api",
        retrieved_at="2026-01-01T00:00:00Z",
    )
