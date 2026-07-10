"""Pure parsing helpers for MediaWiki article responses."""

from __future__ import annotations

import urllib.parse
from typing import Any

from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    count_words,
    estimate_tokens,
    html_to_plain_text,
)
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import FetchResult, WikipediaArticle


def revision_id_from_query(data: dict[str, Any]) -> int:
    """Return the first revision identifier, or zero for malformed data."""
    pages = (data.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict) or not pages:
        return 0
    page = next(iter(pages.values()))
    if not isinstance(page, dict):
        return 0
    revisions = page.get("revisions") or []
    if not isinstance(revisions, list) or not revisions or not isinstance(revisions[0], dict):
        return 0
    return int(revisions[0].get("revid", 0))


def plain_text_from_parse_response(data: dict[str, Any]) -> str:
    """Extract plain text from an Action API ``parse`` response."""
    parsed = data.get("parse") or {}
    if not isinstance(parsed, dict):
        return ""
    text = parsed.get("text", "")
    if isinstance(text, dict):
        text = text.get("*", "")
    return html_to_plain_text(text) if isinstance(text, str) else ""


def query_with_extract(data: dict[str, Any], extract: str) -> dict[str, Any]:
    """Copy a query response with the first page's extract replaced."""
    query = data.get("query") or {}
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, dict) or not pages:
        return data
    key, raw_page = next(iter(pages.items()))
    if not isinstance(raw_page, dict):
        return data
    page = dict(raw_page)
    page["extract"] = extract
    return {"query": {"pages": {key: page}}}


def parse_wikipedia_batch_response(
    language: str,
    site: str,
    requested: list[str],
    data: dict[str, Any],
    *,
    fetch_full_text: bool,
) -> dict[str, FetchResult]:
    """Map an Action API multi-page response back to requested titles."""
    query = data.get("query")
    if not isinstance(query, dict):
        raise ValueError("missing query in batch response")
    raw_pages = query.get("pages")
    if not isinstance(raw_pages, dict):
        raise ValueError("missing query.pages in batch response")
    pages_by_title = {
        str(page.get("title")): page
        for page in raw_pages.values()
        if isinstance(page, dict) and page.get("title")
    }
    aliases: dict[str, str] = {}
    for key in ("normalized", "redirects"):
        entries = query.get(key, [])
        if isinstance(entries, list):
            for entry in entries:
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("from"), str)
                    and isinstance(entry.get("to"), str)
                ):
                    aliases[entry["from"]] = entry["to"]

    results: dict[str, FetchResult] = {}
    for title in requested:
        resolved = title
        seen: set[str] = set()
        while resolved in aliases and resolved not in seen:
            seen.add(resolved)
            resolved = aliases[resolved]
        page = pages_by_title.get(resolved)
        results[title] = (
            FetchResult("article_not_found", None, "page missing")
            if page is None
            else parse_wikipedia_response(
                language,
                site,
                title,
                {"query": {"pages": {"0": page}}},
                fetch_full_text=fetch_full_text,
            )
        )
    return results


def parse_wikipedia_response(
    language: str,
    site: str,
    title: str,
    data: dict[str, Any],
    *,
    wikidata_label: str = "",
    wikidata_description: str = "",
    fetch_full_text: bool = True,
) -> FetchResult:
    """Parse an Action API query response into a fetch result."""
    del wikidata_label, wikidata_description, fetch_full_text
    try:
        pages = (data.get("query") or {}).get("pages") or {}
    except (AttributeError, TypeError):
        return FetchResult("parse_error", None, "missing query.pages")
    if not pages:
        return FetchResult("article_not_found", None, "no pages in response")
    page = next(iter(pages.values()))
    if page.get("missing") is not None or "pageid" not in page:
        return FetchResult("article_not_found", None, "page missing")
    revisions = page.get("revisions") or []
    if not revisions:
        return FetchResult("parse_error", None, "no revisions")
    revision = revisions[0]
    revision_id = int(revision.get("revid", 0))
    extract = page.get("extract", "") or ""
    full_text = clean_article_text(extract)
    lead_text = clean_article_text(extract.strip().split("\n\n", 1)[0])[:500] if extract else ""
    canonical_title = page.get("title", title)
    url = page.get("fullurl") or (
        f"https://{language}.wikipedia.org/wiki/"
        f"{urllib.parse.quote(canonical_title.replace(' ', '_'))}"
    )
    thumbnail = page.get("thumbnail") or {}
    attribution = (
        f'Text from Wikipedia article "{canonical_title}" ({language}.wikipedia.org); '
        f"contributors; revision {revision_id}; accessed {utc_now_iso()}; "
        "licensed under CC BY-SA."
    )
    article = WikipediaArticle(
        language=language,
        site=site,
        title=canonical_title,
        page_id=int(page.get("pageid", 0)),
        revision_id=revision_id,
        revision_timestamp=revision.get("timestamp", ""),
        url=url,
        lead_text=lead_text,
        extract=clean_article_text(extract),
        full_text=full_text,
        full_text_format="plain_text",
        thumbnail_url=thumbnail.get("source", ""),
        thumbnail_width=thumbnail.get("width"),
        thumbnail_height=thumbnail.get("height"),
        categories=[],
        license="CC BY-SA 4.0",
        attribution=attribution,
        source_api="mediawiki_action_api",
        retrieved_at=utc_now_iso(),
    )
    if not full_text:
        return FetchResult("empty_text", article, "no extract returned by API")
    _ = (count_words(full_text), estimate_tokens(full_text))
    return FetchResult("ok", article, "")


__all__ = ["parse_wikipedia_response"]
