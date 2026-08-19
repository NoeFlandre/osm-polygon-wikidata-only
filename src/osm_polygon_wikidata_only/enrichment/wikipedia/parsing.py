"""Pure parsing helpers for MediaWiki article responses."""

from __future__ import annotations

import urllib.parse
from typing import Any, TypeGuard

from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    html_to_plain_text,
)
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import FetchResult, WikipediaArticle

_DEFAULT_WIKIDATA_LABEL = ""
_DEFAULT_WIKIDATA_DESCRIPTION = ""
_DEFAULT_FETCH_FULL_TEXT = True


def revision_id_from_query(data: dict[str, Any]) -> int:
    """Return the first revision identifier, or zero for malformed data."""
    page = _first_query_page(data)
    if page is None:
        return 0
    revision = _first_revision(page)
    if revision is None:
        return 0
    revision_id = revision.get("revid")
    return int(revision_id) if revision_id is not None else 0


def _first_query_page(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first page from a query response, if its shape is valid."""
    query = data.get("query")
    if not isinstance(query, dict):
        return None
    pages = query.get("pages")
    if not isinstance(pages, dict) or not pages:
        return None
    page = next(iter(pages.values()))
    return page if _is_page(page) else None


def _is_page(value: object) -> TypeGuard[dict[str, Any]]:
    """Identify a dictionary with the page shape used by the parser."""
    return isinstance(value, dict)


def _first_revision(page: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first revision from a page, if its shape is valid."""
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        return None
    revision = revisions[0]
    return revision if isinstance(revision, dict) else None


def plain_text_from_parse_response(data: dict[str, Any]) -> str:
    """Extract plain text from an Action API ``parse`` response."""
    parsed = data.get("parse")
    if not isinstance(parsed, dict):
        return ""
    if "text" not in parsed:
        return ""
    return _plain_text_value(parsed["text"])


def _plain_text_value(value: object) -> str:
    """Convert one Action API text value to cleaned plain text."""
    text = value
    if isinstance(text, dict):
        text = text.get("*")
    return html_to_plain_text(text) if isinstance(text, str) else ""


def query_with_extract(data: dict[str, Any], extract: str) -> dict[str, Any]:
    """Copy a query response with the first page's extract replaced."""
    first_page = _first_query_page_entry(data)
    if first_page is None:
        return data
    key, raw_page = first_page
    page = dict(raw_page)
    page["extract"] = extract
    return {"query": {"pages": {key: page}}}


def _first_query_page_entry(data: dict[str, Any]) -> tuple[Any, dict[str, Any]] | None:
    """Return the first query page together with its response key."""
    query = data.get("query")
    if not isinstance(query, dict):
        return None
    pages = query.get("pages")
    if not isinstance(pages, dict) or not pages:
        return None
    key, page = next(iter(pages.items()))
    return (key, page) if _is_page(page) else None


def parse_wikipedia_batch_response(
    language: str,
    site: str,
    requested: list[str],
    data: dict[str, Any],
    *,
    fetch_full_text: bool,
) -> dict[str, FetchResult]:
    """Map an Action API multi-page response back to requested titles."""
    query = _batch_query(data)
    pages_by_title = _batch_pages_by_title(query)
    aliases = _batch_aliases(query)
    return {
        title: _batch_result(language, site, title, pages_by_title, aliases) for title in requested
    }


def _batch_query(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the query container from a batch response."""
    query = data.get("query")
    if not isinstance(query, dict):
        raise ValueError("missing query in batch response")
    if not isinstance(query.get("pages"), dict):
        raise ValueError("missing query.pages in batch response")
    return query


def _batch_pages_by_title(query: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index valid batch pages by canonical title or response key."""
    pages = query["pages"]
    return {
        str(page.get("title") or key): page for key, page in pages.items() if isinstance(page, dict)
    }


def _batch_aliases(query: dict[str, Any]) -> dict[str, str]:
    """Collect valid normalized-title and redirect aliases."""
    aliases: dict[str, str] = {}
    for key in ("normalized", "redirects"):
        entries = query.get(key)
        if isinstance(entries, list):
            aliases.update(_valid_aliases(entries))
    return aliases


def _valid_aliases(entries: list[Any]) -> dict[str, str]:
    """Return aliases with string endpoints from one API field."""
    return {
        entry["from"]: entry["to"]
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("from"), str)
        and isinstance(entry.get("to"), str)
    }


def _batch_result(
    language: str,
    site: str,
    title: str,
    pages_by_title: dict[str, dict[str, Any]],
    aliases: dict[str, str],
) -> FetchResult:
    """Parse one requested title after alias resolution."""
    page = pages_by_title.get(_resolve_alias(title, aliases))
    if page is None:
        return FetchResult("article_not_found", None, "page missing")
    return _parse_wikipedia_page(language, site, title, page)


def _resolve_alias(title: str, aliases: dict[str, str]) -> str:
    """Follow aliases until a title is stable or a cycle is detected."""
    resolved = title
    seen: set[str] = set()
    for _ in aliases:
        if resolved not in aliases or resolved in seen:
            break
        seen.add(resolved)
        resolved = aliases[resolved]
    return resolved


def parse_wikipedia_response(
    language: str,
    site: str,
    title: str,
    data: dict[str, Any],
    *,
    wikidata_label: str = _DEFAULT_WIKIDATA_LABEL,
    wikidata_description: str = _DEFAULT_WIKIDATA_DESCRIPTION,
    fetch_full_text: bool = _DEFAULT_FETCH_FULL_TEXT,
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
    return _parse_wikipedia_page(language, site, title, page)


def _parse_wikipedia_page(
    language: str,
    site: str,
    title: str,
    page: dict[str, Any],
) -> FetchResult:
    """Parse one already-selected MediaWiki page without another query walk."""
    error = _page_shape_error(page)
    if error is not None:
        return error
    revision = page["revisions"][0]
    extract = page.get("extract") or ""
    full_text = clean_article_text(extract)
    article = _build_article(language, site, title, page, _revision_id(revision), full_text)
    if not full_text:
        return FetchResult("empty_text", article, "no extract returned by API")
    return FetchResult.ok(article)


def _page_shape_error(page: dict[str, Any]) -> FetchResult | None:
    """Return a terminal result for an invalid page shape, if any."""
    if page.get("missing") is not None or "pageid" not in page:
        return FetchResult("article_not_found", None, "page missing")
    if not page.get("revisions"):
        return FetchResult("parse_error", None, "no revisions")
    return None


def _revision_id(revision: dict[str, Any]) -> int:
    """Return a revision identifier, defaulting malformed omission to zero."""
    value = revision.get("revid")
    return int(value) if value is not None else 0


def _lead_text(extract: str) -> str:
    """Return the cleaned, bounded first paragraph of an extract."""
    if not extract:
        return ""
    lead_source = extract.strip().partition("\n\n")[0]
    return clean_article_text(lead_source)[:500]


def _article_url(language: str, title: str, page: dict[str, Any]) -> str:
    """Return the API URL or a deterministic Wikipedia fallback URL."""
    return page.get("fullurl") or (
        f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    )


def _build_article(
    language: str,
    site: str,
    requested_title: str,
    page: dict[str, Any],
    revision_id: int,
    full_text: str,
) -> WikipediaArticle:
    """Build the stable article value object from one selected page."""
    title = page.get("title", requested_title)
    revision = page["revisions"][0]
    extract = page.get("extract") or ""
    thumbnail = page.get("thumbnail") or {}
    attribution = (
        f'Text from Wikipedia article "{title}" ({language}.wikipedia.org); '
        f"contributors; revision {revision_id}; accessed {utc_now_iso()}; "
        "licensed under CC BY-SA."
    )
    return WikipediaArticle(
        language=language,
        site=site,
        title=title,
        page_id=int(page["pageid"]),
        revision_id=revision_id,
        revision_timestamp=revision.get("timestamp", ""),
        url=_article_url(language, title, page),
        lead_text=_lead_text(extract),
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


__all__ = ["parse_wikipedia_response"]
