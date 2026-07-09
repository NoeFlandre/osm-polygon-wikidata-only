"""Wikipedia article fetching via the MediaWiki Action API.

Layered the same way as :mod:`wikidata_client`:

* :class:`WikipediaClient` — abstract interface.
* :class:`HttpWikipediaClient` — concrete client using
  ``urllib.request`` (no extra HTTP dependency).
* :class:`InMemoryWikipediaClient` — test double.
* :class:`CachedWikipediaClient` — wraps another client and adds the
  local cache.

The Action API endpoint used is
``https://{lang}.wikipedia.org/w/api.php?action=query&prop=...`` and we
combine:

* ``revisions`` — page id, revision id, revision timestamp
* ``extracts`` — plain-text lead + full article text
* ``pageimages`` — thumbnail metadata
* ``categories`` — list of categories
* ``info`` — canonical URL hint
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from osm_polygon_wikidata_only.config.settings import MEDIAWIKI_API_URL_TEMPLATE, Settings
from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    count_words,
    estimate_tokens,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.rate_limit import sleep_after_429, wait_for_host
from osm_polygon_wikidata_only.utils.retry import with_retries
from osm_polygon_wikidata_only.utils.time import utc_now_iso

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WikipediaArticle:
    """One Wikipedia article fetched for a given language + page."""

    language: str
    site: str  # e.g. "enwiki"
    title: str
    page_id: int
    revision_id: int
    revision_timestamp: str
    url: str
    lead_text: str
    extract: str
    full_text: str
    full_text_format: str
    thumbnail_url: str
    thumbnail_width: int | None
    thumbnail_height: int | None
    categories: list[str]
    license: str
    attribution: str
    source_api: str
    retrieved_at: str


@dataclass(frozen=True)
class FetchResult:
    """Result of one article fetch: the article plus a status string.

    ``status`` is one of the documented :data:`FETCH_STATUSES` from
    :mod:`domain.schema`. On failure, ``article`` is ``None``.
    """

    status: str
    article: WikipediaArticle | None
    error: str = ""


class WikipediaClient(ABC):
    """Abstract interface for fetching a single article."""

    @abstractmethod
    def fetch_article(
        self,
        language: str,
        site: str,
        title: str,
        *,
        wikidata_label: str = "",
        wikidata_description: str = "",
        wikidata_aliases: list[str] | None = None,
        fetch_full_text: bool = True,
    ) -> FetchResult: ...


class InMemoryWikipediaClient(WikipediaClient):
    """Test double: returns canned responses keyed by ``(site, title)``."""

    def __init__(self, responses: dict[tuple[str, str], FetchResult]) -> None:
        self._responses = dict(responses)

    def fetch_article(
        self,
        language: str,
        site: str,
        title: str,
        *,
        wikidata_label: str = "",
        wikidata_description: str = "",
        wikidata_aliases: list[str] | None = None,
        fetch_full_text: bool = True,
    ) -> FetchResult:
        return self._responses.get((site, title), FetchResult("article_not_found", None))


class HttpWikipediaClient(WikipediaClient):
    """Real Wikipedia client using the MediaWiki Action API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def fetch_article(
        self,
        language: str,
        site: str,
        title: str,
        *,
        wikidata_label: str = "",
        wikidata_description: str = "",
        wikidata_aliases: list[str] | None = None,
        fetch_full_text: bool = True,
    ) -> FetchResult:
        url = self._build_url(language, title, fetch_full_text=fetch_full_text)
        try:
            data = with_retries(
                lambda: self._http_get(url),
                attempts=self._settings.request_max_retries,
                base_delay=self._settings.request_base_delay_s,
                retry_on=(urllib.error.URLError, TimeoutError, OSError),
            )
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return FetchResult("article_not_found", None, str(e))
            if e.code == 429:
                return FetchResult("rate_limited", None, str(e))
            return FetchResult("http_error", None, f"HTTP {e.code}: {e}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return FetchResult("http_error", None, str(e))
        return parse_wikipedia_response(
            language,
            site,
            title,
            data,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            fetch_full_text=fetch_full_text,
        )

    def _build_url(self, language: str, title: str, *, fetch_full_text: bool) -> str:
        endpoint = MEDIAWIKI_API_URL_TEMPLATE.format(lang=language)
        params: dict[str, str] = {
            "action": "query",
            "format": "json",
            "formatversion": "1",
            "prop": "revisions|extracts|pageimages|info",
            "titles": title,
            "explaintext": "1",
            "exsectionformat": "plain",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "redirects": "1",
        }
        if not fetch_full_text:
            # ``exintro`` makes the API return only the lead section.
            params["exintro"] = "1"
        return f"{endpoint}?{urllib.parse.urlencode(params)}"

    def _http_get(self, url: str) -> dict[str, Any]:
        host = urllib.parse.urlparse(url).netloc
        wait_for_host(host, min_interval_s=self._settings.wikipedia_min_interval_s)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self._settings.user_agent, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._settings.request_timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                sleep_after_429(e, default_s=self._settings.rate_limit_retry_after_default_s)
            raise
        parsed: object = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
        return parsed


class CachedWikipediaClient(WikipediaClient):
    """Wrap another client and cache successful + failed fetches."""

    def __init__(
        self,
        inner: WikipediaClient,
        cache: JsonFileCache,
        *,
        failed_ttl_s: int = 60 * 60,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._failed_ttl_s = failed_ttl_s

    def fetch_article(
        self,
        language: str,
        site: str,
        title: str,
        *,
        wikidata_label: str = "",
        wikidata_description: str = "",
        wikidata_aliases: list[str] | None = None,
        fetch_full_text: bool = True,
    ) -> FetchResult:
        key = f"wikipedia/{site}/{self._safe_title(title)}.json"
        hit = self._cache.get(key)
        if hit is not None:
            if hit.status == "ok" and isinstance(hit.parsed_result, dict):
                return FetchResult("ok", _article_from_dict(hit.parsed_result))
            return FetchResult(
                hit.parsed_result if isinstance(hit.parsed_result, str) else "http_error", None
            )
        result = self._inner.fetch_article(
            language,
            site,
            title,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            wikidata_aliases=wikidata_aliases,
            fetch_full_text=fetch_full_text,
        )
        if result.status == "ok" and result.article is not None:
            self._cache.set(
                key,
                payload=_article_to_dict(result.article),
                request_url=self._endpoint_for(language, title, fetch_full_text),
                response_metadata={"language": language, "site": site, "title": title},
                status="ok",
            )
        else:
            self._cache.set(
                key,
                payload=result.status,
                request_url=self._endpoint_for(language, title, fetch_full_text),
                response_metadata={"status": result.status, "error": result.error},
                status="error",
                ttl_s=self._failed_ttl_s,
            )
        return result

    @staticmethod
    def _safe_title(title: str) -> str:
        # Cache file system can't have slashes; keep them encoded.
        return title.replace("/", "_").replace(" ", "_")

    def _endpoint_for(self, language: str, title: str, fetch_full_text: bool) -> str:
        if isinstance(self._inner, HttpWikipediaClient):
            return self._inner._build_url(language, title, fetch_full_text=fetch_full_text)
        return ""


def _article_to_dict(a: WikipediaArticle) -> dict[str, Any]:
    return {
        "language": a.language,
        "site": a.site,
        "title": a.title,
        "page_id": a.page_id,
        "revision_id": a.revision_id,
        "revision_timestamp": a.revision_timestamp,
        "url": a.url,
        "lead_text": a.lead_text,
        "extract": a.extract,
        "full_text": a.full_text,
        "full_text_format": a.full_text_format,
        "thumbnail_url": a.thumbnail_url,
        "thumbnail_width": a.thumbnail_width,
        "thumbnail_height": a.thumbnail_height,
        "categories": list(a.categories),
        "license": a.license,
        "attribution": a.attribution,
        "source_api": a.source_api,
        "retrieved_at": a.retrieved_at,
    }


def _article_from_dict(d: dict[str, Any]) -> WikipediaArticle:
    return WikipediaArticle(
        language=d["language"],
        site=d["site"],
        title=d["title"],
        page_id=int(d.get("page_id", 0)),
        revision_id=int(d.get("revision_id", 0)),
        revision_timestamp=d.get("revision_timestamp", ""),
        url=d.get("url", ""),
        lead_text=d.get("lead_text", ""),
        extract=d.get("extract", ""),
        full_text=d.get("full_text", ""),
        full_text_format=d.get("full_text_format", "plain_text"),
        thumbnail_url=d.get("thumbnail_url", ""),
        thumbnail_width=d.get("thumbnail_width"),
        thumbnail_height=d.get("thumbnail_height"),
        categories=list(d.get("categories", [])),
        license=d.get("license", ""),
        attribution=d.get("attribution", ""),
        source_api=d.get("source_api", ""),
        retrieved_at=d.get("retrieved_at", ""),
    )


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
    """Parse the JSON returned by the MediaWiki Action API.

    Returns an :class:`FetchResult` whose ``status`` is ``ok`` on
    success, or one of the documented failure statuses otherwise.
    """
    try:
        pages = (data.get("query") or {}).get("pages") or {}
    except (AttributeError, TypeError):
        return FetchResult("parse_error", None, "missing query.pages")
    if not pages:
        return FetchResult("article_not_found", None, "no pages in response")
    page = next(iter(pages.values()))
    if page.get("missing") is not None or "pageid" not in page:
        return FetchResult("article_not_found", None, "page missing")
    page_id = int(page.get("pageid", 0))
    revisions = page.get("revisions") or []
    if not revisions:
        return FetchResult("parse_error", None, "no revisions")
    revision = revisions[0]
    revision_id = int(revision.get("revid", 0))
    revision_timestamp = revision.get("timestamp", "")
    extract = page.get("extract", "") or ""
    full_text = clean_article_text(extract)
    # ``exintro`` mode (fetch_full_text=False) returns the lead in ``extract``;
    # full-text mode returns the entire article in ``extract``. We treat
    # ``lead_text`` as the first paragraph-ish chunk: the first 500 chars
    # of the cleaned text. The full body is in ``full_text``.
    lead_text = ""
    if extract:
        snippet = extract.strip().split("\n\n", 1)[0]
        lead_text = clean_article_text(snippet)[:500]
    canonical_title = page.get("title", title)
    url = (
        page.get("fullurl")
        or f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(canonical_title.replace(' ', '_'))}"
    )
    thumb = page.get("thumbnail") or {}
    thumbnail_url = thumb.get("source", "")
    thumbnail_width = thumb.get("width")
    thumbnail_height = thumb.get("height")
    attribution = (
        f'Text from Wikipedia article "{canonical_title}" ({language}.wikipedia.org); '
        f"contributors; revision {revision_id}; accessed {utc_now_iso()}; "
        "licensed under CC BY-SA."
    )
    article = WikipediaArticle(
        language=language,
        site=site,
        title=canonical_title,
        page_id=page_id,
        revision_id=revision_id,
        revision_timestamp=revision_timestamp,
        url=url,
        lead_text=lead_text,
        extract=clean_article_text(extract),
        full_text=full_text,
        full_text_format="plain_text",
        thumbnail_url=thumbnail_url,
        thumbnail_width=thumbnail_width,
        thumbnail_height=thumbnail_height,
        categories=[],
        license="CC BY-SA 4.0",
        attribution=attribution,
        source_api="mediawiki_action_api",
        retrieved_at=utc_now_iso(),
    )
    if not full_text:
        return FetchResult("empty_text", None, "no extract returned by API")
    # Touch the helpers so they remain referenced in case future code
    # wants to compute per-article metrics from the article text.
    _ = (count_words(full_text), estimate_tokens(full_text))
    return FetchResult("ok", article, "")


__all__ = [
    "CachedWikipediaClient",
    "FetchResult",
    "HttpWikipediaClient",
    "InMemoryWikipediaClient",
    "WikipediaArticle",
    "WikipediaClient",
    "parse_wikipedia_response",
]
