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

import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from osm_polygon_wikidata_only.config.settings import MEDIAWIKI_API_URL_TEMPLATE, Settings
from osm_polygon_wikidata_only.enrichment.wikimedia import read_wikimedia_json
from osm_polygon_wikidata_only.enrichment.wikimedia.transport import (
    _NonObjectJsonError,
)
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaHttpSession,
    WikimediaSession,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.request_scheduler import (
    AdaptiveRequestScheduler,
    default_scheduler,
)
from osm_polygon_wikidata_only.utils.retry import with_retries

from .wikipedia.models import (
    BatchWikipediaClient,
    FetchResult,
    WikipediaArticle,
    WikipediaClient,
)
from .wikipedia.parsing import (
    parse_wikipedia_batch_response as _parse_wikipedia_batch_response,
)
from .wikipedia.parsing import (
    parse_wikipedia_response,
    plain_text_from_parse_response,
    query_with_extract,
    revision_id_from_query,
)

LOGGER = logging.getLogger(__name__)


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

    def __init__(
        self,
        settings: Settings,
        *,
        scheduler: AdaptiveRequestScheduler | None = None,
        session: WikimediaHttpSession | None = None,
    ) -> None:
        self._settings = settings
        self._scheduler = scheduler or default_scheduler()
        self._session = session or WikimediaSession(
            scheduler=self._scheduler,
            timeout_s=settings.request_timeout_s,
            user_agent=settings.user_agent,
        )

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
            if e.code in (429, 503):
                return FetchResult("rate_limited", None, str(e))
            return FetchResult("http_error", None, f"HTTP {e.code}: {e}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return FetchResult("http_error", None, str(e))
        result = parse_wikipedia_response(
            language,
            site,
            title,
            data,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            fetch_full_text=fetch_full_text,
        )
        if result.status != "empty_text" or not fetch_full_text:
            return result
        revision_id = revision_id_from_query(data)
        if revision_id <= 0:
            return result
        fallback_url = self._build_parse_url(language, revision_id)
        try:
            fallback_data = with_retries(
                lambda: self._http_get(fallback_url),
                attempts=self._settings.request_max_retries,
                base_delay=self._settings.request_base_delay_s,
                retry_on=(urllib.error.URLError, TimeoutError, OSError),
            )
        except urllib.error.HTTPError as error:
            status = "rate_limited" if error.code in (429, 503) else "http_error"
            return FetchResult(status, None, f"parse fallback HTTP {error.code}: {error}")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return FetchResult("http_error", None, f"parse fallback failed: {error}")
        parsed_text = plain_text_from_parse_response(fallback_data)
        if not parsed_text:
            if result.article is not None:
                return FetchResult(
                    "empty_text",
                    result.article,
                    "extract and exact-revision parse were empty",
                )
            return FetchResult("empty_text", None, "extract and exact-revision parse were empty")
        fallback_result = parse_wikipedia_response(
            language,
            site,
            title,
            query_with_extract(data, parsed_text),
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            fetch_full_text=True,
        )
        if fallback_result.article is not None:
            return FetchResult(
                fallback_result.status,
                replace(
                    fallback_result.article,
                    source_api="mediawiki_action_api_parse_fallback",
                ),
                fallback_result.error,
            )
        return fallback_result

    def fetch_articles(
        self,
        language: str,
        site: str,
        titles: Iterable[str],
        *,
        fetch_full_text: bool = True,
    ) -> dict[str, FetchResult]:
        """Fetch a same-site title batch and return a result for every title."""
        requested = list(dict.fromkeys(titles))
        if not requested:
            return {}
        if fetch_full_text:
            # TextExtracts only returns multiple extracts for lead-only
            # (`exintro`) requests. Full-text batches silently omit all but
            # one extract, so preserve complete per-article retrieval here.
            return {
                title: self.fetch_article(language, site, title, fetch_full_text=True)
                for title in requested
            }
        url = self._build_url(language, "|".join(requested), fetch_full_text=fetch_full_text)
        try:
            data = with_retries(
                lambda: self._http_get(url),
                attempts=self._settings.request_max_retries,
                base_delay=self._settings.request_base_delay_s,
                retry_on=(urllib.error.URLError, TimeoutError, OSError),
            )
            return _parse_wikipedia_batch_response(
                language, site, requested, data, fetch_full_text=fetch_full_text
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
            # A batch is only an optimization. Reuse the established per-title
            # path so a malformed or transient batch response cannot drop work.
            return {
                title: self.fetch_article(language, site, title, fetch_full_text=fetch_full_text)
                for title in requested
            }

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
            "maxlag": "5",
        }
        if not fetch_full_text:
            # ``exintro`` makes the API return only the lead section.
            params["exintro"] = "1"
        return f"{endpoint}?{urllib.parse.urlencode(params)}"

    def _build_parse_url(self, language: str, revision_id: int) -> str:
        endpoint = MEDIAWIKI_API_URL_TEMPLATE.format(lang=language)
        params = {
            "action": "parse",
            "format": "json",
            "formatversion": "2",
            "oldid": str(revision_id),
            "prop": "text",
            "disableeditsection": "1",
            "disablelimitreport": "1",
            "maxlag": "5",
        }
        return f"{endpoint}?{urllib.parse.urlencode(params)}"

    def _http_get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self._settings.user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
        )
        host = urllib.parse.urlparse(url).netloc
        try:
            return read_wikimedia_json(
                req,
                self._session,
                host=host,
                anonymous_interval_s=self._settings.wikipedia_min_interval_s,
                authenticated_interval_s=self._settings.wikimedia_authenticated_min_interval_s,
                throttle_callback=self._scheduler.report_host_throttled,
                default_throttle_s=self._settings.rate_limit_retry_after_default_s,
            )
        except _NonObjectJsonError as error:
            raise ValueError(f"Expected JSON object from {url}, got {error.value_type}") from None


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
        key = self._cache_key(site, title, fetch_full_text)
        cached = self._cached_result(key)
        if cached is not None:
            return cached
        result = self._inner.fetch_article(
            language,
            site,
            title,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            wikidata_aliases=wikidata_aliases,
            fetch_full_text=fetch_full_text,
        )
        self._store_result(key, result, language, site, title, fetch_full_text)
        return result

    def fetch_articles(
        self,
        language: str,
        site: str,
        titles: Iterable[str],
        *,
        fetch_full_text: bool = True,
    ) -> dict[str, FetchResult]:
        """Serve cached titles and fetch only the missing titles as a batch."""
        requested = list(dict.fromkeys(titles))
        results: dict[str, FetchResult] = {}
        missing: list[str] = []
        for title in requested:
            key = self._cache_key(site, title, fetch_full_text)
            cached = self._cached_result(key)
            if cached is None:
                missing.append(title)
            else:
                results[title] = cached

        batch_fetch = getattr(self._inner, "fetch_articles", None)
        if callable(batch_fetch):
            fetched = batch_fetch(language, site, missing, fetch_full_text=fetch_full_text)
        else:
            fetched = {
                title: self._inner.fetch_article(
                    language, site, title, fetch_full_text=fetch_full_text
                )
                for title in missing
            }
        for title in missing:
            result = fetched.get(title)
            if result is None:
                result = self._inner.fetch_article(
                    language, site, title, fetch_full_text=fetch_full_text
                )
            key = self._cache_key(site, title, fetch_full_text)
            self._store_result(key, result, language, site, title, fetch_full_text)
            results[title] = result
        return results

    def _cached_result(self, key: str) -> FetchResult | None:
        hit = self._cache.get(key)
        if hit is None or hit.status != "ok" or not isinstance(hit.parsed_result, dict):
            return None
        return FetchResult("ok", _article_from_dict(hit.parsed_result))

    def _store_result(
        self,
        key: str,
        result: FetchResult,
        language: str,
        site: str,
        title: str,
        fetch_full_text: bool,
    ) -> None:
        request_url = self._endpoint_for(language, title, fetch_full_text)
        if result.status == "ok" and result.article is not None:
            self._cache.set(
                key,
                payload=_article_to_dict(result.article),
                request_url=request_url,
                response_metadata={"language": language, "site": site, "title": title},
                status="ok",
            )
            return
        self._cache.set(
            key,
            payload=result.status,
            request_url=request_url,
            response_metadata={"status": result.status, "error": result.error},
            status="error",
            ttl_s=self._failed_ttl_s,
        )

    @staticmethod
    def _safe_title(title: str) -> str:
        # Cache file system can't have slashes; keep them encoded.
        return title.replace("/", "_").replace(" ", "_")

    @classmethod
    def _cache_key(cls, site: str, title: str, fetch_full_text: bool) -> str:
        policy = "full-text-v2" if fetch_full_text else "lead-only-v2"
        return f"wikipedia/{policy}/{site}/{cls._safe_title(title)}.json"

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


__all__ = [
    "BatchWikipediaClient",
    "CachedWikipediaClient",
    "FetchResult",
    "HttpWikipediaClient",
    "InMemoryWikipediaClient",
    "WikipediaArticle",
    "WikipediaClient",
    "parse_wikipedia_response",
]
