"""Wikipedia HTTP and in-memory client implementations.

Responsibility:
    Build Action API URLs, perform the read+gzip+JSON+throttle call
    against the shared :func:`read_wikimedia_json` helper, and map
    transport failures and successful parses to the domain
    :class:`FetchResult` and :class:`WikipediaArticle` shapes.

Out of scope (intentionally retained by other modules):
    * Caching (see :mod:`enrichment.wikipedia.cache`).
    * Parsing helpers (see :mod:`enrichment.wikipedia.parsing`).
    * Wikimedia transport mechanics (see
      :mod:`enrichment.wikimedia.transport`).
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
from osm_polygon_wikidata_only.utils.request_scheduler import (
    AdaptiveRequestScheduler,
    default_scheduler,
)
from osm_polygon_wikidata_only.utils.retry import (
    is_transient_network_error,
    transient_retry_log_callback,
    with_retries,
)

from .models import FetchResult, WikipediaClient
from .parsing import (
    parse_wikipedia_batch_response as _parse_wikipedia_batch_response,
)
from .parsing import (
    parse_wikipedia_response,
    plain_text_from_parse_response,
    query_with_extract,
    revision_id_from_query,
)

LOGGER = logging.getLogger("osm_polygon_wikidata_only.enrichment.wikipedia_client")


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
        data, error = self._request_article_data(url, fallback=False)
        if error is not None:
            return error
        assert data is not None
        result = parse_wikipedia_response(
            language,
            site,
            title,
            data,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            fetch_full_text=fetch_full_text,
        )
        if not _needs_parse_fallback(result, fetch_full_text):
            return result
        revision_id = revision_id_from_query(data)
        if revision_id <= 0:
            return result
        fallback_url = self._build_parse_url(language, revision_id)
        return self._parse_fallback(
            language,
            site,
            title,
            data,
            result,
            fallback_url,
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
        )

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
            return self._fetch_full_text_batch(language, site, requested)
        return self._fetch_lead_batch(language, site, requested)

    def _request_article_data(
        self, url: str, *, fallback: bool
    ) -> tuple[dict[str, Any] | None, FetchResult | None]:
        try:
            return (
                with_retries(
                    lambda: self._http_get(url),
                    attempts=self._settings.request_max_retries,
                    base_delay=self._settings.request_base_delay_s,
                    retry_on=(urllib.error.URLError, TimeoutError, OSError),
                    should_retry=is_transient_network_error,
                    on_retry=transient_retry_log_callback("Wikipedia", logger=LOGGER),
                ),
                None,
            )
        except urllib.error.HTTPError as error:
            return None, _article_http_error(error, fallback=fallback)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return None, _article_network_error(error, fallback=fallback)

    def _parse_fallback(
        self,
        language: str,
        site: str,
        title: str,
        data: dict[str, Any],
        result: FetchResult,
        fallback_url: str,
        *,
        wikidata_label: str,
        wikidata_description: str,
    ) -> FetchResult:
        fallback_data, error = self._request_article_data(fallback_url, fallback=True)
        if error is not None:
            return error
        assert fallback_data is not None
        parsed_text = plain_text_from_parse_response(fallback_data)
        if not parsed_text:
            return _empty_fallback_result(result)
        fallback_result = parse_wikipedia_response(
            language,
            site,
            title,
            query_with_extract(data, parsed_text),
            wikidata_label=wikidata_label,
            wikidata_description=wikidata_description,
            fetch_full_text=True,
        )
        return _mark_fallback_result(fallback_result)

    def _fetch_full_text_batch(
        self, language: str, site: str, requested: list[str]
    ) -> dict[str, FetchResult]:
        return {
            title: self.fetch_article(language, site, title, fetch_full_text=True)
            for title in requested
        }

    def _fetch_lead_batch(
        self, language: str, site: str, requested: list[str]
    ) -> dict[str, FetchResult]:
        url = self._build_url(language, "|".join(requested), fetch_full_text=False)
        data, error = self._request_article_data(url, fallback=False)
        if error is not None:
            return self._fetch_full_text_batch(language, site, requested)
        try:
            assert data is not None
            return _parse_wikipedia_batch_response(
                language, site, requested, data, fetch_full_text=False
            )
        except ValueError:
            return self._fetch_full_text_batch(language, site, requested)

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


def _needs_parse_fallback(result: FetchResult, fetch_full_text: bool) -> bool:
    return result.status == "empty_text" and fetch_full_text


def _article_http_error(error: urllib.error.HTTPError, *, fallback: bool) -> FetchResult:
    if fallback:
        status = "rate_limited" if error.code in (429, 503) else "http_error"
        return FetchResult(status, None, f"parse fallback HTTP {error.code}: {error}")
    if error.code == 404:
        return FetchResult("article_not_found", None, str(error))
    if error.code in (429, 503):
        return FetchResult("rate_limited", None, str(error))
    return FetchResult("http_error", None, f"HTTP {error.code}: {error}")


def _article_network_error(error: BaseException, *, fallback: bool) -> FetchResult:
    if fallback:
        return FetchResult("http_error", None, f"parse fallback failed: {error}")
    return FetchResult("http_error", None, str(error))


def _empty_fallback_result(result: FetchResult) -> FetchResult:
    message = "extract and exact-revision parse were empty"
    if result.article is not None:
        return FetchResult("empty_text", result.article, message)
    return FetchResult("empty_text", None, message)


def _mark_fallback_result(result: FetchResult) -> FetchResult:
    if result.article is None:
        return result
    return FetchResult(
        result.status,
        replace(result.article, source_api="mediawiki_action_api_parse_fallback"),
        result.error,
    )


__all__ = ["HttpWikipediaClient", "InMemoryWikipediaClient"]
