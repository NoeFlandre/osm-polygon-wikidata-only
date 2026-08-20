"""Cached Wikimedia transport for augmentation-only reads."""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, cast

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    count_words,
    estimate_tokens,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import is_valid_qid
from osm_polygon_wikidata_only.enrichment.wikimedia import read_wikimedia_json
from osm_polygon_wikidata_only.enrichment.wikimedia.transport import (
    _NonObjectJsonError,
)
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaSession,
    load_wikimedia_credentials,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.request_scheduler import (
    SYSTEMIC_ACTIVE_HOST_WINDOW_S,
    SYSTEMIC_HOST_FRACTION,
    SYSTEMIC_MINIMUM_HOSTS,
    AdaptiveRequestScheduler,
)
from osm_polygon_wikidata_only.utils.retry import (
    is_transient_network_error,
    transient_retry_log_callback,
    with_retries,
)
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import Document, document_id

LOGGER = logging.getLogger(__name__)
_TRANSIENT_API_ERROR_CODES = frozenset({"maxlag", "ratelimited", "readonly", "readonlytext"})


class MediaWikiApiError(RuntimeError):
    """A structured error returned by a Wikimedia API."""

    def __init__(self, message: str, *, code: str | None = None, info: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.info = info


class _TransientMediaWikiApiError(MediaWikiApiError):
    """A retryable structured Wikimedia API error."""


def _build_scheduler(rate: float, authenticated: bool) -> AdaptiveRequestScheduler:
    return AdaptiveRequestScheduler(
        max_in_flight=3,
        requests_per_minute=rate,
        max_requests_per_minute=rate,
        minimum_requests_per_minute=min(200.0 if authenticated else 60.0, rate),
        active_host_window_s=SYSTEMIC_ACTIVE_HOST_WINDOW_S,
        minimum_systemic_hosts=SYSTEMIC_MINIMUM_HOSTS,
        systemic_host_fraction=SYSTEMIC_HOST_FRACTION,
    )


def _build_session(
    scheduler: AdaptiveRequestScheduler,
    settings: Settings,
    credentials: Any,
) -> WikimediaSession:
    return WikimediaSession(
        scheduler=scheduler,
        timeout_s=settings.request_timeout_s,
        user_agent=settings.user_agent,
        credentials=credentials,
    )


def _log_throttled_http_error(
    host: str,
    error: urllib.error.HTTPError,
    delay: float | None,
) -> None:
    if error.code in (429, 503):
        LOGGER.warning(
            "Wikimedia throttled %s (HTTP %d); retrying after %.1fs",
            host,
            error.code,
            delay if delay is not None else 0.0,
        )


def _normalize_entities(raw_entities: object) -> dict[str, dict[str, Any]]:
    if isinstance(raw_entities, dict):
        return _normalize_entity_map(cast(dict[object, object], raw_entities))
    return _normalize_entity_list(raw_entities)


def _normalize_entity_map(raw_entities: dict[object, object]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for response_qid, entity in raw_entities.items():
        if not isinstance(entity, dict) or not entity.get("id"):
            continue
        normalized = cast(dict[str, Any], dict(entity))
        normalized["id"] = str(response_qid)
        out[str(response_qid)] = normalized
    return out


def _normalize_entity_list(raw_entities: object) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entity in raw_entities:
        if not isinstance(entity, dict) or not entity.get("id"):
            continue
        normalized = cast(dict[str, Any], dict(entity))
        out[str(normalized["id"])] = normalized
    return out


def _project_host(project: str, language: str) -> str:
    site = "wikipedia" if project == "wikipedia" else "wikivoyage"
    return f"{language}.{site}.org"


def _parsed_html_text(data: dict[str, Any]) -> str:
    parsed = data.get("parse", {})
    text = parsed.get("text", "") if isinstance(parsed, dict) else ""
    if isinstance(text, dict):
        text = text.get("*", "")
    return str(text)


def _raise_for_api_error(data: dict[str, Any]) -> None:
    error = data.get("error")
    if not isinstance(error, Mapping):
        return
    code, info = _api_error_details(error)
    exception = _api_error_exception(code)
    raise exception(f"Wikimedia API error {code}: {info}", code=code, info=info)


def _api_error_details(error: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(error.get("code") or "unknown"),
        str(error.get("info") or "No error details supplied"),
    )


def _api_error_exception(code: str) -> type[MediaWikiApiError]:
    if code in _TRANSIENT_API_ERROR_CODES or code.startswith("internal_api_error_"):
        return _TransientMediaWikiApiError
    return MediaWikiApiError


class AugmentationWikimediaClient:
    """Read exact Wikimedia revisions and Wikidata entities with a shared scheduler."""

    def __init__(
        self,
        settings: Settings,
        cache: JsonFileCache,
        *,
        environ: Mapping[str, str] | None = None,
        scheduler: AdaptiveRequestScheduler | None = None,
        session: WikimediaSession | None = None,
    ) -> None:
        source = os.environ if environ is None else environ
        credentials = load_wikimedia_credentials(source)
        rate = 1_200.0 if credentials else 180.0
        effective = replace(settings, request_timeout_s=max(settings.request_timeout_s, 60.0))
        self._settings = effective
        self._scheduler = scheduler or _build_scheduler(rate, bool(credentials))
        self._session = session or _build_session(self._scheduler, effective, credentials)
        self._cache = cache

    def get_json(self, url: str, *, key: str) -> dict[str, Any]:
        # Cache hits short-circuit BEFORE any URL validation or transport
        # invocation: even a malformed cached URL must not be re-parsed.
        cached = self._cached_json(key)
        if cached is not None:
            return cached
        parsed, _host = self._request_json(url)
        self._cache.set(key, parsed, request_url=url, status="ok")
        return parsed

    def _cached_json(self, key: str) -> dict[str, Any] | None:
        hit = self._cache.get(key)
        if hit is None or hit.status != "ok" or not isinstance(hit.parsed_result, dict):
            return None
        try:
            _raise_for_api_error(hit.parsed_result)
        except MediaWikiApiError:
            self._cache.delete(key)
            return None
        return hit.parsed_result

    def _request_json(self, url: str) -> tuple[dict[str, Any], str]:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme != "https":
            raise ValueError(f"Only HTTPS Wikimedia URLs are allowed: {url}")
        host = parsed_url.netloc
        request = urllib.request.Request(  # noqa: S310 - HTTPS is validated above
            url,
            headers={"User-Agent": self._settings.user_agent, "Accept-Encoding": "gzip"},
        )
        delay_state: list[float | None] = [None]

        def on_throttled(throttled_host: str, delay: float) -> None:
            delay_state[0] = delay
            self._scheduler.report_host_throttled(throttled_host, delay)

        def read() -> dict[str, Any]:
            try:
                parsed = read_wikimedia_json(
                    request,
                    self._session,
                    host=host,
                    anonymous_interval_s=self._settings.augmentation_min_interval_s,
                    authenticated_interval_s=self._settings.wikimedia_authenticated_min_interval_s,
                    throttle_callback=on_throttled,
                    default_throttle_s=self._settings.rate_limit_retry_after_default_s,
                )
                _raise_for_api_error(parsed)
                return parsed
            except _NonObjectJsonError:
                raise ValueError(f"Expected JSON object from {url}") from None

        try:
            parsed = with_retries(
                read,
                attempts=self._settings.request_max_retries,
                base_delay=self._settings.request_base_delay_s,
                retry_on=(
                    urllib.error.URLError,
                    TimeoutError,
                    OSError,
                    _TransientMediaWikiApiError,
                ),
                should_retry=lambda error: (
                    isinstance(error, _TransientMediaWikiApiError)
                    or is_transient_network_error(error)
                ),
                on_retry=transient_retry_log_callback(f"Wikimedia host {host}", logger=LOGGER),
            )
        except urllib.error.HTTPError as error:
            _log_throttled_http_error(host, error, delay_state[0])
            raise
        return parsed, host

    def entities(self, qids: Iterable[str], *, props: str) -> dict[str, dict[str, Any]]:
        # Callers pass canonical QIDs, never raw OSM tag values. Filter again
        # at the transport boundary so malformed or compound values cannot turn
        # an otherwise recoverable region into a MediaWiki ``no-such-entity``
        # failure.
        ids = sorted({qid for qid in qids if is_valid_qid(qid)})
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(ids), 50):
            out.update(self._entity_chunk(ids[start : start + 50], props))
        return out

    def _entity_chunk(self, chunk: list[str], props: str) -> dict[str, dict[str, Any]]:
        params = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": props,
                "format": "json",
                "formatversion": "2",
                "maxlag": "5",
            }
        )
        data = self.get_json(
            f"https://www.wikidata.org/w/api.php?{params}",
            key=f"entities/{props.replace('|', '-')}/{'-'.join(chunk)}.json",
        )
        return _normalize_entities(data.get("entities", []))

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        host = _project_host(project, language)
        params = urllib.parse.urlencode(
            {
                "action": "parse",
                "oldid": str(revision_id),
                "prop": "text",
                "format": "json",
                "formatversion": "2",
                "maxlag": "5",
            }
        )
        try:
            data = self.get_json(
                f"https://{host}/w/api.php?{params}",
                key=f"sections/{project}/{language}/{revision_id}.json",
            )
        except MediaWikiApiError as error:
            if error.code not in {"nosuchrevid", "permissiondenied"}:
                raise
            LOGGER.warning(
                "Wikimedia revision %d is unavailable on %s; continuing without sections",
                revision_id,
                host,
            )
            return ""
        return _parsed_html_text(data)

    def wikivoyage_document(
        self, qid: str, language: str, site: str, title: str
    ) -> Document | None:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "revisions|extracts|info",
                "titles": title,
                "rvprop": "ids|timestamp",
                "explaintext": "1",
                "inprop": "url",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
                "maxlag": "5",
            }
        )
        data = self.get_json(
            f"https://{language}.wikivoyage.org/w/api.php?{params}",
            key=f"wikivoyage/{language}/{urllib.parse.quote(title, safe='')}.json",
        )
        page_revision = _voyage_page(data)
        if page_revision is None:
            return None
        page, revision = page_revision
        return _voyage_document(
            qid=qid,
            language=language,
            site=site,
            title=title,
            page=page,
            revision=revision,
            retrieved=utc_now_iso(),
        )


def _voyage_page(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return the first available Wikivoyage page and revision."""
    page = _first_voyage_page(data)
    if page is None:
        return None
    revision = _first_revision(page)
    if revision is None:
        return None
    return page, revision


def _first_voyage_page(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first non-missing page in a query response."""
    pages = (data.get("query") or {}).get("pages") or []
    if not pages or pages[0].get("missing"):
        return None
    return pages[0]


def _first_revision(page: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first revision attached to a page."""
    revisions = page.get("revisions") or []
    if not revisions:
        return None
    return revisions[0]


def _voyage_document(
    *,
    qid: str,
    language: str,
    site: str,
    title: str,
    page: dict[str, Any],
    revision: dict[str, Any],
    retrieved: str,
) -> Document:
    """Build a Wikivoyage document from a selected API page and revision."""
    text = clean_article_text(str(page.get("extract", "")))
    page_id, revision_id = int(page.get("pageid", 0)), int(revision.get("revid", 0))
    return Document(
        document_id(qid, "wikivoyage", language, page_id, revision_id),
        "",
        qid,
        "wikivoyage",
        language,
        site,
        str(page.get("title", title)),
        str(page.get("fullurl", "")),
        page_id,
        revision_id,
        str(revision.get("timestamp", "")),
        retrieved,
        text,
        "plain_text",
        len(text),
        count_words(text),
        estimate_tokens(text),
        "CC BY-SA 4.0",
        f'Text from Wikivoyage article "{page.get("title", title)}"; revision {revision_id}; CC BY-SA.',
        "mediawiki_action_api",
        "ok" if text else "empty_text",
        "",
        __import__("hashlib").sha256(text.encode()).hexdigest(),
    )


__all__ = ["AugmentationWikimediaClient"]
