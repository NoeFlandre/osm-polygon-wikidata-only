"""Wikidata HTTP and in-memory client implementations + transport error.

Responsibility:
    Build the ``wbgetentities`` Action API URL, perform the
    read+gzip+JSON+throttle call against the shared
    :func:`read_wikimedia_json` helper, batch multiple QIDs into one
    request, and surface failures as ``None`` (mapped to ``WikidataError``
    when callers raise).

Logger-name preservation:
    The warning emitted on a failed batch request is logged under the
    *legacy* module path ``osm_polygon_wikidata_only.enrichment.wikidata_client``
    (NOT under the new transport path). This keeps downstream
    consumers and tests filtering on the legacy name working after
    the implementation moved out of ``enrichment.wikidata_client``.

Out of scope (intentionally retained by other modules):
    * Caching (see :mod:`enrichment.wikidata.cache`).
    * Parsing helpers (see :mod:`enrichment.wikidata.parsing`).
    * Wikimedia transport mechanics (see
      :mod:`enrichment.wikimedia.transport`).
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import Any

from osm_polygon_wikidata_only.config.settings import WIKIDATA_API_URL, Settings
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

from .models import WikidataClient, WikidataEntity
from .parsing import is_valid_qid, parse_wikidata_entity

# Note: the logger name is pinned to the legacy module path so that
# downstream consumers and tests filtering on
# ``osm_polygon_wikidata_only.enrichment.wikidata_client`` keep working
# even though the implementation now lives in
# ``enrichment.wikidata.transport``.
LOGGER = logging.getLogger(
    "osm_polygon_wikidata_only.enrichment.wikidata_client",
)


class WikidataError(RuntimeError):
    """Raised when the Wikidata client cannot return a result."""


class InMemoryWikidataClient(WikidataClient):
    """Test double backed by a hand-built :class:`dict`."""

    def __init__(self, mapping: dict[str, WikidataEntity]) -> None:
        self._mapping = dict(mapping)

    def get_entity(self, qid: str) -> WikidataEntity | None:
        if not is_valid_qid(qid):
            return None
        return self._mapping.get(qid)


class HttpWikidataClient(WikidataClient):
    """Real Wikidata client using ``urllib.request`` (stdlib only)."""

    def __init__(
        self,
        settings: Settings,
        *,
        endpoint: str = WIKIDATA_API_URL,
        scheduler: AdaptiveRequestScheduler | None = None,
        session: WikimediaHttpSession | None = None,
    ) -> None:
        self._settings = settings
        self._endpoint = endpoint
        self._scheduler = scheduler or default_scheduler()
        self._session = session or WikimediaSession(
            scheduler=self._scheduler,
            timeout_s=settings.request_timeout_s,
            user_agent=settings.user_agent,
        )

    def get_entity(self, qid: str) -> WikidataEntity | None:
        return self.get_entities([qid])[0]

    def get_entities(self, qids: Iterable[str]) -> list[WikidataEntity | None]:
        """Resolve several QIDs in one Action API request, preserving order."""
        requested = list(qids)
        valid = [qid for qid in dict.fromkeys(requested) if is_valid_qid(qid)]
        if not valid:
            return [None for _ in requested]
        url = self._build_url("|".join(valid))
        try:
            data = with_retries(
                lambda: self._http_get(url),
                attempts=self._settings.request_max_retries,
                base_delay=self._settings.request_base_delay_s,
                retry_on=(urllib.error.URLError, TimeoutError, OSError),
                should_retry=is_transient_network_error,
                on_retry=transient_retry_log_callback("Wikidata", logger=LOGGER),
            )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            LOGGER.warning("Wikidata batch request failed for %d QIDs: %s", len(valid), e)
            return [None for _ in requested]
        parsed = {qid: parse_wikidata_entity(qid, data) for qid in valid}
        return [parsed.get(qid) for qid in requested]

    def _build_url(self, qid: str) -> str:
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks|labels|descriptions|aliases",
            "sitefilter": "wiki",
            "languages": "en",
            "format": "json",
            "maxlag": "5",
        }
        return f"{self._endpoint}?{urllib.parse.urlencode(params)}"

    def _http_get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self._settings.user_agent, "Accept-Encoding": "gzip"},
        )
        host = urllib.parse.urlparse(url).netloc
        try:
            return read_wikimedia_json(
                req,
                self._session,
                host=host,
                anonymous_interval_s=self._settings.wikidata_min_interval_s,
                authenticated_interval_s=self._settings.wikimedia_authenticated_min_interval_s,
                throttle_callback=self._scheduler.report_host_throttled,
                default_throttle_s=self._settings.rate_limit_retry_after_default_s,
            )
        except _NonObjectJsonError as error:
            raise ValueError(f"Expected JSON object from {url}, got {error.value_type}") from None


__all__ = ["HttpWikidataClient", "InMemoryWikidataClient", "WikidataError"]
