"""Wikidata QID lookups: QID -> list of Wikipedia sitelinks.

The client is split into two layers:

* :class:`WikidataClient` — abstract interface used by the pipeline.
* :class:`HttpWikidataClient` — concrete client that calls the
  Wikidata Action API via ``urllib`` (no extra HTTP dependency).
* :class:`InMemoryWikidataClient` — fake client used in tests.

A :class:`CachedWikidataClient` wraps any client and adds the
:mod:`io.cache` layer for transparent reuse on reruns.
"""

from __future__ import annotations

import gzip
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import Any

from osm_polygon_wikidata_only.config.settings import WIKIDATA_API_URL, Settings
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaHttpSession,
    WikimediaSession,
)
from osm_polygon_wikidata_only.io.cache import CacheEntry, JsonFileCache
from osm_polygon_wikidata_only.utils.rate_limit import (
    defer_host,
    retry_after_seconds,
    wait_for_host,
)
from osm_polygon_wikidata_only.utils.request_scheduler import (
    AdaptiveRequestScheduler,
    default_scheduler,
)
from osm_polygon_wikidata_only.utils.retry import with_retries

from .wikidata.models import BatchWikidataClient, Sitelinks, WikidataClient, WikidataEntity
from .wikidata.parsing import is_valid_qid, language_from_site, parse_wikidata_entity

LOGGER = logging.getLogger(__name__)


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
        wait_for_host("www.wikidata.org", min_interval_s=self._settings.wikidata_min_interval_s)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self._settings.user_agent, "Accept-Encoding": "gzip"},
        )
        try:
            raw, encoding = self._session.read(req)
            if encoding == "gzip":
                raw = gzip.decompress(raw)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                delay = retry_after_seconds(
                    e, default_s=self._settings.rate_limit_retry_after_default_s
                )
                defer_host("www.wikidata.org", delay)
                if e.code == 429:
                    self._scheduler.report_throttled(delay)
                else:
                    self._scheduler.report_host_throttled("www.wikidata.org", delay)
            raise
        parsed: object = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
        return parsed


class CachedWikidataClient(WikidataClient):
    """Wrap another client and serve responses from a local cache.

    Successful results are stored with the configured TTL; failed
    results are cached with a shorter ``failed_ttl_s`` so reruns do
    not hammer the API.
    """

    def __init__(
        self,
        inner: WikidataClient,
        cache: JsonFileCache,
        *,
        failed_ttl_s: int = 60 * 60,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._failed_ttl_s = failed_ttl_s

    def get_entity(self, qid: str) -> WikidataEntity | None:
        return self.get_entities([qid])[0]

    def get_entities(self, qids: Iterable[str]) -> list[WikidataEntity | None]:
        """Resolve cache misses together while preserving input order."""
        requested = list(qids)
        resolved: dict[str, WikidataEntity | None] = {}
        misses: list[str] = []
        for qid in dict.fromkeys(requested):
            if not is_valid_qid(qid):
                resolved[qid] = None
                continue
            key = f"wikidata/{qid}.json"
            hit = self._cache.get(key)
            if hit is None:
                misses.append(qid)
            elif hit.status == "ok" and isinstance(hit.parsed_result, dict):
                resolved[qid] = _entity_from_dict(hit.parsed_result)
            else:
                resolved[qid] = None

        batch_get = getattr(self._inner, "get_entities", None)
        if callable(batch_get):
            fetched = batch_get(misses)
        else:
            fetched = [self._inner.get_entity(qid) for qid in misses]
        for qid, entity in zip(misses, fetched, strict=True):
            key = f"wikidata/{qid}.json"
            if entity is None:
                self._cache.set(
                    key,
                    payload=None,
                    status="error",
                    ttl_s=self._failed_ttl_s,
                    response_metadata={"reason": "wikidata_not_found"},
                )
            else:
                self._cache.set(
                    key,
                    payload=_entity_to_dict(entity),
                    request_url=self._endpoint_for(qid),
                    status="ok",
                )
            resolved[qid] = entity
        return [resolved.get(qid) for qid in requested]

    def _endpoint_for(self, qid: str) -> str:
        if isinstance(self._inner, HttpWikidataClient):
            return self._inner._build_url(qid)
        return ""


def _entity_to_dict(entity: WikidataEntity) -> dict[str, Any]:
    return {
        "qid": entity.qid,
        "sitelinks": dict(entity.sitelinks),
        "labels": dict(entity.labels),
        "descriptions": dict(entity.descriptions),
        "aliases": {k: list(v) for k, v in entity.aliases.items()},
    }


def _entity_from_dict(d: dict[str, Any]) -> WikidataEntity:
    return WikidataEntity(
        qid=d["qid"],
        sitelinks=dict(d.get("sitelinks", {})),
        labels=dict(d.get("labels", {})),
        descriptions=dict(d.get("descriptions", {})),
        aliases={k: list(v) for k, v in d.get("aliases", {}).items()},
    )


__all__ = [
    "BatchWikidataClient",
    "CachedWikidataClient",
    "HttpWikidataClient",
    "InMemoryWikidataClient",
    "Sitelinks",
    "WikidataClient",
    "WikidataEntity",
    "WikidataError",
    "is_valid_qid",
    "language_from_site",
    "parse_wikidata_entity",
]


# Note: re-exported to make the linter happy in tests that use
# ``CacheEntry`` indirectly through the cache module.
_ = CacheEntry
