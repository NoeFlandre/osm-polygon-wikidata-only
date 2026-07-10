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
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from osm_polygon_wikidata_only.config.settings import WIKIDATA_API_URL, Settings
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

LOGGER = logging.getLogger(__name__)

# Sitelink entry: site -> title (e.g. ``{"enwiki": "Monaco"}``).
Sitelinks = dict[str, str]


@dataclass(frozen=True)
class WikidataEntity:
    """A minimal Wikidata entity used by this pipeline.

    Only the fields we actually consume are kept. ``labels`` and
    ``descriptions`` are keyed by language code.
    """

    qid: str
    sitelinks: Sitelinks = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)


class WikidataError(RuntimeError):
    """Raised when the Wikidata client cannot return a result."""


_QID_PATTERN = __import__("re").compile(r"^Q[1-9]\d*$")


def is_valid_qid(qid: str) -> bool:
    """Return True if ``qid`` looks like a valid Wikidata identifier.

    A QID is the letter ``Q`` followed by a positive integer.
    """
    if not qid:
        return False
    return bool(_QID_PATTERN.match(qid))


class WikidataClient(ABC):
    """Abstract interface — concrete implementations below."""

    @abstractmethod
    def get_entity(self, qid: str) -> WikidataEntity | None:
        """Return the entity for ``qid`` or ``None`` if it does not exist."""


@runtime_checkable
class BatchWikidataClient(Protocol):
    """Optional capability for resolving several QIDs in one request."""

    def get_entities(self, qids: Iterable[str]) -> list[WikidataEntity | None]: ...


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
    ) -> None:
        self._settings = settings
        self._endpoint = endpoint
        self._scheduler = scheduler or default_scheduler()

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

            def request() -> tuple[bytes, str]:
                with urllib.request.urlopen(req, timeout=self._settings.request_timeout_s) as resp:
                    return resp.read(), resp.headers.get("Content-Encoding", "")

            raw, encoding = self._scheduler.run(request)
            if encoding == "gzip":
                raw = gzip.decompress(raw)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                delay = retry_after_seconds(
                    e, default_s=self._settings.rate_limit_retry_after_default_s
                )
                defer_host("www.wikidata.org", delay)
                self._scheduler.defer(delay)
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


def parse_wikidata_entity(qid: str, data: dict[str, Any]) -> WikidataEntity | None:
    """Parse the JSON returned by ``wbgetentities`` into a :class:`WikidataEntity`.

    Returns ``None`` if the entity is missing or the response is
    malformed.
    """
    entities = data.get("entities") or {}
    if qid not in entities:
        return None
    raw = entities[qid]
    if raw.get("missing") is not None:
        return None
    sitelinks: Sitelinks = {}
    for site, info in (raw.get("sitelinks") or {}).items():
        if not _is_language_wiki(site):
            continue
        title = info.get("title")
        if title:
            sitelinks[site] = title
    labels = {k: v.get("value", "") for k, v in (raw.get("labels") or {}).items()}
    descriptions = {k: v.get("value", "") for k, v in (raw.get("descriptions") or {}).items()}
    aliases: dict[str, list[str]] = {}
    for k, vals in (raw.get("aliases") or {}).items():
        aliases[k] = [v.get("value", "") for v in vals if v.get("value")]
    return WikidataEntity(
        qid=qid,
        sitelinks=sitelinks,
        labels=labels,
        descriptions=descriptions,
        aliases=aliases,
    )


def language_from_site(site: str) -> str:
    """Convert a Wikidata sitelink site to a language code.

    ``enwiki`` -> ``en``; ``frwiki`` -> ``fr``; etc. Returns the
    original site if it does not end in ``wiki``.
    """
    if site.endswith("wiki"):
        language = site[: -len("wiki")]
        aliases = {"be_x_old": "be-tarask"}
        return aliases.get(language, language.replace("_", "-"))
    return site


def _is_language_wiki(site: str) -> bool:
    """True iff ``site`` is a language-Wikipedia sitelink key.

    Wikimedia database names include long and compound language identifiers,
    so length is not a valid discriminator. Explicitly exclude non-Wikipedia
    projects and accept lowercase language database identifiers.
    """
    if not site.endswith("wiki") or len(site) <= len("wiki"):
        return False
    lang = site[: -len("wiki")]
    non_language_projects = {
        "commons",
        "foundation",
        "incubator",
        "mediawiki",
        "meta",
        "outreach",
        "sources",
        "species",
        "strategy",
        "test",
        "test2",
        "wikidata",
    }
    return (
        lang not in non_language_projects
        and lang == lang.lower()
        and all(character.isalnum() or character in "_-" for character in lang)
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
