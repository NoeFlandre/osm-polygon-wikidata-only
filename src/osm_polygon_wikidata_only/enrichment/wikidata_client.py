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

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from osm_polygon_wikidata_only.config.settings import WIKIDATA_API_URL, Settings
from osm_polygon_wikidata_only.io.cache import CacheEntry, JsonFileCache
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
    ) -> None:
        self._settings = settings
        self._endpoint = endpoint

    def get_entity(self, qid: str) -> WikidataEntity | None:
        if not is_valid_qid(qid):
            return None
        url = self._build_url(qid)
        try:
            data = with_retries(
                lambda: self._http_get(url),
                attempts=self._settings.request_max_retries,
                base_delay=self._settings.request_base_delay_s,
                retry_on=(urllib.error.URLError, TimeoutError, OSError),
            )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            LOGGER.warning("Wikidata request failed for %s: %s", qid, e)
            return None
        return parse_wikidata_entity(qid, data)

    def _build_url(self, qid: str) -> str:
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks|labels|descriptions|aliases",
            "sitefilter": "wiki",
            "languages": "en",
            "format": "json",
        }
        return f"{self._endpoint}?{urllib.parse.urlencode(params)}"

    def _http_get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": self._settings.user_agent})
        with urllib.request.urlopen(req, timeout=self._settings.request_timeout_s) as resp:
            raw = resp.read()
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
        if not is_valid_qid(qid):
            return None
        key = f"wikidata/{qid}.json"
        hit = self._cache.get(key)
        if hit is not None:
            if hit.status == "ok":
                return _entity_from_dict(hit.parsed_result)
            return None
        entity = self._inner.get_entity(qid)
        if entity is None:
            self._cache.set(
                key,
                payload=None,
                status="error",
                ttl_s=self._failed_ttl_s,
                response_metadata={"reason": "wikidata_not_found"},
            )
            return None
        self._cache.set(
            key,
            payload=_entity_to_dict(entity),
            request_url=self._endpoint_for(qid),
            status="ok",
        )
        return entity

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
        return site[: -len("wiki")]
    return site


def _is_language_wiki(site: str) -> bool:
    """True iff ``site`` is a language-Wikipedia sitelink key.

    The Wikidata convention is ``<lang>wiki`` where ``<lang>`` is a
    2- or 3-letter lowercase language code. Sitelinks like
    ``commonswiki`` or ``wikidatawiki`` are NOT language Wikipedias
    and are filtered out.
    """
    if not site.endswith("wiki") or len(site) <= len("wiki"):
        return False
    lang = site[: -len("wiki")]
    return 2 <= len(lang) <= 3 and lang.isalpha() and lang == lang.lower()


__all__ = [
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
