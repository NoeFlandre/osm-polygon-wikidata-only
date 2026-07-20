"""Cached Wikidata client and entity serialization helpers.

Responsibility:
    Wrap any :class:`WikidataClient` with the
    :class:`io.cache.JsonFileCache` layer, including cache-key
    construction, success / failure TTL selection, batch dedup +
    ordering, and the entity dict ``(de)serialization`` used by the
    on-disk cache.

Out of scope (intentionally retained elsewhere):
    * WikidataClient implementations (see
      :mod:`enrichment.wikidata.transport`).
    * Parsing helpers (see :mod:`enrichment.wikidata.parsing`).
    * HTTP transport mechanics (see
      :mod:`enrichment.wikimedia.transport`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from osm_polygon_wikidata_only.io.cache import JsonFileCache

from .models import WikidataClient, WikidataEntity
from .parsing import is_valid_qid
from .transport import HttpWikidataClient


class CachedWikidataClient(WikidataClient):
    """Wrap another client and serve responses from a local cache.

    Successful results are stored with the configured TTL. Only an
    authoritative Wikidata ``missing`` result is stored as
    ``status="not_found"`` with the shorter ``failed_ttl_s``. Legacy
    ambiguous error records are cache misses and transport failures are
    never cached.
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
        self._last_batch_cache_hits = 0

    @property
    def last_batch_cache_hits(self) -> int:
        return self._last_batch_cache_hits

    def get_entity(self, qid: str) -> WikidataEntity | None:
        return self.get_entities([qid])[0]

    def get_entities(self, qids: Iterable[str]) -> list[WikidataEntity | None]:
        """Resolve cache misses together while preserving input order."""
        requested = list(qids)
        resolved: dict[str, WikidataEntity | None] = {}
        misses: list[str] = []
        cache_hits = 0
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
                cache_hits += 1
            elif hit.status == "not_found":
                resolved[qid] = None
                cache_hits += 1
            else:
                misses.append(qid)

        self._last_batch_cache_hits = cache_hits
        batch_get = getattr(self._inner, "get_entities", None)
        if not misses:
            fetched: list[WikidataEntity | None] = []
        elif callable(batch_get):
            fetched = batch_get(misses)
        else:
            fetched = [self._inner.get_entity(qid) for qid in misses]
        for qid, entity in zip(misses, fetched, strict=True):
            key = f"wikidata/{qid}.json"
            if entity is None:
                self._cache.set(
                    key,
                    payload=None,
                    status="not_found",
                    ttl_s=self._failed_ttl_s,
                    response_metadata={"reason": "wikidata_entity_missing"},
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


__all__ = ["CachedWikidataClient"]
