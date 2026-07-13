"""Cached Wikipedia client and article serialization helpers.

Responsibility:
    Wrap any :class:`WikipediaClient` with the
    :class:`io.cache.JsonFileCache` layer, including cache-key
    construction, success / failure TTL selection, and the
    article dict ``(de)serialization`` used by the on-disk cache.

Out of scope (intentionally retained elsewhere):
    * WikipediaClient implementations (see
      :mod:`enrichment.wikipedia.transport`).
    * Parsing helpers (see :mod:`enrichment.wikipedia.parsing`).
    * HTTP transport mechanics (see
      :mod:`enrichment.wikimedia.transport`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from osm_polygon_wikidata_only.io.cache import JsonFileCache

from .models import FetchResult, WikipediaArticle, WikipediaClient
from .transport import HttpWikipediaClient


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


__all__ = ["CachedWikipediaClient"]
