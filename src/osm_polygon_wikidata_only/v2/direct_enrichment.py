"""Direct Wikipedia-tag enrichment with V1 reuse and durable caching."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from osm_polygon_wikidata_only.enrichment.text_cleaning import count_words, estimate_tokens
from osm_polygon_wikidata_only.enrichment.wikipedia.cache import CachedWikipediaClient
from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult, WikipediaClient
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.v2.config import V2_CACHE_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.models import article_id, document_id
from osm_polygon_wikidata_only.v2.v1_index import V1ReuseIndex
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef


@dataclass(frozen=True, slots=True)
class DirectWikipediaStatus:
    """Terminal state for one normalized direct Wikipedia reference."""

    ref: WikipediaTagRef
    status: str
    error: str = ""
    reused_v1: bool = False


@dataclass(frozen=True, slots=True)
class DirectEnrichmentResult:
    """Documents, links, and statuses produced for one polygon."""

    documents: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    statuses: tuple[DirectWikipediaStatus, ...]


def _link_row(
    polygon_id: str,
    document: Mapping[str, Any],
    *,
    polygon_context: Mapping[str, Any],
    sources: Sequence[str],
) -> dict[str, Any]:
    return {
        "polygon_id": polygon_id,
        "document_id": document["document_id"],
        "project": "wikipedia",
        "wikidata": document.get("wikidata"),
        "language": document["language"],
        "source_pbf": polygon_context.get("source_pbf", ""),
        "region": polygon_context.get("region", ""),
        "osm_type": polygon_context.get("osm_type", ""),
        "osm_id": int(polygon_context.get("osm_id", 0)),
        "page_id": int(document["page_id"]),
        "revision_id": int(document["revision_id"]),
        "link_sources": json_dumps(sorted(set(sources))),
    }


def _article_document(article: Any) -> dict[str, Any]:
    wikidata: str | None = None
    return {
        "document_id": document_id(
            wikidata, article.language, article.page_id, article.revision_id
        ),
        "article_id": article_id(wikidata, article.language, article.page_id, article.revision_id),
        "wikidata": wikidata,
        "project": "wikipedia",
        "language": article.language,
        "site": article.site,
        "title": article.title,
        "url": article.url,
        "page_id": article.page_id,
        "revision_id": article.revision_id,
        "revision_timestamp": article.revision_timestamp,
        "retrieved_at": article.retrieved_at,
        "wikidata_label": "",
        "wikidata_description": "",
        "wikidata_aliases": "[]",
        "lead_text": article.lead_text,
        "extract": article.extract,
        "full_text": article.full_text,
        "full_text_format": article.full_text_format,
        "article_length_chars": len(article.full_text),
        "article_length_words": count_words(article.full_text),
        "article_length_tokens_estimate": estimate_tokens(article.full_text),
        "thumbnail_url": article.thumbnail_url,
        "thumbnail_width": article.thumbnail_width,
        "thumbnail_height": article.thumbnail_height,
        "categories": json_dumps(article.categories),
        "license": article.license,
        "attribution": article.attribution,
        "source_api": article.source_api,
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": hashlib.sha256(article.full_text.encode("utf-8")).hexdigest(),
    }


def _cached_client(
    client: WikipediaClient,
    cache: JsonFileCache | None,
) -> WikipediaClient:
    if cache is None:
        return client
    if cache.contract_version != V2_CACHE_CONTRACT_VERSION:
        raise ValueError("V2 Wikipedia cache has the wrong contract version")
    return CachedWikipediaClient(client, cache)


def enrich_wikipedia_refs(
    polygon_id: str,
    refs: Sequence[WikipediaTagRef],
    *,
    index: V1ReuseIndex,
    wikipedia_client: WikipediaClient,
    polygon_context: Mapping[str, Any] | None = None,
    cache: JsonFileCache | None = None,
    fetch_full_text: bool = True,
) -> DirectEnrichmentResult:
    """Resolve direct references, reusing V1 before making HTTP calls."""
    context = polygon_context or {}
    client = _cached_client(wikipedia_client, cache)
    documents: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    statuses: list[DirectWikipediaStatus] = []

    for ref in refs:
        existing = index.by_title(ref.language, ref.title)
        if existing:
            document = dict(existing[0])
            documents[str(document["document_id"])] = document
            links[str(document["document_id"])] = _link_row(
                polygon_id,
                document,
                polygon_context=context,
                sources=("osm_wikipedia_tag",),
            )
            statuses.append(DirectWikipediaStatus(ref, "reused_v1", reused_v1=True))
            continue

        result: FetchResult = client.fetch_article(
            ref.language,
            f"{ref.language}wiki",
            ref.title,
            fetch_full_text=fetch_full_text,
        )
        if result.article is None:
            statuses.append(DirectWikipediaStatus(ref, result.status, result.error))
            continue
        document = _article_document(result.article)
        if result.status != "ok":
            document["fetch_status"] = result.status
            document["fetch_error"] = result.error
        key = str(document["document_id"])
        documents[key] = document
        links[key] = _link_row(
            polygon_id,
            document,
            polygon_context=context,
            sources=("osm_wikipedia_tag",),
        )
        statuses.append(DirectWikipediaStatus(ref, result.status, result.error))

    return DirectEnrichmentResult(
        documents=tuple(documents.values()),
        links=tuple(links.values()),
        statuses=tuple(statuses),
    )


__all__ = [
    "DirectEnrichmentResult",
    "DirectWikipediaStatus",
    "enrich_wikipedia_refs",
]
