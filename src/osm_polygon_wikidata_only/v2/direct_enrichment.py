"""Direct Wikipedia-tag enrichment with V1 reuse and durable caching."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
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

LOGGER = logging.getLogger(__name__)


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
    deferred_errors: tuple[tuple[int, Exception], ...] = ()


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
    wait_for_index: bool = True,
) -> DirectEnrichmentResult:
    """Resolve direct references, optionally without waiting for V1 indexing.

    The speculative mode is used by the V2 merger while the persistent V1
    reuse index is still scanning.  Its results must be reconciled with the
    completed index before they are persisted because an index miss is not
    authoritative until every V1 shard has been checked.
    """
    context = polygon_context or {}
    client = _cached_client(wikipedia_client, cache)
    documents: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    statuses: dict[int, DirectWikipediaStatus] = {}
    pending: list[tuple[int, WikipediaTagRef]] = []
    speculative: dict[int, FetchResult | Exception] = {}
    deferred_errors: dict[int, Exception] = {}

    def reuse(position: int, ref: WikipediaTagRef, document: Mapping[str, Any]) -> None:
        row = dict(document)
        documents[str(row["document_id"])] = row
        links[str(row["document_id"])] = _link_row(
            polygon_id,
            row,
            polygon_context=context,
            sources=("osm_wikipedia_tag",),
        )
        statuses[position] = DirectWikipediaStatus(ref, "reused_v1", reused_v1=True)

    for position, ref in enumerate(refs):
        existing = index.by_title(ref.language, ref.title)
        if existing:
            reuse(position, ref, existing[0])
            continue
        pending.append((position, ref))

    # A miss is not authoritative until the background index has finished.
    # Fetch unresolved direct pages speculatively while the index continues so
    # this region does not sit idle.  The final lookup below always wins, so a
    # page discovered in a later V1 shard is still reused and the speculative
    # response is discarded.
    if pending and not index.is_ready:
        LOGGER.info(
            "V2 direct enrichment fetching %d unresolved title(s) while the V1 index continues",
            len(pending),
        )
        for position, ref in pending:
            if index.is_ready:
                break
            try:
                speculative[position] = client.fetch_article(
                    ref.language,
                    f"{ref.language}wiki",
                    ref.title,
                    fetch_full_text=fetch_full_text,
                )
            except Exception as exc:
                # Do not fail early for a page that may still be found in a
                # later V1 shard.  If it remains absent after the final index
                # check, re-raise the original exception below.
                speculative[position] = exc
        if wait_for_index:
            LOGGER.info(
                "V2 direct enrichment waiting for final V1 index state after %d speculative fetch(es)",
                len(speculative),
            )
            index.wait_until_ready()
            LOGGER.info("V2 direct enrichment resumed after V1 reuse index completion")

    for position, ref in pending:
        existing = index.by_title(ref.language, ref.title)
        if existing:
            reuse(position, ref, existing[0])
            continue
        result_or_error = speculative.get(position)
        if isinstance(result_or_error, Exception):
            if wait_for_index:
                raise result_or_error
            deferred_errors[position] = result_or_error
            statuses[position] = DirectWikipediaStatus(
                ref,
                "deferred_error",
                str(result_or_error),
            )
            continue
        result = result_or_error
        if result is None:
            result = client.fetch_article(
                ref.language,
                f"{ref.language}wiki",
                ref.title,
                fetch_full_text=fetch_full_text,
            )
        if result.article is None:
            statuses[position] = DirectWikipediaStatus(ref, result.status, result.error)
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
        statuses[position] = DirectWikipediaStatus(ref, result.status, result.error)

    return DirectEnrichmentResult(
        documents=tuple(documents.values()),
        links=tuple(links.values()),
        statuses=tuple(statuses[position] for position in range(len(refs))),
        deferred_errors=tuple(sorted(deferred_errors.items())),
    )


def _title_key(language: str, title: str) -> tuple[str, str]:
    return language.casefold(), title.replace("_", " ").casefold()


def reconcile_wikipedia_refs(
    polygon_id: str,
    refs: Sequence[WikipediaTagRef],
    result: DirectEnrichmentResult,
    *,
    index: V1ReuseIndex,
    polygon_context: Mapping[str, Any] | None = None,
) -> DirectEnrichmentResult:
    """Prefer final V1 rows over speculative direct results.

    ``result`` must have been produced with ``wait_for_index=False`` and the
    caller must wait for the index before invoking this function.  Any
    deferred fetch error is raised only when the completed index confirms
    that the requested page is not available for reuse.
    """
    context = polygon_context or {}
    direct_by_title: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for document in result.documents:
        direct_by_title.setdefault(
            _title_key(str(document.get("language", "")), str(document.get("title", ""))),
            [],
        ).append(document)
    for documents in direct_by_title.values():
        documents.sort(key=lambda row: str(row.get("document_id", "")))
    deferred_errors = dict(result.deferred_errors)
    documents: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    statuses: list[DirectWikipediaStatus] = []
    for position, ref in enumerate(refs):
        existing = index.by_title(ref.language, ref.title)
        if existing:
            row = dict(existing[0])
            key = str(row["document_id"])
            documents[key] = row
            links[key] = _link_row(
                polygon_id,
                row,
                polygon_context=context,
                sources=("osm_wikipedia_tag",),
            )
            statuses.append(DirectWikipediaStatus(ref, "reused_v1", reused_v1=True))
            continue
        error = deferred_errors.get(position)
        if error is not None:
            raise error
        candidates = direct_by_title.get(_title_key(ref.language, ref.title), [])
        if not candidates:
            prior = result.statuses[position]
            statuses.append(prior)
            continue
        row = dict(candidates[0])
        key = str(row["document_id"])
        documents[key] = row
        links[key] = _link_row(
            polygon_id,
            row,
            polygon_context=context,
            sources=("osm_wikipedia_tag",),
        )
        statuses.append(result.statuses[position])
    return DirectEnrichmentResult(
        documents=tuple(documents.values()),
        links=tuple(links.values()),
        statuses=tuple(statuses),
    )


__all__ = [
    "DirectEnrichmentResult",
    "DirectWikipediaStatus",
    "enrich_wikipedia_refs",
    "reconcile_wikipedia_refs",
]
