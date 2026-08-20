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
    initial_matches: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]] | None = None,
    defer_final_lookup: bool = False,
) -> DirectEnrichmentResult:
    """Resolve direct references, optionally without waiting for V1 indexing.

    The speculative mode is used by the V2 merger while the persistent V1
    reuse index is still scanning.  Its results must be reconciled with the
    completed index before they are persisted because an index miss is not
    authoritative until every V1 shard has been checked.
    """
    context = _polygon_context(polygon_context)
    client = _cached_client(wikipedia_client, cache)
    documents: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    statuses: dict[int, DirectWikipediaStatus] = {}
    initial_matches = _provided_or_lookup(index, refs, initial_matches)
    pending = _partition_references(
        polygon_id,
        refs,
        initial_matches,
        context,
        documents,
        links,
        statuses,
    )

    # A miss is not authoritative until the background index has finished.
    # Fetch unresolved direct pages speculatively while the index continues so
    # this region does not sit idle.  The final lookup below always wins, so a
    # page discovered in a later V1 shard is still reused and the speculative
    # response is discarded.
    speculative = _speculative_fetches(
        index,
        pending,
        client,
        fetch_full_text=fetch_full_text,
        wait_for_index=wait_for_index,
    )

    final_matches = _final_matches(index, pending, defer_final_lookup)
    deferred_errors = _resolve_pending(
        polygon_id,
        pending,
        final_matches,
        speculative,
        client,
        context,
        fetch_full_text,
        wait_for_index,
        documents,
        links,
        statuses,
    )

    return DirectEnrichmentResult(
        documents=tuple(documents.values()),
        links=tuple(links.values()),
        statuses=tuple(statuses[position] for position in range(len(refs))),
        deferred_errors=tuple(sorted(deferred_errors.items())),
    )


def _partition_references(
    polygon_id: str,
    refs: Sequence[WikipediaTagRef],
    matches: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]],
    context: Mapping[str, Any],
    documents: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
    statuses: dict[int, DirectWikipediaStatus],
) -> list[tuple[int, WikipediaTagRef]]:
    pending: list[tuple[int, WikipediaTagRef]] = []
    for position, ref in enumerate(refs):
        existing = matches.get(_title_key(ref.language, ref.title), ())
        if existing:
            _record_reuse(
                polygon_id,
                position,
                ref,
                existing[0],
                context,
                documents,
                links,
                statuses,
            )
        else:
            pending.append((position, ref))
    return pending


def _polygon_context(polygon_context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return polygon_context or {}


def _provided_or_lookup(
    index: V1ReuseIndex,
    refs: Sequence[WikipediaTagRef],
    provided: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]] | None,
) -> Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]]:
    return _lookup_titles(index, refs) if provided is None else provided


def _final_matches(
    index: V1ReuseIndex,
    pending: Sequence[tuple[int, WikipediaTagRef]],
    defer_lookup: bool,
) -> Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]]:
    if defer_lookup:
        return {}
    return _lookup_titles(index, [ref for _position, ref in pending])


def _record_reuse(
    polygon_id: str,
    position: int,
    ref: WikipediaTagRef,
    document: Mapping[str, Any],
    context: Mapping[str, Any],
    documents: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
    statuses: dict[int, DirectWikipediaStatus],
) -> None:
    row = dict(document)
    key = str(row["document_id"])
    documents[key] = row
    links[key] = _link_row(
        polygon_id,
        row,
        polygon_context=context,
        sources=("osm_wikipedia_tag",),
    )
    statuses[position] = DirectWikipediaStatus(ref, "reused_v1", reused_v1=True)


def _speculative_fetches(
    index: V1ReuseIndex,
    pending: Sequence[tuple[int, WikipediaTagRef]],
    client: WikipediaClient,
    *,
    fetch_full_text: bool,
    wait_for_index: bool,
) -> dict[int, FetchResult | Exception]:
    if not pending or index.is_ready:
        return {}
    LOGGER.info(
        "V2 direct enrichment fetching %d unresolved title(s) while the V1 index continues",
        len(pending),
    )
    speculative = _fetch_speculative_results(index, pending, client, fetch_full_text)
    if wait_for_index:
        LOGGER.info(
            "V2 direct enrichment waiting for final V1 index state after %d speculative fetch(es)",
            len(speculative),
        )
        index.wait_until_ready()
        LOGGER.info("V2 direct enrichment resumed after V1 reuse index completion")
    return speculative


def _fetch_speculative_results(
    index: V1ReuseIndex,
    pending: Sequence[tuple[int, WikipediaTagRef]],
    client: WikipediaClient,
    fetch_full_text: bool,
) -> dict[int, FetchResult | Exception]:
    speculative: dict[int, FetchResult | Exception] = {}
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
            # Keep the exception until the completed index confirms that the
            # title cannot be reused from a later V1 shard.
            speculative[position] = exc
    return speculative


def _resolve_pending(
    polygon_id: str,
    pending: Sequence[tuple[int, WikipediaTagRef]],
    final_matches: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]],
    speculative: Mapping[int, FetchResult | Exception],
    client: WikipediaClient,
    context: Mapping[str, Any],
    fetch_full_text: bool,
    wait_for_index: bool,
    documents: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
    statuses: dict[int, DirectWikipediaStatus],
) -> dict[int, Exception]:
    deferred_errors: dict[int, Exception] = {}
    for position, ref in pending:
        existing = final_matches.get(_title_key(ref.language, ref.title), ())
        if existing:
            _record_reuse(
                polygon_id,
                position,
                ref,
                existing[0],
                context,
                documents,
                links,
                statuses,
            )
            continue
        outcome = _resolve_pending_outcome(
            position,
            ref,
            speculative.get(position),
            client,
            fetch_full_text=fetch_full_text,
            wait_for_index=wait_for_index,
        )
        _apply_pending_outcome(
            polygon_id,
            position,
            outcome,
            ref,
            context,
            documents,
            links,
            statuses,
            deferred_errors,
        )
    return deferred_errors


def _resolve_pending_outcome(
    position: int,
    ref: WikipediaTagRef,
    result_or_error: FetchResult | Exception | None,
    client: WikipediaClient,
    *,
    fetch_full_text: bool,
    wait_for_index: bool,
) -> tuple[DirectWikipediaStatus, FetchResult | None, Exception | None]:
    if isinstance(result_or_error, Exception):
        if wait_for_index:
            raise result_or_error
        return (
            DirectWikipediaStatus(ref, "deferred_error", str(result_or_error)),
            None,
            result_or_error,
        )
    result = result_or_error or client.fetch_article(
        ref.language,
        f"{ref.language}wiki",
        ref.title,
        fetch_full_text=fetch_full_text,
    )
    return DirectWikipediaStatus(ref, result.status, result.error), result, None


def _apply_pending_outcome(
    polygon_id: str,
    position: int,
    outcome: tuple[DirectWikipediaStatus, FetchResult | None, Exception | None],
    ref: WikipediaTagRef,
    context: Mapping[str, Any],
    documents: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
    statuses: dict[int, DirectWikipediaStatus],
    deferred_errors: dict[int, Exception],
) -> None:
    status, result, deferred_error = outcome
    statuses[position] = status
    if deferred_error is not None:
        deferred_errors[position] = deferred_error
    if result is None or result.article is None:
        return
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


def _title_key(language: str, title: str) -> tuple[str, str]:
    return language.casefold(), title.replace("_", " ").casefold()


def _lookup_titles(
    index: Any,
    refs: Sequence[WikipediaTagRef],
) -> dict[tuple[str, str], tuple[Mapping[str, Any], ...]]:
    """Resolve references in one batch when the index supports it."""
    keys = tuple(dict.fromkeys((ref.language, ref.title) for ref in refs))
    if not keys:
        return {}
    batch_lookup = getattr(index, "by_titles", None)
    if callable(batch_lookup):
        return batch_lookup(keys)
    return {
        _title_key(language, title): index.by_title(language, title) for language, title in keys
    }


def reconcile_wikipedia_refs(
    polygon_id: str,
    refs: Sequence[WikipediaTagRef],
    result: DirectEnrichmentResult,
    *,
    index: V1ReuseIndex,
    polygon_context: Mapping[str, Any] | None = None,
    title_matches: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]] | None = None,
) -> DirectEnrichmentResult:
    """Prefer final V1 rows over speculative direct results.

    ``result`` must have been produced with ``wait_for_index=False`` and the
    caller must wait for the index before invoking this function.  Any
    deferred fetch error is raised only when the completed index confirms
    that the requested page is not available for reuse.
    """
    context = _polygon_context(polygon_context)
    direct_by_title = _direct_documents_by_title(result.documents)
    deferred_errors = dict(result.deferred_errors)
    documents: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    statuses: list[DirectWikipediaStatus] = []
    final_matches = _provided_or_lookup(index, refs, title_matches)
    for position, ref in enumerate(refs):
        status, row = _reconcile_reference(
            position,
            ref,
            result,
            final_matches,
            direct_by_title,
            deferred_errors,
        )
        _store_reconciled_row(polygon_id, row, context, documents, links)
        statuses.append(status)
    return DirectEnrichmentResult(
        documents=tuple(documents.values()),
        links=tuple(links.values()),
        statuses=tuple(statuses),
    )


def _direct_documents_by_title(
    documents: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        key = _title_key(str(document.get("language", "")), str(document.get("title", "")))
        grouped[key].append(dict(document))
    for candidates in grouped.values():
        candidates.sort(key=lambda row: str(row.get("document_id", "")))
    return grouped


def _reconcile_reference(
    position: int,
    ref: WikipediaTagRef,
    result: DirectEnrichmentResult,
    final_matches: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]],
    direct_by_title: Mapping[tuple[str, str], list[dict[str, Any]]],
    deferred_errors: Mapping[int, Exception],
) -> tuple[DirectWikipediaStatus, dict[str, Any] | None]:
    existing = final_matches.get(_title_key(ref.language, ref.title), ())
    if existing:
        return DirectWikipediaStatus(ref, "reused_v1", reused_v1=True), dict(existing[0])
    error = deferred_errors.get(position)
    if error is not None:
        raise error
    candidates = direct_by_title.get(_title_key(ref.language, ref.title), [])
    if not candidates:
        return result.statuses[position], None
    return result.statuses[position], dict(candidates[0])


def _store_reconciled_row(
    polygon_id: str,
    row: dict[str, Any] | None,
    context: Mapping[str, Any],
    documents: dict[str, dict[str, Any]],
    links: dict[str, dict[str, Any]],
) -> None:
    if row is None:
        return
    key = str(row["document_id"])
    documents[key] = row
    links[key] = _link_row(
        polygon_id,
        row,
        polygon_context=context,
        sources=("osm_wikipedia_tag",),
    )


__all__ = [
    "DirectEnrichmentResult",
    "DirectWikipediaStatus",
    "enrich_wikipedia_refs",
    "reconcile_wikipedia_refs",
]
