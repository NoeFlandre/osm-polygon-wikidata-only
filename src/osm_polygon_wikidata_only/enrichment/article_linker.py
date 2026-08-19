"""High-level orchestrator: QID -> linked Wikipedia articles.

Given a Wikidata QID, this module:

1. Asks the :class:`WikidataClient` for the entity.
2. Selects the available Wikipedia sitelinks (filtered by an optional
   language allow-list).
3. Asks the :class:`WikipediaClient` to fetch each article.
4. Returns a per-QID summary that the processor can turn into
   ``Article`` and ``PolygonArticleLink`` rows.

The linker is intentionally test-friendly: any client conforming to
the abstract interface can be plugged in.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .progress import EnrichmentProgress
from .wikidata_client import (
    BatchWikidataClient,
    WikidataClient,
    WikidataEntity,
    is_valid_qid,
    language_from_site,
)
from .wikipedia_client import BatchWikipediaClient, FetchResult, WikipediaArticle, WikipediaClient

LOGGER = logging.getLogger(__name__)


PREFERRED_LANGUAGES: tuple[str, ...] = ("en", "fr", "de", "es", "it")
DEFAULT_BATCH_SIZE = 50
DEFAULT_SITE_WORKERS = 5

type SiteKey = tuple[str, str]
type SiteRequest = tuple[int, str, str]
type SiteWork = tuple[SiteKey, list[str]]


@dataclass
class LinkSummary:
    """The per-QID result of :func:`link_qid`."""

    qid: str
    entity: WikidataEntity | None
    articles: list[WikipediaArticle] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)  # site -> status
    errors: dict[str, str] = field(default_factory=dict)  # site -> error message

    @property
    def has_any_article(self) -> bool:
        return any(self.articles)

    def best_language(self, preference: Iterable[str] = PREFERRED_LANGUAGES) -> str:
        """Pick a deterministic preferred language from the loaded articles.

        Iterates ``preference`` first, then falls back to the
        lexicographically smallest available article language.
        """
        available = {a.language for a in self.articles}
        for lang in preference:
            if lang in available:
                return lang
        return min(available) if available else ""


def link_qid(
    qid: str,
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    languages: Iterable[str] | None = None,
    fetch_full_text: bool = True,
) -> LinkSummary:
    """Resolve ``qid`` to a list of :class:`WikipediaArticle` instances.

    Parameters
    ----------
    qid:
        Wikidata identifier (e.g. ``Q42``).
    wikidata_client:
        Any :class:`WikidataClient`.
    wikipedia_client:
        Any :class:`WikipediaClient`.
    languages:
        Optional allow-list of language codes. ``None`` means
        "fetch every available sitelink".
    fetch_full_text:
        Passed to :meth:`WikipediaClient.fetch_article`. ``False`` means
        "lead + extract only".
    """
    if not is_valid_qid(qid):
        return LinkSummary(qid=qid, entity=None)
    entity = wikidata_client.get_entity(qid)
    if entity is None:
        return LinkSummary(qid=qid, entity=None)

    summary = LinkSummary(qid=qid, entity=entity)
    allow = {lang for lang in languages} if languages is not None else None
    _link_entity_articles(
        summary,
        wikipedia_client,
        allow=allow,
        fetch_full_text=fetch_full_text,
    )
    return summary


def _site_allowed(site: str, allow: set[str] | None) -> bool:
    return allow is None or language_from_site(site) in allow


def _link_entity_articles(
    summary: LinkSummary,
    wikipedia_client: WikipediaClient,
    *,
    allow: set[str] | None,
    fetch_full_text: bool,
) -> None:
    entity = summary.entity
    assert entity is not None

    for site, title in sorted(entity.sitelinks.items()):
        language = language_from_site(site)
        if not _site_allowed(site, allow):
            continue
        result = _fetch_entity_article(
            wikipedia_client,
            entity,
            language,
            site,
            title,
            fetch_full_text=fetch_full_text,
        )
        _record_link_result(summary, site, result)


def _fetch_entity_article(
    wikipedia_client: WikipediaClient,
    entity: WikidataEntity,
    language: str,
    site: str,
    title: str,
    *,
    fetch_full_text: bool,
) -> FetchResult:
    return wikipedia_client.fetch_article(
        language,
        site,
        title,
        wikidata_label=entity.labels.get(language) or entity.labels.get("en", ""),
        wikidata_description=entity.descriptions.get(language) or entity.descriptions.get("en", ""),
        wikidata_aliases=entity.aliases.get(language) or entity.aliases.get("en", []),
        fetch_full_text=fetch_full_text,
    )


def _record_link_result(summary: LinkSummary, site: str, result: FetchResult) -> None:
    summary.statuses[site] = result.status
    if result.status == "article_not_found" or result.article is None:
        summary.errors[site] = result.error
        return
    summary.articles.append(result.article)
    if result.status != "ok":
        summary.errors[site] = result.error


def fetch_qids(
    qids: Iterable[str],
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    languages: Iterable[str] | None = None,
    fetch_full_text: bool = True,
    max_articles_per_qid: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    site_workers: int = DEFAULT_SITE_WORKERS,
    progress: EnrichmentProgress | None = None,
) -> list[LinkSummary]:
    """Fetch and link several QIDs, returning one :class:`LinkSummary` each."""
    _validate_fetch_options(batch_size, site_workers)
    requested = list(qids)
    if progress is not None:
        progress.set_qids_total(len(requested))
    if isinstance(wikidata_client, BatchWikidataClient) and isinstance(
        wikipedia_client, BatchWikipediaClient
    ):
        return _fetch_batched_qids(
            requested,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            languages=languages,
            fetch_full_text=fetch_full_text,
            max_articles_per_qid=max_articles_per_qid,
            batch_size=batch_size,
            site_workers=site_workers,
            progress=progress,
        )
    return _fetch_compatibility_qids(
        requested,
        wikidata_client=wikidata_client,
        wikipedia_client=wikipedia_client,
        languages=languages,
        fetch_full_text=fetch_full_text,
        max_articles_per_qid=max_articles_per_qid,
        progress=progress,
    )


def _validate_fetch_options(batch_size: int, site_workers: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if site_workers < 1:
        raise ValueError("site_workers must be >= 1")


def _fetch_batched_entities(
    requested: list[str],
    client: BatchWikidataClient,
    *,
    batch_size: int,
    progress: EnrichmentProgress | None,
) -> list[LinkSummary]:
    entities: list[WikidataEntity | None] = []
    for chunk in _chunks(requested, batch_size):
        entities.extend(client.get_entities(chunk))
        if progress is not None:
            progress.advance_qids(len(chunk))
    return [
        LinkSummary(qid=qid, entity=entity) for qid, entity in zip(requested, entities, strict=True)
    ]


def _build_site_requests(
    summaries: list[LinkSummary],
    languages: Iterable[str] | None,
) -> tuple[dict[SiteKey, list[SiteRequest]], set[str] | None]:
    requests: dict[SiteKey, list[SiteRequest]] = {}
    allow = {lang for lang in languages} if languages is not None else None
    for index, summary in enumerate(summaries):
        for key, rows in _summary_site_requests(index, summary, allow).items():
            requests.setdefault(key, []).extend(rows)
    return requests, allow


def _summary_site_requests(
    index: int,
    summary: LinkSummary,
    allow: set[str] | None,
) -> dict[SiteKey, list[SiteRequest]]:
    if summary.entity is None:
        return {}
    requests: dict[SiteKey, list[SiteRequest]] = {}
    for site, title in sorted(summary.entity.sitelinks.items()):
        if _site_allowed(site, allow):
            key = (language_from_site(site), site)
            requests.setdefault(key, []).append((index, site, title))
    return requests


def _plan_site_work(
    requests: dict[SiteKey, list[SiteRequest]],
    *,
    batch_size: int,
) -> tuple[list[SiteWork], dict[SiteKey, int]]:
    site_titles = _unique_site_titles(requests)
    site_chunks = _chunk_site_titles(site_titles, batch_size)
    return _round_robin_site_work(site_chunks)


def _unique_site_titles(
    requests: dict[SiteKey, list[SiteRequest]],
) -> dict[SiteKey, list[str]]:
    return {
        key: list(dict.fromkeys(title for _, _, title in rows)) for key, rows in requests.items()
    }


def _chunk_site_titles(
    site_titles: dict[SiteKey, list[str]],
    batch_size: int,
) -> dict[SiteKey, list[list[str]]]:
    return {key: list(_chunks(titles, batch_size)) for key, titles in site_titles.items()}


def _round_robin_site_work(
    site_chunks: dict[SiteKey, list[list[str]]],
) -> tuple[list[SiteWork], dict[SiteKey, int]]:
    work: list[SiteWork] = []
    for index in range(max(map(len, site_chunks.values()), default=0)):
        work.extend(
            (key, chunks[index]) for key, chunks in site_chunks.items() if index < len(chunks)
        )
    return work, {key: len(chunks) for key, chunks in site_chunks.items()}


def _fetch_site_work(
    work: list[SiteWork],
    requests: dict[SiteKey, list[SiteRequest]],
    chunks_remaining: dict[SiteKey, int],
    client: BatchWikipediaClient,
    *,
    fetch_full_text: bool,
    site_workers: int,
    progress: EnrichmentProgress | None,
) -> dict[SiteKey, dict[str, FetchResult]]:
    remaining = dict(chunks_remaining)
    fetched: dict[SiteKey, dict[str, FetchResult]] = {key: {} for key in requests}
    with ThreadPoolExecutor(max_workers=min(site_workers, max(1, len(work)))) as executor:
        futures = {
            executor.submit(_fetch_one_site_chunk, client, key, titles, fetch_full_text): len(
                titles
            )
            for key, titles in work
        }
        for future in as_completed(futures):
            _consume_site_future(future, futures, fetched, remaining, progress)
    return fetched


def _fetch_one_site_chunk(
    client: BatchWikipediaClient,
    key: SiteKey,
    titles: list[str],
    fetch_full_text: bool,
) -> tuple[SiteKey, dict[str, FetchResult]]:
    language, site = key
    return key, client.fetch_articles(language, site, titles, fetch_full_text=fetch_full_text)


def _consume_site_future(
    future: Future[tuple[SiteKey, dict[str, FetchResult]]],
    futures: dict[Future[tuple[SiteKey, dict[str, FetchResult]]], int],
    fetched: dict[SiteKey, dict[str, FetchResult]],
    chunks_remaining: dict[SiteKey, int],
    progress: EnrichmentProgress | None,
) -> None:
    key, chunk_results = future.result()
    fetched[key].update(chunk_results)
    if progress is not None:
        progress.advance_articles(futures[future])
    chunks_remaining[key] -= 1
    if chunks_remaining[key] == 0 and progress is not None:
        progress.complete_site(0)


def _apply_article_result(
    summary: LinkSummary,
    site: str,
    result: FetchResult,
) -> None:
    summary.statuses[site] = result.status
    if result.status != "article_not_found" and result.article is not None:
        summary.articles.append(result.article)
        if result.status != "ok":
            summary.errors[site] = result.error
    else:
        summary.errors[site] = result.error


def _populate_batched_summaries(
    summaries: list[LinkSummary],
    fetched: dict[SiteKey, dict[str, FetchResult]],
    allow: set[str] | None,
    *,
    max_articles_per_qid: int | None,
) -> None:
    for summary in summaries:
        _populate_one_summary(summary, fetched, allow, max_articles_per_qid)


def _populate_one_summary(
    summary: LinkSummary,
    fetched: dict[SiteKey, dict[str, FetchResult]],
    allow: set[str] | None,
    max_articles_per_qid: int | None,
) -> None:
    entity = summary.entity
    if entity is None:
        return
    for site, title in sorted(entity.sitelinks.items()):
        if not _site_allowed(site, allow):
            continue
        language = language_from_site(site)
        _apply_article_result(summary, site, fetched[(language, site)][title])
    if max_articles_per_qid is not None:
        summary.articles = summary.articles[:max_articles_per_qid]


def _fetch_batched_qids(
    requested: list[str],
    *,
    wikidata_client: BatchWikidataClient,
    wikipedia_client: BatchWikipediaClient,
    languages: Iterable[str] | None,
    fetch_full_text: bool,
    max_articles_per_qid: int | None,
    batch_size: int,
    site_workers: int,
    progress: EnrichmentProgress | None,
) -> list[LinkSummary]:
    summaries = _fetch_batched_entities(
        requested, wikidata_client, batch_size=batch_size, progress=progress
    )
    requests, allow = _build_site_requests(summaries, languages)
    if progress is not None:
        progress.start_wikipedia(len(requests))
    work, chunks_remaining = _plan_site_work(requests, batch_size=batch_size)
    fetched = _fetch_site_work(
        work,
        requests,
        chunks_remaining,
        wikipedia_client,
        fetch_full_text=fetch_full_text,
        site_workers=site_workers,
        progress=progress,
    )
    _populate_batched_summaries(
        summaries,
        fetched,
        allow,
        max_articles_per_qid=max_articles_per_qid,
    )
    return summaries


def _fetch_compatibility_qids(
    requested: list[str],
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    languages: Iterable[str] | None,
    fetch_full_text: bool,
    max_articles_per_qid: int | None,
    progress: EnrichmentProgress | None,
) -> list[LinkSummary]:
    out: list[LinkSummary] = []
    for qid in requested:
        summary = link_qid(
            qid,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            languages=languages,
            fetch_full_text=fetch_full_text,
        )
        if progress is not None:
            progress.advance_qids()
        if max_articles_per_qid is not None:
            summary.articles = summary.articles[:max_articles_per_qid]
        out.append(summary)
    return out


def _chunks[T](items: list[T], size: int) -> Iterable[list[T]]:
    """Yield stable, bounded list chunks without copying the full input again."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SITE_WORKERS",
    "PREFERRED_LANGUAGES",
    "LinkSummary",
    "fetch_qids",
    "link_qid",
]
