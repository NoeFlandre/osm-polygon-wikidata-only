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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

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

    for site, title in sorted(entity.sitelinks.items()):
        language = language_from_site(site)
        if allow is not None and language not in allow:
            continue
        result = wikipedia_client.fetch_article(
            language,
            site,
            title,
            wikidata_label=entity.labels.get(language) or entity.labels.get("en", ""),
            wikidata_description=entity.descriptions.get(language)
            or entity.descriptions.get("en", ""),
            wikidata_aliases=entity.aliases.get(language) or entity.aliases.get("en", []),
            fetch_full_text=fetch_full_text,
        )
        summary.statuses[site] = result.status
        if result.status != "ok" or result.article is None:
            summary.errors[site] = result.error
            continue
        summary.articles.append(result.article)

    return summary


def fetch_qids(
    qids: Iterable[str],
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    languages: Iterable[str] | None = None,
    fetch_full_text: bool = True,
    max_articles_per_qid: int | None = None,
) -> list[LinkSummary]:
    """Fetch and link several QIDs, returning one :class:`LinkSummary` each."""
    requested = list(qids)
    if isinstance(wikidata_client, BatchWikidataClient) and isinstance(
        wikipedia_client, BatchWikipediaClient
    ):
        entities = wikidata_client.get_entities(requested)
        summaries = [
            LinkSummary(qid=qid, entity=entity)
            for qid, entity in zip(requested, entities, strict=True)
        ]
        requests: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
        allow = {lang for lang in languages} if languages is not None else None
        for index, summary in enumerate(summaries):
            if summary.entity is None:
                continue
            for site, title in sorted(summary.entity.sitelinks.items()):
                language = language_from_site(site)
                if allow is None or language in allow:
                    requests.setdefault((language, site), []).append((index, site, title))

        def fetch_site(
            key: tuple[str, str], rows: list[tuple[int, str, str]]
        ) -> tuple[tuple[str, str], dict[str, FetchResult]]:
            language, site = key
            titles = list(dict.fromkeys(title for _, _, title in rows))
            return key, wikipedia_client.fetch_articles(
                language, site, titles, fetch_full_text=fetch_full_text
            )

        fetched: dict[tuple[str, str], dict[str, FetchResult]] = {}
        with ThreadPoolExecutor(max_workers=min(5, max(1, len(requests)))) as executor:
            for key, site_results in executor.map(lambda item: fetch_site(*item), requests.items()):
                fetched[key] = site_results
        for summary in summaries:
            entity = summary.entity
            if entity is None:
                continue
            for site, title in sorted(entity.sitelinks.items()):
                language = language_from_site(site)
                if allow is not None and language not in allow:
                    continue
                article_result = fetched[(language, site)][title]
                summary.statuses[site] = article_result.status
                if article_result.status == "ok" and article_result.article is not None:
                    summary.articles.append(article_result.article)
                else:
                    summary.errors[site] = article_result.error
            if max_articles_per_qid is not None:
                summary.articles = summary.articles[:max_articles_per_qid]
        return summaries

    out: list[LinkSummary] = []
    for qid in requested:
        summary = link_qid(
            qid,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            languages=languages,
            fetch_full_text=fetch_full_text,
        )
        if max_articles_per_qid is not None:
            summary.articles = summary.articles[:max_articles_per_qid]
        out.append(summary)
    return out


__all__ = [
    "PREFERRED_LANGUAGES",
    "LinkSummary",
    "fetch_qids",
    "link_qid",
]
