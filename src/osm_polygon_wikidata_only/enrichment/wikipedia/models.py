"""Typed contracts and value objects for Wikipedia enrichment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class WikipediaArticle:
    """One exact Wikipedia article revision and its cleaned text."""

    language: str
    site: str
    title: str
    page_id: int
    revision_id: int
    revision_timestamp: str
    url: str
    lead_text: str
    extract: str
    full_text: str
    full_text_format: str
    thumbnail_url: str
    thumbnail_width: int | None
    thumbnail_height: int | None
    categories: list[str]
    license: str
    attribution: str
    source_api: str
    retrieved_at: str


@dataclass(frozen=True)
class FetchResult:
    """Terminal result of one article fetch attempt."""

    status: str
    article: WikipediaArticle | None
    error: str = ""


class WikipediaClient(ABC):
    """Stable single-article client contract used by the pipeline."""

    @abstractmethod
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
    ) -> FetchResult: ...


@runtime_checkable
class BatchWikipediaClient(Protocol):
    """Optional capability for fetching same-site article batches."""

    def fetch_articles(
        self,
        language: str,
        site: str,
        titles: Iterable[str],
        *,
        fetch_full_text: bool = True,
    ) -> dict[str, FetchResult]: ...


__all__ = ["BatchWikipediaClient", "FetchResult", "WikipediaArticle", "WikipediaClient"]
