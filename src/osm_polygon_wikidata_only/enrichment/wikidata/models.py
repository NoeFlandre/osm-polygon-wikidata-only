"""Typed contracts and value objects for Wikidata enrichment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

Sitelinks = dict[str, str]


@dataclass(frozen=True)
class WikidataEntity:
    """Wikidata fields required for multilingual article linking."""

    qid: str
    sitelinks: Sitelinks = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)


class WikidataClient(ABC):
    """Stable single-entity client contract used by the pipeline."""

    @abstractmethod
    def get_entity(self, qid: str) -> WikidataEntity | None: ...


@runtime_checkable
class BatchWikidataClient(Protocol):
    """Optional capability for resolving several QIDs in one request."""

    def get_entities(self, qids: Iterable[str]) -> list[WikidataEntity | None]: ...


__all__ = ["BatchWikidataClient", "Sitelinks", "WikidataClient", "WikidataEntity"]
