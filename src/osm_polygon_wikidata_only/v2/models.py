"""Small immutable V2 identity models."""

from __future__ import annotations

from dataclasses import dataclass


def document_id(wikidata: str | None, language: str, page_id: int, revision_id: int) -> str:
    """Build a stable identity for a QID-backed or direct-only page."""
    prefix = wikidata or "osm"
    return f"{prefix}:wikipedia:{language}:{page_id}:{revision_id}"


def article_id(wikidata: str | None, language: str, page_id: int, revision_id: int) -> str:
    """Build the V2 article identity paired with :func:`document_id`."""
    prefix = wikidata or "osm"
    return f"{prefix}:{language}:{page_id}:{revision_id}"


@dataclass(frozen=True, slots=True)
class V2DocumentIdentity:
    """Stable identity fields used to reuse or deduplicate a page."""

    document_id: str
    article_id: str
    wikidata: str | None
    language: str
    page_id: int
    revision_id: int


@dataclass(frozen=True, slots=True)
class V2LinkIdentity:
    """One polygon-to-document relationship and its discovery sources."""

    polygon_id: str
    document_id: str
    link_sources: tuple[str, ...]


__all__ = [
    "V2DocumentIdentity",
    "V2LinkIdentity",
    "article_id",
    "document_id",
]
