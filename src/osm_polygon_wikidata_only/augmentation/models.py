"""Typed rows and deterministic identifiers for augmentation sidecars."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


def document_id(wikidata: str, project: str, language: str, page_id: int, revision_id: int) -> str:
    return f"{wikidata}:{project}:{language}:{page_id}:{revision_id}"


def stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def _as_int(value: object) -> int:
    return int(str(value))


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    article_id: str
    wikidata: str
    project: str
    language: str
    site: str
    title: str
    url: str
    page_id: int
    revision_id: int
    revision_timestamp: str
    retrieved_at: str
    full_text: str
    full_text_format: str
    article_length_chars: int
    article_length_words: int
    article_length_tokens_estimate: int
    license: str
    attribution: str
    source_api: str
    fetch_status: str
    fetch_error: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Section:
    section_id: str
    document_id: str
    article_id: str
    wikidata: str
    project: str
    language: str
    site: str
    page_id: int
    revision_id: int
    section_index: int
    heading: str
    anchor: str
    level: int
    parent_section_id: str
    section_path: str
    text: str
    text_length_chars: int
    text_length_words: int
    text_length_tokens_estimate: int
    content_hash: str
    license: str
    attribution: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WikidataFact:
    fact_id: str
    wikidata: str
    property_id: str
    property_label_en: str
    property_labels: str
    value_type: str
    value_entity_id: str
    value_label_en: str
    value_labels: str
    value_text: str
    numeric_value: float | None
    unit_entity_id: str
    rank: str
    qualifiers: str
    references: str
    retrieved_at: str
    source_api: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def document_from_article_row(row: dict[str, object]) -> Document:
    wikidata = str(row["wikidata"])
    language = str(row["language"])
    page_id = _as_int(row["page_id"])
    revision_id = _as_int(row["revision_id"])
    return Document(
        document_id=document_id(wikidata, "wikipedia", language, page_id, revision_id),
        article_id=str(row["article_id"]),
        wikidata=wikidata,
        project="wikipedia",
        language=language,
        site=str(row["site"]),
        title=str(row["title"]),
        url=str(row["url"]),
        page_id=page_id,
        revision_id=revision_id,
        revision_timestamp=str(row["revision_timestamp"]),
        retrieved_at=str(row["retrieved_at"]),
        full_text=str(row["full_text"]),
        full_text_format=str(row["full_text_format"]),
        article_length_chars=_as_int(row["article_length_chars"]),
        article_length_words=_as_int(row["article_length_words"]),
        article_length_tokens_estimate=_as_int(row["article_length_tokens_estimate"]),
        license=str(row["license"]),
        attribution=str(row["attribution"]),
        source_api=str(row["source_api"]),
        fetch_status=str(row["fetch_status"]),
        fetch_error=str(row["fetch_error"]),
        content_hash=str(row["content_hash"]),
    )


__all__ = ["Document", "Section", "WikidataFact", "document_from_article_row", "document_id"]
