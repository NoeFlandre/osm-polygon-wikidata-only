"""Read-only index of V1 Wikipedia documents for V2 reuse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import article_schema

DocumentRow = dict[str, object]
PageKey = tuple[str, int]
RevisionKey = tuple[str, int, int]
IndexKey = TypeVar("IndexKey")


def _title_key(language: str, title: str) -> tuple[str, str]:
    return language.casefold(), " ".join(title.replace("_", " ").split()).casefold()


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"V1 document field {field!r} is not an integer")
    return value


@dataclass(frozen=True, slots=True)
class V1ReuseIndex:
    """Immutable lookup maps built from V1 Wikipedia document shards."""

    by_page_index: Mapping[PageKey, tuple[DocumentRow, ...]]
    by_title_index: Mapping[tuple[str, str], tuple[DocumentRow, ...]]
    by_qid_index: Mapping[str, tuple[DocumentRow, ...]]
    files: tuple[Path, ...]
    row_count: int

    def by_page(self, language: str, page_id: int) -> tuple[DocumentRow, ...]:
        return self.by_page_index.get((language.casefold(), page_id), ())

    def by_title(self, language: str, title: str) -> tuple[DocumentRow, ...]:
        return self.by_title_index.get(_title_key(language, title), ())

    def by_qid(self, qid: str) -> tuple[DocumentRow, ...]:
        return self.by_qid_index.get(qid, ())


def _read_rows(path: Path, *, legacy_articles: bool = False) -> list[DocumentRow]:
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise ValueError(f"V1 document shard is unreadable: {path}: {exc}") from exc
    expected_schema = article_schema() if legacy_articles else wikipedia_document_schema()
    if not table.schema.equals(expected_schema, check_metadata=True):
        label = "legacy article" if legacy_articles else "V1 document"
        raise ValueError(f"V1 {label} shard has an invalid schema: {path}")
    if legacy_articles:
        try:
            return [wikipedia_document_from_article_row(row).to_dict() for row in table.to_pylist()]
        except Exception as exc:
            raise ValueError(f"V1 legacy article shard is invalid: {path}: {exc}") from exc
    return [dict(row) for row in table.to_pylist()]


def build_v1_reuse_index(processed_dir: Path) -> V1ReuseIndex:
    """Read and validate all V1 Wikipedia document shards without writing."""
    document_dir = processed_dir / "wikipedia" / "documents"
    article_dir = processed_dir / "articles"
    canonical_paths = {path.stem: path for path in document_dir.glob("*.parquet")}
    # V1 releases before the canonical-document migration stored Wikipedia
    # rows under ``articles/``. Prefer canonical shards when both exist, but
    # keep the legacy fallback so V2 never refetches an already-finalized page.
    effective_paths = dict(canonical_paths)
    for path in article_dir.glob("*.parquet"):
        effective_paths.setdefault(path.stem, path)
    files = tuple(sorted(effective_paths.values(), key=lambda path: path.name))
    by_page: dict[PageKey, list[DocumentRow]] = {}
    by_title: dict[tuple[str, str], list[DocumentRow]] = {}
    by_qid: dict[str, list[DocumentRow]] = {}
    seen_revisions: dict[RevisionKey, str] = {}
    seen_documents: set[str] = set()
    row_count = 0

    for path in files:
        for row in _read_rows(path, legacy_articles=path.parent == article_dir):
            document_id = str(row["document_id"])
            language = str(row["language"])
            page_id = _required_int(row["page_id"], "page_id")
            revision_id = _required_int(row["revision_id"], "revision_id")
            revision_key = (language.casefold(), page_id, revision_id)
            prior_id = seen_revisions.get(revision_key)
            if prior_id is not None and prior_id != document_id:
                raise ValueError(f"duplicate V1 page revision identity: {revision_key!r}")
            if document_id in seen_documents:
                continue
            seen_revisions[revision_key] = document_id
            seen_documents.add(document_id)
            row_count += 1
            by_page.setdefault((language.casefold(), page_id), []).append(row)
            by_title.setdefault(_title_key(language, str(row["title"])), []).append(row)
            qid = row.get("wikidata")
            if qid:
                by_qid.setdefault(str(qid), []).append(row)

    def freeze(
        mapping: dict[IndexKey, list[DocumentRow]],
    ) -> Mapping[IndexKey, tuple[DocumentRow, ...]]:
        return MappingProxyType(
            {
                key: tuple(sorted(rows, key=lambda item: str(item["document_id"])))
                for key, rows in mapping.items()
            }
        )

    return V1ReuseIndex(
        by_page_index=freeze(by_page),
        by_title_index=freeze(by_title),
        by_qid_index=freeze(by_qid),
        files=files,
        row_count=row_count,
    )


__all__ = ["V1ReuseIndex", "build_v1_reuse_index"]
