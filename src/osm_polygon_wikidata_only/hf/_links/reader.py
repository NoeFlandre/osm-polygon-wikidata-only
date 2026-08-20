"""Validated, schema-aware readers for polygon-to-document links."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema
from osm_polygon_wikidata_only.hf._geographic.models import CoverageMapError
from osm_polygon_wikidata_only.hf._geographic.parquet_inputs import (
    read_required_columns,
    sorted_parquets,
)


@dataclass(frozen=True, slots=True)
class DocumentLink:
    polygon_id: str
    project: str
    document_id: str
    wikidata: str
    language: str


def read_document_links(
    processed_root: Path,
    *,
    links_dir: Path | None = None,
) -> tuple[DocumentLink, ...]:
    """Read canonical links deterministically and validate their schema.

    ``links_dir`` is used by isolated dataset contracts such as V2.  The
    default remains the V1 ``polygon_articles`` directory.
    """
    expected = polygon_document_link_schema()
    source_dir = links_dir or processed_root / "polygon_articles"
    rows: list[DocumentLink] = []
    for path in sorted_parquets(source_dir):
        source_rows = _read_source_rows(processed_root, path, expected)
        if source_rows is None:
            continue
        rows.extend(_document_links(source_rows))
    return tuple(sorted(rows, key=lambda row: (row.polygon_id, row.project, row.document_id)))


def _read_source_rows(
    processed_root: Path,
    path: Path,
    expected: Any,
) -> list[dict[str, Any]] | None:
    schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    if _is_canonical_schema(schema, expected):
        return read_required_columns(
            path,
            ("polygon_id", "project", "document_id", "wikidata", "language"),
            label="polygon links",
        )
    if _is_legacy_schema(schema):
        return _read_legacy_rows(processed_root, path)
    raise CoverageMapError(
        f"polygon link parquet {path} does not use the canonical or supported legacy schema"
    )


def _is_canonical_schema(schema: Any, expected: Any) -> bool:
    return schema.equals(expected, check_metadata=True) or (
        set(expected.names).issubset(schema.names) and "link_sources" in schema.names
    )


def _is_legacy_schema(schema: Any) -> bool:
    return schema.equals(polygon_article_schema(), check_metadata=True) or {
        "polygon_id",
        "article_id",
    }.issubset(schema.names)


def _read_legacy_rows(processed_root: Path, path: Path) -> list[dict[str, Any]] | None:
    documents_path = _legacy_documents_path(processed_root, path)
    if documents_path is None:
        return None
    article_to_document = _article_document_map(documents_path)
    legacy_rows = pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
    source_rows = [_legacy_row(row, article_to_document) for row in legacy_rows]
    if any(not row["document_id"] for row in source_rows):
        raise CoverageMapError(
            f"legacy polygon link parquet {path} contains unresolved article IDs"
        )
    return source_rows


def _legacy_documents_path(processed_root: Path, path: Path) -> Path | None:
    candidates = (
        processed_root / "wikipedia" / "documents" / path.name,
        processed_root / "articles" / path.name,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _article_document_map(documents_path: Path) -> dict[str, str]:
    document_schema = pq.read_schema(documents_path)  # type: ignore[no-untyped-call]
    columns = (
        ("article_id", "document_id") if "document_id" in document_schema.names else ("article_id",)
    )
    return {
        str(row["article_id"]): str(row.get("document_id", row["article_id"]))
        for row in read_required_columns(documents_path, columns, label="wikipedia documents")
    }


def _legacy_row(row: dict[str, Any], article_to_document: dict[str, str]) -> dict[str, Any]:
    return {
        "polygon_id": row["polygon_id"],
        "project": "wikipedia",
        "document_id": article_to_document.get(str(row["article_id"]), ""),
        "wikidata": row.get("wikidata", ""),
        "language": row.get("language", ""),
    }


def _document_links(rows: list[dict[str, Any]]) -> list[DocumentLink]:
    return [
        DocumentLink(
            polygon_id=str(row["polygon_id"]),
            project=str(row["project"]),
            document_id=str(row["document_id"]),
            wikidata=str(row.get("wikidata") or ""),
            language=str(row["language"]),
        )
        for row in rows
    ]


__all__ = ["DocumentLink", "read_document_links"]
