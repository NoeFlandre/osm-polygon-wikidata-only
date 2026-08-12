"""Validated, schema-aware readers for polygon-to-document links."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    rows: list[DocumentLink] = []
    expected = polygon_document_link_schema()
    source_dir = links_dir or processed_root / "polygon_articles"
    for path in sorted_parquets(source_dir):
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
        is_v2_schema = set(expected.names).issubset(schema.names) and "link_sources" in schema.names
        if schema.equals(expected, check_metadata=True) or is_v2_schema:
            source_rows = read_required_columns(
                path,
                ("polygon_id", "project", "document_id", "wikidata", "language"),
                label="polygon links",
            )
        elif schema.equals(polygon_article_schema(), check_metadata=True) or {
            "polygon_id",
            "article_id",
        }.issubset(schema.names):
            documents_path = processed_root / "wikipedia" / "documents" / path.name
            if not documents_path.is_file():
                documents_path = processed_root / "articles" / path.name
            if not documents_path.is_file():
                continue
            document_schema = pq.read_schema(documents_path)  # type: ignore[no-untyped-call]
            if "document_id" in document_schema.names:
                article_to_document = {
                    str(row["article_id"]): str(row["document_id"])
                    for row in read_required_columns(
                        documents_path,
                        ("article_id", "document_id"),
                        label="wikipedia documents",
                    )
                }
            else:
                article_to_document = {
                    str(row["article_id"]): str(row["article_id"])
                    for row in read_required_columns(
                        documents_path,
                        ("article_id",),
                        label="wikipedia documents",
                    )
                }
            legacy_rows = pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
            source_rows = [
                {
                    "polygon_id": row["polygon_id"],
                    "project": "wikipedia",
                    "document_id": article_to_document.get(str(row["article_id"]), ""),
                    "wikidata": row.get("wikidata", ""),
                    "language": row.get("language", ""),
                }
                for row in legacy_rows
            ]
            if any(not row["document_id"] for row in source_rows):
                raise CoverageMapError(
                    f"legacy polygon link parquet {path} contains unresolved article IDs"
                )
        else:
            raise CoverageMapError(
                f"polygon link parquet {path} does not use the canonical or supported legacy schema"
            )
        for row in source_rows:
            rows.append(
                DocumentLink(
                    polygon_id=str(row["polygon_id"]),
                    project=str(row["project"]),
                    document_id=str(row["document_id"]),
                    wikidata=str(row.get("wikidata") or ""),
                    language=str(row["language"]),
                )
            )
    return tuple(sorted(rows, key=lambda row: (row.polygon_id, row.project, row.document_id)))


__all__ = ["DocumentLink", "read_document_links"]
