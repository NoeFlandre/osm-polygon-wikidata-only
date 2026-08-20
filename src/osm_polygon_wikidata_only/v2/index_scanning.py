"""Read and validate V1 Parquet identity columns for the V2 reuse index.

This module owns the file-system boundary of the V1 reuse index.  It keeps
schema validation, row-group projection, and descriptor cleanup separate from
the SQLite lifecycle and lookup code in :mod:`v1_index`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import article_schema
from osm_polygon_wikidata_only.io.parquet import _open_parquet_file

DocumentRow = dict[str, object]
_INDEX_PROJECTION = ("document_id", "language", "title", "page_id", "revision_id", "wikidata")


def required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"V1 document field {field!r} is not an integer")
    return value


def effective_paths(processed_dir: Path) -> tuple[Path, ...]:
    """Return canonical V1 document shards with legacy article fallbacks."""
    document_dir = processed_dir / "wikipedia" / "documents"
    article_dir = processed_dir / "articles"
    canonical_paths = {path.stem: path for path in document_dir.glob("*.parquet")}
    # V1 releases before the canonical-document migration stored Wikipedia
    # rows under ``articles/``. Prefer canonical shards when both exist, but
    # keep the legacy fallback so V2 never refetches an already-finalized page.
    effective = dict(canonical_paths)
    for path in article_dir.glob("*.parquet"):
        effective.setdefault(path.stem, path)
    return tuple(sorted(effective.values(), key=lambda path: path.name))


def read_rows(path: Path, *, legacy_articles: bool = False) -> list[DocumentRow]:
    """Read and schema-check one complete V1 shard for the in-memory index."""
    table = _read_table(path)
    _validate_table_schema(table.schema, path, legacy_articles)
    return _rows_from_table(table, path, legacy_articles)


def _read_table(path: Path) -> Any:
    try:
        with _open_parquet_file(path) as parquet_file:
            return parquet_file.read()
    except Exception as exc:
        raise ValueError(f"V1 document shard is unreadable: {path}: {exc}") from exc


def _validate_table_schema(schema: Any, path: Path, legacy_articles: bool) -> None:
    expected = article_schema() if legacy_articles else wikipedia_document_schema()
    if schema.equals(expected, check_metadata=True):
        return
    label = "legacy article" if legacy_articles else "V1 document"
    raise ValueError(f"V1 {label} shard has an invalid schema: {path}")


def _rows_from_table(table: Any, path: Path, legacy_articles: bool) -> list[DocumentRow]:
    if legacy_articles:
        return _legacy_rows(table, path)
    return [dict(row) for row in table.to_pylist()]


def _legacy_rows(table: Any, path: Path) -> list[DocumentRow]:
    try:
        return [wikipedia_document_from_article_row(row).to_dict() for row in table.to_pylist()]
    except Exception as exc:
        raise ValueError(f"V1 legacy article shard is invalid: {path}: {exc}") from exc


def _index_columns(legacy_articles: bool) -> tuple[str, ...]:
    return tuple(
        column for column in _INDEX_PROJECTION if not legacy_articles or column != "document_id"
    )


def validated_parquet_file(path: Path, *, legacy_articles: bool):
    """Open and schema-check one V1 shard, closing handles on failure."""
    parquet_file: pq.ParquetFile | None = None
    try:
        parquet_file = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
        _validate_table_schema(parquet_file.schema_arrow, path, legacy_articles)
        opened = parquet_file
        parquet_file = None
        return opened
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"V1 document shard is unreadable: {path}: {exc}") from exc
    finally:
        if parquet_file is not None:
            parquet_file.close()  # type: ignore[no-untyped-call]


def scan_index_row_group(
    path: Path,
    *,
    legacy_articles: bool,
    row_group: int,
    parquet_file: pq.ParquetFile | None = None,
) -> list[tuple[str, str, str, int, int, str, int, int]]:
    """Read identity columns and row positions for one committed row group."""
    owns_parquet_file = parquet_file is None
    validated: pq.ParquetFile | None = None
    try:
        validated = parquet_file or validated_parquet_file(path, legacy_articles=legacy_articles)
        return _read_index_rows(validated, path, legacy_articles, row_group)
    finally:
        if owns_parquet_file and validated is not None:
            validated.close()


def _read_row_group(
    parquet_file: pq.ParquetFile,
    row_group: int,
    legacy_articles: bool,
) -> Any:
    return parquet_file.read_row_group(row_group, columns=list(_index_columns(legacy_articles)))


def _read_index_rows(
    parquet_file: pq.ParquetFile,
    path: Path,
    legacy_articles: bool,
    row_group: int,
) -> list[tuple[str, str, str, int, int, str, int, int]]:
    try:
        table = _read_row_group(parquet_file, row_group, legacy_articles)
        return index_rows_from_table(table, legacy_articles=legacy_articles, row_group=row_group)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"V1 document shard is unreadable: {path}: {exc}") from exc


def index_rows_from_table(
    table: Any,
    *,
    legacy_articles: bool,
    row_group: int,
) -> list[tuple[str, str, str, int, int, str, int, int]]:
    """Extract index fields column-wise, avoiding one Python dict per row."""
    columns = {name: table.column(name).to_pylist() for name in _index_columns(legacy_articles)}
    indexed: list[tuple[str, str, str, int, int, str, int, int]] = []
    for row_index in range(table.num_rows):
        language = str(columns["language"][row_index])
        page_id = required_int(columns["page_id"][row_index], "page_id")
        revision_id = required_int(columns["revision_id"][row_index], "revision_id")
        wikidata = columns["wikidata"][row_index]
        if legacy_articles:
            document_id = f"{wikidata}:wikipedia:{language}:{page_id}:{revision_id}"
        else:
            document_id = str(columns["document_id"][row_index])
        indexed.append(
            (
                document_id,
                language,
                str(columns["title"][row_index]),
                page_id,
                revision_id,
                str(wikidata or ""),
                row_group,
                row_index,
            )
        )
    return indexed


def scan_index_rows(
    path: Path, *, legacy_articles: bool
) -> list[tuple[str, str, str, int, int, str, int, int]]:
    """Read identity columns and row positions for one V1 shard."""
    with validated_parquet_file(path, legacy_articles=legacy_articles) as parquet_file:
        indexed: list[tuple[str, str, str, int, int, str, int, int]] = []
        seen_documents: set[str] = set()
        for row_group in range(parquet_file.num_row_groups):
            for row in scan_index_row_group(
                path,
                legacy_articles=legacy_articles,
                row_group=row_group,
                parquet_file=parquet_file,
            ):
                if row[0] in seen_documents:
                    continue
                seen_documents.add(row[0])
                indexed.append(row)
        return indexed


__all__ = [
    "DocumentRow",
    "effective_paths",
    "index_rows_from_table",
    "read_rows",
    "required_int",
    "scan_index_row_group",
    "scan_index_rows",
    "validated_parquet_file",
]
