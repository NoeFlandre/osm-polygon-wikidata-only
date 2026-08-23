"""Lookup and row-materialization behavior for the persistent V1 index.

The durable index store owns SQLite lifecycle and Parquet scan checkpoints.
This module keeps query normalization, bounded title lookups, and payload
materialization in a separate mixin so those concerns can be tested without
coupling them to the scan loop.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Protocol

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
)
from osm_polygon_wikidata_only.v2 import index_scanning

DocumentRow = index_scanning.DocumentRow
_TITLE_QUERY_BATCH_SIZE = 256
_QUERY_SQL = {
    "page": """
        SELECT document_id, source_path, legacy, row_group, row_index
        FROM documents WHERE language=? AND page_id=?
        ORDER BY document_id, source_path
    """,
    "title": """
        SELECT document_id, source_path, legacy, row_group, row_index
        FROM documents WHERE language=? AND title_key=?
        ORDER BY document_id, source_path
    """,
    "qid": """
        SELECT document_id, source_path, legacy, row_group, row_index
        FROM documents WHERE qid=?
        ORDER BY document_id, source_path
    """,
}


def title_key(language: str, title: str) -> tuple[str, str]:
    """Return the canonical language/title lookup key."""
    return language.casefold(), " ".join(title.replace("_", " ").split()).casefold()


def normalized_title_keys(keys: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Normalize title lookup keys while retaining first-seen order."""
    return tuple(dict.fromkeys(title_key(language, title) for language, title in keys))


def group_title_rows(
    rows: Sequence[sqlite3.Row],
) -> tuple[dict[tuple[str, str], tuple[sqlite3.Row, ...]], tuple[sqlite3.Row, ...]]:
    """Group title query rows and keep one reference per document identity."""
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    references_by_document: dict[str, sqlite3.Row] = {}
    for row in rows:
        key = (str(row["language"]), str(row["title_key"]))
        grouped.setdefault(key, []).append(row)
        references_by_document.setdefault(str(row["document_id"]), row)
    return {key: tuple(value) for key, value in grouped.items()}, tuple(
        references_by_document.values()
    )


def title_chunk_results(
    chunk: Sequence[tuple[str, str]],
    grouped: Mapping[tuple[str, str], Sequence[sqlite3.Row]],
    materialized: Mapping[str, DocumentRow],
) -> dict[tuple[str, str], tuple[DocumentRow, ...]]:
    """Build deterministic lookup results for a fetched title chunk."""
    results: dict[tuple[str, str], tuple[DocumentRow, ...]] = {}
    for key in chunk:
        document_ids = {str(row["document_id"]) for row in grouped.get(key, ())}
        results[key] = tuple(materialized[document_id] for document_id in sorted(document_ids))
    return results


def partition_title_cache(
    normalized: Sequence[tuple[str, str]],
    cached: Mapping[tuple[str, str], tuple[DocumentRow, ...]],
    *,
    complete: bool,
) -> tuple[dict[tuple[str, str], tuple[DocumentRow, ...]], tuple[tuple[str, str], ...]]:
    """Split normalized title keys into cached hits and ordered misses."""
    if not complete:
        return {}, tuple(normalized)
    results: dict[tuple[str, str], tuple[DocumentRow, ...]] = {}
    missing: list[tuple[str, str]] = []
    for key in normalized:
        result = cached.get(key)
        if result is None:
            missing.append(key)
        else:
            results[key] = result
    return results, tuple(missing)


def first_query_rows_by_document(rows: list[sqlite3.Row]) -> dict[str, sqlite3.Row]:
    """Keep the first deterministic reference for each document identity."""
    by_document: dict[str, sqlite3.Row] = {}
    for row in rows:
        by_document.setdefault(str(row["document_id"]), row)
    return by_document


def group_materialization_references(
    references: tuple[sqlite3.Row, ...],
    row_cache: OrderedDict[str, DocumentRow],
    result: dict[str, DocumentRow],
) -> dict[tuple[str, bool, int], list[sqlite3.Row]]:
    """Separate cached rows from Parquet row-group reads."""
    grouped: dict[tuple[str, bool, int], list[sqlite3.Row]] = {}
    for reference in references:
        document_id = str(reference["document_id"])
        cached = row_cache.get(document_id)
        if cached is not None:
            row_cache.move_to_end(document_id)
            result[document_id] = cached
            continue
        group = (
            str(reference["source_path"]),
            bool(reference["legacy"]),
            int(reference["row_group"]),
        )
        grouped.setdefault(group, []).append(reference)
    return grouped


def materialize_group(
    source_path: str,
    legacy: bool,
    row_group: int,
    references: list[sqlite3.Row],
    row_cache: OrderedDict[str, DocumentRow],
    row_cache_limit: int,
) -> dict[str, DocumentRow]:
    """Materialize one Parquet row group into the bounded document cache."""
    with pq.ParquetFile(source_path) as parquet_file:
        table = parquet_file.read_row_group(row_group)
        raw_rows = table.to_pylist()
    result: dict[str, DocumentRow] = {}
    for reference in references:
        document_id = str(reference["document_id"])
        row = raw_rows[int(reference["row_index"])]
        normalized = wikipedia_document_from_article_row(row).to_dict() if legacy else dict(row)
        row_cache[document_id] = normalized
        row_cache.move_to_end(document_id)
        while len(row_cache) > row_cache_limit:
            row_cache.popitem(last=False)
        result[document_id] = normalized
    return result


class _QueryStore(Protocol):
    _complete: threading.Event
    _initialized: threading.Event
    _materialize_lock: threading.Lock
    _query_cache: OrderedDict[tuple[str, tuple[object, ...]], tuple[DocumentRow, ...]]
    _query_cache_limit: int
    _query_cache_lock: threading.Lock
    _reader_query_lock: threading.Lock
    _row_cache: OrderedDict[str, DocumentRow]
    _row_cache_limit: int

    def _cache_query(
        self,
        cache_key: tuple[str, tuple[object, ...]],
        result: tuple[DocumentRow, ...],
    ) -> None: ...

    def _cached_query(
        self,
        cache_key: tuple[str, tuple[object, ...]],
    ) -> tuple[DocumentRow, ...] | None: ...

    def _ensure_secondary_index(self, query_name: str) -> None: ...

    def _raise_error(self) -> None: ...

    def _reader(self) -> sqlite3.Connection: ...

    def _query(
        self, query_name: str, parameters: tuple[object, ...]
    ) -> tuple[DocumentRow, ...]: ...

    def _query_rows(
        self,
        query_name: str,
        parameters: tuple[object, ...],
    ) -> list[sqlite3.Row]: ...

    def _cached_title_results(
        self,
        normalized: Sequence[tuple[str, str]],
    ) -> tuple[dict[tuple[str, str], tuple[DocumentRow, ...]], tuple[tuple[str, str], ...]]: ...

    def _fetch_title_chunk(
        self,
        chunk: Sequence[tuple[str, str]],
    ) -> tuple[dict[tuple[str, str], tuple[sqlite3.Row, ...]], tuple[sqlite3.Row, ...]]: ...

    def _materialize(
        self,
        references: tuple[sqlite3.Row, ...],
    ) -> dict[str, DocumentRow]: ...


class PersistentIndexQueries:
    """Lookup/materialization mixin for the durable V1 index store."""

    def _cached_query(
        self: _QueryStore,
        cache_key: tuple[str, tuple[object, ...]],
    ) -> tuple[DocumentRow, ...] | None:
        if not self._complete.is_set():
            return None
        with self._query_cache_lock:
            cached = self._query_cache.get(cache_key)
            if cached is None:
                return None
            self._query_cache.move_to_end(cache_key)
            return cached

    def _cache_query(
        self: _QueryStore,
        cache_key: tuple[str, tuple[object, ...]],
        result: tuple[DocumentRow, ...],
    ) -> None:
        if not self._complete.is_set():
            return
        with self._query_cache_lock:
            self._query_cache[cache_key] = result
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > self._query_cache_limit:
                self._query_cache.popitem(last=False)

    def _query(
        self: _QueryStore,
        query_name: str,
        parameters: tuple[object, ...],
    ) -> tuple[DocumentRow, ...]:
        if not self._initialized.is_set():
            return ()
        self._raise_error()
        cache_key = (query_name, parameters)
        cached = self._cached_query(cache_key)
        if cached is not None:
            return cached
        rows = self._query_rows(query_name, parameters)
        if not rows:
            self._cache_query(cache_key, ())
            return ()
        by_document = first_query_rows_by_document(rows)
        with self._materialize_lock:
            materialized = self._materialize(tuple(by_document.values()))
        result = tuple(materialized[document_id] for document_id in sorted(materialized))
        self._cache_query(cache_key, result)
        return result

    def _query_rows(
        self: _QueryStore,
        query_name: str,
        parameters: tuple[object, ...],
    ) -> list[sqlite3.Row]:
        query = _QUERY_SQL[query_name]
        self._ensure_secondary_index(query_name)
        with self._reader_query_lock:
            return list(self._reader().execute(query, parameters))

    def _cached_title_results(
        self: _QueryStore,
        normalized: Sequence[tuple[str, str]],
    ) -> tuple[dict[tuple[str, str], tuple[DocumentRow, ...]], tuple[tuple[str, str], ...]]:
        """Read completed title results from the bounded cache."""
        cached: dict[tuple[str, str], tuple[DocumentRow, ...]] = {}
        complete = self._complete.is_set()
        if complete:
            with self._query_cache_lock:
                for key in normalized:
                    cache_key = ("title", key)
                    result = self._query_cache.get(cache_key)
                    if result is not None:
                        self._query_cache.move_to_end(cache_key)
                        cached[key] = result
        return partition_title_cache(normalized, cached, complete=complete)

    def _fetch_title_chunk(
        self: _QueryStore,
        chunk: Sequence[tuple[str, str]],
    ) -> tuple[dict[tuple[str, str], tuple[sqlite3.Row, ...]], tuple[sqlite3.Row, ...]]:
        """Fetch one bounded title chunk from SQLite."""
        predicates = " OR ".join("(language=? AND title_key=?)" for _ in chunk)
        parameters = tuple(value for key in chunk for value in key)
        # The predicate is composed only from generated ``?`` placeholders;
        # all title values remain bound parameters.
        with self._reader_query_lock:
            rows = list(
                self._reader().execute(
                    f"""
                    SELECT language, title_key, document_id, source_path, legacy, row_group, row_index
                    FROM documents
                    WHERE {predicates}
                    ORDER BY language, title_key, document_id, source_path
                    """,  # noqa: S608
                    parameters,
                )
            )
        return group_title_rows(rows)

    def by_titles(
        self: _QueryStore,
        keys: Sequence[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[DocumentRow, ...]]:
        """Resolve title keys in bounded batches without changing row order."""
        normalized = normalized_title_keys(keys)
        if not self._initialized.is_set():
            return {key: () for key in normalized}
        self._raise_error()

        results, missing = self._cached_title_results(normalized)

        for offset in range(0, len(missing), _TITLE_QUERY_BATCH_SIZE):
            chunk = tuple(missing[offset : offset + _TITLE_QUERY_BATCH_SIZE])
            grouped, references = self._fetch_title_chunk(chunk)
            with self._materialize_lock:
                materialized = self._materialize(references)
            chunk_results = title_chunk_results(chunk, grouped, materialized)
            results.update(chunk_results)
            for key, result in chunk_results.items():
                self._cache_query(("title", key), result)
        return results

    def _materialize(
        self: _QueryStore,
        references: tuple[sqlite3.Row, ...],
    ) -> dict[str, DocumentRow]:
        result: dict[str, DocumentRow] = {}
        grouped = group_materialization_references(references, self._row_cache, result)

        for (source_path, legacy, row_group), group_references in grouped.items():
            result.update(
                materialize_group(
                    source_path,
                    legacy,
                    row_group,
                    group_references,
                    self._row_cache,
                    self._row_cache_limit,
                )
            )
        return result

    def by_page(
        self: _QueryStore,
        language: str,
        page_id: int,
    ) -> tuple[DocumentRow, ...]:
        return self._query("page", (language.casefold(), page_id))

    def by_title(
        self: _QueryStore,
        language: str,
        title: str,
    ) -> tuple[DocumentRow, ...]:
        language_key, title_key_value = title_key(language, title)
        return self._query("title", (language_key, title_key_value))

    def by_qid(self: _QueryStore, qid: str) -> tuple[DocumentRow, ...]:
        return self._query("qid", (qid,))


__all__ = [
    "PersistentIndexQueries",
    "first_query_rows_by_document",
    "group_materialization_references",
    "group_title_rows",
    "materialize_group",
    "normalized_title_keys",
    "partition_title_cache",
    "title_chunk_results",
    "title_key",
]
