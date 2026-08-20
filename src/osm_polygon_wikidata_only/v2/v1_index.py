"""Read-only, resumable indexes of V1 Wikipedia document shards.

The V2 runner uses a disk-backed SQLite index by default.  It validates and
indexes one V1 shard at a time under the external V2 cache, so the full V1
document corpus is never held in memory and an interrupted build resumes from
the last committed Parquet row group.  The persistent store maintains the
title index needed by V2 during the scan; compatibility page and QID indexes
are created lazily only if those lookups are requested after completion.
Small callers and tests can omit ``cache_dir`` to use the original in-memory
index.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
)
from osm_polygon_wikidata_only.v2 import index_scanning
from osm_polygon_wikidata_only.v2.fingerprints import FileStatFingerprint

LOGGER = logging.getLogger(__name__)

DocumentRow = index_scanning.DocumentRow
_effective_paths = index_scanning.effective_paths
_index_columns = index_scanning._index_columns
_index_rows_from_table = index_scanning.index_rows_from_table
_read_rows = index_scanning.read_rows
_required_int = index_scanning.required_int
_scan_index_row_group = index_scanning.scan_index_row_group
_scan_index_rows = index_scanning.scan_index_rows
_validated_parquet_file = index_scanning.validated_parquet_file

PageKey = tuple[str, int]
_INDEX_SCHEMA_VERSION = 4
_INDEX_FILENAME = "v1_reuse_index.sqlite3"
_INDEX_SHUTDOWN_TIMEOUT_S = 5.0
_TITLE_QUERY_BATCH_SIZE = 256
_SECONDARY_INDEX_SQL = {
    "page": (
        "documents_page",
        "CREATE INDEX IF NOT EXISTS documents_page ON documents(language, page_id, document_id)",
    ),
    "qid": (
        "documents_qid",
        "CREATE INDEX IF NOT EXISTS documents_qid ON documents(qid, document_id)",
    ),
}
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


def _title_key(language: str, title: str) -> tuple[str, str]:
    return language.casefold(), " ".join(title.replace("_", " ").split()).casefold()


def _normalized_title_keys(keys: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Normalize title lookup keys while retaining first-seen order."""
    return tuple(dict.fromkeys(_title_key(language, title) for language, title in keys))


def _group_title_rows(
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


def _title_chunk_results(
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


def _partition_title_cache(
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


def _resumable_row_group(
    previous: tuple[int, int, int, int, bool, int, int] | None,
    fingerprint: tuple[int, int, int, int, bool],
    total_row_groups: int,
) -> int:
    """Return the next safe row group for a matching scan checkpoint."""
    if (
        previous is None
        or previous[:5] != fingerprint
        or previous[5] != total_row_groups
        or not 0 <= previous[6] <= total_row_groups
    ):
        return 0
    return previous[6]


def _count_distinct_documents(connection: sqlite3.Connection) -> int:
    """Count indexed document identities for a newly changed index."""
    return int(
        connection.execute("SELECT COUNT(DISTINCT document_id) FROM documents").fetchone()[0]
    )


def _read_cached_row_count(connection: sqlite3.Connection) -> int | None:
    row = connection.execute("SELECT value FROM index_metadata WHERE key='row_count'").fetchone()
    if row is None:
        return None
    try:
        value = int(row[0])
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _store_cached_row_count(connection: sqlite3.Connection, row_count: int) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO index_metadata(key, value) VALUES ('row_count', ?)",
        (str(row_count),),
    )


def _first_query_rows_by_document(rows: list[sqlite3.Row]) -> dict[str, sqlite3.Row]:
    """Keep the first deterministic reference for each document identity."""
    by_document: dict[str, sqlite3.Row] = {}
    for row in rows:
        by_document.setdefault(str(row["document_id"]), row)
    return by_document


def _group_materialization_references(
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


def _materialize_group(
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


def _current_index_files(files: tuple[Path, ...]) -> dict[str, tuple[Path, bool]]:
    return {str(path.resolve()): (path, path.parent.name == "articles") for path in files}


def _known_index_files(
    connection: sqlite3.Connection,
) -> dict[str, tuple[int, int, int, int, bool]]:
    return {
        str(row["path"]): (
            int(row["size"]),
            int(row["mtime_ns"]),
            int(row["ctime_ns"]),
            int(row["inode"]),
            bool(row["legacy"]),
        )
        for row in connection.execute("SELECT * FROM file_state")
    }


def _index_scan_progress(
    connection: sqlite3.Connection,
) -> dict[str, tuple[int, int, int, int, bool, int, int]]:
    return {
        str(row["path"]): (
            int(row["size"]),
            int(row["mtime_ns"]),
            int(row["ctime_ns"]),
            int(row["inode"]),
            bool(row["legacy"]),
            int(row["total_row_groups"]),
            int(row["next_row_group"]),
        )
        for row in connection.execute("SELECT * FROM scan_progress")
    }


class _PersistentV1Index:
    """Incremental SQLite metadata index with bounded Parquet row loading."""

    def __init__(
        self,
        cache_dir: Path,
        files: tuple[Path, ...],
        *,
        background: bool = False,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_dir / _INDEX_FILENAME
        self._connection: sqlite3.Connection | None = None
        self._row_cache: OrderedDict[str, DocumentRow] = OrderedDict()
        self._row_cache_limit = 10_000
        self._files = files
        self._ready = threading.Event()
        self._initialized = threading.Event()
        self._complete = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._index_reader_executor: ThreadPoolExecutor | None = None
        self._reader_connections: list[sqlite3.Connection] = []
        self._reader_lock = threading.Lock()
        self._reader_query_lock = threading.Lock()
        self._materialize_lock = threading.Lock()
        self._secondary_index_lock = threading.Lock()
        self._secondary_indexes: set[str] = set()
        self._query_cache_lock = threading.Lock()
        self._query_cache: OrderedDict[tuple[str, tuple[object, ...]], tuple[DocumentRow, ...]] = (
            OrderedDict()
        )
        self._query_cache_limit = 4096
        if background:
            self._thread = threading.Thread(
                target=self._run_sync,
                name="v2-v1-index",
                daemon=True,
            )
            self._thread.start()
        else:
            self._connection = self._open_connection()
            self._initialize_schema()
            self._initialized.set()
            self._run_sync()
            self._raise_error()

    def _open_connection(self) -> sqlite3.Connection:
        """Open and tune the writer connection on the indexing worker."""
        connection = sqlite3.connect(self._db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("PRAGMA wal_autocheckpoint=10000")
        return connection

    @property
    def _writer_connection(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise RuntimeError("V2 V1 reuse index writer is not initialized")
        return connection

    @property
    def is_ready(self) -> bool:
        """Return whether the current file set has been fully indexed."""
        return self._ready.is_set() and self._complete.is_set() and self._error is None

    def wait_until_ready(self) -> None:
        """Wait for indexing and propagate a background indexing failure."""
        self._ready.wait()
        self._raise_error()
        if not self._complete.is_set():
            raise RuntimeError("V2 V1 reuse index stopped before completion")

    def cancel(self) -> None:
        """Request stop after the current row-group transaction."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_INDEX_SHUTDOWN_TIMEOUT_S)
        if thread is not None and thread.is_alive():
            LOGGER.warning(
                "V2 V1 reuse index is still finishing a row group after %.1fs; "
                "committed checkpoints remain safe",
                _INDEX_SHUTDOWN_TIMEOUT_S,
            )

    def close(self) -> None:
        """Stop indexing and close connections only after the worker exits."""
        self.cancel()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # The daemon worker owns the writer while it finishes its current
            # row group.  Closing it here could corrupt the active transaction.
            return
        with self._reader_query_lock, self._reader_lock:
            for connection in self._reader_connections:
                connection.close()
            self._reader_connections.clear()
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def row_count(self) -> int:
        """Return the number of identities committed so far."""
        connection = sqlite3.connect(self._db_path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA query_only=ON")
        try:
            cached = _read_cached_row_count(connection)
            if cached is not None:
                return cached
            return _count_distinct_documents(connection)
        finally:
            connection.close()

    def _reader(self) -> sqlite3.Connection:
        with self._reader_lock:
            if self._reader_connections:
                return self._reader_connections[0]
            connection = sqlite3.connect(
                self._db_path,
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA query_only=ON")
            self._reader_connections.append(connection)
            return connection

    def _run_sync(self) -> None:
        try:
            if not self._initialized.is_set():
                self._connection = self._open_connection()
                self._initialize_schema()
                self._initialized.set()
                LOGGER.info("V2 V1 reuse index storage initialized; scanning in background")
            if self._sync(self._files):
                self._complete.set()
        except BaseException as exc:
            self._error = exc
            LOGGER.exception("V2 V1 reuse index failed")
        finally:
            reader = self._index_reader_executor
            if reader is not None:
                reader.shutdown(wait=True, cancel_futures=True)
                self._index_reader_executor = None
            self._initialized.set()
            self._ready.set()

    def _initialize_schema(self) -> None:
        connection = self._writer_connection
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, 1, 2, 3, _INDEX_SCHEMA_VERSION):
            with connection:
                connection.executescript(
                    "DROP TABLE IF EXISTS documents; DROP TABLE IF EXISTS file_state; "
                    "DROP TABLE IF EXISTS scan_progress; DROP TABLE IF EXISTS index_metadata;"
                )
        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS file_state (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    ctime_ns INTEGER NOT NULL,
                    inode INTEGER NOT NULL,
                    legacy INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    legacy INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    title_key TEXT NOT NULL,
                    page_id INTEGER NOT NULL,
                    revision_id INTEGER NOT NULL,
                    qid TEXT NOT NULL,
                    row_group INTEGER NOT NULL,
                    row_index INTEGER NOT NULL,
                    PRIMARY KEY (document_id, source_path)
                );
                CREATE TABLE IF NOT EXISTS scan_progress (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    ctime_ns INTEGER NOT NULL,
                    inode INTEGER NOT NULL,
                    legacy INTEGER NOT NULL,
                    total_row_groups INTEGER NOT NULL,
                    next_row_group INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Page and QID lookups are compatibility APIs, but V2 only uses
            # title lookups.  Building both secondary indexes while inserting
            # every document dominates the persistent index build, so defer
            # them until a caller actually requests one after the scan is
            # complete.  Dropping them here also migrates existing caches to
            # the cheaper write path without discarding indexed rows.
            title_columns = tuple(
                str(row[2]) for row in connection.execute("PRAGMA index_info(documents_title)")
            )
            if title_columns != ("language", "title_key"):
                connection.execute("DROP INDEX IF EXISTS documents_title")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS documents_title ON documents(language, title_key)"
            )
            connection.execute("DROP INDEX IF EXISTS documents_page")
            connection.execute("DROP INDEX IF EXISTS documents_qid")
            connection.execute(f"PRAGMA user_version={_INDEX_SCHEMA_VERSION}")

    def _ensure_secondary_index(self, query_name: str) -> None:
        if query_name not in _SECONDARY_INDEX_SQL or not self._complete.is_set():
            return
        index_name, statement = _SECONDARY_INDEX_SQL[query_name]
        with self._secondary_index_lock:
            if index_name in self._secondary_indexes:
                return
            with self._writer_connection:
                self._writer_connection.execute(statement)
            self._secondary_indexes.add(index_name)

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int, int, int]:
        return FileStatFingerprint.from_path(path).index_tuple()

    def _commit_indexed_row_group(
        self,
        connection: sqlite3.Connection,
        indexed: list[tuple[str, str, str, int, int, str, int, int]],
        *,
        resolved: str,
        fingerprint: tuple[int, int, int, int, bool],
        total_row_groups: int,
        row_group: int,
    ) -> None:
        with connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO documents(
                    document_id, source_path, legacy, language, title_key,
                    page_id, revision_id, qid, row_group, row_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(document_id),
                        resolved,
                        int(fingerprint[-1]),
                        str(language).casefold(),
                        _title_key(str(language), str(title))[1],
                        int(page_id),
                        int(revision_id),
                        str(qid),
                        int(indexed_row_group),
                        int(row_index),
                    )
                    for (
                        document_id,
                        language,
                        title,
                        page_id,
                        revision_id,
                        qid,
                        indexed_row_group,
                        row_index,
                    ) in indexed
                ],
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO scan_progress(
                    path, size, mtime_ns, ctime_ns, inode, legacy,
                    total_row_groups, next_row_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (resolved, *fingerprint, total_row_groups, row_group + 1),
            )

    def _remove_stale_paths(
        self,
        current: Mapping[str, tuple[Path, bool]],
        known: Mapping[str, tuple[int, int, int, int, bool]],
    ) -> set[str]:
        """Remove cached rows for shards no longer present in the file set."""
        stale = set(known) - set(current)
        if not stale:
            return stale
        self._delete_stale_paths(stale)
        return stale

    def _delete_stale_paths(self, stale: set[str]) -> None:
        connection = self._writer_connection
        with connection:
            for path in stale:
                connection.execute("DELETE FROM documents WHERE source_path=?", (path,))
                connection.execute("DELETE FROM file_state WHERE path=?", (path,))
                connection.execute("DELETE FROM scan_progress WHERE path=?", (path,))
            connection.execute("DELETE FROM index_metadata WHERE key='row_count'")
        self._row_cache.clear()

    def _invalidate_row_count(self, already_invalidated: bool) -> bool:
        """Invalidate the cached count once before scanning changed data."""
        if already_invalidated:
            return True
        with self._writer_connection:
            self._writer_connection.execute("DELETE FROM index_metadata WHERE key='row_count'")
        return True

    def _unchanged_shard(
        self,
        resolved: str,
        fingerprint: tuple[int, int, int, int, bool],
        known: Mapping[str, tuple[int, int, int, int, bool]],
        progress: Mapping[str, tuple[int, int, int, int, bool, int, int]],
    ) -> bool:
        """Return whether a shard is already indexed with the same fingerprint."""
        if known.get(resolved) != fingerprint:
            return False
        if resolved in progress:
            with self._writer_connection:
                self._writer_connection.execute(
                    "DELETE FROM scan_progress WHERE path=?", (resolved,)
                )
        return True

    def _prepare_shard(
        self,
        path: Path,
        resolved: str,
        fingerprint: tuple[int, int, int, int, bool],
        progress: Mapping[str, tuple[int, int, int, int, bool, int, int]],
    ) -> tuple[pq.ParquetFile, bool, int, int]:
        """Validate a shard and initialize or resume its row-group checkpoint."""
        legacy_articles = bool(fingerprint[-1])
        parquet_file = _validated_parquet_file(path, legacy_articles=legacy_articles)
        total_row_groups = parquet_file.num_row_groups
        start_row_group = _resumable_row_group(
            progress.get(resolved), fingerprint, total_row_groups
        )
        if start_row_group == 0:
            with self._writer_connection:
                self._writer_connection.execute(
                    "DELETE FROM documents WHERE source_path=?", (resolved,)
                )
                self._writer_connection.execute("DELETE FROM file_state WHERE path=?", (resolved,))
                self._writer_connection.execute(
                    """
                    INSERT OR REPLACE INTO scan_progress(
                        path, size, mtime_ns, ctime_ns, inode, legacy,
                        total_row_groups, next_row_group
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (resolved, *fingerprint, total_row_groups),
                )
            self._row_cache.clear()
        return parquet_file, legacy_articles, total_row_groups, start_row_group

    def _index_reader(self) -> ThreadPoolExecutor:
        """Return the single bounded reader executor used for row-group scans."""
        reader = self._index_reader_executor
        if reader is None:
            reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v2-index-reader")
            self._index_reader_executor = reader
        return reader

    def _submit_row_group(
        self,
        reader: ThreadPoolExecutor,
        path: Path,
        *,
        legacy_articles: bool,
        row_group: int,
        total_row_groups: int,
        parquet_file: pq.ParquetFile,
    ) -> Future[list[tuple[str, str, str, int, int, str, int, int]]] | None:
        """Submit a row group unless it is past the shard or cancellation boundary."""
        if row_group >= total_row_groups or self._stop.is_set():
            return None
        return reader.submit(
            _scan_index_row_group,
            path,
            legacy_articles=legacy_articles,
            row_group=row_group,
            parquet_file=parquet_file,
        )

    def _stop_row_group_scan(
        self,
        next_group: Future[list[tuple[str, str, str, int, int, str, int, int]]] | None,
        *,
        position: int,
        total_files: int,
        row_group: int,
        total_row_groups: int,
    ) -> bool:
        """Cancel a prefetched row group and report the resumable stop boundary."""
        if not self._stop.is_set():
            return False
        if next_group is not None:
            next_group.cancel()
        self._log_row_group_stop(position, total_files, row_group, total_row_groups)
        return True

    @staticmethod
    def _log_row_group_stop(
        position: int,
        total_files: int,
        row_group: int,
        total_row_groups: int,
    ) -> None:
        LOGGER.info(
            "V2 V1 reuse index stopped in shard %d/%d at row group %d/%d; "
            "completed work is resumable",
            position,
            total_files,
            row_group,
            total_row_groups,
        )

    @staticmethod
    def _resolve_row_group(
        next_group: Future[list[tuple[str, str, str, int, int, str, int, int]]] | None,
    ) -> list[tuple[str, str, str, int, int, str, int, int]]:
        """Resolve a prefetched row group, preserving the loop invariant error."""
        if next_group is None:  # pragma: no cover - loop invariant
            raise RuntimeError("V2 index reader lost the next row group")
        return next_group.result()

    def _scan_row_groups(
        self,
        reader: ThreadPoolExecutor,
        path: Path,
        *,
        legacy_articles: bool,
        start_row_group: int,
        total_row_groups: int,
        parquet_file: pq.ParquetFile,
        resolved: str,
        fingerprint: tuple[int, int, int, int, bool],
        position: int,
        total_files: int,
    ) -> bool:
        """Scan and commit a shard's row groups with one-row-group lookahead."""
        next_group = self._submit_row_group(
            reader,
            path,
            legacy_articles=legacy_articles,
            row_group=start_row_group,
            total_row_groups=total_row_groups,
            parquet_file=parquet_file,
        )
        try:
            for row_group in range(start_row_group, total_row_groups):
                if self._stop_row_group_scan(
                    next_group,
                    position=position,
                    total_files=total_files,
                    row_group=row_group,
                    total_row_groups=total_row_groups,
                ):
                    return False
                indexed = self._resolve_row_group(next_group)
                next_group = self._submit_row_group(
                    reader,
                    path,
                    legacy_articles=legacy_articles,
                    row_group=row_group + 1,
                    total_row_groups=total_row_groups,
                    parquet_file=parquet_file,
                )
                self._commit_indexed_row_group(
                    self._writer_connection,
                    indexed,
                    resolved=resolved,
                    fingerprint=fingerprint,
                    total_row_groups=total_row_groups,
                    row_group=row_group,
                )
                LOGGER.info(
                    "V2 V1 reuse index: shard %d/%d row group %d/%d ready (%d identities)",
                    position,
                    total_files,
                    row_group + 1,
                    total_row_groups,
                    len(indexed),
                )
            return True
        finally:
            if next_group is not None:
                next_group.cancel()

    def _finalize_shard(
        self,
        resolved: str,
        fingerprint: tuple[int, int, int, int, bool],
        *,
        position: int,
        total_files: int,
    ) -> None:
        """Commit the completed shard marker and clear materialized row state."""
        with self._writer_connection:
            self._writer_connection.execute(
                """
                INSERT OR REPLACE INTO file_state(path, size, mtime_ns, ctime_ns, inode, legacy)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (resolved, *fingerprint),
            )
            self._writer_connection.execute("DELETE FROM scan_progress WHERE path=?", (resolved,))
        self._row_cache.clear()
        LOGGER.info("V2 V1 reuse index: shard ready %d/%d", position, total_files)

    def _sync_shard(
        self,
        path: Path,
        resolved: str,
        fingerprint: tuple[int, int, int, int, bool],
        progress: Mapping[str, tuple[int, int, int, int, bool, int, int]],
        *,
        position: int,
        total_files: int,
    ) -> bool:
        """Validate, resume, scan, and finalize one changed shard."""
        parquet_file, legacy_articles, total_row_groups, start_row_group = self._prepare_shard(
            path, resolved, fingerprint, progress
        )
        try:
            return_value = self._scan_row_groups(
                self._index_reader(),
                path,
                legacy_articles=legacy_articles,
                start_row_group=start_row_group,
                total_row_groups=total_row_groups,
                parquet_file=parquet_file,
                resolved=resolved,
                fingerprint=fingerprint,
                position=position,
                total_files=total_files,
            )
        finally:
            parquet_file.close()
        if return_value:
            self._finalize_shard(
                resolved,
                fingerprint,
                position=position,
                total_files=total_files,
            )
        return return_value

    def _finish_sync(self, changed: int, stale: set[str], total_files: int) -> bool:
        """Cache the final identity count and emit the completion status."""
        connection = self._writer_connection
        row_count = _read_cached_row_count(connection)
        if row_count is None:
            row_count = _count_distinct_documents(connection)
            with connection:
                _store_cached_row_count(connection, row_count)
        status = "ready" if changed or stale else "reused"
        LOGGER.info(
            "V2 V1 reuse index %s: %d/%d shards; %d document identities; cache=%s",
            status,
            total_files,
            total_files,
            row_count,
            self._db_path,
        )
        return True

    def _sync_file(
        self,
        path: Path,
        *,
        position: int,
        total_files: int,
        known: Mapping[str, tuple[int, int, int, int, bool]],
        progress: Mapping[str, tuple[int, int, int, int, bool, int, int]],
        row_count_invalidated: bool,
    ) -> tuple[bool, bool, bool]:
        """Process one shard and report continuation, change, and cache state."""
        if self._stop.is_set():
            LOGGER.info(
                "V2 V1 reuse index stopped after %d/%d shards; completed work is resumable",
                position - 1,
                total_files,
            )
            return False, False, row_count_invalidated
        resolved = str(path.resolve())
        fingerprint = (*self._fingerprint(path), path.parent.name == "articles")
        if self._unchanged_shard(resolved, fingerprint, known, progress):
            return True, False, row_count_invalidated
        row_count_invalidated = self._invalidate_row_count(row_count_invalidated)
        LOGGER.info(
            "V2 V1 reuse index: scanning shard %d/%d (%s)", position, total_files, path.name
        )
        completed = self._sync_shard(
            path,
            resolved,
            fingerprint,
            progress,
            position=position,
            total_files=total_files,
        )
        return completed, completed, row_count_invalidated

    def _sync_files(
        self,
        files: tuple[Path, ...],
        *,
        known: Mapping[str, tuple[int, int, int, int, bool]],
        progress: Mapping[str, tuple[int, int, int, int, bool, int, int]],
        stale: set[str],
    ) -> int | None:
        """Process all shards, returning ``None`` when cancellation stopped the scan."""
        row_count_invalidated = bool(stale)
        changed = 0
        for position, path in enumerate(files, start=1):
            completed, did_change, row_count_invalidated = self._sync_file(
                path,
                position=position,
                total_files=len(files),
                known=known,
                progress=progress,
                row_count_invalidated=row_count_invalidated,
            )
            if not completed:
                return None
            changed += did_change
        return changed

    def _sync(self, files: tuple[Path, ...]) -> bool:
        connection = self._writer_connection
        current = _current_index_files(files)
        known = _known_index_files(connection)
        progress = _index_scan_progress(connection)
        stale = self._remove_stale_paths(current, known)
        changed = self._sync_files(files, known=known, progress=progress, stale=stale)
        if changed is None:
            return False
        return self._finish_sync(changed, stale, len(files))

    def _query(self, query_name: str, parameters: tuple[object, ...]) -> tuple[DocumentRow, ...]:
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
        by_document = _first_query_rows_by_document(rows)
        with self._materialize_lock:
            materialized = self._materialize(tuple(by_document.values()))
        result = tuple(materialized[document_id] for document_id in sorted(materialized))
        self._cache_query(cache_key, result)
        return result

    def _cached_query(
        self, cache_key: tuple[str, tuple[object, ...]]
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
        self,
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

    def _query_rows(self, query_name: str, parameters: tuple[object, ...]) -> list[sqlite3.Row]:
        query = _QUERY_SQL[query_name]
        self._ensure_secondary_index(query_name)
        with self._reader_query_lock:
            return list(self._reader().execute(query, parameters))

    def _cached_title_results(
        self,
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
        return _partition_title_cache(normalized, cached, complete=complete)

    def _fetch_title_chunk(
        self,
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
        return _group_title_rows(rows)

    def by_titles(
        self,
        keys: Sequence[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[DocumentRow, ...]]:
        """Resolve title keys in bounded batches without changing row order."""
        normalized = _normalized_title_keys(keys)
        if not self._initialized.is_set():
            return {key: () for key in normalized}
        self._raise_error()

        results, missing = self._cached_title_results(normalized)

        for offset in range(0, len(missing), _TITLE_QUERY_BATCH_SIZE):
            chunk = tuple(missing[offset : offset + _TITLE_QUERY_BATCH_SIZE])
            grouped, references = self._fetch_title_chunk(chunk)
            with self._materialize_lock:
                materialized = self._materialize(references)
            chunk_results = _title_chunk_results(chunk, grouped, materialized)
            results.update(chunk_results)
            for key, result in chunk_results.items():
                self._cache_query(("title", key), result)
        return results

    def _materialize(self, references: tuple[sqlite3.Row, ...]) -> dict[str, DocumentRow]:
        result: dict[str, DocumentRow] = {}
        grouped = _group_materialization_references(references, self._row_cache, result)

        for (source_path, legacy, row_group), group_references in grouped.items():
            result.update(
                _materialize_group(
                    source_path,
                    legacy,
                    row_group,
                    group_references,
                    self._row_cache,
                    self._row_cache_limit,
                )
            )
        return result

    def by_page(self, language: str, page_id: int) -> tuple[DocumentRow, ...]:
        return self._query(
            "page",
            (language.casefold(), page_id),
        )

    def by_title(self, language: str, title: str) -> tuple[DocumentRow, ...]:
        language_key, title_key = _title_key(language, title)
        return self._query("title", (language_key, title_key))

    def by_qid(self, qid: str) -> tuple[DocumentRow, ...]:
        return self._query("qid", (qid,))


@dataclass(frozen=True, slots=True)
class V1ReuseIndex:
    """Immutable lookup maps built from V1 Wikipedia document shards.

    A single Wikipedia page revision may appear under multiple Wikidata QIDs
    because V1 stores each polygon relationship's QID-backed document row.
    Lookup values therefore remain tuples and are sorted deterministically.
    """

    by_page_index: Mapping[PageKey, tuple[DocumentRow, ...]]
    by_title_index: Mapping[tuple[str, str], tuple[DocumentRow, ...]]
    by_qid_index: Mapping[str, tuple[DocumentRow, ...]]
    files: tuple[Path, ...]
    row_count: int
    _store: _PersistentV1Index | None = field(default=None, repr=False, compare=False)

    @property
    def is_ready(self) -> bool:
        """Return whether all configured V1 shards have been indexed."""
        return self._store is None or self._store.is_ready

    def wait_until_ready(self) -> None:
        """Wait for a background index build, if this is a persistent index."""
        if self._store is not None:
            self._store.wait_until_ready()

    def close(self) -> None:
        """Close a persistent index and preserve its committed checkpoints."""
        if self._store is not None:
            self._store.close()

    def by_page(self, language: str, page_id: int) -> tuple[DocumentRow, ...]:
        if self._store is not None:
            return self._store.by_page(language, page_id)
        return self.by_page_index.get((language.casefold(), page_id), ())

    def by_title(self, language: str, title: str) -> tuple[DocumentRow, ...]:
        if self._store is not None:
            return self._store.by_title(language, title)
        return self.by_title_index.get(_title_key(language, title), ())

    def by_titles(
        self,
        keys: Sequence[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[DocumentRow, ...]]:
        """Resolve several titles and return canonicalized lookup keys."""
        if self._store is not None:
            return self._store.by_titles(keys)
        return {
            _title_key(language, title): self.by_title(language, title) for language, title in keys
        }

    def by_qid(self, qid: str) -> tuple[DocumentRow, ...]:
        if self._store is not None:
            return self._store.by_qid(qid)
        return self.by_qid_index.get(qid, ())


def _freeze[Key](
    mapping: dict[Key, list[DocumentRow]],
) -> Mapping[Key, tuple[DocumentRow, ...]]:
    return MappingProxyType(
        {
            key: tuple(sorted(rows, key=lambda item: str(item["document_id"])))
            for key, rows in mapping.items()
        }
    )


def _build_in_memory_index(files: tuple[Path, ...], article_dir: Path) -> V1ReuseIndex:
    by_page: dict[PageKey, list[DocumentRow]] = {}
    by_title: dict[tuple[str, str], list[DocumentRow]] = {}
    by_qid: dict[str, list[DocumentRow]] = {}
    seen_documents: set[str] = set()
    row_count = 0

    for path in files:
        for row in _read_rows(path, legacy_articles=path.parent == article_dir):
            document_id = str(row["document_id"])
            language = str(row["language"])
            page_id = _required_int(row["page_id"], "page_id")
            _required_int(row["revision_id"], "revision_id")
            if document_id in seen_documents:
                continue
            seen_documents.add(document_id)
            row_count += 1
            by_page.setdefault((language.casefold(), page_id), []).append(row)
            by_title.setdefault(_title_key(language, str(row["title"])), []).append(row)
            qid = row.get("wikidata")
            if qid:
                by_qid.setdefault(str(qid), []).append(row)

    return V1ReuseIndex(
        by_page_index=_freeze(by_page),
        by_title_index=_freeze(by_title),
        by_qid_index=_freeze(by_qid),
        files=files,
        row_count=row_count,
    )


def build_v1_reuse_index(
    processed_dir: Path,
    *,
    cache_dir: Path | None = None,
) -> V1ReuseIndex:
    """Build a V1 reuse index, optionally persisted under ``cache_dir``.

    With a cache directory, each shard is fingerprinted and committed to a
    SQLite metadata index independently.  Unchanged shards are not rescanned,
    changed shards replace only their own rows, and document payloads are read
    from bounded Parquet row-group loads when queried.
    """
    files = _effective_paths(processed_dir)
    if cache_dir is None:
        return _build_in_memory_index(files, processed_dir / "articles")
    store = _PersistentV1Index(cache_dir, files)
    return _persistent_index(store, files)


def start_v1_reuse_index(
    processed_dir: Path,
    *,
    cache_dir: Path,
) -> V1ReuseIndex:
    """Start a resumable V1 index build and return its live lookup handle.

    The builder commits one shard at a time on a daemon thread.  Lookups can
    reuse rows from committed shards while the remaining shards are scanned.
    Callers must wait for readiness before treating a miss as absent.  The V2
    direct-enrichment layer may fetch a miss speculatively while indexing, then
    performs this final lookup before accepting the network result.
    """
    files = _effective_paths(processed_dir)
    store = _PersistentV1Index(cache_dir, files, background=True)
    return _persistent_index(store, files, row_count=0)


def _persistent_index(
    store: _PersistentV1Index,
    files: tuple[Path, ...],
    *,
    row_count: int | None = None,
) -> V1ReuseIndex:
    if row_count is None:
        row_count = store.row_count()
    return V1ReuseIndex(
        by_page_index=MappingProxyType({}),
        by_title_index=MappingProxyType({}),
        by_qid_index=MappingProxyType({}),
        files=files,
        row_count=row_count,
        _store=store,
    )


__all__ = ["V1ReuseIndex", "build_v1_reuse_index", "start_v1_reuse_index"]
