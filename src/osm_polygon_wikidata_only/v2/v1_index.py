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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from osm_polygon_wikidata_only.v2 import index_queries, index_scanning
from osm_polygon_wikidata_only.v2.index_queries import (
    PersistentIndexQueries,
    first_query_rows_by_document,
    group_materialization_references,
    group_title_rows,
    materialize_group,
    normalized_title_keys,
    partition_title_cache,
    title_chunk_results,
    title_key,
)
from osm_polygon_wikidata_only.v2.index_sync import (
    PersistentIndexSync,
    count_distinct_documents,
    read_cached_row_count,
)

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

# Keep these private aliases stable for existing internal callers and tests
# while the query/materialization implementation lives in ``index_queries``.
_title_key = title_key
_normalized_title_keys = normalized_title_keys
_group_title_rows = group_title_rows
_title_chunk_results = title_chunk_results
_partition_title_cache = partition_title_cache
_first_query_rows_by_document = first_query_rows_by_document
_group_materialization_references = group_materialization_references
_materialize_group = materialize_group
_QUERY_SQL = index_queries._QUERY_SQL
_TITLE_QUERY_BATCH_SIZE = index_queries._TITLE_QUERY_BATCH_SIZE

PageKey = tuple[str, int]
_INDEX_SCHEMA_VERSION = 4
_INDEX_FILENAME = "v1_reuse_index.sqlite3"
_INDEX_SHUTDOWN_TIMEOUT_S = 5.0
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


class _PersistentV1Index(PersistentIndexQueries, PersistentIndexSync):
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
            cached = read_cached_row_count(connection)
            if cached is not None:
                return cached
            return count_distinct_documents(connection)
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
