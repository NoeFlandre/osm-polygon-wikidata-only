"""Read-only, resumable indexes of V1 Wikipedia document shards.

The V2 runner uses a disk-backed SQLite index by default.  It validates and
indexes one V1 shard at a time under the external V2 cache, so the full V1
document corpus is never held in memory and an interrupted build resumes from
the last committed Parquet row group.  Small callers and tests can omit
``cache_dir`` to use the original in-memory index.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import article_schema

LOGGER = logging.getLogger(__name__)

DocumentRow = dict[str, object]
PageKey = tuple[str, int]
_INDEX_SCHEMA_VERSION = 2
_INDEX_FILENAME = "v1_reuse_index.sqlite3"
_INDEX_PROJECTION = ("document_id", "language", "title", "page_id", "revision_id", "wikidata")


def _title_key(language: str, title: str) -> tuple[str, str]:
    return language.casefold(), " ".join(title.replace("_", " ").split()).casefold()


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"V1 document field {field!r} is not an integer")
    return value


def _effective_paths(processed_dir: Path) -> tuple[Path, ...]:
    document_dir = processed_dir / "wikipedia" / "documents"
    article_dir = processed_dir / "articles"
    canonical_paths = {path.stem: path for path in document_dir.glob("*.parquet")}
    # V1 releases before the canonical-document migration stored Wikipedia
    # rows under ``articles/``. Prefer canonical shards when both exist, but
    # keep the legacy fallback so V2 never refetches an already-finalized page.
    effective_paths = dict(canonical_paths)
    for path in article_dir.glob("*.parquet"):
        effective_paths.setdefault(path.stem, path)
    return tuple(sorted(effective_paths.values(), key=lambda path: path.name))


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


def _index_columns(legacy_articles: bool) -> tuple[str, ...]:
    return tuple(
        column for column in _INDEX_PROJECTION if not legacy_articles or column != "document_id"
    )


def _validated_parquet_file(path: Path, *, legacy_articles: bool):
    """Open and schema-check one V1 shard before reading its row groups."""
    try:
        parquet_file = pq.ParquetFile(path)
        expected_schema = article_schema() if legacy_articles else wikipedia_document_schema()
        if not parquet_file.schema_arrow.equals(expected_schema, check_metadata=True):
            label = "legacy article" if legacy_articles else "V1 document"
            raise ValueError(f"V1 {label} shard has an invalid schema: {path}")
        return parquet_file
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"V1 document shard is unreadable: {path}: {exc}") from exc


def _scan_index_row_group(
    path: Path,
    *,
    legacy_articles: bool,
    row_group: int,
    parquet_file: pq.ParquetFile | None = None,
) -> list[tuple[str, str, str, int, int, str, int, int]]:
    """Read identity columns and row positions for one committed row group."""
    try:
        validated = parquet_file or _validated_parquet_file(path, legacy_articles=legacy_articles)
        table = validated.read_row_group(
            row_group,
            columns=list(_index_columns(legacy_articles)),
        )
        indexed: list[tuple[str, str, str, int, int, str, int, int]] = []
        for row_index, row in enumerate(table.to_pylist()):
            language = str(row["language"])
            page_id = _required_int(row["page_id"], "page_id")
            revision_id = _required_int(row["revision_id"], "revision_id")
            if legacy_articles:
                document_id = f"{row['wikidata']}:wikipedia:{language}:{page_id}:{revision_id}"
            else:
                document_id = str(row["document_id"])
            indexed.append(
                (
                    document_id,
                    language,
                    str(row["title"]),
                    page_id,
                    revision_id,
                    str(row.get("wikidata") or ""),
                    row_group,
                    row_index,
                )
            )
        return indexed
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"V1 document shard is unreadable: {path}: {exc}") from exc


def _scan_index_rows(
    path: Path, *, legacy_articles: bool
) -> list[tuple[str, str, str, int, int, str, int, int]]:
    """Read identity columns and row positions for one V1 shard."""
    parquet_file = _validated_parquet_file(path, legacy_articles=legacy_articles)
    indexed: list[tuple[str, str, str, int, int, str, int, int]] = []
    seen_documents: set[str] = set()
    for row_group in range(parquet_file.num_row_groups):
        for row in _scan_index_row_group(
            path,
            legacy_articles=legacy_articles,
            row_group=row_group,
        ):
            if row[0] in seen_documents:
                continue
            seen_documents.add(row[0])
            indexed.append(row)
    return indexed


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
        self._reader_local = threading.local()
        self._reader_connections: list[sqlite3.Connection] = []
        self._reader_lock = threading.Lock()
        self._materialize_lock = threading.Lock()
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
        """Stop after the current row-group transaction, preserving checkpoints."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30)
        if thread is not None and thread.is_alive():
            LOGGER.warning("V2 V1 reuse index did not stop within 30 seconds")

    def close(self) -> None:
        """Stop indexing if needed and close the writer connection."""
        self.cancel()
        with self._reader_lock:
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
            return int(
                connection.execute("SELECT COUNT(DISTINCT document_id) FROM documents").fetchone()[
                    0
                ]
            )
        finally:
            connection.close()

    def _reader(self) -> sqlite3.Connection:
        connection = getattr(self._reader_local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                self._db_path,
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA query_only=ON")
            self._reader_local.connection = connection
            with self._reader_lock:
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
            self._initialized.set()
            self._ready.set()

    def _initialize_schema(self) -> None:
        connection = self._writer_connection
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, 1, _INDEX_SCHEMA_VERSION):
            with connection:
                connection.executescript(
                    "DROP TABLE IF EXISTS documents; DROP TABLE IF EXISTS file_state; "
                    "DROP TABLE IF EXISTS scan_progress;"
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
                CREATE INDEX IF NOT EXISTS documents_title
                    ON documents(language, title_key, document_id);
                CREATE INDEX IF NOT EXISTS documents_page
                    ON documents(language, page_id, document_id);
                CREATE INDEX IF NOT EXISTS documents_qid
                    ON documents(qid, document_id);
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
                """
            )
            connection.execute(f"PRAGMA user_version={_INDEX_SCHEMA_VERSION}")

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def _sync(self, files: tuple[Path, ...]) -> bool:
        connection = self._writer_connection
        current = {str(path.resolve()): (path, path.parent.name == "articles") for path in files}
        known = {
            str(row["path"]): (
                int(row["size"]),
                int(row["mtime_ns"]),
                int(row["ctime_ns"]),
                int(row["inode"]),
                bool(row["legacy"]),
            )
            for row in connection.execute("SELECT * FROM file_state")
        }
        progress = {
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
        stale = set(known) - set(current)
        if stale:
            with connection:
                for path in stale:
                    connection.execute("DELETE FROM documents WHERE source_path=?", (path,))
                    connection.execute("DELETE FROM file_state WHERE path=?", (path,))
                    connection.execute("DELETE FROM scan_progress WHERE path=?", (path,))
            self._row_cache.clear()

        changed = 0
        for position, path in enumerate(files, start=1):
            if self._stop.is_set():
                LOGGER.info(
                    "V2 V1 reuse index stopped after %d/%d shards; completed work is resumable",
                    position - 1,
                    len(files),
                )
                return False
            resolved = str(path.resolve())
            fingerprint = (*self._fingerprint(path), path.parent.name == "articles")
            if known.get(resolved) == fingerprint:
                with connection:
                    connection.execute("DELETE FROM scan_progress WHERE path=?", (resolved,))
                continue
            LOGGER.info(
                "V2 V1 reuse index: scanning shard %d/%d (%s)", position, len(files), path.name
            )
            legacy_articles = bool(fingerprint[-1])
            parquet_file = _validated_parquet_file(path, legacy_articles=legacy_articles)
            total_row_groups = parquet_file.num_row_groups
            previous = progress.get(resolved)
            matching_progress = (
                previous is not None
                and previous[:5] == fingerprint
                and previous[5] == total_row_groups
                and 0 <= previous[6] <= total_row_groups
            )
            start_row_group = previous[6] if matching_progress and previous is not None else 0
            if start_row_group == 0:
                with connection:
                    connection.execute("DELETE FROM documents WHERE source_path=?", (resolved,))
                    connection.execute("DELETE FROM file_state WHERE path=?", (resolved,))
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO scan_progress(
                            path, size, mtime_ns, ctime_ns, inode, legacy,
                            total_row_groups, next_row_group
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                        """,
                        (resolved, *fingerprint, total_row_groups),
                    )
                self._row_cache.clear()
            for row_group in range(start_row_group, total_row_groups):
                if self._stop.is_set():
                    LOGGER.info(
                        "V2 V1 reuse index stopped in shard %d/%d at row group %d/%d; "
                        "completed work is resumable",
                        position,
                        len(files),
                        row_group,
                        total_row_groups,
                    )
                    return False
                indexed = _scan_index_row_group(
                    path,
                    legacy_articles=legacy_articles,
                    row_group=row_group,
                    parquet_file=parquet_file,
                )
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
                                int(legacy_articles),
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
                LOGGER.info(
                    "V2 V1 reuse index: shard %d/%d row group %d/%d ready (%d identities)",
                    position,
                    len(files),
                    row_group + 1,
                    total_row_groups,
                    len(indexed),
                )
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO file_state(path, size, mtime_ns, ctime_ns, inode, legacy)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (resolved, *fingerprint),
                )
                connection.execute("DELETE FROM scan_progress WHERE path=?", (resolved,))
            self._row_cache.clear()
            changed += 1
            LOGGER.info(
                "V2 V1 reuse index: shard ready %d/%d",
                position,
                len(files),
            )

        row_count = connection.execute(
            "SELECT COUNT(DISTINCT document_id) FROM documents"
        ).fetchone()[0]
        if changed or stale:
            LOGGER.info(
                "V2 V1 reuse index ready: %d/%d shards; %d document identities; cache=%s",
                len(files),
                len(files),
                row_count,
                self._db_path,
            )
        else:
            LOGGER.info(
                "V2 V1 reuse index reused: %d cached shards; %d document identities; cache=%s",
                len(files),
                row_count,
                self._db_path,
            )
        return True

    def _query(self, query_name: str, parameters: tuple[object, ...]) -> tuple[DocumentRow, ...]:
        if not self._initialized.is_set():
            return ()
        self._raise_error()
        queries = {
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
        rows = list(self._reader().execute(queries[query_name], parameters))
        if not rows:
            return ()
        by_document: dict[str, sqlite3.Row] = {}
        for row in rows:
            by_document.setdefault(str(row["document_id"]), row)
        with self._materialize_lock:
            materialized = self._materialize(tuple(by_document.values()))
        return tuple(materialized[document_id] for document_id in sorted(materialized))

    def _materialize(self, references: tuple[sqlite3.Row, ...]) -> dict[str, DocumentRow]:
        result: dict[str, DocumentRow] = {}
        grouped: dict[tuple[str, bool, int], list[sqlite3.Row]] = {}
        for reference in references:
            document_id = str(reference["document_id"])
            cached = self._row_cache.get(document_id)
            if cached is not None:
                self._row_cache.move_to_end(document_id)
                result[document_id] = cached
                continue
            group = (
                str(reference["source_path"]),
                bool(reference["legacy"]),
                int(reference["row_group"]),
            )
            grouped.setdefault(group, []).append(reference)

        for (source_path, legacy, row_group), group_references in grouped.items():
            parquet_file = pq.ParquetFile(source_path)
            table = parquet_file.read_row_group(row_group)
            raw_rows = table.to_pylist()
            for reference in group_references:
                document_id = str(reference["document_id"])
                row = raw_rows[int(reference["row_index"])]
                normalized = (
                    wikipedia_document_from_article_row(row).to_dict() if legacy else dict(row)
                )
                self._row_cache[document_id] = normalized
                self._row_cache.move_to_end(document_id)
                while len(self._row_cache) > self._row_cache_limit:
                    self._row_cache.popitem(last=False)
                result[document_id] = normalized
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
