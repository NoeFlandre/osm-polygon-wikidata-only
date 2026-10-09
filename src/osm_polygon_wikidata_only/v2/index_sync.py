"""Shard synchronization for the persistent V1 reuse index.

Fingerprints each V1 shard, scans changed shards one row group at a time and
commits progress so an interrupted build resumes where it stopped.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.v2 import index_scanning
from osm_polygon_wikidata_only.v2.fingerprints import FileStatFingerprint
from osm_polygon_wikidata_only.v2.index_queries import title_key as _title_key

LOGGER = logging.getLogger("osm_polygon_wikidata_only.v2.v1_index")


@dataclass(frozen=True, slots=True)
class _ShardScan:
    """Parquet shard path, open file handle and row-group layout for one scan."""

    path: Path
    legacy_articles: bool
    total_row_groups: int
    parquet_file: pq.ParquetFile


DocumentRow = index_scanning.DocumentRow
_effective_paths = index_scanning.effective_paths
_scan_index_row_group = index_scanning.scan_index_row_group
_scan_index_rows = index_scanning.scan_index_rows
_validated_parquet_file = index_scanning.validated_parquet_file


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


def count_distinct_documents(connection: sqlite3.Connection) -> int:
    """Count indexed document identities for a newly changed index."""
    return int(
        connection.execute("SELECT COUNT(DISTINCT document_id) FROM documents").fetchone()[0]
    )


def read_cached_row_count(connection: sqlite3.Connection) -> int | None:
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


class PersistentIndexSync:
    """Mixin that keeps the SQLite document index in step with V1 shards."""

    _writer_connection: sqlite3.Connection
    _stop: threading.Event
    _index_reader_executor: ThreadPoolExecutor | None
    _db_path: Path
    _row_cache: OrderedDict[str, DocumentRow]
    _query_cache: OrderedDict[tuple[str, tuple[object, ...]], tuple[DocumentRow, ...]]
    _query_cache_lock: threading.Lock
    _secondary_indexes: set[str]

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
        shard: _ShardScan,
        *,
        start_row_group: int,
        resolved: str,
        fingerprint: tuple[int, int, int, int, bool],
        position: int,
        total_files: int,
    ) -> bool:
        """Scan and commit a shard's row groups with one-row-group lookahead."""
        path, legacy_articles, total_row_groups, parquet_file = (
            shard.path,
            shard.legacy_articles,
            shard.total_row_groups,
            shard.parquet_file,
        )
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
                _ShardScan(
                    path=path,
                    legacy_articles=legacy_articles,
                    total_row_groups=total_row_groups,
                    parquet_file=parquet_file,
                ),
                start_row_group=start_row_group,
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
        row_count = read_cached_row_count(connection)
        if row_count is None:
            row_count = count_distinct_documents(connection)
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
