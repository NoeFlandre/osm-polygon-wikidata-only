"""Read-only, resumable indexes of V1 Wikipedia document shards.

The V2 runner uses a disk-backed SQLite index by default.  It validates and
indexes one V1 shard at a time under the external V2 cache, so the full V1
document corpus is never held in memory and an interrupted build resumes from
the last committed shard.  Small callers and tests can omit ``cache_dir`` to
use the original in-memory index.
"""

from __future__ import annotations

import logging
import sqlite3
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
_INDEX_SCHEMA_VERSION = 1
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


def _scan_index_rows(
    path: Path, *, legacy_articles: bool
) -> list[tuple[str, str, str, int, int, str, int, int]]:
    """Read only identity columns and row positions for one V1 shard."""
    try:
        parquet_file = pq.ParquetFile(path)
        expected_schema = article_schema() if legacy_articles else wikipedia_document_schema()
        if not parquet_file.schema_arrow.equals(expected_schema, check_metadata=True):
            label = "legacy article" if legacy_articles else "V1 document"
            raise ValueError(f"V1 {label} shard has an invalid schema: {path}")
        columns = tuple(
            column for column in _INDEX_PROJECTION if not legacy_articles or column != "document_id"
        )
        indexed: list[tuple[str, str, str, int, int, str, int, int]] = []
        seen_documents: set[str] = set()
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group, columns=list(columns))
            for row_index, row in enumerate(table.to_pylist()):
                language = str(row["language"])
                page_id = _required_int(row["page_id"], "page_id")
                revision_id = _required_int(row["revision_id"], "revision_id")
                if legacy_articles:
                    document_id = f"{row['wikidata']}:wikipedia:{language}:{page_id}:{revision_id}"
                else:
                    document_id = str(row["document_id"])
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
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


class _PersistentV1Index:
    """Incremental SQLite metadata index with bounded Parquet row loading."""

    def __init__(self, cache_dir: Path, files: tuple[Path, ...]) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_dir / _INDEX_FILENAME
        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._row_cache: OrderedDict[str, DocumentRow] = OrderedDict()
        self._row_cache_limit = 10_000
        self._initialize_schema()
        self._sync(files)

    def _initialize_schema(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, _INDEX_SCHEMA_VERSION):
            with self._connection:
                self._connection.executescript(
                    "DROP TABLE IF EXISTS documents; DROP TABLE IF EXISTS file_state;"
                )
        with self._connection:
            self._connection.executescript(
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
                """
            )
            self._connection.execute(f"PRAGMA user_version={_INDEX_SCHEMA_VERSION}")

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def _sync(self, files: tuple[Path, ...]) -> None:
        current = {str(path.resolve()): (path, path.parent.name == "articles") for path in files}
        known = {
            str(row["path"]): (
                int(row["size"]),
                int(row["mtime_ns"]),
                int(row["ctime_ns"]),
                int(row["inode"]),
                bool(row["legacy"]),
            )
            for row in self._connection.execute("SELECT * FROM file_state")
        }
        stale = set(known) - set(current)
        if stale:
            with self._connection:
                for path in stale:
                    self._connection.execute("DELETE FROM documents WHERE source_path=?", (path,))
                    self._connection.execute("DELETE FROM file_state WHERE path=?", (path,))
            self._row_cache.clear()

        changed = 0
        for position, path in enumerate(files, start=1):
            resolved = str(path.resolve())
            fingerprint = (*self._fingerprint(path), path.parent.name == "articles")
            if known.get(resolved) == fingerprint:
                continue
            LOGGER.info(
                "V2 V1 reuse index: scanning shard %d/%d (%s)", position, len(files), path.name
            )
            indexed = _scan_index_rows(path, legacy_articles=bool(fingerprint[-1]))
            with self._connection:
                self._connection.execute("DELETE FROM documents WHERE source_path=?", (resolved,))
                self._connection.executemany(
                    """
                    INSERT OR REPLACE INTO documents(
                        document_id, source_path, legacy, language, title_key,
                        page_id, revision_id, qid, row_group, row_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(document_id),
                            resolved,
                            int(bool(fingerprint[-1])),
                            str(language).casefold(),
                            _title_key(str(language), str(title))[1],
                            int(page_id),
                            int(revision_id),
                            str(qid),
                            int(row_group),
                            int(row_index),
                        )
                        for (
                            document_id,
                            language,
                            title,
                            page_id,
                            revision_id,
                            qid,
                            row_group,
                            row_index,
                        ) in indexed
                    ],
                )
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO file_state(path, size, mtime_ns, ctime_ns, inode, legacy)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (resolved, *fingerprint),
                )
            self._row_cache.clear()
            changed += 1
            LOGGER.info(
                "V2 V1 reuse index: shard ready %d/%d (%d document identities)",
                position,
                len(files),
                len(indexed),
            )

        row_count = self._connection.execute(
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

    def _query(self, query_name: str, parameters: tuple[object, ...]) -> tuple[DocumentRow, ...]:
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
        rows = list(self._connection.execute(queries[query_name], parameters))
        if not rows:
            return ()
        by_document: dict[str, sqlite3.Row] = {}
        for row in rows:
            by_document.setdefault(str(row["document_id"]), row)
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
    row_count = int(
        store._connection.execute("SELECT COUNT(DISTINCT document_id) FROM documents").fetchone()[0]
    )
    return V1ReuseIndex(
        by_page_index=MappingProxyType({}),
        by_title_index=MappingProxyType({}),
        by_qid_index=MappingProxyType({}),
        files=files,
        row_count=row_count,
        _store=store,
    )


__all__ = ["V1ReuseIndex", "build_v1_reuse_index"]
