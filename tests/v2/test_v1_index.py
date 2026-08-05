import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.schema import ARTICLE_COLUMNS, article_schema
from osm_polygon_wikidata_only.v2.v1_index import (
    V1ReuseIndex,
    build_v1_reuse_index,
    start_v1_reuse_index,
)


def _document(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "document_id": "Q42:wikipedia:en:1:2",
        "article_id": "Q42:en:1:2",
        "wikidata": "Q42",
        "project": "wikipedia",
        "language": "en",
        "site": "enwiki",
        "title": "Douglas Adams",
        "url": "https://en.wikipedia.org/wiki/Douglas_Adams",
        "page_id": 1,
        "revision_id": 2,
        "revision_timestamp": "2020-01-01T00:00:00Z",
        "retrieved_at": "2020-01-01T00:00:00Z",
        "wikidata_label": "Douglas Adams",
        "wikidata_description": "writer",
        "wikidata_aliases": "[]",
        "lead_text": "lead",
        "extract": "extract",
        "full_text": "text",
        "full_text_format": "plain_text",
        "article_length_chars": 4,
        "article_length_words": 1,
        "article_length_tokens_estimate": 1,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "[]",
        "license": "CC BY-SA",
        "attribution": "",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "hash",
    }
    row.update(overrides)
    return row


def _write_documents(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "wikipedia" / "documents" / "region.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=wikipedia_document_schema()), path)
    return path


def _write_legacy_articles(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "articles" / "region.parquet"
    path.parent.mkdir(parents=True)
    legacy_rows = [{column: row[column] for column in ARTICLE_COLUMNS} for row in rows]
    pq.write_table(pa.Table.from_pylist(legacy_rows, schema=article_schema()), path)
    return path


def test_index_lookup_by_title_page_and_qid(tmp_path: Path) -> None:
    path = _write_documents(tmp_path, [_document()])
    index = build_v1_reuse_index(tmp_path)
    assert isinstance(index, V1ReuseIndex)
    assert index.by_page("en", 1)[0]["document_id"] == "Q42:wikipedia:en:1:2"
    assert index.by_title("en", "Douglas Adams")[0]["wikidata"] == "Q42"
    assert index.by_qid("Q42")[0]["page_id"] == 1
    assert path.is_file()


def test_index_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "wikipedia" / "documents" / "region.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"title": ["bad"]}), path)
    with pytest.raises(ValueError, match="schema"):
        build_v1_reuse_index(tmp_path)


def test_validated_parquet_file_closes_handle_when_schema_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema mismatch must not leak the opened ParquetFile handle.

    ``_validated_parquet_file`` opens the shard before checking its schema.
    When the schema check fails it must close the handle it opened rather
    than relying on callers (which observe the function raising and so never
    receive a handle to close).
    """
    path = tmp_path / "wikipedia" / "documents" / "region.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"title": ["bad"]}), path)

    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    original = v1_index.pq.ParquetFile
    opened: list[object] = []
    closed: list[object] = []

    class _TrackedParquetFile:
        def __init__(self, source: Path) -> None:
            self._inner = original(source)
            opened.append(self)

        @property
        def schema_arrow(self):
            return self._inner.schema_arrow

        @property
        def num_row_groups(self):
            return self._inner.num_row_groups

        def read_row_group(self, row_group: int, columns=None):
            return self._inner.read_row_group(row_group, columns=columns)

        def read(self):
            return self._inner.read()

        def close(self) -> None:
            if self not in closed:
                self._inner.close()
                closed.append(self)

    monkeypatch.setattr(v1_index.pq, "ParquetFile", _TrackedParquetFile)

    with pytest.raises(ValueError, match="schema"):
        v1_index._validated_parquet_file(path, legacy_articles=False)

    assert opened, "_validated_parquet_file must open a ParquetFile"
    assert closed == opened, "ParquetFile handle leaked on schema validation failure"


def test_index_allows_shared_page_revision_for_distinct_qids(tmp_path: Path) -> None:
    _write_documents(
        tmp_path,
        [
            _document(),
            _document(
                document_id="Q43:wikipedia:en:1:2",
                article_id="Q43:en:1:2",
                wikidata="Q43",
            ),
        ],
    )
    index = build_v1_reuse_index(tmp_path)
    assert [row["document_id"] for row in index.by_page("en", 1)] == [
        "Q42:wikipedia:en:1:2",
        "Q43:wikipedia:en:1:2",
    ]
    assert index.by_title("en", "Douglas Adams") == index.by_page("en", 1)


def test_index_row_group_builder_uses_columnar_values_without_row_dicts() -> None:
    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    class Column:
        def __init__(self, values: list[object]) -> None:
            self._values = values

        def to_pylist(self) -> list[object]:
            return self._values

    class ColumnarTable:
        num_rows = 1

        def __init__(self) -> None:
            self._columns = {
                "document_id": Column(["Q42:wikipedia:en:1:2"]),
                "language": Column(["en"]),
                "title": Column(["Douglas Adams"]),
                "page_id": Column([1]),
                "revision_id": Column([2]),
                "wikidata": Column(["Q42"]),
            }

        def column(self, name: str) -> Column:
            return self._columns[name]

        def to_pylist(self) -> list[dict[str, object]]:
            raise AssertionError("row dictionaries must not be materialized")

    assert v1_index._index_rows_from_table(ColumnarTable(), legacy_articles=False, row_group=4) == [
        ("Q42:wikipedia:en:1:2", "en", "Douglas Adams", 1, 2, "Q42", 4, 0)
    ]


def test_index_reuses_legacy_articles_when_canonical_documents_are_absent(
    tmp_path: Path,
) -> None:
    path = _write_legacy_articles(tmp_path, [_document()])
    index = build_v1_reuse_index(tmp_path)
    assert index.files == (path,)
    assert index.by_title("en", "Douglas Adams")[0]["document_id"] == "Q42:wikipedia:en:1:2"


def test_persistent_index_reuses_completed_shards_without_rescanning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_documents(tmp_path, [_document()])
    cache_dir = tmp_path / "v2-cache" / "v1-index"
    first = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert first.by_title("en", "Douglas Adams")
    assert (cache_dir / "v1_reuse_index.sqlite3").is_file()

    def fail_if_scanned(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        raise AssertionError("an unchanged V1 shard was rescanned")

    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    monkeypatch.setattr(v1_index, "_scan_index_rows", fail_if_scanned)
    second = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert second.by_title("en", "Douglas Adams")[0]["document_id"] == "Q42:wikipedia:en:1:2"


def test_persistent_index_defers_unused_secondary_indexes_until_lookup(
    tmp_path: Path,
) -> None:
    """V2 title reuse must not build page/QID indexes it never queries."""
    _write_documents(tmp_path, [_document()])
    cache_dir = tmp_path / "v2-cache" / "v1-index"
    index = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)

    import sqlite3

    connection = sqlite3.connect(cache_dir / "v1_reuse_index.sqlite3")
    try:
        indexes = {str(row[1]) for row in connection.execute("PRAGMA index_list(documents)")}
    finally:
        connection.close()
    assert "documents_title" in indexes
    assert "documents_page" not in indexes
    assert "documents_qid" not in indexes

    assert index.by_page("en", 1)[0]["document_id"] == "Q42:wikipedia:en:1:2"

    connection = sqlite3.connect(cache_dir / "v1_reuse_index.sqlite3")
    try:
        indexes_after_lookup = {
            str(row[1]) for row in connection.execute("PRAGMA index_list(documents)")
        }
    finally:
        connection.close()
    assert "documents_page" in indexes_after_lookup
    index.close()


def test_persistent_index_migrates_existing_secondary_indexes_to_lazy_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reopening an older cache removes write-only indexes before rescanning."""
    _write_documents(tmp_path, [_document()])
    cache_dir = tmp_path / "v2-cache" / "v1-index"
    first = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    first.close()

    import sqlite3

    connection = sqlite3.connect(cache_dir / "v1_reuse_index.sqlite3")
    try:
        with connection:
            connection.execute(
                "CREATE INDEX documents_page ON documents(language, page_id, document_id)"
            )
            connection.execute("CREATE INDEX documents_qid ON documents(qid, document_id)")
            connection.execute("PRAGMA user_version=2")
    finally:
        connection.close()

    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    def fail_scan(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        raise AssertionError("unchanged shard was rescanned during cache migration")

    monkeypatch.setattr(v1_index, "_scan_index_row_group", fail_scan)
    reopened = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    try:
        connection = sqlite3.connect(cache_dir / "v1_reuse_index.sqlite3")
        try:
            indexes = {str(row[1]) for row in connection.execute("PRAGMA index_list(documents)")}
        finally:
            connection.close()
        assert "documents_page" not in indexes
        assert "documents_qid" not in indexes
        assert reopened.by_qid("Q42")[0]["document_id"] == "Q42:wikipedia:en:1:2"
    finally:
        reopened.close()


def test_persistent_index_caches_completed_title_lookups(
    tmp_path: Path,
) -> None:
    """Repeated title reuse should not rerun the same SQLite query."""
    _write_documents(tmp_path, [_document()])
    index = build_v1_reuse_index(tmp_path, cache_dir=tmp_path / "v2-cache" / "v1-index")
    store = index._store
    assert store is not None
    reader = store._reader()
    statements: list[str] = []
    reader.set_trace_callback(statements.append)

    assert index.by_title("en", "Douglas Adams")
    assert index.by_title("en", "Douglas Adams")

    assert sum("FROM documents" in statement for statement in statements) == 1
    index.close()


def test_persistent_index_batches_title_lookups(
    tmp_path: Path,
) -> None:
    """A group of title lookups uses one bounded SQLite query."""
    rows = [
        _document(document_id="Q42:wikipedia:en:1:2", title="Douglas Adams", page_id=1),
        _document(document_id="Q43:wikipedia:en:2:3", title="Douglas Noel", page_id=2),
    ]
    _write_documents(tmp_path, rows)
    index = build_v1_reuse_index(tmp_path, cache_dir=tmp_path / "v2-cache" / "v1-index")
    store = index._store
    assert store is not None
    reader = store._reader()
    statements: list[str] = []
    reader.set_trace_callback(statements.append)

    matches = index.by_titles((("en", "Douglas Adams"), ("en", "Douglas Noel")))

    assert matches[("en", "douglas adams")][0]["document_id"] == "Q42:wikipedia:en:1:2"
    assert matches[("en", "douglas noel")][0]["document_id"] == "Q43:wikipedia:en:2:3"
    assert sum("FROM documents" in statement for statement in statements) == 1
    index.close()


def test_persistent_lookup_closes_parquet_handles_after_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated V1 lookups must not consume one file descriptor per query."""
    _write_documents(tmp_path, [_document()])
    index = build_v1_reuse_index(tmp_path, cache_dir=tmp_path / "v2-cache" / "v1-index")

    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    original = v1_index.pq.ParquetFile
    opened: list[object] = []
    closed: list[object] = []

    class TrackedParquetFile:
        def __init__(self, path: Path) -> None:
            self._inner = original(path)
            opened.append(self)

        def __enter__(self) -> "TrackedParquetFile":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def close(self) -> None:
            if self not in closed:
                self._inner.close()
                closed.append(self)

        def read_row_group(self, row_group: int):
            return self._inner.read_row_group(row_group)

    monkeypatch.setattr(v1_index.pq, "ParquetFile", TrackedParquetFile)
    assert index.by_title("en", "Douglas Adams")
    assert opened
    assert closed == opened
    index.close()


def test_index_close_does_not_close_sqlite_while_worker_is_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted run must not close SQLite under an active index worker."""
    _write_documents(tmp_path, [_document()])
    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    started = threading.Event()
    release = threading.Event()

    def blocked_sync(_store: object, _files: tuple[Path, ...]) -> bool:
        started.set()
        release.wait(timeout=2)
        return True

    monkeypatch.setattr(v1_index._PersistentV1Index, "_sync", blocked_sync)
    monkeypatch.setattr(v1_index, "_INDEX_SHUTDOWN_TIMEOUT_S", 0.01)
    index = start_v1_reuse_index(tmp_path, cache_dir=tmp_path / "v2-cache" / "v1-index")
    store = index._store
    assert store is not None
    assert started.wait(timeout=2)
    connection = store._connection
    assert connection is not None

    index.close()

    assert store._thread is not None and store._thread.is_alive()
    assert store._connection is connection
    release.set()
    store._thread.join(timeout=2)
    assert not store._thread.is_alive()

    index.close()
    assert store._connection is None


def test_persistent_index_does_not_write_for_unchanged_shards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_documents(tmp_path, [_document()])
    cache_dir = tmp_path / "v2-cache" / "v1-index"
    build_v1_reuse_index(tmp_path, cache_dir=cache_dir)

    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    statements: list[str] = []
    original = v1_index._PersistentV1Index._open_connection

    def open_traced(store: object):
        connection = original(store)  # type: ignore[arg-type]
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(v1_index._PersistentV1Index, "_open_connection", open_traced)
    build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert not any("DELETE FROM scan_progress" in statement for statement in statements)


def test_persistent_index_reads_next_row_group_during_current_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "wikipedia" / "documents" / "region.parquet"
    path.parent.mkdir(parents=True)
    rows = [
        _document(),
        _document(
            document_id="Q43:wikipedia:en:2:3",
            article_id="Q43:en:2:3",
            wikidata="Q43",
            title="Second page",
            page_id=2,
            revision_id=3,
        ),
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=wikipedia_document_schema()),
        path,
        row_group_size=1,
    )
    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    second_scanned = threading.Event()
    original_scan = v1_index._scan_index_row_group

    def scan(
        path_arg: Path,
        *,
        legacy_articles: bool,
        row_group: int,
        parquet_file: object = None,
    ):
        if row_group == 1:
            second_scanned.set()
        return original_scan(
            path_arg,
            legacy_articles=legacy_articles,
            row_group=row_group,
            parquet_file=parquet_file,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(v1_index, "_scan_index_row_group", scan)
    original_commit = v1_index._PersistentV1Index._commit_indexed_row_group

    def commit(store: object, *args: object, **kwargs: object) -> None:
        if kwargs["row_group"] == 0:
            assert second_scanned.wait(timeout=2)
        original_commit(store, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(v1_index._PersistentV1Index, "_commit_indexed_row_group", commit)
    index = build_v1_reuse_index(tmp_path, cache_dir=tmp_path / "cache")
    assert index.by_title("en", "Second page")[0]["document_id"] == "Q43:wikipedia:en:2:3"


def test_persistent_index_adds_new_shards_incrementally(tmp_path: Path) -> None:
    _write_documents(tmp_path, [_document()])
    cache_dir = tmp_path / "v2-cache" / "v1-index"
    build_v1_reuse_index(tmp_path, cache_dir=cache_dir)

    second = _document(
        document_id="Q43:wikipedia:fr:2:3",
        article_id="Q43:fr:2:3",
        wikidata="Q43",
        language="fr",
        site="frwiki",
        title="Douglas Adams FR",
        page_id=2,
        revision_id=3,
    )
    path = tmp_path / "wikipedia" / "documents" / "second.parquet"
    pq.write_table(pa.Table.from_pylist([second], schema=wikipedia_document_schema()), path)

    index = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert index.by_title("en", "Douglas Adams")
    assert index.by_title("fr", "Douglas Adams FR")[0]["document_id"] == "Q43:wikipedia:fr:2:3"


def test_persistent_index_resumes_after_an_interrupted_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_documents(tmp_path, [_document()])
    cache_dir = tmp_path / "v2-cache" / "v1-index"
    build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    second = _document(
        document_id="Q43:wikipedia:fr:2:3",
        article_id="Q43:fr:2:3",
        wikidata="Q43",
        language="fr",
        site="frwiki",
        title="Douglas Adams FR",
        page_id=2,
        revision_id=3,
    )
    second_path = tmp_path / "wikipedia" / "documents" / "second.parquet"
    pq.write_table(
        pa.Table.from_pylist([second], schema=wikipedia_document_schema()),
        second_path,
    )

    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    original = v1_index._scan_index_row_group
    scanned: list[str] = []

    def fail_on_second(
        path: Path,
        *,
        legacy_articles: bool,
        row_group: int,
        parquet_file: object = None,
    ):
        scanned.append(path.name)
        if path.name == second_path.name:
            raise OSError("interrupted index build")
        return original(
            path,
            legacy_articles=legacy_articles,
            row_group=row_group,
            parquet_file=parquet_file,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(v1_index, "_scan_index_row_group", fail_on_second)
    with pytest.raises(OSError, match="interrupted index build"):
        build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert scanned == [second_path.name]

    monkeypatch.setattr(v1_index, "_scan_index_row_group", original)
    index = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert index.by_title("fr", "Douglas Adams FR")[0]["document_id"] == "Q43:wikipedia:fr:2:3"


def test_background_index_exposes_committed_rows_before_final_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_documents(tmp_path, [_document()])
    second = tmp_path / "wikipedia" / "documents" / "second.parquet"
    second_row = _document(
        document_id="Q43:wikipedia:fr:2:3",
        article_id="Q43:fr:2:3",
        wikidata="Q43",
        language="fr",
        site="frwiki",
        title="Douglas Adams FR",
        page_id=2,
        revision_id=3,
    )
    pq.write_table(pa.Table.from_pylist([second_row], schema=wikipedia_document_schema()), second)

    import threading

    second_started = threading.Event()
    release_second = threading.Event()
    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    original = v1_index._scan_index_row_group

    def scan(
        path: Path,
        *,
        legacy_articles: bool,
        row_group: int,
        parquet_file: object = None,
    ):
        if path == second:
            second_started.set()
            assert release_second.wait(timeout=2)
        return original(
            path,
            legacy_articles=legacy_articles,
            row_group=row_group,
            parquet_file=parquet_file,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(v1_index, "_scan_index_row_group", scan)
    index = start_v1_reuse_index(tmp_path, cache_dir=tmp_path / "v2-cache" / "v1-index")
    assert second_started.wait(timeout=2)
    assert index.by_title("en", "Douglas Adams")[0]["document_id"] == "Q42:wikipedia:en:1:2"
    assert not index.is_ready
    release_second.set()
    index.wait_until_ready()
    assert index.by_title("fr", "Douglas Adams FR")[0]["document_id"] == "Q43:wikipedia:fr:2:3"
    index.close()


def test_background_index_handle_returns_before_storage_initialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_documents(tmp_path, [_document()])
    import threading

    initialized = threading.Event()
    release = threading.Event()
    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    original = v1_index._PersistentV1Index._initialize_schema

    def initialize_slow(store: object) -> None:
        initialized.set()
        assert release.wait(timeout=2)
        original(store)  # type: ignore[arg-type]

    monkeypatch.setattr(v1_index._PersistentV1Index, "_initialize_schema", initialize_slow)
    index = start_v1_reuse_index(tmp_path, cache_dir=tmp_path / "v2-cache" / "v1-index")
    assert initialized.wait(timeout=2)
    assert not index.is_ready
    assert index.by_title("en", "Douglas Adams") == ()
    release.set()
    index.wait_until_ready()
    assert index.by_title("en", "Douglas Adams")
    index.close()


def test_background_index_opens_sqlite_on_worker_thread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The caller must not wait for SQLite WAL setup on the external disk."""
    _write_documents(tmp_path, [_document()])
    import threading

    opened = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    result: dict[str, V1ReuseIndex] = {}
    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    original = v1_index._PersistentV1Index._open_connection

    def open_slow(store: object):
        opened.set()
        assert release.wait(timeout=2)
        return original(store)  # type: ignore[arg-type]

    monkeypatch.setattr(v1_index._PersistentV1Index, "_open_connection", open_slow)

    def launch() -> None:
        result["index"] = start_v1_reuse_index(
            tmp_path,
            cache_dir=tmp_path / "v2-cache" / "v1-index",
        )
        returned.set()

    thread = threading.Thread(target=launch)
    thread.start()
    assert opened.wait(timeout=2)
    assert returned.wait(timeout=0.2)
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    index = result["index"]
    index.wait_until_ready()
    index.close()


def test_persistent_index_resumes_inside_an_interrupted_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "wikipedia" / "documents" / "region.parquet"
    path.parent.mkdir(parents=True)
    rows = [
        _document(),
        _document(
            document_id="Q43:wikipedia:en:2:3",
            article_id="Q43:en:2:3",
            wikidata="Q43",
            title="Second page",
            page_id=2,
            revision_id=3,
        ),
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=wikipedia_document_schema()),
        path,
        row_group_size=1,
    )
    cache_dir = tmp_path / "v2-cache" / "v1-index"

    import osm_polygon_wikidata_only.v2.v1_index as v1_index

    original = v1_index._scan_index_row_group
    calls: list[int] = []

    def scan(
        path_arg: Path,
        *,
        legacy_articles: bool,
        row_group: int,
        parquet_file: object = None,
    ):
        calls.append(row_group)
        if row_group == 1 and len(calls) == 2:
            raise OSError("interrupted row group")
        return original(
            path_arg,
            legacy_articles=legacy_articles,
            row_group=row_group,
            parquet_file=parquet_file,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(v1_index, "_scan_index_row_group", scan)
    with pytest.raises(OSError, match="interrupted row group"):
        build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert calls == [0, 1]

    connection = __import__("sqlite3").connect(cache_dir / "v1_reuse_index.sqlite3")
    progress = connection.execute(
        "SELECT next_row_group FROM scan_progress WHERE path=?",
        (str(path.resolve()),),
    ).fetchone()
    assert progress == (1,)
    connection.close()

    monkeypatch.setattr(v1_index, "_scan_index_row_group", original)
    resumed = build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert resumed.by_title("en", "Douglas Adams")[0]["document_id"] == "Q42:wikipedia:en:1:2"
    assert resumed.by_title("en", "Second page")[0]["document_id"] == "Q43:wikipedia:en:2:3"
