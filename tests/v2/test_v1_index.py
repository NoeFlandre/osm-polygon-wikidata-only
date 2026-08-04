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

    original = v1_index._scan_index_rows
    scanned: list[str] = []

    def fail_on_second(path: Path, *, legacy_articles: bool):
        scanned.append(path.name)
        if path.name == second_path.name:
            raise OSError("interrupted index build")
        return original(path, legacy_articles=legacy_articles)

    monkeypatch.setattr(v1_index, "_scan_index_rows", fail_on_second)
    with pytest.raises(OSError, match="interrupted index build"):
        build_v1_reuse_index(tmp_path, cache_dir=cache_dir)
    assert scanned == [second_path.name]

    monkeypatch.setattr(v1_index, "_scan_index_rows", original)
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

    original = v1_index._scan_index_rows

    def scan(path: Path, *, legacy_articles: bool):
        if path == second:
            second_started.set()
            assert release_second.wait(timeout=2)
        return original(path, legacy_articles=legacy_articles)

    monkeypatch.setattr(v1_index, "_scan_index_rows", scan)
    index = start_v1_reuse_index(tmp_path, cache_dir=tmp_path / "v2-cache" / "v1-index")
    assert second_started.wait(timeout=2)
    assert index.by_title("en", "Douglas Adams")[0]["document_id"] == "Q42:wikipedia:en:1:2"
    assert not index.is_ready
    release_second.set()
    index.wait_until_ready()
    assert index.by_title("fr", "Douglas Adams FR")[0]["document_id"] == "Q43:wikipedia:fr:2:3"
    index.close()
