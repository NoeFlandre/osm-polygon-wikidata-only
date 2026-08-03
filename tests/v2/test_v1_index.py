from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.schema import ARTICLE_COLUMNS, article_schema
from osm_polygon_wikidata_only.v2.v1_index import V1ReuseIndex, build_v1_reuse_index


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


def test_index_rejects_duplicate_page_identity(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="duplicate"):
        build_v1_reuse_index(tmp_path)


def test_index_reuses_legacy_articles_when_canonical_documents_are_absent(
    tmp_path: Path,
) -> None:
    path = _write_legacy_articles(tmp_path, [_document()])
    index = build_v1_reuse_index(tmp_path)
    assert index.files == (path,)
    assert index.by_title("en", "Douglas Adams")[0]["document_id"] == "Q42:wikipedia:en:1:2"
