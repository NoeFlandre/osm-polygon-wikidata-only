"""Behavioral tests for the V1 Parquet scan boundary used by V2 reuse."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.schema import article_schema
from osm_polygon_wikidata_only.v2.index_scanning import (
    effective_paths,
    index_rows_from_table,
    read_rows,
    required_int,
    scan_index_rows,
    validated_parquet_file,
)


def _document_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=wikipedia_document_schema())


def _article_row() -> dict[str, object]:
    row: dict[str, object] = {}
    for field in article_schema():
        row[field.name] = 0 if pa.types.is_integer(field.type) else ""
    row.update(
        {
            "article_id": "Q42:en:7:9",
            "wikidata": "Q42",
            "language": "en",
            "title": "Douglas Adams",
            "page_id": 7,
            "revision_id": 9,
            "thumbnail_width": None,
            "thumbnail_height": None,
        }
    )
    return row


def test_effective_paths_prefers_canonical_shard_over_legacy_fallback(tmp_path: Path) -> None:
    canonical = tmp_path / "wikipedia" / "documents"
    legacy = tmp_path / "articles"
    canonical.mkdir(parents=True)
    legacy.mkdir()
    (canonical / "shared.parquet").touch()
    (legacy / "shared.parquet").touch()
    (legacy / "legacy-only.parquet").touch()

    assert effective_paths(tmp_path) == (
        legacy / "legacy-only.parquet",
        canonical / "shared.parquet",
    )


@pytest.mark.parametrize("value", [True, False, "1", 1.0, None])
def test_required_int_rejects_non_integer_values(value: object) -> None:
    with pytest.raises(ValueError, match="is not an integer"):
        required_int(value, "page_id")


def test_index_rows_from_table_derives_legacy_document_identity() -> None:
    table = pa.table(
        {
            "language": ["fr"],
            "title": ["Paris"],
            "page_id": [10],
            "revision_id": [20],
            "wikidata": ["Q90"],
        }
    )

    rows = index_rows_from_table(table, legacy_articles=True, row_group=3)

    assert rows == [("Q90:wikipedia:fr:10:20", "fr", "Paris", 10, 20, "Q90", 3, 0)]


def test_index_rows_from_table_preserves_canonical_document_identity() -> None:
    table = pa.table(
        {
            "document_id": ["Q90:wikipedia:fr:10:20"],
            "language": ["fr"],
            "title": ["Paris"],
            "page_id": [10],
            "revision_id": [20],
            "wikidata": ["Q90"],
        }
    )

    rows = index_rows_from_table(table, legacy_articles=False, row_group=0)

    assert rows[0][:6] == ("Q90:wikipedia:fr:10:20", "fr", "Paris", 10, 20, "Q90")


def test_scan_index_rows_deduplicates_identity_across_row_groups(tmp_path: Path) -> None:
    path = tmp_path / "documents.parquet"
    pq.write_table(
        _document_table(
            [
                {
                    "document_id": "Q1:wikipedia:en:1:1",
                    "language": "en",
                    "title": "One",
                    "page_id": 1,
                    "revision_id": 1,
                    "wikidata": "Q1",
                },
                {
                    "document_id": "Q1:wikipedia:en:1:1",
                    "language": "en",
                    "title": "Duplicate",
                    "page_id": 1,
                    "revision_id": 1,
                    "wikidata": "Q1",
                },
            ]
        ),
        path,
        row_group_size=1,
    )

    rows = scan_index_rows(path, legacy_articles=False)

    assert len(rows) == 1
    assert rows[0][2] == "One"


def test_read_rows_converts_legacy_article_schema_to_documents(tmp_path: Path) -> None:
    path = tmp_path / "articles.parquet"
    pq.write_table(pa.Table.from_pylist([_article_row()], schema=article_schema()), path)

    rows = read_rows(path, legacy_articles=True)

    assert rows[0]["document_id"] == "Q42:wikipedia:en:7:9"
    assert rows[0]["project"] == "wikipedia"


def test_read_rows_rejects_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"unexpected": [1]}), path)

    with pytest.raises(ValueError, match="invalid schema"):
        read_rows(path, legacy_articles=False)


def test_read_rows_reports_missing_shard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="V1 document shard is unreadable"):
        read_rows(tmp_path / "missing.parquet", legacy_articles=False)


def test_index_rows_from_table_rejects_boolean_identity_fields() -> None:
    table = pa.table(
        {
            "document_id": ["Q1:wikipedia:en:1:1"],
            "language": ["en"],
            "title": ["One"],
            "page_id": [True],
            "revision_id": [1],
            "wikidata": ["Q1"],
        }
    )

    with pytest.raises(ValueError, match=r"page_id.*not an integer"):
        index_rows_from_table(table, legacy_articles=False, row_group=0)


def test_validated_parquet_file_closes_and_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unreadable"):
        validated_parquet_file(tmp_path / "missing.parquet", legacy_articles=False)


def test_shard_errors_reraises_value_error_and_wraps_other_errors(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.v2.index_scanning import _shard_errors

    path = tmp_path / "x.parquet"
    original = ValueError("already specific")
    with pytest.raises(ValueError, match="already specific") as same, _shard_errors(path):
        raise original
    assert same.value is original

    with pytest.raises(
        ValueError, match=f"V1 document shard is unreadable: {path}: boom"
    ) as wrapped:
        with _shard_errors(path):
            raise OSError("boom")
    assert isinstance(wrapped.value.__cause__, OSError)


def test_unreadable_shard_message_is_consistent_across_entry_points(tmp_path: Path) -> None:
    bad = tmp_path / "missing.parquet"
    expected = f"V1 document shard is unreadable: {bad}"

    with pytest.raises(ValueError, match="V1 document shard is unreadable") as read_error:
        read_rows(bad)
    with pytest.raises(ValueError, match="V1 document shard is unreadable") as open_error:
        validated_parquet_file(bad, legacy_articles=False)
    assert str(read_error.value).startswith(expected)
    assert str(open_error.value).startswith(expected)


def test_scan_row_group_keeps_arrow_value_errors_unwrapped(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.v2.index_scanning import scan_index_row_group

    path = tmp_path / "ok.parquet"
    pq.write_table(_document_table([]), path)
    with pytest.raises(ValueError, match="row_group_indices") as error:
        scan_index_row_group(path, legacy_articles=False, row_group=5)
    assert "unreadable" not in str(error.value)
