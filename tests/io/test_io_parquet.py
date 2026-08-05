"""Tests for io.parquet."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
)
from osm_polygon_wikidata_only.io.parquet import (
    _open_parquet_file,
    read_table,
    write_articles,
    write_polygon_articles,
    write_polygons,
    write_table,
)


def _sample_polygon() -> dict:
    return {
        "polygon_id": "monaco-latest:way:1",
        "region": "monaco",
        "source_pbf": "monaco-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": 1,
        "wikidata": "Q235",
        "name": "Monaco",
        "tags": json.dumps({"name": "Monaco"}, sort_keys=True),
        "tag_keys": json.dumps(["name"]),
        "tag_count": 1,
        "osm_primary_tag": "",
        "centroid": json.dumps({"type": "Point", "coordinates": [7.42, 43.73]}),
        "lat": 43.73,
        "lon": 7.42,
        "bbox": json.dumps([7.42, 43.73, 7.43, 43.74]),
        "area_m2": 1_000.0,
        "area_km2": 0.001,
        "area_bucket": "100m2-1k_m2",
        "has_name": True,
        "has_wikidata": True,
        "has_wikipedia": False,
        "wikipedia_language_count": 0,
        "wikipedia_languages": "[]",
        "wikipedia_article_count": 0,
        "has_english_wikipedia": False,
        "has_french_wikipedia": False,
        "text_available": False,
        "best_language": "",
        "extraction_version": "0.1.0",
        "extracted_at": "2026-01-01T00:00:00Z",
    }


def test_write_polygons_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "polygons" / "monaco.parquet"
    n = write_polygons(out, [_sample_polygon()])
    assert n == 1
    table = read_table(out)
    assert table.num_rows == 1
    assert table.column("wikidata").to_pylist() == ["Q235"]


def test_write_table_handles_empty_input(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.domain.schema import polygon_schema

    out = tmp_path / "empty.parquet"
    n = write_table(out, [], columns=POLYGON_COLUMNS, schema=polygon_schema())
    assert n == 0
    table = read_table(out)
    assert table.num_rows == 0
    # Schema columns are preserved on an empty file.
    assert [f.name for f in table.schema] == list(POLYGON_COLUMNS)


def test_write_polygon_articles_writes_proper_schema(tmp_path: Path) -> None:
    out = tmp_path / "links.parquet"
    rows = [
        {
            "polygon_id": "monaco-latest:way:1",
            "article_id": "Q235:en:1:1",
            "wikidata": "Q235",
            "language": "en",
            "source_pbf": "monaco-latest.osm.pbf",
            "region": "monaco",
            "osm_type": "way",
            "osm_id": 1,
            "page_id": 1,
            "revision_id": 1,
            "is_best_language": True,
        }
    ]
    n = write_polygon_articles(out, rows)
    assert n == 1
    table = read_table(out)
    assert set(table.column_names) == set(POLYGON_ARTICLE_COLUMNS)


def test_write_articles_handles_optional_ints(tmp_path: Path) -> None:
    out = tmp_path / "articles.parquet"
    rows = [
        {
            "article_id": "Q1:en:1:1",
            "wikidata": "Q1",
            "language": "en",
            "site": "enwiki",
            "title": "T",
            "url": "https://en.wikipedia.org/wiki/T",
            "page_id": 1,
            "revision_id": 1,
            "revision_timestamp": "2026-01-01T00:00:00Z",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "wikidata_label": "T",
            "wikidata_description": "",
            "wikidata_aliases": "[]",
            "lead_text": "",
            "extract": "",
            "full_text": "hello world",
            "full_text_format": "plain_text",
            "article_length_chars": 11,
            "article_length_words": 2,
            "article_length_tokens_estimate": 2,
            "thumbnail_url": "",
            "thumbnail_width": None,
            "thumbnail_height": None,
            "categories": "[]",
            "license": "CC BY-SA",
            "attribution": "Wikipedia",
            "source_api": "mediawiki_action_api",
            "fetch_status": "ok",
            "fetch_error": "",
            "content_hash": "deadbeef",
        }
    ]
    n = write_articles(out, rows)
    assert n == 1
    table = read_table(out)
    assert table.column("full_text").to_pylist() == ["hello world"]
    assert table.column("thumbnail_width").to_pylist() == [None]


# ---------------------------------------------------------------------------
# Resource ownership: _open_parquet_file context manager
#
# ``_open_parquet_file`` is the single Parquet handle owner in
# :mod:`io.parquet`. It must close the underlying file descriptor on every
# exit path: successful reads, exceptions raised inside the ``with`` body,
# and failures raised by the read itself. Closure is observed through an
# injected fake ``ParquetFile`` rather than OS-level descriptor counts.
# ---------------------------------------------------------------------------


def _install_tracked_parquet_file(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[object], list[object]]:
    """Patch ``io.parquet.pq.ParquetFile`` with a handle that records closes."""
    import osm_polygon_wikidata_only.io.parquet as parquet_mod

    original = parquet_mod.pq.ParquetFile
    opened: list[object] = []
    closed: list[object] = []

    class _TrackedParquetFile:
        def __init__(self, source: Path) -> None:
            self._inner = original(source)
            opened.append(self)

        def read(self):
            return self._inner.read()

        @property
        def schema_arrow(self):
            return self._inner.schema_arrow

        @property
        def num_row_groups(self):
            return self._inner.num_row_groups

        def read_row_group(self, row_group: int, columns=None):
            return self._inner.read_row_group(row_group, columns=columns)

        def close(self) -> None:
            if self not in closed:
                self._inner.close()
                closed.append(self)

    monkeypatch.setattr(parquet_mod.pq, "ParquetFile", _TrackedParquetFile)
    return opened, closed


def test_open_parquet_file_closes_after_successful_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "polygons" / "monaco.parquet"
    write_polygons(out, [_sample_polygon()])
    opened, closed = _install_tracked_parquet_file(monkeypatch)

    with _open_parquet_file(out) as parquet_file:
        table = parquet_file.read()

    assert table.num_rows == 1
    assert opened, "_open_parquet_file must open a ParquetFile"
    assert closed == opened, "ParquetFile handle leaked after a successful read"


def test_open_parquet_file_closes_when_body_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "polygons" / "monaco.parquet"
    write_polygons(out, [_sample_polygon()])
    opened, closed = _install_tracked_parquet_file(monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        with _open_parquet_file(out):
            raise RuntimeError("boom")

    assert closed == opened, "ParquetFile handle leaked when the with-body raised"


def test_open_parquet_file_closes_when_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "polygons" / "monaco.parquet"
    write_polygons(out, [_sample_polygon()])
    opened, closed = _install_tracked_parquet_file(monkeypatch)

    class _FailingParquetFile:
        def __init__(self, source: Path) -> None:
            opened.append(self)

        def read(self):
            raise OSError("read failed")

        def close(self) -> None:
            closed.append(self)

    import osm_polygon_wikidata_only.io.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod.pq, "ParquetFile", _FailingParquetFile)

    with pytest.raises(OSError, match="read failed"):
        with _open_parquet_file(out) as parquet_file:
            parquet_file.read()

    assert closed == opened, "ParquetFile handle leaked when the read itself failed"


def test_open_parquet_file_repeated_reads_retain_no_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "polygons" / "monaco.parquet"
    write_polygons(out, [_sample_polygon()])
    opened, closed = _install_tracked_parquet_file(monkeypatch)

    for _ in range(5):
        with _open_parquet_file(out) as parquet_file:
            assert parquet_file.read().num_rows == 1

    assert len(opened) == 5
    assert closed == opened, "repeated reads must not retain one handle per read"


def test_open_parquet_file_reads_identical_schema_rows_column_order_and_empty_table(
    tmp_path: Path,
) -> None:
    """Helper reads must produce a byte-equivalent table to the eager reader.

    The equivalence covers schema (including key-value metadata), column
    order, row order, and row values, both for a non-empty shard and for an
    empty shard written via :func:`write_table`. This pins the helper as a
    drop-in replacement for the eager :func:`pq.read_table` path.
    """
    from osm_polygon_wikidata_only.domain.schema import polygon_schema

    non_empty = tmp_path / "polygons" / "monaco.parquet"
    write_polygons(non_empty, [_sample_polygon()])

    with _open_parquet_file(non_empty) as parquet_file:
        via_helper = parquet_file.read()
    direct = pq.read_table(non_empty)

    assert via_helper.schema.equals(direct.schema, check_metadata=True)
    assert [field.name for field in via_helper.schema] == list(POLYGON_COLUMNS)
    assert via_helper.column_names == direct.column_names
    assert via_helper.num_rows == direct.num_rows
    assert via_helper.to_pylist() == direct.to_pylist()
    assert via_helper.column("wikidata").to_pylist() == direct.column("wikidata").to_pylist()

    empty_path = tmp_path / "empty.parquet"
    write_table(empty_path, [], columns=POLYGON_COLUMNS, schema=polygon_schema())

    with _open_parquet_file(empty_path) as parquet_file:
        empty_via_helper = parquet_file.read()
    empty_direct = pq.read_table(empty_path)

    assert empty_via_helper.num_rows == 0
    assert empty_via_helper.schema.equals(empty_direct.schema, check_metadata=True)
    assert [field.name for field in empty_via_helper.schema] == list(POLYGON_COLUMNS)
    assert empty_via_helper.column_names == empty_direct.column_names
