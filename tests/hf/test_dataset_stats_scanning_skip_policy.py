"""Skip-on-error policy of the dataset-stats Parquet scanning primitives."""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._dataset_stats.scanning import (
    safe_metadata_row_count,
    safe_table,
    sorted_parquets,
)


def _write(path: Path, rows: int = 3) -> Path:
    pq.write_table(pa.table({"a": list(range(rows)), "b": ["x"] * rows}), path)
    return path


def test_sorted_parquets_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert sorted_parquets(tmp_path / "absent") == []


def test_sorted_parquets_lists_only_parquet_files_in_order(tmp_path: Path) -> None:
    _write(tmp_path / "b.parquet")
    _write(tmp_path / "a.parquet")
    (tmp_path / "c.txt").write_text("ignored", encoding="utf-8")
    assert [path.name for path in sorted_parquets(tmp_path)] == ["a.parquet", "b.parquet"]


def test_safe_table_reads_selected_columns(tmp_path: Path) -> None:
    table = safe_table(_write(tmp_path / "t.parquet"), ["a"])
    assert table is not None
    assert table.column_names == ["a"]


def test_safe_table_skips_corrupt_file_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not parquet")
    with caplog.at_level(logging.WARNING):
        assert safe_table(broken, ["a"]) is None
    assert f"Skipping {broken}" in caplog.text


def test_safe_metadata_row_count_reads_row_count(tmp_path: Path) -> None:
    assert safe_metadata_row_count(_write(tmp_path / "t.parquet", rows=5)) == 5


def test_safe_metadata_row_count_skips_unreadable_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not parquet")
    with caplog.at_level(logging.WARNING):
        assert safe_metadata_row_count(broken) is None
    assert f"Skipping {broken}" in caplog.text


def test_safe_metadata_row_count_skips_missing_file(tmp_path: Path) -> None:
    assert safe_metadata_row_count(tmp_path / "missing.parquet") is None
