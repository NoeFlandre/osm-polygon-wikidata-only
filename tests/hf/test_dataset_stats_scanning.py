from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._dataset_stats.scanning import safe_table


def test_safe_table_reads_selected_columns_without_dataset_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "values.parquet"
    pq.write_table(pa.table({"kept": [1, 2], "ignored": [3, 4]}), parquet_path)

    def fail_if_dataset_reader_is_used(*args: object, **kwargs: object) -> None:
        raise AssertionError("single-file dataset reader was used")

    monkeypatch.setattr(pq, "read_table", fail_if_dataset_reader_is_used)

    table = safe_table(parquet_path, ["kept"])

    assert table is not None
    assert table.column_names == ["kept"]
    assert table.column("kept").to_pylist() == [1, 2]
