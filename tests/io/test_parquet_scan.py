from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches_if_columns_present


def test_projected_batch_scan_preserves_columns_and_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"polygon_id": ["p1", "p2"], "unused": [1, 2]}), path)

    batches = list(
        iter_record_batches_if_columns_present(
            path,
            columns=("polygon_id",),
            batch_size=1,
        )
    )

    assert [batch.schema.names for batch in batches] == [["polygon_id"], ["polygon_id"]]
    assert [batch.column(0).to_pylist() for batch in batches] == [["p1"], ["p2"]]


def test_projected_batch_scan_skips_a_file_missing_a_required_column(tmp_path: Path) -> None:
    path = tmp_path / "legacy.parquet"
    pq.write_table(pa.table({"polygon_id": ["p1"], "document_id": ["doc1"]}), path)

    batches = list(
        iter_record_batches_if_columns_present(
            path,
            columns=("polygon_id", "document_id", "project"),
            batch_size=65_536,
        )
    )

    assert batches == []
