from __future__ import annotations

from osm_polygon_wikidata_only.v2 import index_scanning, v1_index


def test_index_scanning_owns_parquet_read_and_scan_helpers() -> None:
    assert v1_index._effective_paths is index_scanning.effective_paths
    assert v1_index._required_int is index_scanning.required_int
    assert v1_index._read_rows is index_scanning.read_rows
    assert v1_index._validated_parquet_file is index_scanning.validated_parquet_file
    assert v1_index._scan_index_row_group is index_scanning.scan_index_row_group
    assert v1_index._scan_index_rows is index_scanning.scan_index_rows
