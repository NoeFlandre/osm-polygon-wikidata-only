from __future__ import annotations

import sqlite3
from collections import OrderedDict
from typing import Any, cast

import osm_polygon_wikidata_only.v2.v1_index as v1_index
from osm_polygon_wikidata_only.v2 import index_queries
from osm_polygon_wikidata_only.v2.index_queries import (
    PersistentIndexQueries,
    partition_title_cache,
)


def test_persistent_index_uses_separate_query_boundary() -> None:
    """The durable index keeps lookup/materialization separate from scanning."""
    assert issubclass(v1_index._PersistentV1Index, PersistentIndexQueries)


def test_partition_title_cache_returns_hits_and_ordered_misses_only_when_complete() -> None:
    normalized = (("en", "page"), ("fr", "missing"), ("de", "also-missing"))
    cached: dict[tuple[str, str], tuple[dict[str, object], ...]] = {
        ("en", "page"): ({"document_id": "doc-1"},)
    }

    assert partition_title_cache(normalized, cached, complete=True) == (
        {("en", "page"): ({"document_id": "doc-1"},)},
        (("fr", "missing"), ("de", "also-missing")),
    )
    assert partition_title_cache(normalized, cached, complete=False) == ({}, normalized)


def test_cached_materialization_references_skip_parquet_groups() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE refs (document_id TEXT, source_path TEXT, legacy INTEGER, row_group INTEGER)"
    )
    connection.executemany(
        "INSERT INTO refs VALUES (?, ?, ?, ?)",
        [("cached", "one.parquet", 0, 1), ("miss", "two.parquet", 1, 2)],
    )
    references = tuple(connection.execute("SELECT * FROM refs"))
    cached_row = {"title": "cached"}
    row_cache = OrderedDict([("cached", cached_row)])
    result: dict[str, object] = {}

    groups = cast(Any, index_queries.group_materialization_references)(
        references, row_cache, result
    )

    assert result == {"cached": cached_row}
    assert list(groups) == [("two.parquet", True, 2)]
    assert groups[("two.parquet", True, 2)][0]["document_id"] == "miss"
    connection.close()
