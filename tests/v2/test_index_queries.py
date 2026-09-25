from __future__ import annotations

import osm_polygon_wikidata_only.v2.v1_index as v1_index
from osm_polygon_wikidata_only.v2.index_queries import (
    PersistentIndexQueries,
    partition_title_cache,
)


def test_persistent_index_uses_separate_query_boundary() -> None:
    """The durable index keeps lookup/materialization separate from scanning."""
    assert issubclass(v1_index._PersistentV1Index, PersistentIndexQueries)


def test_partition_title_cache_returns_hits_and_ordered_misses_only_when_complete() -> None:
    normalized = (("en", "page"), ("fr", "missing"), ("de", "also-missing"))
    cached = {("en", "page"): ({"document_id": "doc-1"},)}

    assert partition_title_cache(normalized, cached, complete=True) == (
        {("en", "page"): ({"document_id": "doc-1"},)},
        (("fr", "missing"), ("de", "also-missing")),
    )
    assert partition_title_cache(normalized, cached, complete=False) == ({}, normalized)
