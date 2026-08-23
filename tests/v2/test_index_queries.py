from __future__ import annotations

import osm_polygon_wikidata_only.v2.v1_index as v1_index
from osm_polygon_wikidata_only.v2.index_queries import PersistentIndexQueries


def test_persistent_index_uses_separate_query_boundary() -> None:
    """The durable index keeps lookup/materialization separate from scanning."""
    assert issubclass(v1_index._PersistentV1Index, PersistentIndexQueries)
