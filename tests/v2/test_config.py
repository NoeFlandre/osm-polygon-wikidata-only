from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot


def test_v2_data_root_paths_are_separate_from_v1() -> None:
    root = DataRoot(Path("/data"))
    assert root.processed_v2 == Path("/data/processed_v2")
    assert root.v2_cache == Path("/data/cache/v2")
    assert root.processed_v2 != root.processed
    assert root.v2_cache != root.cache
