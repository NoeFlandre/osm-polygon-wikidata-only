from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.v2.config import (
    V2_CACHE_CONTRACT_VERSION,
    V2_CONTRACT_VERSION,
    V2_REPO_ID,
    DatasetVersion,
)


def test_v2_constants_are_explicit_and_separate() -> None:
    assert V2_CONTRACT_VERSION == "wikipedia-tags-v2"
    assert V2_CACHE_CONTRACT_VERSION == "wikipedia-tags-v2-cache"
    assert V2_REPO_ID == "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
    assert DatasetVersion.V1.value == "v1"
    assert DatasetVersion.V2.value == "v2"


def test_v2_data_root_paths_are_separate_from_v1() -> None:
    root = DataRoot(Path("/data"))
    assert root.processed_v2 == Path("/data/processed_v2")
    assert root.v2_cache == Path("/data/cache/v2")
    assert root.processed_v2 != root.processed
    assert root.v2_cache != root.cache
