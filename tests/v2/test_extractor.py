from types import SimpleNamespace
from typing import Any, cast

from osm_polygon_wikidata_only.pipeline import extractor as pipeline_extractor
from osm_polygon_wikidata_only.v2 import extractor as v2_extractor
from osm_polygon_wikidata_only.v2.extractor import candidate_to_v2_row

_SQUARE = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'


def test_v2_keeps_wikipedia_only_polygon() -> None:
    row = candidate_to_v2_row(
        ("way", 7, {"name": "Only Wiki", "wikipedia": "fr:Only Wiki"}, _SQUARE),
        source_pbf_stem="test-latest",
        region="test",
        source_pbf="test-latest.osm.pbf",
    )
    assert row is not None
    assert row["wikidata"] is None
    assert row["has_wikidata"] is False
    assert row["discovery_sources"] == '["wikipedia_tag"]'
    assert '"language":"fr"' in row["wikipedia_tag_refs"]


def test_v2_preserves_both_discovery_sources() -> None:
    row = candidate_to_v2_row(
        (
            "relation",
            8,
            {"wikidata": "Q42", "wikipedia": "en:Douglas Adams"},
            _SQUARE,
        ),
        source_pbf_stem="test-latest",
        region="test",
        source_pbf="test-latest.osm.pbf",
    )
    assert row is not None
    assert row["wikidata"] == "Q42"
    assert row["has_wikidata"] is True
    assert row["discovery_sources"] == '["wikidata","wikipedia_tag"]'


def test_v2_drops_candidate_without_valid_reference() -> None:
    row = candidate_to_v2_row(
        ("way", 9, {"wikidata": "not-a-qid", "wikipedia": "bad"}, _SQUARE),
        source_pbf_stem="test-latest",
        region="test",
        source_pbf="test-latest.osm.pbf",
    )
    assert row is None


def test_geometry_helpers_reject_malformed_and_non_polygon_inputs() -> None:
    assert v2_extractor._geometry(_SQUARE) is not None
    for invalid in ("{broken", "[]", '{"type":"Point","coordinates":[0,0]}'):
        assert v2_extractor._geometry(invalid) is None
    assert pipeline_extractor._compute_geom(_SQUARE) is not None
    for invalid in ("{broken", "[]", '{"type":"Point","coordinates":[0,0]}'):
        assert pipeline_extractor._compute_geom(invalid) is None


def test_candidate_consumer_accepts_streaming_and_collected_candidate_sources() -> None:
    streamed: list[object] = []
    cast(Any, v2_extractor._consume_candidates)(
        SimpleNamespace(iter_polygon_candidates=lambda add: add("streamed")), streamed.append
    )
    cast(Any, v2_extractor._consume_candidates)(
        SimpleNamespace(collect_polygon_candidates=lambda: ["collected"]), streamed.append
    )
    assert streamed == ["streamed", "collected"]
