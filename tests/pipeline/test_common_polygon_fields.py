"""Pinned extraction contracts recorded from main at 816c121, before sharing fields."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest

from osm_polygon_wikidata_only.domain.models import Polygon
from osm_polygon_wikidata_only.io.pbf_reader import PolygonCandidate
from osm_polygon_wikidata_only.pipeline.extractor import candidate_to_polygon, polygon_to_dict
from osm_polygon_wikidata_only.utils import time as time_module
from osm_polygon_wikidata_only.v2.extractor import candidate_to_v2_row

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "extraction"
_PROVENANCE = {
    "source_pbf_stem": "fixture-latest",
    "region": "fixture",
    "source_pbf": "fixture-latest.osm.pbf",
}
_TIMESTAMP = "2026-01-02T03:04:05+00:00"
_SQUARE = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'
_V2_METADATA = {
    "discovery_sources": '["wikidata"]',
    "wikipedia_tag_refs": "[]",
    "wikipedia_tag_rejections": "[]",
}


def _rows(
    candidate: PolygonCandidate, extracted_at: str | None = _TIMESTAMP
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    polygon = candidate_to_polygon(candidate, **_PROVENANCE, extracted_at=extracted_at)
    row = candidate_to_v2_row(candidate, **_PROVENANCE, extracted_at=extracted_at)
    if polygon is not None:
        assert isinstance(polygon, Polygon)
    return polygon_to_dict(polygon) if polygon is not None else None, row


@pytest.mark.parametrize("fixture", ["polygon", "multipolygon_with_hole", "empty_cleaned_tags"])
def test_common_fields_match_pinned_rows_and_each_other(fixture: str) -> None:
    data = json.loads((_FIXTURES / f"{fixture}.json").read_text())
    candidate = cast(PolygonCandidate, tuple(data["candidate"]))
    original_tags = dict(candidate[2])
    expected = data["expected"]

    v1, v2 = _rows(candidate)

    # Full fixed snapshots pin field sets, defaults, coordinate order and JSON spacing.
    assert v1 == expected
    assert v2 == {**expected, **_V2_METADATA}
    assert v1 == {key: value for key, value in v2.items() if key not in _V2_METADATA}
    assert candidate[2] == original_tags

    # Input insertion order must not affect serialized tags or primary-tag precedence.
    reordered = (candidate[0], candidate[1], dict(reversed(candidate[2].items())), candidate[3])
    assert _rows(reordered) == (v1, v2)


@pytest.mark.parametrize(
    ("tags", "v1_qid", "v2_qid", "v2_kept"),
    [
        ({}, None, None, False),
        ({"wikidata": " \t "}, None, None, False),
        ({"wikidata": "not-a-qid"}, "not-a-qid", None, False),
        ({"wikidata": " Q0 "}, "Q0", None, False),
        ({"wikipedia": "fr:École"}, None, None, True),
        ({"wikidata": "  ", "wikipedia": "fr:École"}, None, None, True),
        ({"wikidata": "bad", "wikipedia": "fr:École"}, "bad", None, True),
        ({"wikidata": " Q42 "}, "Q42", "Q42", True),
        ({"wikidata": " Q7 ; Q42 ; Q7 "}, "Q7 ; Q42 ; Q7", "Q7;Q42", True),
        ({"wikidata": "Q42;bad"}, "Q42;bad", None, False),
        ({"wikidata": "Q42;"}, "Q42;", None, False),
    ],
)
def test_version_specific_admission_is_unchanged(
    tags: dict[str, str], v1_qid: str | None, v2_qid: str | None, v2_kept: bool
) -> None:
    v1, v2 = _rows(("way", 42, tags, _SQUARE))
    if v1_qid is None:
        assert v1 is None
    else:
        assert v1 is not None
        assert v1["wikidata"] == v1_qid
        assert v1["has_wikidata"] is True
    if not v2_kept:
        assert v2 is None
        return
    assert v2 is not None
    assert v2["wikidata"] == v2_qid
    assert v2["has_wikidata"] is (v2_qid is not None)
    assert v2["discovery_sources"] == (
        '["wikidata"]' if v2_qid is not None else '["wikipedia_tag"]'
    )
    assert "wikidata" not in json.loads(v2["tags"])


def test_wikipedia_metadata_and_enrichment_defaults_remain_separate() -> None:
    candidate: PolygonCandidate = (
        "relation",
        3,
        {"wikidata": "Q42", "wikipedia": "fr:École", "wikipedia:en": "School", "wikipedia:de": ""},
        _SQUARE,
    )
    v1, v2 = _rows(candidate)
    assert v1 is not None and v2 is not None
    assert v2["discovery_sources"] == '["wikidata","wikipedia_tag"]'
    assert v2["wikipedia_tag_refs"] == (
        '[{"language":"en","raw_key":"wikipedia:en","raw_value":"School","title":"School"},'
        '{"language":"fr","raw_key":"wikipedia","raw_value":"fr:École","title":"École"}]'
    )
    assert v2["wikipedia_tag_rejections"] == (
        '[{"raw_key":"wikipedia:de","raw_value":"","reason":"empty value"}]'
    )
    assert v1 == {key: value for key, value in v2.items() if key not in _V2_METADATA}
    assert v2["has_wikipedia"] is False
    assert v2["wikipedia_languages"] == "[]"
    assert v2["wikipedia_article_count"] == 0


@pytest.mark.parametrize(
    "geometry",
    [
        "{not json",
        "[]",
        '{"type":"Point","coordinates":[1,2]}',
        '{"type":"Polygon"}',
        '{"type":"Polygon","coordinates":[]}',
        '{"type":"MultiPolygon","coordinates":[]}',
        '{"type":"Polygon","coordinates":[[[0,0],[1,1],[0,0]]]}',
    ],
)
def test_invalid_geometry_is_rejected_by_both_extractors(geometry: str) -> None:
    assert _rows(("way", 1, {"wikidata": "Q42"}, geometry)) == (None, None)


@pytest.mark.parametrize("name", ["", " ", "École"])
def test_name_presence_keeps_existing_truthiness(name: str) -> None:
    v1, v2 = _rows(("way", 1, {"wikidata": "Q42", "name": name}, _SQUARE))
    assert v1 is not None and v2 is not None
    for row in (v1, v2):
        assert row["name"] == name
        assert row["has_name"] is bool(name)


@pytest.mark.parametrize("supplied", [None, "", _TIMESTAMP, "not-normalized"])
def test_timestamp_fallback_and_supplied_values_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, supplied: str | None
) -> None:
    clock = Mock()
    clock.now.return_value = datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=UTC)
    monkeypatch.setattr(time_module, "datetime", clock)

    v1, v2 = _rows(("way", 1, {"wikidata": "Q42"}, _SQUARE), supplied)

    assert v1 is not None and v2 is not None
    for row in (v1, v2):
        assert row["extracted_at"] == (supplied or "2026-03-04T05:06:07Z")
    assert clock.now.call_count == (0 if supplied else 2)
    if not supplied:
        clock.now.assert_called_with(UTC)
