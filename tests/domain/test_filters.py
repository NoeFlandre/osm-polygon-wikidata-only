from __future__ import annotations

from osm_polygon_wikidata_only.domain.filters import has_wikidata


def test_has_wikidata_requires_a_non_blank_value() -> None:
    assert has_wikidata({}) is False
    assert has_wikidata({"wikidata": "   "}) is False
    assert has_wikidata({"wikidata": " Q42 "}) is True
