from __future__ import annotations

from osm_polygon_wikidata_only.domain.filters import has_wikidata, is_polygon_relation


def test_has_wikidata_requires_a_non_blank_value() -> None:
    assert has_wikidata({}) is False
    assert has_wikidata({"wikidata": "   "}) is False
    assert has_wikidata({"wikidata": " Q42 "}) is True


def test_is_polygon_relation_requires_the_exact_relation_type() -> None:
    assert is_polygon_relation({"type": "multipolygon"}) is True
    assert is_polygon_relation({"type": " multipolygon "}) is True
    assert is_polygon_relation({"type": "boundary"}) is False
    assert is_polygon_relation({}) is False
