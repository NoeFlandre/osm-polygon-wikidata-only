"""Focused contracts for the PBF area callback helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import osmium.geom
import pytest

from osm_polygon_wikidata_only.io.pbf_reader import (
    _area_osm_type,
    _candidate_tags,
    _geometry_json,
    _is_wikipedia_key,
    _reference_tags,
)


class _Tag:
    def __init__(self, key: str, value: str) -> None:
        self.k = key
        self.v = value


def _tags(*items: tuple[str, str]) -> list[_Tag]:
    return [_Tag(key, value) for key, value in items]


@pytest.mark.parametrize(
    ("items", "include_wikipedia", "expected"),
    [
        ([("wikidata", "Q42")], False, {"wikidata": "Q42"}),
        ([("wikidata", "  ")], False, None),
        ([("wikipedia:fr", "Titre")], True, {"wikipedia:fr": "Titre"}),
        ([("wikipedia:fr", "Titre")], False, None),
    ],
)
def test_candidate_tags_preserve_relevant_tag_contract(
    items: list[tuple[str, str]], include_wikipedia: bool, expected: dict[str, str] | None
) -> None:
    assert _candidate_tags(_tags(*items), include_wikipedia_tagged=include_wikipedia) == expected


@pytest.mark.parametrize(
    ("multipolygon", "from_way", "expected"),
    [(True, False, "relation"), (False, True, "way"), (False, False, None)],
)
def test_area_osm_type_matches_osmium_area_identity(
    multipolygon: bool, from_way: bool, expected: str | None
) -> None:
    area = SimpleNamespace(
        is_multipolygon=lambda: multipolygon,
        from_way=lambda: from_way,
    )
    assert _area_osm_type(area) == expected


def test_geometry_json_returns_factory_output_and_swallows_geometry_errors() -> None:
    class Factory:
        def __init__(self, value: str | None = None, error: Exception | None = None) -> None:
            self.value = value
            self.error = error
            self.received: list[object] = []

        def create_multipolygon(self, area: object) -> str:
            self.received.append(area)
            if self.error is not None:
                raise self.error
            assert self.value is not None
            return self.value

    area = object()
    assert _geometry_json(Factory(value="{}"), area) == "{}"
    assert _geometry_json(Factory(error=ValueError("bad")), area) is None


@pytest.mark.parametrize(
    ("key", "expected"),
    [("wikipedia", True), ("wikipedia:fr", True), ("wikipediaish", False), ("WIKIPEDIA", False)],
)
def test_wikipedia_key_matching_is_exact(key: str, expected: bool) -> None:
    assert _is_wikipedia_key(key) is expected


def test_reference_tags_exclude_wikipedia_without_v2_opt_in() -> None:
    assert _reference_tags(_tags(("wikipedia:fr", "Titre")), include_wikipedia_tagged=False) == {}


def test_handler_initializes_v1_defaults_and_geometry_factory() -> None:
    from osm_polygon_wikidata_only.io.pbf_reader import _PolygonHandler

    handler = _PolygonHandler(lambda _candidate: None)

    assert handler._include_wikipedia_tagged is False
    assert isinstance(handler._factory, osmium.geom.GeoJSONFactory)


def test_iter_polygon_candidates_forwards_reader_wikipedia_flag(tmp_path, monkeypatch) -> None:
    from osm_polygon_wikidata_only.io import pbf_reader

    path = tmp_path / "fixture.osm.pbf"
    path.touch()
    captured: dict[str, object] = {}

    class FakeHandler:
        def __init__(self, _callback: object, *, include_wikipedia_tagged: bool) -> None:
            captured["include_wikipedia_tagged"] = include_wikipedia_tagged

        def apply_file(self, _path: str, *, locations: bool) -> None:
            captured["locations"] = locations

    monkeypatch.setattr(pbf_reader, "_PolygonHandler", FakeHandler)
    pbf_reader.PBFReader(path, include_wikipedia_tagged=True).iter_polygon_candidates(
        lambda _x: None
    )

    assert captured == {"include_wikipedia_tagged": True, "locations": True}


def test_area_passes_original_area_to_geometry_factory() -> None:
    from osm_polygon_wikidata_only.io.pbf_reader import _PolygonHandler

    class Factory:
        def __init__(self) -> None:
            self.received: list[object] = []

        def create_multipolygon(self, area: object) -> str:
            self.received.append(area)
            return "{}"

    factory = Factory()
    handler = _PolygonHandler(lambda _candidate: None)
    handler_any = cast(Any, handler)
    handler_any._factory = factory
    area = SimpleNamespace(
        id=7,
        tags=_tags(("wikidata", "Q42")),
        is_multipolygon=lambda: True,
        from_way=lambda: False,
    )

    handler_any.area(area)

    assert factory.received == [area]


def test_area_forwards_wikipedia_opt_in_to_candidate_filter() -> None:
    from osm_polygon_wikidata_only.io.pbf_reader import _PolygonHandler

    seen: list[tuple[str, int]] = []
    handler = _PolygonHandler(
        lambda candidate: seen.append((candidate[0], candidate[1])),
        include_wikipedia_tagged=True,
    )
    handler_any = cast(Any, handler)
    handler_any._factory = SimpleNamespace(create_multipolygon=lambda _area: "{}")
    area = SimpleNamespace(
        id=7,
        tags=_tags(("wikipedia:fr", "Titre")),
        is_multipolygon=lambda: True,
        from_way=lambda: False,
    )

    handler_any.area(area)

    assert seen == [("relation", 7)]


@pytest.mark.parametrize(
    ("tags", "multipolygon", "from_way", "geometry", "expected"),
    [
        (_tags(("name", "ignored")), True, False, "{}", []),
        (_tags(("wikidata", "Q42")), False, False, "{}", []),
        (_tags(("wikidata", "Q42")), True, False, "{}", [("relation", 7)]),
        (_tags(("wikidata", "Q42")), False, True, "{}", [("way", 7)]),
        (_tags(("wikidata", "Q42")), True, False, None, []),
    ],
)
def test_area_filters_invalid_candidates_and_delivers_valid_ones(
    tags: list[_Tag],
    multipolygon: bool,
    from_way: bool,
    geometry: str | None,
    expected: list[tuple[str, int]],
) -> None:
    from osm_polygon_wikidata_only.io.pbf_reader import _PolygonHandler

    class Factory:
        def create_multipolygon(self, _area: object) -> str:
            if geometry is None:
                raise ValueError("invalid geometry")
            return geometry

    seen: list[tuple[str, int]] = []
    handler = _PolygonHandler(lambda candidate: seen.append((candidate[0], candidate[1])))
    untyped_handler = cast(Any, handler)
    untyped_handler._factory = Factory()
    area = SimpleNamespace(
        id=7,
        tags=tags,
        is_multipolygon=lambda: multipolygon,
        from_way=lambda: from_way,
    )

    untyped_handler.area(area)

    assert seen == expected
