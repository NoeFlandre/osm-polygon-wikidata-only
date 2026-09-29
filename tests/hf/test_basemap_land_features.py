"""Behaviour of the landmass GeoJSON loader and ring conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.hf._geographic.basemap import _convert_ring, load_land_features


def test_missing_file_yields_no_features(tmp_path: Path) -> None:
    assert load_land_features(tmp_path / "absent.geojson") is None


def test_empty_file_yields_no_features(tmp_path: Path) -> None:
    empty = tmp_path / "empty.geojson"
    empty.write_text("", encoding="utf-8")

    assert load_land_features(empty) is None


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3]"])
def test_unparseable_or_non_mapping_file_yields_no_features(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.geojson"
    path.write_text(content, encoding="utf-8")

    assert load_land_features(path) is None


def test_features_that_are_not_a_list_become_an_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "odd.geojson"
    path.write_text(json.dumps({"features": "nope"}), encoding="utf-8")

    assert load_land_features(path) == []


def test_valid_file_returns_its_features(tmp_path: Path) -> None:
    path = tmp_path / "land.geojson"
    features = [{"type": "Feature", "geometry": None}]
    path.write_text(json.dumps({"features": features}), encoding="utf-8")

    assert load_land_features(path) == features


def test_convert_ring_returns_float_points() -> None:
    assert _convert_ring([[0, 0], [1, 0], [1, 1]]) == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


@pytest.mark.parametrize(
    "ring",
    [
        [[0, 0], [1, "x"], [1, 1]],
        [[0, 0], [1], [1, 1]],
        [[0, 0], None, [1, 1]],
        [[0, 0], [1, 1]],
    ],
)
def test_convert_ring_rejects_malformed_or_short_rings(ring: list[object]) -> None:
    assert _convert_ring(ring) is None
