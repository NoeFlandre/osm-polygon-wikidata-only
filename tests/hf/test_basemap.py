"""Focused contracts for Natural Earth basemap geometry rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osm_polygon_wikidata_only.hf._geographic import basemap


class _AxesSpy:
    def __init__(self) -> None:
        self.patches: list[Any] = []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def add_patch(self, patch: Any) -> None:
        self.patches.append(patch)

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def set_facecolor(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_facecolor", *args, **kwargs)

    def set_xlim(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_xlim", *args, **kwargs)

    def set_ylim(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_ylim", *args, **kwargs)

    def set_xticks(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_xticks", *args, **kwargs)

    def set_yticks(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_yticks", *args, **kwargs)

    def grid(self, *args: Any, **kwargs: Any) -> None:
        self._record("grid", *args, **kwargs)

    def tick_params(self, *args: Any, **kwargs: Any) -> None:
        self._record("tick_params", *args, **kwargs)

    def set_aspect(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_aspect", *args, **kwargs)


def test_feature_rings_accepts_polygon_and_multipolygon() -> None:
    polygon = {"type": "Polygon", "coordinates": [[[1, 2], [3, 4], [5, 6]]]}
    multipolygon = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[1, 2], [3, 4], [5, 6]]],
            [],
            [[[7, 8], [9, 10], [11, 12]]],
        ],
    }

    assert basemap._feature_rings({"geometry": polygon}) == [[[1, 2], [3, 4], [5, 6]]]
    assert basemap._feature_rings({"geometry": multipolygon}) == [
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
    ]


@pytest.mark.parametrize(
    "feature",
    [
        None,
        [],
        {},
        {"geometry": None},
        {"geometry": {"coordinates": []}},
        {"geometry": {"type": "Point", "coordinates": [1, 2]}},
        {"geometry": {"type": "Polygon", "coordinates": []}},
        {"geometry": {"type": "MultiPolygon", "coordinates": []}},
    ],
)
def test_feature_rings_ignores_non_area_features(feature: object) -> None:
    assert basemap._feature_rings(feature) == []


def test_feature_geometry_returns_only_mapping_geometry() -> None:
    assert basemap._feature_geometry({"geometry": {"type": "Point"}}) == {"type": "Point"}
    assert basemap._feature_geometry({"geometry": None}) is None


def test_geometry_rings_ignores_unsupported_geometry() -> None:
    assert basemap._geometry_rings({"type": "Point", "coordinates": [1, 2]}) == []


def test_draw_landmasses_adds_one_patch_per_ring(monkeypatch: pytest.MonkeyPatch) -> None:
    patches: list[tuple[Any, ...]] = []

    def fake_polygon(points: list[tuple[float, float]], **kwargs: Any) -> tuple[Any, ...]:
        patches.append((points, kwargs))
        return patches[-1]

    monkeypatch.setattr(basemap.mpatches, "Polygon", fake_polygon)
    axes = _AxesSpy()
    basemap.draw_landmasses(
        axes,
        [
            {"geometry": {"type": "Polygon", "coordinates": [[[1, 2], [3, 4], [5, 6]]]}},
            {
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[7, 8], [9, 10], [11, 12]]]],
                }
            },
        ],
    )

    assert len(axes.patches) == 2
    assert len(patches) == 2
    assert patches[0][0] == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


def test_draw_land_ring_ignores_short_rings(monkeypatch: pytest.MonkeyPatch) -> None:
    axes = _AxesSpy()
    monkeypatch.setattr(basemap.mpatches, "Polygon", lambda *_args, **_kwargs: pytest.fail())
    basemap._draw_land_ring(axes, [[0, 0], [1, 1]])
    assert axes.patches == []


def test_load_land_basemap_reads_cached_features(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = [{"type": "Feature", "geometry": None}]
    (cache / "ne_110m_land.geojson").write_text(
        json.dumps({"features": expected}), encoding="utf-8"
    )
    assert basemap.load_land_basemap(cache) == expected


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"features": []}'])
def test_load_land_basemap_handles_invalid_or_empty_cache(tmp_path: Path, payload: str) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ne_110m_land.geojson").write_text(payload, encoding="utf-8")
    expected = [] if payload == '{"features": []}' else None
    assert basemap.load_land_basemap(cache) == expected


def test_load_land_basemap_returns_none_when_cache_is_missing(tmp_path: Path) -> None:
    assert basemap.load_land_basemap(tmp_path) is None


def test_init_axes_applies_shared_world_style() -> None:
    axes = _AxesSpy()
    basemap.init_axes(axes)
    names = [name for name, _args, _kwargs in axes.calls]
    assert names == [
        "set_facecolor",
        "set_xlim",
        "set_ylim",
        "set_xticks",
        "set_yticks",
        "grid",
        "tick_params",
        "set_aspect",
    ]
