"""Focused contracts for Natural Earth basemap geometry rendering."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from osm_polygon_wikidata_only.hf._geographic import basemap


class _AxesSpy:
    def __init__(self) -> None:
        self.patches: list[Any] = []
        self.collections: list[tuple[Any, bool]] = []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def add_patch(self, patch: Any) -> None:
        self.patches.append(patch)

    def add_collection(self, collection: Any, *, autolim: bool = True) -> None:
        self.collections.append((collection, autolim))

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


def test_draw_landmasses_batches_features_into_one_collection() -> None:
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

    assert axes.patches == []
    assert len(axes.collections) == 1
    collection, autolim = axes.collections[0]
    assert autolim is False
    assert len(collection.get_paths()) == 2


def test_draw_landmasses_batches_many_features_into_one_collection() -> None:
    axes = _AxesSpy()
    features = [
        {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[index, 0], [index + 0.5, 0], [index, 0.5]]],
            }
        }
        for index in range(1100)
    ]

    basemap.draw_landmasses(axes, features)

    assert len(axes.collections) == 1
    assert len(axes.collections[0][0].get_paths()) == 1100


def test_draw_landmasses_ignores_short_rings() -> None:
    axes = _AxesSpy()
    basemap.draw_landmasses(
        axes,
        [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]}}],
    )
    assert axes.collections == []


def test_draw_landmasses_preserves_polygon_holes_in_rendered_image() -> None:
    ocean = np.array([207, 226, 243], dtype=np.uint8)
    land = np.array([232, 224, 208], dtype=np.uint8)
    feature = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]],
                [[-5, -5], [-5, 5], [5, 5], [5, -5], [-5, -5]],
            ],
        }
    }
    figure = Figure(figsize=(2, 2), dpi=40)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    figure.set_facecolor("#cfe2f3")
    axes.set_facecolor("#cfe2f3")
    axes.set_xlim(-10, 10)
    axes.set_ylim(-10, 10)
    axes.set_axis_off()
    basemap.draw_landmasses(axes, [feature])
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[..., :3]

    assert np.array_equal(pixels[40, 40], ocean)
    assert np.array_equal(pixels[40, 68], land)


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


def test_load_land_basemap_caches_by_path_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "ne_110m_land.geojson"
    path.write_text('{"features": [{"name": "first"}]}', encoding="utf-8")
    reads = 0
    read_text = Path.read_text

    def count_reads(candidate: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal reads
        if candidate.resolve() == path.resolve():
            reads += 1
        return read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_reads)

    first = basemap.load_land_basemap(cache)
    second = basemap.load_land_basemap(cache)
    assert second is first
    assert reads == 1

    previous_mtime = path.stat().st_mtime_ns
    path.write_text('{"features": [{"name": "second"}]}', encoding="utf-8")
    changed_mtime = previous_mtime + 2_000_000_000
    os.utime(path, ns=(changed_mtime, changed_mtime))
    third = basemap.load_land_basemap(cache)

    assert third == [{"name": "second"}]
    assert reads == 2


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
