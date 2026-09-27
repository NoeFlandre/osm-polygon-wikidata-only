"""Contracts for lightweight map assets in orchestration-only tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osm_polygon_wikidata_only.hf._geographic.basemap import draw_landmasses
from osm_polygon_wikidata_only.hf._geographic.polygon_count import render_count_map
from osm_polygon_wikidata_only.hf.coverage_map import generate_coverage_map
from osm_polygon_wikidata_only.v2 import maps as v2_maps
from tests._support import write_tiny_png

pytestmark = pytest.mark.map_orchestration


def test_coverage_map_is_replaced_with_valid_placeholder_png(tmp_path: Path) -> None:
    output = tmp_path / "coverage.png"

    result = generate_coverage_map([1.0], [2.0], output)

    expected = write_tiny_png(tmp_path / "expected.png")
    assert result == output
    assert output.read_bytes() == expected.read_bytes()


def test_count_map_is_replaced_with_valid_placeholder_png(tmp_path: Path) -> None:
    output = tmp_path / "count.png"

    result = render_count_map(
        [],
        output,
        title="Test",
        caption="Orchestration only",
        colorbar_label="Count",
        allow_empty=True,
    )

    expected = write_tiny_png(tmp_path / "expected.png")
    assert result.output_path == output
    assert output.read_bytes() == expected.read_bytes()


def test_v2_coverage_renderer_alias_uses_placeholder_png(tmp_path: Path) -> None:
    output = tmp_path / "v2-coverage.png"

    result = v2_maps.generate_coverage_map([1.0], [2.0], output)

    expected = write_tiny_png(tmp_path / "expected.png")
    assert result == output
    assert output.read_bytes() == expected.read_bytes()


def test_basemap_draw_is_skipped_for_orchestration_tests() -> None:
    class Axes:
        def __init__(self) -> None:
            self.patches: list[Any] = []

        def add_patch(self, patch: Any) -> None:
            self.patches.append(patch)

    axes = Axes()
    feature = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]],
        }
    }

    draw_landmasses(axes, [feature])

    assert axes.patches == []
