"""Micro-benchmarks for the pure functions every extracted polygon passes through.

Run with ``just bench``. A plain ``pytest`` run does not collect this directory.
"""

from __future__ import annotations

import math

import pytest

from osm_polygon_wikidata_only.domain.geometry import compute_polygon_geometry
from osm_polygon_wikidata_only.domain.ids import content_hash, polygon_id

pytest.importorskip("pytest_benchmark")


def _circle(vertices: int) -> dict[str, object]:
    ring = [
        [
            10.0 + 0.01 * math.cos(2 * math.pi * i / vertices),
            50.0 + 0.01 * math.sin(2 * math.pi * i / vertices),
        ]
        for i in range(vertices)
    ]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


@pytest.mark.parametrize("vertices", [8, 500, 5000])
def test_compute_polygon_geometry(benchmark, vertices: int) -> None:
    geometry = _circle(vertices)

    result = benchmark(compute_polygon_geometry, geometry)

    assert result.area_m2 > 0


def test_content_hash_of_a_long_article(benchmark) -> None:
    text = "Ünïcode paragraph. " * 20_000

    assert len(benchmark(content_hash, text)) == 64


def test_polygon_id(benchmark) -> None:
    assert benchmark(polygon_id, "europe-latest", "way", 123456)
