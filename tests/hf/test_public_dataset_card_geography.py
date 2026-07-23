"""Regression tests for public geographic coverage presentation."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf import geographic_text_presence as text_presence_module
from osm_polygon_wikidata_only.hf._geographic.h3_geometry import split_antimeridian
from osm_polygon_wikidata_only.hf.continent_stats import (
    assign_continents,
    render_continent_stats,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card
from osm_polygon_wikidata_only.hf.geographic_text_density import (
    aggregate_geographic_text_density,
)
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    generate_geographic_text_presence,
    load_text_presence,
)


def test_antimeridian_polygon_is_split_into_closed_local_rings() -> None:
    source = [(179.0, 65.0), (-179.0, 65.0), (-178.0, 64.0), (178.0, 64.0)]

    rings = split_antimeridian(source)

    assert len(rings) == 2
    for ring in rings:
        assert len(ring) >= 3
        assert all(-180.0 <= lon <= 180.0 for lon, _ in ring)
        closed = [*ring, ring[0]]
        assert all(abs(right[0] - left[0]) <= 180.0 for left, right in pairwise(closed))


def test_non_crossing_polygon_is_preserved() -> None:
    source = [(1.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0)]
    assert split_antimeridian(source) == [source]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_combined_text_presence_counts_each_polygon_once(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "x.parquet",
        [
            {"polygon_id": "p1", "wikidata": "Q1", "lon": 2.0, "lat": 48.0},
            {"polygon_id": "p2", "wikidata": "Q2", "lon": 3.0, "lat": 49.0},
            {"polygon_id": "p3", "wikidata": "Q3", "lon": 4.0, "lat": 50.0},
        ],
    )
    _write(
        processed / "wikipedia" / "documents" / "x.parquet",
        [
            {"article_id": "a1", "wikidata": "Q1", "full_text": "text"},
            {"article_id": "a2", "wikidata": "Q1", "full_text": "more"},
            {"article_id": "a3", "wikidata": "Q3", "full_text": "   "},
        ],
    )
    _write(
        processed / "polygon_articles" / "x.parquet",
        [
            {"polygon_id": "p1", "article_id": "a1"},
            {"polygon_id": "p1", "article_id": "a2"},
            {"polygon_id": "p3", "article_id": "a3"},
        ],
    )
    _write(
        processed / "wikivoyage" / "documents" / "x.parquet",
        [
            {"document_id": "v1", "wikidata": "Q1", "full_text": "duplicate route"},
            {"document_id": "v2", "wikidata": "Q2", "full_text": "voyage text"},
            {"document_id": "v3", "wikidata": "Q3", "full_text": None},
        ],
    )

    snapshot = load_text_presence(processed)

    assert snapshot.polygon_count == 3
    assert snapshot.wikipedia_covered_polygon_ids == frozenset({"p1"})
    assert snapshot.combined_covered_polygon_ids == frozenset({"p1", "p2"})
    assert snapshot.wikipedia_document_ids == frozenset({"a1", "a2"})
    assert snapshot.wikivoyage_document_ids == frozenset({"v1", "v2"})
    assert [(point.polygon_id, point.wikidata) for point in snapshot.covered_points] == [
        ("p1", "Q1"),
        ("p2", "Q2"),
    ]
    output = tmp_path / "combined.png"
    result = generate_geographic_text_presence(processed, output)
    assert result.output_path == output
    assert output.read_bytes().startswith(b"\x89PNG")


def test_combined_text_map_uses_blue_points_without_changing_default_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "x.parquet",
        [{"polygon_id": "p1", "wikidata": "Q1", "lon": 2.0, "lat": 48.0}],
    )
    _write(
        processed / "wikipedia" / "documents" / "x.parquet",
        [{"article_id": "a1", "wikidata": "Q1", "full_text": "text"}],
    )
    _write(
        processed / "polygon_articles" / "x.parquet",
        [{"polygon_id": "p1", "article_id": "a1"}],
    )
    captured: dict[str, Any] = {}

    def capture_map(*args: Any, **kwargs: Any) -> Path:
        captured.update(kwargs)
        return args[2]

    monkeypatch.setattr(text_presence_module, "generate_coverage_map", capture_map)
    generate_geographic_text_presence(processed, tmp_path / "combined.png")

    assert captured["point_color"] == "#2563EB"
    assert captured["point_edge"] == "#1E40AF"


def test_continent_assignment_and_public_rendering() -> None:
    features = [
        {
            "properties": {"CONTINENT": "Europe"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-10, 35], [30, 35], [30, 70], [-10, 70], [-10, 35]]],
            },
        },
        {
            "properties": {"CONTINENT": "Africa"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-20, -40], [55, -40], [55, 34], [-20, 34], [-20, -40]]],
            },
        },
    ]
    assignments = assign_continents([(2.0, 48.0), (20.0, 0.0), (150.0, 0.0)], features)
    assert assignments == ["Europe", "Africa", "Unassigned"]

    rendered = render_continent_stats(
        [
            ("Europe", 2, 1, 1, 1, 2),
            ("Africa", 4, 2, 3, 2, 3),
        ]
    )
    assert "## Geographic distribution by continent" in rendered
    assert "| Africa | 4 | 2 | 3 | 2 | 3 | 75.0% |" in rendered
    assert "augmentation" not in rendered.lower()
    assert "WGS84 centroid" in rendered
    assert "Natural Earth 1:110m Admin-0" in rendered
    assert "`Polygons`" in rendered
    assert "`Wikipedia documents`" in rendered
    assert "`Wikivoyage documents`" in rendered
    assert "`Polygons with Wikipedia text`" in rendered
    assert "`Polygons with Wikipedia or Wikivoyage text`" in rendered
    assert "`Text coverage`" in rendered
    assert "combined text-covered polygons / all dataset polygons" in rendered
    assert "one continent" in rendered
    assert "more than one continent" in rendered
    assert "`Unassigned`" in rendered
    assert "finalized Parquet tables" in rendered


def test_dataset_card_uses_public_language_and_combined_map_first() -> None:
    card = render_dataset_card(
        repo_id="owner/dataset",
        stats={},
        polygon_columns=[],
        polygon_descriptions={},
        article_columns=[],
        article_descriptions={},
        link_columns=[],
        link_descriptions={},
    )
    assert "and no per-QID article cap" not in card
    assert "lossless" not in card
    assert "Additional derived text and fact tables are" not in card
    assert "Wikipedia and Wikivoyage text" in card
    assert "assets/geographic_text_presence.png" in card
    assert card.index("assets/geographic_text_presence.png") < card.index("assets/coverage_map.png")
    assert "assets/geographic_text_density.png" in card
    assert "assets/geographic_wikipedia_text_coverage.png" not in card
    assert "assets/geographic_polygon_count.png" not in card
    assert card.count("![") == 3
    assert "Each point represents one dataset polygon carrying an OSM" in card
    assert "raw number of polygons with non-empty Wikipedia or Wikivoyage text" in card


def test_combined_text_density_counts_overlap_once(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "x.parquet",
        [
            {"polygon_id": "p1", "wikidata": "Q1", "lon": 2.0, "lat": 48.0},
            {"polygon_id": "p2", "wikidata": "Q2", "lon": 2.01, "lat": 48.01},
            {"polygon_id": "p3", "wikidata": "Q3", "lon": 20.0, "lat": 0.0},
        ],
    )
    _write(
        processed / "wikipedia" / "documents" / "x.parquet",
        [{"article_id": "a1", "wikidata": "Q1", "full_text": "Wikipedia"}],
    )
    _write(
        processed / "polygon_articles" / "x.parquet",
        [{"polygon_id": "p1", "article_id": "a1"}],
    )
    _write(
        processed / "wikivoyage" / "documents" / "x.parquet",
        [
            {"document_id": "v1", "wikidata": "Q1", "full_text": "Both"},
            {"document_id": "v2", "wikidata": "Q2", "full_text": "Voyage"},
        ],
    )

    cells = aggregate_geographic_text_density(processed, h3_resolution=2)

    assert sum(cell.polygon_count for cell in cells) == 2


def test_combined_text_presence_deduplicates_polygon_ids_across_files(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    duplicate = {"polygon_id": "p1", "wikidata": "Q1", "lon": 2.0, "lat": 48.0}
    _write(processed / "polygons" / "a.parquet", [duplicate])
    _write(processed / "polygons" / "b.parquet", [duplicate])
    _write(
        processed / "wikipedia" / "documents" / "a.parquet",
        [{"article_id": "a1", "wikidata": "Q1", "full_text": "Wikipedia"}],
    )
    _write(
        processed / "polygon_articles" / "a.parquet",
        [{"polygon_id": "p1", "article_id": "a1"}],
    )

    snapshot = load_text_presence(processed)
    cells = aggregate_geographic_text_density(processed, snapshot=snapshot)

    assert snapshot.polygon_count == 1
    assert len(snapshot.covered_points) == 1
    assert sum(cell.polygon_count for cell in cells) == 1
    assert [cell.h3_cell for cell in cells] == sorted(cell.h3_cell for cell in cells)
