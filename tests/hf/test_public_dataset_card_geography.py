"""Regression tests for public geographic coverage presentation."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.hf import geographic_text_presence as text_presence_module
from osm_polygon_wikidata_only.hf._geographic.h3_geometry import split_antimeridian
from osm_polygon_wikidata_only.hf._geographic.models import CoverageMapError
from osm_polygon_wikidata_only.hf._links.reader import is_canonical_link_schema
from osm_polygon_wikidata_only.hf.continent_stats import (
    assign_continents,
    compute_continent_stats,
    render_continent_stats,
)
from osm_polygon_wikidata_only.hf.geographic_text_density import (
    aggregate_geographic_text_density,
)
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    _document_identity_column,
    generate_geographic_text_presence,
    load_text_presence,
)
from osm_polygon_wikidata_only.v2.schema import polygon_document_link_v2_schema


def test_link_schema_classifier_accepts_v1_and_v2_but_rejects_legacy() -> None:
    assert is_canonical_link_schema(polygon_document_link_schema())
    assert is_canonical_link_schema(polygon_document_link_v2_schema())
    assert not is_canonical_link_schema(
        pa.schema(
            [
                pa.field("polygon_id", pa.string()),
                pa.field("article_id", pa.string()),
            ]
        )
    )


def test_text_presence_recognizes_v2_links_as_canonical(tmp_path: Path) -> None:
    links_dir = tmp_path / "polygon_document_links"
    links_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist([], schema=polygon_document_link_v2_schema()),
        links_dir / "x.parquet",
    )

    assert text_presence_module._has_canonical_links(links_dir)


def test_document_identity_column_prefers_canonical_id() -> None:
    assert _document_identity_column({"document_id", "article_id"}) == "document_id"
    assert _document_identity_column({"article_id"}) == "article_id"


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


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
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
    assert "combined text-covered identities / all unique polygon identities" in rendered
    assert "one continent" in rendered
    assert "more than one continent" in rendered
    assert "`Unassigned`" in rendered
    assert "finalized Parquet tables" in rendered


def test_continent_document_counts_resolve_duplicate_polygon_aliases(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "a-region.parquet",
        [
            {
                "polygon_id": "north:way:7",
                "osm_type": "way",
                "osm_id": 7,
                "wikidata": "Q7",
                "lon": 2.0,
                "lat": 48.0,
            }
        ],
    )
    _write(
        processed / "polygons" / "b-region.parquet",
        [
            {
                "polygon_id": "south:way:7",
                "osm_type": "way",
                "osm_id": 7,
                "wikidata": "Q7",
                "lon": 3.0,
                "lat": 49.0,
            }
        ],
    )
    wikipedia_dir = processed / "wikipedia" / "documents"
    wikipedia_dir.mkdir(parents=True, exist_ok=True)
    document = {field.name: None for field in wikipedia_document_schema()}
    document.update(
        {
            "document_id": "Q7:wikipedia:en:7:1",
            "article_id": "Q7:en:7:1",
            "wikidata": "Q7",
            "project": "wikipedia",
            "language": "en",
            "full_text": "body",
            "fetch_status": "ok",
        }
    )
    pq.write_table(
        pa.Table.from_pylist([document], schema=wikipedia_document_schema()),
        wikipedia_dir / "a-region.parquet",
    )
    links_dir = processed / "polygon_articles"
    links_dir.mkdir(parents=True, exist_ok=True)
    link = {field.name: None for field in polygon_document_link_schema()}
    link.update(
        {
            "polygon_id": "south:way:7",
            "document_id": "Q7:wikipedia:en:7:1",
            "project": "wikipedia",
            "wikidata": "Q7",
            "language": "en",
        }
    )
    pq.write_table(
        pa.Table.from_pylist([link], schema=polygon_document_link_schema()),
        links_dir / "a-region.parquet",
    )
    country_path = tmp_path / "countries.geojson"
    country_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "properties": {"CONTINENT": "Europe"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-10, 35], [30, 35], [30, 70], [-10, 70], [-10, 35]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert compute_continent_stats(processed, country_path) == [("Europe", 1, 1, 0, 1, 1)]


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


def test_text_reporting_deduplicates_overlapping_regions_by_typed_identity(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "a-region.parquet",
        [
            {
                "polygon_id": "north:way:7",
                "osm_type": "way",
                "osm_id": 7,
                "wikidata": "Q7",
                "lon": 2.0,
                "lat": 48.0,
            },
            {
                "polygon_id": "relation:9",
                "osm_type": "relation",
                "osm_id": 9,
                "wikidata": "Q9",
                "lon": 4.0,
                "lat": 50.0,
            },
        ],
    )
    _write(
        processed / "polygons" / "b-region.parquet",
        [
            {
                "polygon_id": "south:way:7",
                "osm_type": "way",
                "osm_id": 7,
                "wikidata": "Q7",
                "lon": 3.0,
                "lat": 49.0,
            }
        ],
    )
    _write(
        processed / "wikipedia" / "documents" / "a-region.parquet",
        [
            {
                "document_id": "wiki-7",
                "full_text": "successful Wikipedia body",
                "fetch_status": "ok",
            },
            {
                "document_id": "wiki-9-failed",
                "full_text": "looks non-empty but failed",
                "fetch_status": "http_error",
            },
        ],
    )
    _write(
        processed / "wikivoyage" / "documents" / "a-region.parquet",
        [
            {
                "document_id": "voy-9",
                "wikidata": "Q9",
                "full_text": "successful Wikivoyage body",
                "fetch_status": "ok",
            }
        ],
    )
    links_dir = processed / "polygon_articles"
    links_dir.mkdir(parents=True, exist_ok=True)
    link_rows = []
    for polygon_id, project, document_id, osm_type, osm_id in (
        ("north:way:7", "wikipedia", "wiki-7", "way", 7),
        ("relation:9", "wikipedia", "wiki-9-failed", "relation", 9),
        ("relation:9", "wikivoyage", "voy-9", "relation", 9),
        ("south:way:7", "wikipedia", "wiki-7", "way", 7),
    ):
        row = {field.name: None for field in polygon_document_link_schema()}
        row.update(
            {
                "polygon_id": polygon_id,
                "project": project,
                "document_id": document_id,
                "wikidata": f"Q{osm_id}",
                "language": "en",
                "osm_type": osm_type,
                "osm_id": osm_id,
            }
        )
        link_rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(link_rows, schema=polygon_document_link_schema()),
        links_dir / "a-region.parquet",
    )

    snapshot = load_text_presence(processed)
    cells = aggregate_geographic_text_density(processed, snapshot=snapshot)

    assert snapshot.polygon_count == 2
    assert snapshot.wikipedia_polygon_identities == frozenset({("way", 7)})
    assert snapshot.combined_polygon_identities == frozenset({("way", 7), ("relation", 9)})
    assert [(point.identity, point.polygon_id) for point in snapshot.covered_points] == [
        (("relation", 9), "relation:9"),
        (("way", 7), "north:way:7"),
    ]
    assert sum(cell.polygon_count for cell in cells) == 2


# ---------------------------------------------------------------------------
# Vectorized document scanning and snapshot reuse
# ---------------------------------------------------------------------------


def _presence_root(tmp_path: Path, documents: list[dict[str, Any]]) -> Path:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "x.parquet",
        [{"polygon_id": "p1", "wikidata": "Q1", "lon": 2.0, "lat": 48.0}],
    )
    _write(processed / "wikipedia" / "documents" / "x.parquet", documents)
    _write(processed / "polygon_articles" / "x.parquet", [{"polygon_id": "p1", "article_id": "a1"}])
    return processed


def test_document_scan_rejects_rows_whose_fetch_status_is_not_ok(tmp_path: Path) -> None:
    processed = _presence_root(
        tmp_path,
        [{"article_id": "a1", "wikidata": "Q1", "full_text": "text", "fetch_status": "error"}],
    )

    snapshot = load_text_presence(processed)

    assert snapshot.wikipedia_document_ids == frozenset()
    assert snapshot.combined_covered_polygon_ids == frozenset()


def test_document_scan_keeps_rows_whose_fetch_status_is_ok(tmp_path: Path) -> None:
    processed = _presence_root(
        tmp_path,
        [{"article_id": "a1", "wikidata": "Q1", "full_text": "text", "fetch_status": "ok"}],
    )

    assert load_text_presence(processed).wikipedia_document_ids == frozenset({"a1"})


def test_document_scan_treats_whitespace_only_and_null_text_as_empty(tmp_path: Path) -> None:
    processed = _presence_root(
        tmp_path,
        [
            {"article_id": "a1", "wikidata": "Q1", "full_text": " \t\n "},
            {"article_id": "a2", "wikidata": "Q1", "full_text": None},
        ],
    )

    assert load_text_presence(processed).wikipedia_document_ids == frozenset()


def test_document_scan_reports_a_table_without_full_text(tmp_path: Path) -> None:
    processed = _presence_root(tmp_path, [{"article_id": "a1", "wikidata": "Q1"}])

    with pytest.raises(CoverageMapError, match="full_text"):
        load_text_presence(processed)


def test_snapshot_is_reused_until_the_inputs_change(tmp_path: Path) -> None:
    processed = _presence_root(
        tmp_path, [{"article_id": "a1", "wikidata": "Q1", "full_text": "text"}]
    )

    first = load_text_presence(processed)
    assert load_text_presence(processed) is first, "an unchanged root must not be rescanned"

    _write(
        processed / "wikipedia" / "documents" / "y.parquet",
        [{"article_id": "a2", "wikidata": "Q1", "full_text": "more"}],
    )
    second = load_text_presence(processed)

    assert second is not first
    assert second.wikipedia_document_ids == frozenset({"a1", "a2"})


def test_batch_scan_reports_an_unreadable_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.parquet"

    with pytest.raises(CoverageMapError, match="Could not read wikipedia parquet"):
        text_presence_module._scan_text_batches(
            missing, "wikipedia", ["article_id", "full_text"], lambda batch: None
        )


def test_batch_scan_reports_a_file_that_is_not_parquet(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_text("this is not parquet", encoding="utf-8")

    with pytest.raises(CoverageMapError, match="could not be read as columns"):
        text_presence_module._scan_text_batches(
            corrupt, "wikipedia", ["article_id", "full_text"], lambda batch: None
        )
