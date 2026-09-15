"""Tests for the polygon surface and geometry statistics.

Every expected value below is derived by hand from the fixture rows, so
the tests cross-check the implementation rather than restate it. The
cases cover an empty dataset, a single polygon, a MultiPolygon with a
hole, a multi-file run, byte-stable repetition, and the refusals that
protect the scan from a foreign or stale input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_COLUMNS,
    article_schema,
    empty_row,
    polygon_schema,
)
from osm_polygon_wikidata_only.hf._polygon_geometry import aggregation
from osm_polygon_wikidata_only.hf._polygon_geometry.decoding import decode_bbox, decode_geometry
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import (
    PolygonStatsInputError,
    compute_polygon_geometry_stats,
    load_polygon_geometry_stats,
    polygon_directory_fingerprint,
    render_polygon_geometry_stats,
    stats_payload,
    write_polygon_stats_report,
)

# One degree square at the equator, closed per RFC 7946.
SQUARE = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
}
# Two components; the first carries one hole.
HOLED_MULTIPOLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [
            [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.5], [0.5, 1.0], [1.0, 1.0], [1.0, 0.5], [0.5, 0.5]],
        ],
        [[[5.0, 5.0], [5.0, 6.0], [6.0, 6.0], [6.0, 5.0], [5.0, 5.0]]],
    ],
}


def _polygon_row(**overrides: object) -> dict[str, object]:
    row = empty_row(POLYGON_COLUMNS)
    row.update(overrides)
    return row


def _write_polygons(processed: Path, stem: str, rows: list[dict[str, object]]) -> Path:
    directory = processed / "polygons"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{stem}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=polygon_schema()), destination)
    return destination


def _write_manifest(processed: Path, counts: dict[str, int]) -> Path:
    manifest = processed / "manifests" / "processed_pbfs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                f"{stem}.osm.pbf": {
                    "polygons_path": f"polygons/{stem}.parquet",
                    "polygon_count": count,
                }
                for stem, count in counts.items()
            }
        ),
        encoding="utf-8",
    )
    return manifest


# --- empty input --------------------------------------------------------


def test_missing_polygon_directory_yields_an_empty_snapshot(tmp_path: Path) -> None:
    stats = compute_polygon_geometry_stats(tmp_path / "processed")

    assert stats.polygon_count == 0
    assert stats.file_count == 0
    assert stats.area.total_m2 == 0.0
    assert stats.area.median_m2 == 0.0
    assert stats.extent.dataset_max_lon == 0.0
    assert stats.per_source == ()
    # The bucket list is published even when nothing landed in it.
    assert [bucket.count for bucket in stats.area_histogram] == [0] * 15


def test_empty_polygon_file_yields_zero_rows_and_one_file(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(processed, "empty-latest", [])
    _write_manifest(processed, {"empty-latest": 0})

    stats = compute_polygon_geometry_stats(processed)

    assert (stats.file_count, stats.polygon_count) == (1, 0)
    assert stats.shape.total_vertices == 0
    assert "No polygon rows are published yet." in render_polygon_geometry_stats(stats)


# --- single polygon -----------------------------------------------------


def test_single_polygon_reports_its_own_values(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "single-latest",
        [
            _polygon_row(
                source_pbf="single-latest.osm.pbf",
                area_m2=2500.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"single-latest": 1})

    stats = compute_polygon_geometry_stats(processed)

    assert stats.polygon_count == 1
    assert stats.area.total_m2 == 2500.0
    assert stats.area.minimum_m2 == stats.area.maximum_m2 == 2500.0
    assert stats.area.median_m2 == stats.area.mean_m2 == 2500.0
    assert stats.area.p1_m2 == stats.area.p99_m2 == 2500.0
    assert stats.area.non_positive_count == 0
    assert (stats.shape.polygon_count, stats.shape.multipolygon_count) == (1, 0)
    # Four distinct vertices; the repeated closing coordinate is not a fifth.
    assert stats.shape.total_vertices == 4
    assert (stats.shape.total_rings, stats.shape.total_holes) == (1, 0)
    assert stats.shape.with_holes_count == 0
    assert stats.shape.components.maximum == 1.0
    assert (stats.extent.dataset_min_lon, stats.extent.dataset_max_lat) == (0.0, 1.0)
    assert stats.extent.width_deg.median == 1.0
    assert len(stats.per_source) == 1
    assert stats.per_source[0].source_pbf == "single-latest.osm.pbf"
    assert stats.per_source[0].total_area_m2 == 2500.0
    # The single positive area falls in the [1e3, 1e4) decade bucket.
    counted = {bucket.label: bucket.count for bucket in stats.area_histogram}
    assert counted["1e3_to_1e4_m2"] == 1
    assert sum(counted.values()) == 1


# --- holes and MultiPolygon --------------------------------------------


def test_multipolygon_with_a_hole_is_counted_once_with_every_ring(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "multi-latest",
        [
            _polygon_row(
                source_pbf="multi-latest.osm.pbf",
                area_m2=1_000_000.0,
                bbox=json.dumps([0.0, 0.0, 6.0, 6.0]),
                geometry=json.dumps(HOLED_MULTIPOLYGON),
            )
        ],
    )
    _write_manifest(processed, {"multi-latest": 1})

    stats = compute_polygon_geometry_stats(processed)

    assert stats.polygon_count == 1
    assert (stats.shape.polygon_count, stats.shape.multipolygon_count) == (0, 1)
    assert stats.shape.with_holes_count == 1
    assert stats.shape.total_holes == 1
    # Two components: an outer ring plus its hole, and a second outer ring.
    assert stats.shape.total_rings == 3
    assert stats.shape.components.maximum == 2.0
    assert stats.shape.total_vertices == 12
    # The recorded area is used as-is; the hole was already subtracted upstream.
    assert stats.area.total_m2 == 1_000_000.0


def test_undecodable_geometry_and_bbox_are_counted_not_dropped(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "broken-latest",
        [
            _polygon_row(
                source_pbf="broken-latest.osm.pbf",
                area_m2=-5.0,
                bbox="not json",
                geometry='{"type": "LineString", "coordinates": [[0, 0]]}',
            )
        ],
    )
    _write_manifest(processed, {"broken-latest": 1})

    stats = compute_polygon_geometry_stats(processed)

    assert stats.polygon_count == 1
    assert stats.shape.unreadable_count == 1
    assert stats.shape.polygon_count == stats.shape.multipolygon_count == 0
    assert stats.extent.unreadable_count == 1
    assert stats.area.non_positive_count == 1
    assert stats.area_histogram[0].label == "non_positive"
    assert stats.area_histogram[0].count == 1


def test_antimeridian_and_pole_bboxes_are_flagged(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "edges-latest",
        [
            _polygon_row(
                source_pbf="edges-latest.osm.pbf",
                area_m2=1.0,
                bbox=json.dumps([-179.0, -90.0, 179.0, -80.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"edges-latest": 1})

    stats = compute_polygon_geometry_stats(processed)

    assert stats.extent.wider_than_180_deg_count == 1
    assert stats.extent.pole_touching_count == 1


# --- multi-file runs ----------------------------------------------------


def test_multi_file_run_merges_every_file_and_breaks_down_by_source(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "alpha-latest",
        [
            _polygon_row(
                source_pbf="alpha-latest.osm.pbf",
                area_m2=100.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            ),
            _polygon_row(
                source_pbf="alpha-latest.osm.pbf",
                area_m2=300.0,
                bbox=json.dumps([2.0, 2.0, 4.0, 3.0]),
                geometry=json.dumps(SQUARE),
            ),
        ],
    )
    _write_polygons(
        processed,
        "beta-latest",
        [
            _polygon_row(
                source_pbf="beta-latest.osm.pbf",
                area_m2=600.0,
                bbox=json.dumps([-10.0, -5.0, -9.0, -4.0]),
                geometry=json.dumps(HOLED_MULTIPOLYGON),
            )
        ],
    )
    _write_manifest(processed, {"alpha-latest": 2, "beta-latest": 1})

    stats = compute_polygon_geometry_stats(processed)

    assert (stats.file_count, stats.polygon_count) == (2, 3)
    assert stats.area.total_m2 == 1000.0
    assert stats.area.median_m2 == 300.0
    assert [source.source_pbf for source in stats.per_source] == [
        "alpha-latest.osm.pbf",
        "beta-latest.osm.pbf",
    ]
    assert [source.polygon_count for source in stats.per_source] == [2, 1]
    assert [source.total_area_m2 for source in stats.per_source] == [400.0, 600.0]
    assert stats.per_source[0].median_area_m2 == 200.0
    # The envelope spans both files.
    assert (stats.extent.dataset_min_lon, stats.extent.dataset_max_lon) == (-10.0, 4.0)
    assert (stats.extent.dataset_min_lat, stats.extent.dataset_max_lat) == (-5.0, 3.0)


def test_row_group_streaming_does_not_change_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    rows = [
        _polygon_row(
            source_pbf="stream-latest.osm.pbf",
            area_m2=float(index + 1),
            bbox=json.dumps([0.0, 0.0, float(index + 1), 1.0]),
            geometry=json.dumps(SQUARE),
        )
        for index in range(10)
    ]
    _write_polygons(processed, "stream-latest", rows)
    _write_manifest(processed, {"stream-latest": 10})

    whole = compute_polygon_geometry_stats(processed)
    monkeypatch.setattr(aggregation, "BATCH_ROWS", 3)
    batched = compute_polygon_geometry_stats(processed)

    assert whole == batched


# --- determinism --------------------------------------------------------


def test_report_and_card_block_are_byte_stable_for_unchanged_input(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "stable-latest",
        [
            _polygon_row(
                source_pbf="stable-latest.osm.pbf",
                area_m2=1234.5,
                bbox=json.dumps([1.0, 2.0, 3.0, 4.0]),
                geometry=json.dumps(HOLED_MULTIPOLYGON),
            )
        ],
    )
    _write_manifest(processed, {"stable-latest": 1})

    first = compute_polygon_geometry_stats(processed)
    second = compute_polygon_geometry_stats(processed)

    assert first == second
    assert stats_payload(first) == stats_payload(second)
    assert render_polygon_geometry_stats(first) == render_polygon_geometry_stats(second)

    report = write_polygon_stats_report(processed, tmp_path / "stats.json")
    original = report.read_bytes()
    write_polygon_stats_report(processed, report)
    assert report.read_bytes() == original


def test_snapshot_is_reused_until_a_polygon_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    written = _write_polygons(
        processed,
        "memo-latest",
        [
            _polygon_row(
                source_pbf="memo-latest.osm.pbf",
                area_m2=10.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"memo-latest": 1})

    scans: list[Path] = []
    original = aggregation.compute_polygon_geometry_stats

    def counting_scan(directory: Path):
        scans.append(directory)
        return original(directory)

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.polygon_geometry_stats.compute_polygon_geometry_stats",
        counting_scan,
    )

    first = load_polygon_geometry_stats(processed)
    assert load_polygon_geometry_stats(processed) is first
    assert len(scans) == 1

    _write_polygons(
        processed,
        "memo-latest",
        [
            _polygon_row(
                source_pbf="memo-latest.osm.pbf",
                area_m2=20.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    assert written.is_file()
    refreshed = load_polygon_geometry_stats(processed)

    assert len(scans) == 2
    assert refreshed.area.total_m2 == 20.0


# --- invalid and stale inputs -------------------------------------------


def test_a_non_polygon_table_is_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    directory = processed / "polygons"
    directory.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([], schema=article_schema()), directory / "articles-latest.parquet"
    )

    with pytest.raises(PolygonStatsInputError, match="is not the polygon table"):
        compute_polygon_geometry_stats(processed)


def test_a_manifest_listed_file_that_is_missing_is_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(processed, "present-latest", [])
    _write_manifest(processed, {"present-latest": 0, "absent-latest": 3})

    with pytest.raises(PolygonStatsInputError, match="missing manifest-listed polygon files"):
        compute_polygon_geometry_stats(processed)


def test_a_row_count_that_drifted_from_the_manifest_is_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "drift-latest",
        [
            _polygon_row(
                source_pbf="drift-latest.osm.pbf",
                area_m2=1.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"drift-latest": 7})

    with pytest.raises(PolygonStatsInputError, match="row counts drifted"):
        compute_polygon_geometry_stats(processed)


def test_a_malformed_manifest_is_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(processed, "any-latest", [])
    manifest = processed / "manifests" / "processed_pbfs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{ not json", encoding="utf-8")

    with pytest.raises(PolygonStatsInputError, match="Unreadable processed manifest"):
        compute_polygon_geometry_stats(processed)


def test_a_manifest_entry_without_a_polygon_count_is_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(processed, "any-latest", [])
    manifest = processed / "manifests" / "processed_pbfs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"any-latest.osm.pbf": {"polygons_path": "polygons/any-latest.parquet"}}),
        encoding="utf-8",
    )

    with pytest.raises(PolygonStatsInputError, match="lacks polygons_path/polygon_count"):
        compute_polygon_geometry_stats(processed)


def test_a_file_the_manifest_does_not_list_is_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "listed-latest",
        [
            _polygon_row(
                source_pbf="listed-latest.osm.pbf",
                area_m2=50.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_polygons(
        processed,
        "unlisted-latest",
        [
            _polygon_row(
                source_pbf="unlisted-latest.osm.pbf",
                area_m2=99.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"listed-latest": 1})

    with caplog.at_level("WARNING"):
        stats = compute_polygon_geometry_stats(processed)

    assert stats.polygon_count == 1
    assert stats.area.total_m2 == 50.0
    assert "unlisted-latest" in caplog.text


# --- published payload --------------------------------------------------


def test_payload_and_card_expose_the_documented_fields(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "payload-latest",
        [
            _polygon_row(
                source_pbf="payload-latest.osm.pbf",
                area_m2=4242.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"payload-latest": 1})
    stats = compute_polygon_geometry_stats(processed)

    payload = stats_payload(stats)

    assert payload["contract_version"] == "v1"
    assert payload["source"] == {
        "table": "polygons",
        "column_scope": ["source_pbf", "area_m2", "bbox", "geometry"],
        "file_count": 1,
        "polygon_count": 1,
    }
    assert set(payload) == {
        "contract_version",
        "source",
        "area_m2",
        "area_histogram",
        "geometry",
        "extent",
        "per_source_pbf",
    }
    assert payload["area_m2"]["total"] == 4242.0
    assert payload["per_source_pbf"][0]["source_pbf"] == "payload-latest.osm.pbf"

    block = render_polygon_geometry_stats(stats)
    assert block.startswith("## Polygon surface and geometry\n")
    assert "4,242.00 m2" in block
    assert "stats.json" in block


# --- decoder edge cases -------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not json",
        json.dumps([1, 2, 3]),
        json.dumps({"type": "Polygon"}),
        json.dumps({"type": "Polygon", "coordinates": []}),
        json.dumps({"type": "LineString", "coordinates": [[0.0, 0.0]]}),
        json.dumps({"type": "MultiPolygon", "coordinates": ["not-a-part"]}),
        json.dumps({"type": "MultiPolygon", "coordinates": [[]]}),
    ],
)
def test_undecodable_geometry_values_return_none(raw: object) -> None:
    assert decode_geometry(raw) is None


def test_an_unclosed_ring_counts_every_vertex(tmp_path: Path) -> None:
    sample = decode_geometry(
        json.dumps({"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]})
    )

    assert sample is not None
    assert sample.vertices == 3


def test_an_empty_ring_contributes_no_vertex() -> None:
    sample = decode_geometry(json.dumps({"type": "Polygon", "coordinates": [[], [[0.0, 0.0]]]}))

    assert sample is not None
    assert sample.vertices == 1


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not json",
        json.dumps({"min_lon": 0.0}),
        json.dumps([0.0, 0.0, 1.0]),
        json.dumps([0.0, 0.0, 1.0, "north"]),
        json.dumps([0.0, 0.0, 1.0, True]),
        json.dumps([1.0, 0.0, 0.0, 1.0]),
        json.dumps([0.0, 1.0, 1.0, 0.0]),
    ],
)
def test_undecodable_bbox_values_return_none(raw: object) -> None:
    assert decode_bbox(raw) is None


def test_the_memo_fingerprint_marks_missing_inputs_without_a_polygon_directory(
    tmp_path: Path,
) -> None:
    assert polygon_directory_fingerprint(tmp_path / "processed") == (
        ("processed_pbfs.json", -1, -1),
    )
    assert load_polygon_geometry_stats(tmp_path / "processed").polygon_count == 0


def test_a_manifest_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(processed, "any-latest", [])
    manifest = processed / "manifests" / "processed_pbfs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(["any-latest"]), encoding="utf-8")

    with pytest.raises(PolygonStatsInputError, match="is not a JSON object"):
        compute_polygon_geometry_stats(processed)


def test_a_manifest_entry_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_polygons(processed, "any-latest", [])
    manifest = processed / "manifests" / "processed_pbfs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"any-latest.osm.pbf": "not-an-object"}), encoding="utf-8")

    with pytest.raises(PolygonStatsInputError, match="non-object entry"):
        compute_polygon_geometry_stats(processed)


# --- review regressions -------------------------------------------------


def test_per_source_counts_every_row_including_missing_areas(tmp_path: Path) -> None:
    """A row with no recorded area still counts toward its source."""
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "nulls-latest",
        [
            _polygon_row(
                source_pbf="nulls-latest.osm.pbf",
                area_m2=None,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            ),
            _polygon_row(
                source_pbf="nulls-latest.osm.pbf",
                area_m2=None,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            ),
            _polygon_row(
                source_pbf="mixed-latest.osm.pbf",
                area_m2=None,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            ),
            _polygon_row(
                source_pbf="mixed-latest.osm.pbf",
                area_m2=8.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            ),
        ],
    )
    _write_manifest(processed, {"nulls-latest": 4})

    stats = compute_polygon_geometry_stats(processed)

    assert stats.polygon_count == 4
    assert stats.area.null_count == 3
    by_source = {source.source_pbf: source for source in stats.per_source}
    # Every area is missing: the real row count is still reported.
    assert by_source["nulls-latest.osm.pbf"].polygon_count == 2
    assert by_source["nulls-latest.osm.pbf"].total_area_m2 == 0.0
    assert by_source["nulls-latest.osm.pbf"].maximum_area_m2 == 0.0
    # One of the two rows carries an area; both rows are counted.
    assert by_source["mixed-latest.osm.pbf"].polygon_count == 2
    assert by_source["mixed-latest.osm.pbf"].total_area_m2 == 8.0


def test_a_manifest_edit_invalidates_the_memo(tmp_path: Path) -> None:
    """The manifest decides what is scanned, so it is part of the key."""
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "first-latest",
        [
            _polygon_row(
                source_pbf="first-latest.osm.pbf",
                area_m2=10.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_polygons(
        processed,
        "second-latest",
        [
            _polygon_row(
                source_pbf="second-latest.osm.pbf",
                area_m2=90.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"first-latest": 1})
    assert load_polygon_geometry_stats(processed).area.total_m2 == 10.0

    _write_manifest(processed, {"first-latest": 1, "second-latest": 1})
    refreshed = load_polygon_geometry_stats(processed)

    assert refreshed.polygon_count == 2
    assert refreshed.area.total_m2 == 100.0


def test_a_manifest_that_goes_stale_is_refused_after_a_cached_scan(tmp_path: Path) -> None:
    """A memo hit never hides a drift the manifest introduced."""
    processed = tmp_path / "processed"
    _write_polygons(
        processed,
        "drifting-latest",
        [
            _polygon_row(
                source_pbf="drifting-latest.osm.pbf",
                area_m2=10.0,
                bbox=json.dumps([0.0, 0.0, 1.0, 1.0]),
                geometry=json.dumps(SQUARE),
            )
        ],
    )
    _write_manifest(processed, {"drifting-latest": 1})
    assert load_polygon_geometry_stats(processed).polygon_count == 1

    _write_manifest(processed, {"drifting-latest": 99})

    with pytest.raises(PolygonStatsInputError, match="row counts drifted"):
        load_polygon_geometry_stats(processed)
