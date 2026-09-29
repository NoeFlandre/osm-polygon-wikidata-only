"""Tests for deterministic global polygon identity indexing."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._geographic.polygon_identities import (
    CoverageMapError,
    _coordinate,
    _polygon_identity,
    _record_polygon_row,
    load_unique_polygon_records,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_load_unique_polygon_records_deduplicates_typed_identity_deterministically(
    tmp_path: Path,
) -> None:
    polygons = tmp_path / "polygons"
    _write(
        polygons / "a-region.parquet",
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
                "lon": None,
                "lat": None,
            },
        ],
    )
    _write(
        polygons / "b-region.parquet",
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

    index = load_unique_polygon_records(
        [polygons / "b-region.parquet", polygons / "a-region.parquet"]
    )

    assert set(index.records) == {("way", 7), ("relation", 9)}
    assert index.records[("way", 7)].polygon_id == "north:way:7"
    assert index.records[("way", 7)].lon == 2.0
    assert index.records[("way", 7)].lat == 48.0
    assert index.records[("relation", 9)].lon is None
    assert index.by_polygon_id == {
        "north:way:7": ("way", 7),
        "south:way:7": ("way", 7),
        "relation:9": ("relation", 9),
    }


def test_load_unique_polygon_records_keeps_legacy_fixture_identity_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "polygons.parquet"
    _write(
        path,
        [
            {"polygon_id": "source:way:11", "wikidata": "Q11", "lon": 1.0, "lat": 2.0},
            {"polygon_id": "fixture-id", "wikidata": "Q12", "lon": 3.0, "lat": 4.0},
        ],
    )

    index = load_unique_polygon_records([path])

    assert ("way", 11) in index.records
    assert ("legacy", "fixture-id") in index.records


@pytest.mark.parametrize(
    ("row", "polygon_id"),
    [
        ({"osm_type": None, "osm_id": 1}, "way:1"),
        ({"osm_type": "way", "osm_id": "not-an-id"}, "way:bad"),
    ],
)
def test_typed_polygon_identity_rejects_missing_or_invalid_components(
    row: dict[str, object], polygon_id: str
) -> None:
    assert _polygon_identity(row, polygon_id, typed_identity=True) is None


def test_record_polygon_row_rejects_invalid_and_conflicting_identities(tmp_path: Path) -> None:
    path = tmp_path / "polygons.parquet"
    records = {}
    by_polygon_id = {}
    valid = {"polygon_id": "p1", "osm_type": "way", "osm_id": 1}
    _record_polygon_row(
        path,
        0,
        valid,
        typed_identity=True,
        records=records,
        by_polygon_id=by_polygon_id,
    )
    assert by_polygon_id == {"p1": ("way", 1)}

    with pytest.raises(CoverageMapError, match=r"invalid \(osm_type, osm_id\) identity"):
        _record_polygon_row(
            path,
            1,
            {"polygon_id": "p2", "osm_type": None, "osm_id": 2},
            typed_identity=True,
            records=records,
            by_polygon_id=by_polygon_id,
        )
    with pytest.raises(CoverageMapError, match="conflicting identities"):
        _record_polygon_row(
            path,
            2,
            {"polygon_id": "p1", "osm_type": "relation", "osm_id": 1},
            typed_identity=True,
            records=records,
            by_polygon_id=by_polygon_id,
        )


def test_coordinate_returns_none_for_missing_or_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "polygons.parquet"
    assert _coordinate(path, "p1", None, "lon") is None
    assert _coordinate(path, "p1", float("nan"), "lon") is None
    assert _coordinate(path, "p1", "2.5", "lat") == 2.5
    with pytest.raises(CoverageMapError, match="invalid lat"):
        _coordinate(path, "p1", "not-a-number", "lat")
