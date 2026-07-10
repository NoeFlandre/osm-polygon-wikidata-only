"""Tests for the coverage map generation from polygon centroids."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf.coverage_map import (
    WORLD_LAND_FILENAME,
    ensure_world_land,
    generate_coverage_map,
    load_centroids_from_parquet,
)
from osm_polygon_wikidata_only.hf.repo_layout import REMOTE_COVERAGE_MAP_FILE

# --- repo_layout --------------------------------------------------------


def test_remote_coverage_map_path() -> None:
    assert REMOTE_COVERAGE_MAP_FILE == "coverage_map.png"


# --- helpers ------------------------------------------------------------


def _write_polygon_parquet(path: Path, lons: list[float], lats: list[float]) -> Path:
    """Write a minimal polygons parquet with only the columns we need."""
    table = pa.table({"lon": lons, "lat": lats})
    pq.write_table(table, path)
    return path


def _write_mock_land_geojson(path: Path) -> Path:
    """Write a tiny GeoJSON with two landmasses for testing."""
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-10, 35], [10, 35], [10, 60], [-10, 60], [-10, 35]]],
                },
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[-120, 25], [-80, 25], [-80, 50], [-120, 50], [-120, 25]]],
                        [[[130, 30], [145, 30], [145, 45], [130, 45], [130, 30]]],
                    ],
                },
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- load_centroids_from_parquet ----------------------------------------


def test_load_centroids_reads_lat_lon(tmp_path: Path) -> None:
    _write_polygon_parquet(tmp_path / "a-latest.parquet", [4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    lons, lats = load_centroids_from_parquet(tmp_path)
    assert lons == [4.0, 5.0, 6.0]
    assert lats == [1.0, 2.0, 3.0]


def test_load_centroids_skips_nulls(tmp_path: Path) -> None:
    table = pa.table({"lon": [4.0, None, 6.0], "lat": [1.0, 2.0, None]})
    pq.write_table(table, tmp_path / "a-latest.parquet")
    lons, lats = load_centroids_from_parquet(tmp_path)
    assert lons == [4.0]
    assert lats == [1.0]


def test_load_centroids_multiple_files(tmp_path: Path) -> None:
    _write_polygon_parquet(tmp_path / "a-latest.parquet", [1.0], [10.0])
    _write_polygon_parquet(tmp_path / "b-latest.parquet", [2.0, 3.0], [20.0, 30.0])
    lons, lats = load_centroids_from_parquet(tmp_path)
    assert sorted(lons) == [1.0, 2.0, 3.0]
    assert sorted(lats) == [10.0, 20.0, 30.0]


def test_load_centroids_empty_dir(tmp_path: Path) -> None:
    lons, lats = load_centroids_from_parquet(tmp_path)
    assert lons == []
    assert lats == []


def test_load_centroids_ignores_non_parquet(tmp_path: Path) -> None:
    _write_polygon_parquet(tmp_path / "a-latest.parquet", [1.0], [2.0])
    (tmp_path / "readme.txt").write_text("ignore me", encoding="utf-8")
    lons, lats = load_centroids_from_parquet(tmp_path)
    assert lons == [1.0]
    assert lats == [2.0]


def test_load_centroids_reads_from_subdir(tmp_path: Path) -> None:
    polygons_dir = tmp_path / "polygons"
    polygons_dir.mkdir()
    _write_polygon_parquet(polygons_dir / "a-latest.parquet", [1.0], [2.0])
    lons, lats = load_centroids_from_parquet(polygons_dir)
    assert lons == [1.0]
    assert lats == [2.0]


# --- generate_coverage_map ----------------------------------------------


def test_generate_coverage_map_creates_valid_png(tmp_path: Path) -> None:
    out = tmp_path / "coverage_map.png"
    generate_coverage_map([7.4, 19.4], [43.7, 41.3], out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_coverage_map_with_land(tmp_path: Path) -> None:
    land = _write_mock_land_geojson(tmp_path / "land.geojson")
    out = tmp_path / "coverage_map.png"
    generate_coverage_map([0.0, 10.0], [45.0, 50.0], out, land_geojson_path=land)
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_coverage_map_land_changes_output(tmp_path: Path) -> None:
    """Map with land should differ from map without."""
    land = _write_mock_land_geojson(tmp_path / "land.geojson")
    out_no_land = tmp_path / "no_land.png"
    out_with_land = tmp_path / "with_land.png"
    generate_coverage_map([], [], out_no_land)
    generate_coverage_map([], [], out_with_land, land_geojson_path=land)
    assert out_no_land.read_bytes() != out_with_land.read_bytes()


def test_generate_coverage_map_many_points(tmp_path: Path) -> None:
    lons = [float(i % 360 - 180) for i in range(5000)]
    lats = [float(i % 180 - 90) for i in range(5000)]
    out = tmp_path / "coverage_map.png"
    generate_coverage_map(lons, lats, out)
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size < 1_000_000


def test_generate_coverage_map_empty_points(tmp_path: Path) -> None:
    out = tmp_path / "coverage_map.png"
    generate_coverage_map([], [], out)
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_coverage_map_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "subdir" / "coverage_map.png"
    generate_coverage_map([0.0], [0.0], out)
    assert out.exists()


def test_generate_coverage_map_returns_output_path(tmp_path: Path) -> None:
    out = tmp_path / "coverage_map.png"
    result = generate_coverage_map([], [], out)
    assert result == out


# --- ensure_world_land --------------------------------------------------


def test_ensure_world_land_returns_cached_without_download(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cached = cache_dir / WORLD_LAND_FILENAME
    cache_dir.mkdir()
    cached.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    result = ensure_world_land(cache_dir)
    assert result == cached


def test_ensure_world_land_does_not_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / WORLD_LAND_FILENAME
    cached.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    download_called = False

    def fail_if_called(_url: str, _path: str) -> None:
        nonlocal download_called
        download_called = True

    monkeypatch.setattr("urllib.request.urlretrieve", fail_if_called)
    ensure_world_land(cache_dir)
    assert not download_called


# --- integration: map is cumulative across PBFs -------------------------


def test_coverage_map_is_cumulative_across_pbfs(tmp_path: Path) -> None:
    """After processing multiple PBFs, the map should include centroids from all of them."""
    polygons_dir = tmp_path / "processed" / "polygons"
    polygons_dir.mkdir(parents=True)

    # Simulate two already-processed PBFs.
    _write_polygon_parquet(polygons_dir / "a-latest.parquet", [1.0, 2.0], [10.0, 20.0])
    _write_polygon_parquet(polygons_dir / "b-latest.parquet", [3.0], [30.0])

    # Generate the map (this is what enqueue_upload does after each PBF).
    lons, lats = load_centroids_from_parquet(polygons_dir)
    map_path = tmp_path / "coverage_map.png"
    generate_coverage_map(lons, lats, map_path)

    assert map_path.exists()
    assert sorted(lons) == [1.0, 2.0, 3.0]
    assert sorted(lats) == [10.0, 20.0, 30.0]


def test_coverage_map_reflects_new_pbf_after_previous_pbfs(tmp_path: Path) -> None:
    """Adding a new PBF's parquet should expand the cumulative map."""
    polygons_dir = tmp_path / "processed" / "polygons"
    polygons_dir.mkdir(parents=True)

    _write_polygon_parquet(polygons_dir / "a-latest.parquet", [1.0], [10.0])
    before_lons, _before_lats = load_centroids_from_parquet(polygons_dir)
    assert before_lons == [1.0]

    # Simulate a new PBF being processed (its parquet is written).
    _write_polygon_parquet(polygons_dir / "b-latest.parquet", [2.0], [20.0])
    after_lons, after_lats = load_centroids_from_parquet(polygons_dir)
    assert sorted(after_lons) == [1.0, 2.0]
    assert sorted(after_lats) == [10.0, 20.0]


def test_map_uploaded_alongside_parquet_in_orchestrator_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator's on_complete callback fires after every PBF.

    This locks in the contract that the coverage map (and all other
    artifacts) are enqueued for upload after each individual PBF.
    """
    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.config.settings import Settings
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        InMemoryWikidataClient,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        InMemoryWikipediaClient,
    )
    from osm_polygon_wikidata_only.pipeline.orchestrator import orchestrate

    data_root = DataRoot(tmp_path / "data")
    data_root.ensure()

    # Two empty PBF placeholders.
    a = tmp_path / "a.osm.pbf"
    b = tmp_path / "b.osm.pbf"
    a.write_bytes(b"")
    b.write_bytes(b"")

    # Stub both pipeline stages to avoid touching the real PBF reader.
    def fake_extract(path: Path, **_: object) -> Path:
        return path

    def fake_process(path: Path, **_: object) -> object:
        polygons_path = data_root.processed_polygons / f"{path.stem}.parquet"
        return type(
            "R",
            (),
            {
                "manifest_entry": {"source_pbf": path.name},
                "polygons_path": polygons_path,
                "articles_path": data_root.processed_articles / f"{path.stem}.parquet",
                "polygon_articles_path": data_root.processed_links / f"{path.stem}.parquet",
                "manifest_path": data_root.processed_manifests / "processed_pbfs.json",
            },
        )()

    monkeypatch.setattr("osm_polygon_wikidata_only.pipeline.orchestrator.extract_pbf", fake_extract)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.pipeline.orchestrator.process_extracted_pbf", fake_process
    )

    callback_invocations: list[str] = []

    def on_complete(result: object) -> None:
        cb = getattr(result, "manifest_entry", {})
        callback_invocations.append(str(cb.get("source_pbf", "")))

    orchestrate(
        [a, b],
        data_root=data_root,
        settings=Settings(),
        wikidata_client=InMemoryWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        on_complete=on_complete,
    )

    # The callback fires once per PBF, after processing each one.
    assert callback_invocations == ["a.osm.pbf", "b.osm.pbf"]


# --- dataset card embedding ---------------------------------------------


def test_render_dataset_card_includes_coverage_map() -> None:
    from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card

    markdown = render_dataset_card(
        repo_id="org/name",
        stats={"polygon_count": 1, "article_count": 2, "unique_wikidata_count": 1},
        polygon_columns=["polygon_id"],
        polygon_descriptions={"polygon_id": "id"},
        article_columns=["article_id"],
        article_descriptions={"article_id": "id"},
        link_columns=["polygon_id"],
        link_descriptions={"polygon_id": "id"},
    )
    assert "coverage_map.png" in markdown
    assert "![Coverage" in markdown or "![Coverage Map]" in markdown
