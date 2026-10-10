"""Tests for the coverage map generation from polygon centroids."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf import coverage_map
from osm_polygon_wikidata_only.hf.coverage_map import (
    WORLD_COUNTRIES_FILENAME,
    WORLD_LAND_FILENAME,
    _all_centroid_rows,
    _selected_centroid_rows,
    ensure_world_countries,
    ensure_world_land,
    generate_coverage_map,
    load_centroids_from_parquet,
)

pytestmark = pytest.mark.real_map


def test_polygon_id_schema_detection_handles_empty_missing_and_unreadable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_polygon_id = tmp_path / "with-id.parquet"
    pq.write_table(pa.table({"polygon_id": ["p1"]}), with_polygon_id)
    without_polygon_id = tmp_path / "without-id.parquet"
    pq.write_table(pa.table({"lat": [1.0]}), without_polygon_id)

    assert coverage_map._has_polygon_ids([])
    assert coverage_map._has_polygon_ids([with_polygon_id])
    assert not coverage_map._has_polygon_ids([without_polygon_id])

    def unreadable(_path: Path) -> object:
        raise OSError("unreadable parquet")

    monkeypatch.setattr(coverage_map.pq, "read_schema", unreadable)
    assert not coverage_map._has_polygon_ids([with_polygon_id])


# --- helpers ------------------------------------------------------------


def _write_polygon_parquet(path: Path, lons: list[float | None], lats: list[float | None]) -> Path:
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


def _natural_earth_land_path() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "ne_110m_land.geojson"


# --- load_centroids_from_parquet ----------------------------------------


def test_load_centroids_reads_lat_lon(tmp_path: Path) -> None:
    _write_polygon_parquet(tmp_path / "a-latest.parquet", [4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    lons, lats = load_centroids_from_parquet(tmp_path)
    assert lons == [4.0, 5.0, 6.0]
    assert lats == [1.0, 2.0, 3.0]


def test_all_centroid_rows_skip_missing_coordinates() -> None:
    batch = pa.record_batch({"lon": [4.0, None, 6.0], "lat": [1.0, 2.0, None]})

    assert list(_all_centroid_rows(batch)) == [(4.0, 1.0)]


def test_selected_centroid_rows_keep_only_requested_ids_with_coordinates() -> None:
    batch = pa.record_batch(
        {
            "polygon_id": ["keep", "drop", "keep-no-coordinates"],
            "lon": [4.0, 5.0, None],
            "lat": [1.0, 2.0, 3.0],
        }
    )

    assert list(_selected_centroid_rows(batch, {"keep", "keep-no-coordinates"})) == [(4.0, 1.0)]


def test_load_centroids_skips_nulls(tmp_path: Path) -> None:
    table = pa.table({"lon": [4.0, None, 6.0], "lat": [1.0, 2.0, None]})
    pq.write_table(table, tmp_path / "a-latest.parquet")
    lons, lats = load_centroids_from_parquet(tmp_path)
    assert lons == [4.0]
    assert lats == [1.0]


def test_load_centroids_streams_batches_without_read_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lons: list[float | None] = [0.0, 1.0, None, 3.0, 4.0]
    lats: list[float | None] = [0.0, 10.0, 20.0, None, 40.0]
    _write_polygon_parquet(tmp_path / "a-latest.parquet", lons, lats)

    def fail_read_table(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("centroid loading must stream Parquet batches")

    monkeypatch.setattr(coverage_map.pq, "read_table", fail_read_table)
    monkeypatch.setattr(coverage_map, "_CENTROID_BATCH_SIZE", 2)

    actual_lons, actual_lats = load_centroids_from_parquet(tmp_path)

    expected = [
        (lon, lat)
        for lon, lat in zip(lons, lats, strict=True)
        if lon is not None and lat is not None
    ]
    assert list(zip(actual_lons, actual_lats, strict=True)) == expected


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


def test_load_centroids_filters_to_requested_polygon_ids(tmp_path: Path) -> None:
    table = pa.table(
        {
            "polygon_id": ["keep", "drop"],
            "lon": [4.0, 5.0],
            "lat": [1.0, 2.0],
        }
    )
    pq.write_table(table, tmp_path / "a-latest.parquet")

    lons, lats = load_centroids_from_parquet(tmp_path, polygon_ids={"keep"})

    assert lons == [4.0]
    assert lats == [1.0]


def test_load_centroids_deduplicates_overlapping_typed_identities(tmp_path: Path) -> None:
    polygons = tmp_path / "polygons"
    polygons.mkdir()
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["north:way:7", "relation:9"],
                "osm_type": ["way", "relation"],
                "osm_id": [7, 9],
                "lon": [2.0, 4.0],
                "lat": [48.0, 50.0],
            }
        ),
        polygons / "a-region.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["south:way:7"],
                "osm_type": ["way"],
                "osm_id": [7],
                "lon": [3.0],
                "lat": [49.0],
            }
        ),
        polygons / "b-region.parquet",
    )

    lons, lats = load_centroids_from_parquet(polygons)

    assert list(zip(lons, lats, strict=True)) == [(4.0, 50.0), (2.0, 48.0)]


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


def test_generate_coverage_map_matches_tolerant_golden(tmp_path: Path) -> None:
    import matplotlib.image as mpimg
    import numpy as np

    output = tmp_path / "coverage_map.png"
    generate_coverage_map(
        [0.0, 10.0],
        [45.0, 50.0],
        output,
        land_geojson_path=_natural_earth_land_path(),
        title="Golden Coverage",
        figsize=(4, 2),
        dpi=30,
    )

    actual = mpimg.imread(output)
    expected_path = Path(__file__).parents[1] / "fixtures/coverage_map_tolerant_golden.png"
    expected = mpimg.imread(expected_path)
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))

    assert actual.shape == expected.shape
    assert float(difference.mean()) <= 0.002
    assert float((difference.max(axis=2) > 0.04).mean()) <= 0.01


def test_natural_earth_land_fixture_is_pinned() -> None:
    # Natural Earth Vector v5.1.2: geojson/ne_110m_land.geojson.
    assert sha256(_natural_earth_land_path().read_bytes()).hexdigest() == (
        "9e0729ee253ca7d7a5c4ae9395fb1902264c5377c52e224d13dd85010e2835d9"
    )


def test_generate_coverage_map_many_points(tmp_path: Path) -> None:
    lons = [float(i % 360 - 180) for i in range(10_000)]
    lats = [float(i % 180 - 90) for i in range(10_000)]
    durations: list[float] = []
    out = tmp_path / "coverage-map-2.png"
    for run in range(3):
        started = perf_counter()
        generate_coverage_map(
            lons,
            lats,
            tmp_path / f"coverage-map-{run}.png",
            land_geojson_path=_natural_earth_land_path(),
        )
        durations.append(perf_counter() - started)

    # The 0.25s target is for a 4-vCPU dev machine; double it for CI variance.
    assert median(durations) <= 1.0
    assert out.exists()
    assert (tmp_path / "coverage-map-2.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "coverage-map-2.png").stat().st_size < 1_000_000


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


def test_ensure_world_land_copies_the_bundled_reference_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = Path(coverage_map.__file__).with_name(WORLD_LAND_FILENAME)
    monkeypatch.setattr(
        "urllib.request.urlretrieve",
        lambda *_args, **_kwargs: pytest.fail("world land must be available offline"),
    )

    result = ensure_world_land(tmp_path / "cache")

    assert result.read_bytes() == bundled.read_bytes()


def test_bundled_world_land_matches_the_pinned_natural_earth_digest() -> None:
    # Natural Earth Vector v5.1.2: geojson/ne_110m_land.geojson.
    bundled = Path(coverage_map.__file__).with_name(WORLD_LAND_FILENAME)

    assert sha256(bundled.read_bytes()).hexdigest() == (
        "9e0729ee253ca7d7a5c4ae9395fb1902264c5377c52e224d13dd85010e2835d9"
    )


def test_bundled_world_land_renders_nonempty_land_pixels(tmp_path: Path) -> None:
    import matplotlib.image as mpimg
    import numpy as np

    land = Path(coverage_map.__file__).with_name(WORLD_LAND_FILENAME)
    output = tmp_path / "bundled-land.png"
    generate_coverage_map([], [], output, land_geojson_path=land, figsize=(4, 2), dpi=30)

    rendered = mpimg.imread(output)[..., :3]
    land_color = np.array([232, 224, 208], dtype=np.float32) / 255
    land_pixels = np.all(np.isclose(rendered, land_color, atol=0.01), axis=2)
    assert int(land_pixels.sum()) > 10


def test_ensure_world_countries_copies_the_bundled_reference(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    result = ensure_world_countries(cache_dir)

    assert result == cache_dir / WORLD_COUNTRIES_FILENAME
    assert result.is_file()
    assert result.stat().st_size > 0


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


def test_try_ensure_world_land_returns_path_on_success(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf.coverage_map import try_ensure_world_land

    result = try_ensure_world_land(tmp_path)

    assert result is not None
    assert result.is_file()


def test_try_ensure_world_land_logs_cause_and_warns_on_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from osm_polygon_wikidata_only.hf.coverage_map import try_ensure_world_land

    def failing(_cache: Path) -> Path:
        raise OSError("disk exploded")

    warnings: list[str] = []
    with caplog.at_level("WARNING"):
        result = try_ensure_world_land(tmp_path, ensure=failing, warn=warnings.append, message="m")

    assert result is None
    assert warnings == ["m"]
    assert any("disk exploded" in record.getMessage() for record in caplog.records)


def test_try_ensure_world_land_without_warn_callable_still_returns_none(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf.coverage_map import try_ensure_world_land

    def failing(_cache: Path) -> Path:
        raise RuntimeError("nope")

    assert try_ensure_world_land(tmp_path, ensure=failing) is None


@pytest.mark.parametrize("error", [OSError("unreadable"), KeyError("columns")])
def test_centroid_file_scan_skips_unreadable_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(_path: Path):
        raise error

    monkeypatch.setattr(coverage_map, "open_parquet", fail)
    assert coverage_map._load_centroid_file(tmp_path / "corrupt.parquet") == ([], [])
