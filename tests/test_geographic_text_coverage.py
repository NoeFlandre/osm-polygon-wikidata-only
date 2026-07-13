"""Tests for the geographic Wikipedia text coverage visualization.

The feature shows, for each H3 cell at resolution 3, the fraction of
dataset polygons linked to at least one Wikipedia article with
non-empty ``full_text``. The visualization is deterministic and
publication-ready.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
)
from osm_polygon_wikidata_only.hf.geographic_text_coverage import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_MIN_POLYGONS_PER_CELL,
    CoverageCell,
    CoverageMapError,
    aggregate_geographic_text_coverage,
    assign_h3_cell,
    generate_geographic_text_coverage,
    render_geographic_text_coverage,
)

# --- helpers ------------------------------------------------------------


def _write_polygons_parquet(
    path: Path,
    polygon_ids: list[str],
    lats: list[float],
    lons: list[float],
) -> Path:
    table = pa.table(
        {
            "polygon_id": polygon_ids,
            "lat": lats,
            "lon": lons,
            "wikidata": [f"Q{idx + 1}" for idx in range(len(polygon_ids))],
        }
    )
    pq.write_table(table, path)
    return path


def _write_articles_parquet(
    path: Path,
    article_ids: list[str],
    full_texts: list[str | None],
) -> Path:
    rows = [
        {
            "article_id": aid,
            "wikidata": aid.split(":")[0],
            "language": "en",
            "site": "enwiki",
            "title": "T",
            "url": "",
            "page_id": 1,
            "revision_id": 1,
            "revision_timestamp": "",
            "retrieved_at": "",
            "wikidata_label": "",
            "wikidata_description": "",
            "wikidata_aliases": "[]",
            "lead_text": "",
            "extract": "",
            "full_text": text,
            "full_text_format": "plain_text",
            "article_length_chars": len(text) if text else 0,
            "article_length_words": 0,
            "article_length_tokens_estimate": 0,
            "thumbnail_url": "",
            "thumbnail_width": None,
            "thumbnail_height": None,
            "categories": "[]",
            "license": "",
            "attribution": "",
            "source_api": "",
            "fetch_status": "ok" if text else "empty_text",
            "fetch_error": "",
            "content_hash": "",
        }
        for aid, text in zip(article_ids, full_texts, strict=True)
    ]
    table = pa.Table.from_pylist(rows, schema=_minimal_articles_schema())
    pq.write_table(table, path)
    return path


def _minimal_articles_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("article_id", pa.string()),
            pa.field("wikidata", pa.string()),
            pa.field("language", pa.string()),
            pa.field("site", pa.string()),
            pa.field("title", pa.string()),
            pa.field("url", pa.string()),
            pa.field("page_id", pa.int64()),
            pa.field("revision_id", pa.int64()),
            pa.field("revision_timestamp", pa.string()),
            pa.field("retrieved_at", pa.string()),
            pa.field("wikidata_label", pa.string()),
            pa.field("wikidata_description", pa.string()),
            pa.field("wikidata_aliases", pa.string()),
            pa.field("lead_text", pa.string()),
            pa.field("extract", pa.string()),
            pa.field("full_text", pa.string()),
            pa.field("full_text_format", pa.string()),
            pa.field("article_length_chars", pa.int64()),
            pa.field("article_length_words", pa.int64()),
            pa.field("article_length_tokens_estimate", pa.int64()),
            pa.field("thumbnail_url", pa.string()),
            pa.field("thumbnail_width", pa.int64()),
            pa.field("thumbnail_height", pa.int64()),
            pa.field("categories", pa.string()),
            pa.field("license", pa.string()),
            pa.field("attribution", pa.string()),
            pa.field("source_api", pa.string()),
            pa.field("fetch_status", pa.string()),
            pa.field("fetch_error", pa.string()),
            pa.field("content_hash", pa.string()),
        ]
    )


def _write_links_parquet(path: Path, links: Iterable[tuple[str, str]]) -> Path:
    rows = [
        {
            "polygon_id": pid,
            "article_id": aid,
            "wikidata": pid.split(":")[0],
            "language": "en",
            "source_pbf": "fixture",
            "region": "fixture",
            "osm_type": "way",
            "osm_id": 1,
            "page_id": 1,
            "revision_id": 1,
            "is_best_language": True,
        }
        for pid, aid in links
    ]
    table = pa.Table.from_pylist(rows, schema=_minimal_links_schema())
    pq.write_table(table, path)
    return path


def _minimal_links_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("polygon_id", pa.string()),
            pa.field("article_id", pa.string()),
            pa.field("wikidata", pa.string()),
            pa.field("language", pa.string()),
            pa.field("source_pbf", pa.string()),
            pa.field("region", pa.string()),
            pa.field("osm_type", pa.string()),
            pa.field("osm_id", pa.int64()),
            pa.field("page_id", pa.int64()),
            pa.field("revision_id", pa.int64()),
            pa.field("is_best_language", pa.bool_()),
        ]
    )


def _build_processed_root(
    tmp_path: Path,
    *,
    polygons: tuple[tuple[str, float, float], ...],
    articles: tuple[tuple[str, str | None], ...] = (),
    links: tuple[tuple[str, str], ...] = (),
) -> Path:
    """Build a synthetic processed root; returns the ``processed/`` dir."""
    polygons_dir = tmp_path / "processed" / "polygons"
    articles_dir = tmp_path / "processed" / "articles"
    links_dir = tmp_path / "processed" / "polygon_articles"
    polygons_dir.mkdir(parents=True)
    articles_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    if polygons:
        polygon_ids = [p[0] for p in polygons]
        lats = [p[1] for p in polygons]
        lons = [p[2] for p in polygons]
        _write_polygons_parquet(polygons_dir / "fixture-latest.parquet", polygon_ids, lats, lons)
    if articles:
        article_ids = [a[0] for a in articles]
        texts = [a[1] for a in articles]
        _write_articles_parquet(articles_dir / "fixture-latest.parquet", article_ids, texts)
    if links:
        _write_links_parquet(links_dir / "fixture-latest.parquet", links)
    return tmp_path / "processed"


# --- H3 assignment ------------------------------------------------------


def test_assign_h3_cell_returns_string_at_resolution_3() -> None:
    cell = assign_h3_cell(43.73, 7.42, resolution=3)
    assert isinstance(cell, str)
    assert cell.startswith("83")  # res-3 hex IDs begin with "83"
    # Resolution must be encoded in the leading nibble of the second pair.
    assert int(cell[1], 16) == 3


def test_assign_h3_cell_matches_known_value_for_known_coordinate() -> None:
    # Paris is known to map to res-3 cell 831fb4fffffffff (verified against
    # the upstream h3 4.x library during exploration of this feature).
    assert assign_h3_cell(48.8566, 2.3522, resolution=3) == "831fb4fffffffff"


def test_assign_h3_cell_rejects_nan() -> None:
    with pytest.raises(CoverageMapError):
        assign_h3_cell(math.nan, 0.0, resolution=3)


def test_assign_h3_cell_rejects_out_of_range_latitude() -> None:
    with pytest.raises(CoverageMapError):
        assign_h3_cell(91.0, 0.0, resolution=3)


def test_assign_h3_cell_rejects_out_of_range_longitude() -> None:
    with pytest.raises(CoverageMapError):
        assign_h3_cell(0.0, 181.0, resolution=3)


def test_assign_h3_cell_rejects_invalid_resolution() -> None:
    with pytest.raises(CoverageMapError):
        assign_h3_cell(0.0, 0.0, resolution=99)


# --- Aggregation --------------------------------------------------------


def test_aggregate_returns_cell_per_h3_id_sorted(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(
            ("a:way:1", 43.73, 7.42),
            ("a:way:2", 43.74, 7.43),
            ("a:way:3", 43.75, 7.44),
        ),
    )
    cells = aggregate_geographic_text_coverage(processed)
    assert cells
    # Results are sorted by H3 cell ID.
    ids = [cell.h3_cell for cell in cells]
    assert ids == sorted(ids)


def test_aggregate_each_polygon_contributes_once_to_denominator(tmp_path: Path) -> None:
    # Three polygons, all in roughly the same H3 cell.
    processed = _build_processed_root(
        tmp_path,
        polygons=(
            ("a:way:1", 0.0, 0.0),
            ("a:way:2", 0.001, 0.001),
            ("a:way:3", -0.001, -0.001),
        ),
    )
    cells = aggregate_geographic_text_coverage(processed)
    assert sum(c.polygon_count for c in cells) == 3


def test_aggregate_polygon_with_one_qualifying_article_counts_as_covered(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0),),
        articles=(("p:en:1:1", "Some non-empty body"),),
        links=(("p:way:1", "p:en:1:1"),),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.polygon_count == 1
    assert cell.covered_polygon_count == 1
    assert cell.coverage_rate == pytest.approx(1.0)


def test_aggregate_polygon_with_multiple_qualifying_articles_counts_once(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0),),
        articles=(
            ("p:en:1:1", "first body"),
            ("p:fr:2:2", "second body"),
            ("p:de:3:3", "third body"),
        ),
        links=(
            ("p:way:1", "p:en:1:1"),
            ("p:way:1", "p:fr:2:2"),
            ("p:way:1", "p:de:3:3"),
        ),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.covered_polygon_count == 1
    assert cell.coverage_rate == pytest.approx(1.0)


def test_aggregate_polygon_with_only_empty_text_does_not_count(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0),),
        articles=(("p:en:1:1", ""),),
        links=(("p:way:1", "p:en:1:1"),),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.covered_polygon_count == 0
    assert cell.coverage_rate == pytest.approx(0.0)


def test_aggregate_polygon_with_only_null_text_does_not_count(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0),),
        articles=(("p:en:1:1", None),),
        links=(("p:way:1", "p:en:1:1"),),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.covered_polygon_count == 0


def test_aggregate_polygon_with_only_whitespace_text_does_not_count(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0),),
        articles=(("p:en:1:1", "   \n\t  "),),
        links=(("p:way:1", "p:en:1:1"),),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.covered_polygon_count == 0


def test_aggregate_polygon_without_link_does_not_count(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0),),
        articles=(("p:en:1:1", "body"),),
        links=(),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.covered_polygon_count == 0


def test_aggregate_one_polygon_with_one_qualifying_link(tmp_path: Path) -> None:
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0), ("p:way:2", 0.0, 0.0)),
        articles=(("p:en:1:1", "body"), ("p:en:2:2", "")),
        links=(("p:way:1", "p:en:1:1"), ("p:way:2", "p:en:2:2")),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.polygon_count == 2
    assert cell.covered_polygon_count == 1
    assert cell.coverage_rate == pytest.approx(0.5)


def test_aggregate_mixed_articles_distinguish_covered_polygon(tmp_path: Path) -> None:
    # Polygon A links to two articles, one with text and one without.
    # Polygon B links only to an empty-text article.
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:A", 0.0, 0.0), ("p:way:B", 0.0, 0.0)),
        articles=(
            ("p:en:1:1", "body"),
            ("p:en:1:2", ""),
            ("p:en:2:1", ""),
        ),
        links=(
            ("p:way:A", "p:en:1:1"),
            ("p:way:A", "p:en:1:2"),
            ("p:way:B", "p:en:2:1"),
        ),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.polygon_count == 2
    assert cell.covered_polygon_count == 1


def test_aggregate_cell_includes_low_sample_flag(tmp_path: Path) -> None:
    # With DEFAULT_MIN_POLYGONS_PER_CELL = 20, one polygon is low-sample.
    processed = _build_processed_root(
        tmp_path,
        polygons=(("p:way:1", 0.0, 0.0),),
    )
    [cell] = aggregate_geographic_text_coverage(processed)
    assert cell.is_low_sample is True


def test_aggregate_cell_with_default_threshold_just_above_is_not_low_sample(tmp_path: Path) -> None:
    polygons = tuple((f"p:way:{idx}", 0.0 + idx * 1e-6, 0.0 + idx * 1e-6) for idx in range(20))
    processed = _build_processed_root(tmp_path, polygons=polygons)
    cells = aggregate_geographic_text_coverage(processed)
    assert sum(c.polygon_count for c in cells) == 20
    # At least one cell must be exactly at or above 20.
    assert any(not c.is_low_sample for c in cells)


def test_aggregate_custom_threshold_affects_low_sample_flag(tmp_path: Path) -> None:
    polygons = tuple((f"p:way:{idx}", 0.0 + idx * 1e-6, 0.0 + idx * 1e-6) for idx in range(5))
    processed = _build_processed_root(tmp_path, polygons=polygons)
    cells_high = aggregate_geographic_text_coverage(processed, min_polygons_per_cell=10)
    cells_low = aggregate_geographic_text_coverage(processed, min_polygons_per_cell=2)
    assert all(c.is_low_sample for c in cells_high)
    assert all(not c.is_low_sample for c in cells_low)


def test_aggregate_determinism_repeated_calls_identical(tmp_path: Path) -> None:
    polygons = tuple((f"p:way:{idx}", idx * 0.01, idx * 0.01) for idx in range(30))
    articles = tuple((f"p:en:{idx}:1", f"body {idx}") for idx in range(30))
    links = tuple((f"p:way:{idx}", f"p:en:{idx}:1") for idx in range(30))
    processed = _build_processed_root(tmp_path, polygons=polygons, articles=articles, links=links)
    first = aggregate_geographic_text_coverage(processed)
    second = aggregate_geographic_text_coverage(processed)
    assert first == second


def test_aggregate_sorted_by_h3_cell_id(tmp_path: Path) -> None:
    # Spread polygons across multiple H3 cells using valid coordinates.
    polygons = tuple(
        (f"p:way:{idx}", float(idx % 170 - 80), float(idx % 350 - 175)) for idx in range(60)
    )
    processed = _build_processed_root(tmp_path, polygons=polygons)
    cells = aggregate_geographic_text_coverage(processed)
    ids = [cell.h3_cell for cell in cells]
    assert ids == sorted(ids)


# --- Schema/input failures ---------------------------------------------


def test_aggregate_missing_polygons_directory_raises(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "articles").mkdir(parents=True)
    (processed / "polygon_articles").mkdir(parents=True)
    with pytest.raises(CoverageMapError, match=r"polygons"):
        aggregate_geographic_text_coverage(processed)


def test_aggregate_missing_articles_directory_raises(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "polygons").mkdir(parents=True)
    (processed / "polygon_articles").mkdir(parents=True)
    with pytest.raises(CoverageMapError, match=r"articles"):
        aggregate_geographic_text_coverage(processed)


def test_aggregate_missing_links_directory_raises(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "polygons").mkdir(parents=True)
    (processed / "articles").mkdir(parents=True)
    with pytest.raises(CoverageMapError, match=r"polygon_articles"):
        aggregate_geographic_text_coverage(processed)


def test_aggregate_missing_polygon_lat_column_raises(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    polygons_dir = processed / "polygons"
    articles_dir = processed / "articles"
    links_dir = processed / "polygon_articles"
    polygons_dir.mkdir(parents=True)
    articles_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    pq.write_table(pa.table({"polygon_id": ["p:way:1"], "lon": [0.0]}), polygons_dir / "a.parquet")
    pq.write_table(
        pa.table({"article_id": ["p:en:1:1"], "full_text": ["body"]}),
        articles_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"polygon_id": ["p:way:1"], "article_id": ["p:en:1:1"]}),
        links_dir / "a.parquet",
    )
    with pytest.raises(CoverageMapError, match=r"lat"):
        aggregate_geographic_text_coverage(processed)


def test_aggregate_missing_article_full_text_column_raises(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    polygons_dir = processed / "polygons"
    articles_dir = processed / "articles"
    links_dir = processed / "polygon_articles"
    polygons_dir.mkdir(parents=True)
    articles_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    _write_polygons_parquet(polygons_dir / "a.parquet", ["p:way:1"], [0.0], [0.0])
    pq.write_table(
        pa.table({"article_id": ["p:en:1:1"], "title": ["T"]}),
        articles_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"polygon_id": ["p:way:1"], "article_id": ["p:en:1:1"]}),
        links_dir / "a.parquet",
    )
    with pytest.raises(CoverageMapError, match=r"full_text"):
        aggregate_geographic_text_coverage(processed)


def test_aggregate_missing_link_polygon_id_column_raises(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    polygons_dir = processed / "polygons"
    articles_dir = processed / "articles"
    links_dir = processed / "polygon_articles"
    polygons_dir.mkdir(parents=True)
    articles_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    _write_polygons_parquet(polygons_dir / "a.parquet", ["p:way:1"], [0.0], [0.0])
    pq.write_table(
        pa.table({"article_id": ["p:en:1:1"], "full_text": ["body"]}),
        articles_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"article_id": ["p:en:1:1"]}),
        links_dir / "a.parquet",
    )
    with pytest.raises(CoverageMapError, match=r"polygon_id"):
        aggregate_geographic_text_coverage(processed)


# --- Rendering ----------------------------------------------------------


def _cell_fixture() -> list[CoverageCell]:
    return [
        CoverageCell(
            h3_cell="833969fffffffff",
            polygon_count=25,
            covered_polygon_count=15,
            coverage_rate=15 / 25,
            is_low_sample=False,
        ),
        CoverageCell(
            h3_cell="83754efffffffff",
            polygon_count=3,
            covered_polygon_count=1,
            coverage_rate=1 / 3,
            is_low_sample=True,
        ),
    ]


def test_render_creates_valid_png(tmp_path: Path) -> None:
    out = tmp_path / "coverage.png"
    render_geographic_text_coverage(_cell_fixture(), out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "coverage.png"
    render_geographic_text_coverage(_cell_fixture(), out)
    assert out.exists()


def test_render_handles_only_low_sample_cells(tmp_path: Path) -> None:
    cells = [
        CoverageCell(
            h3_cell="83754efffffffff",
            polygon_count=3,
            covered_polygon_count=1,
            coverage_rate=1 / 3,
            is_low_sample=True,
        )
    ]
    out = tmp_path / "coverage.png"
    render_geographic_text_coverage(cells, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_does_not_perform_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda *_args, **_kwargs: pytest.fail("network call")
    )
    out = tmp_path / "coverage.png"
    render_geographic_text_coverage(_cell_fixture(), out)
    assert out.exists()


def test_render_png_is_pillow_openable(tmp_path: Path) -> None:
    from PIL import Image

    out = tmp_path / "coverage.png"
    render_geographic_text_coverage(_cell_fixture(), out)
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.mode in {"RGB", "RGBA"}


def test_render_writes_through_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    seen_replaces: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def tracking_replace(src: str | Path, dst: str | Path) -> None:
        seen_replaces.append((Path(str(src)), Path(str(dst))))
        return original_replace(src, dst)

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.geographic_text_coverage.os.replace",
        tracking_replace,
    )
    out = tmp_path / "coverage.png"
    render_geographic_text_coverage(_cell_fixture(), out)
    assert any(dst == out for _, dst in seen_replaces)


def test_render_deterministic_with_fixed_inputs(tmp_path: Path) -> None:
    cells = _cell_fixture()
    out1 = tmp_path / "a.png"
    out2 = tmp_path / "b.png"
    render_geographic_text_coverage(cells, out1)
    render_geographic_text_coverage(cells, out2)
    # Inputs are identical and config is fixed -> byte-identical output.
    assert out1.read_bytes() == out2.read_bytes()


def test_render_handles_antimeridian_crossing_cells(tmp_path: Path) -> None:
    # Use a H3 cell id known to sit near the antimeridian. We do not assert
    # on geographic correctness of that cell; we just ensure rendering
    # does not crash or produce an empty file.
    import h3

    boundary = h3.cell_to_boundary("83754efffffffff")
    assert boundary, "boundary must be available for the cell we render"
    cells = [
        CoverageCell(
            h3_cell="83754efffffffff",
            polygon_count=25,
            covered_polygon_count=10,
            coverage_rate=10 / 25,
            is_low_sample=False,
        )
    ]
    out = tmp_path / "coverage.png"
    render_geographic_text_coverage(cells, out)
    assert out.exists()


# --- End-to-end generation ---------------------------------------------


def test_generate_writes_deterministic_path(tmp_path: Path) -> None:
    polygons = tuple((f"p:way:{idx}", idx * 0.01, idx * 0.01) for idx in range(30))
    processed = _build_processed_root(tmp_path, polygons=polygons)
    data_root = DataRoot(tmp_path / "data_root")
    data_root.ensure()
    result = generate_geographic_text_coverage(
        processed,
        tmp_path / "assets" / "geographic_wikipedia_text_coverage.png",
    )
    out = result.output_path
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- Domain helpers ----------------------------------------------------


def test_coverage_cell_is_immutable() -> None:
    cell = CoverageCell(
        h3_cell="833969fffffffff",
        polygon_count=1,
        covered_polygon_count=1,
        coverage_rate=1.0,
        is_low_sample=False,
    )
    with pytest.raises(Exception):
        cell.coverage_rate = 0.0  # type: ignore[misc]


def test_default_h3_resolution_is_three() -> None:
    assert DEFAULT_H3_RESOLUTION == 3


def test_default_min_polygons_per_cell_is_twenty() -> None:
    assert DEFAULT_MIN_POLYGONS_PER_CELL == 20


# --- Schema constant smoke check ---------------------------------------


def test_domain_schema_constants_align_with_module_contract() -> None:
    # Defensive: if the upstream schema grows new columns, the visualization
    # should still work because we read only the columns we need. We assert
    # the schema's required columns exist.
    assert "polygon_id" in POLYGON_COLUMNS
    assert "lat" in POLYGON_COLUMNS
    assert "lon" in POLYGON_COLUMNS
    assert "article_id" in ARTICLE_COLUMNS
    assert "full_text" in ARTICLE_COLUMNS
    assert "polygon_id" in POLYGON_ARTICLE_COLUMNS
    assert "article_id" in POLYGON_ARTICLE_COLUMNS


# --- Strict polygon validation (no silent skipping) -------------------


def test_aggregate_does_not_skip_null_lat_or_lon(tmp_path: Path) -> None:
    """A polygon with null lat/lon must raise a CoverageMapError, not be skipped."""
    polygons_dir = tmp_path / "processed" / "polygons"
    articles_dir = tmp_path / "processed" / "articles"
    links_dir = tmp_path / "processed" / "polygon_articles"
    polygons_dir.mkdir(parents=True)
    articles_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["p:way:1", "p:way:2"],
                "lat": [43.73, None],
                "lon": [7.42, 7.43],
            }
        ),
        polygons_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"article_id": ["p:en:1:1"], "full_text": ["body"]}),
        articles_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"polygon_id": ["p:way:1"], "article_id": ["p:en:1:1"]}),
        links_dir / "a.parquet",
    )

    with pytest.raises(CoverageMapError) as excinfo:
        aggregate_geographic_text_coverage(tmp_path / "processed")
    assert "p:way:2" in str(excinfo.value)
    assert str(polygons_dir / "a.parquet") in str(excinfo.value)


def test_aggregate_does_not_skip_non_finite_lat(tmp_path: Path) -> None:
    """A polygon with NaN/inf lat must raise, not silently drop."""
    polygons_dir = tmp_path / "processed" / "polygons"
    articles_dir = tmp_path / "processed" / "articles"
    links_dir = tmp_path / "processed" / "polygon_articles"
    polygons_dir.mkdir(parents=True)
    articles_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["p:way:n"],
                "lat": [float("nan")],
                "lon": [0.0],
            }
        ),
        polygons_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"article_id": ["p:en:1:1"], "full_text": ["body"]}),
        articles_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"polygon_id": ["p:way:n"], "article_id": ["p:en:1:1"]}),
        links_dir / "a.parquet",
    )
    with pytest.raises(CoverageMapError, match=r"p:way:n"):
        aggregate_geographic_text_coverage(tmp_path / "processed")


def test_aggregate_does_not_skip_out_of_range_lon(tmp_path: Path) -> None:
    polygons_dir = tmp_path / "processed" / "polygons"
    articles_dir = tmp_path / "processed" / "articles"
    links_dir = tmp_path / "processed" / "polygon_articles"
    polygons_dir.mkdir(parents=True)
    articles_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["p:way:bad"],
                "lat": [0.0],
                "lon": [200.0],
            }
        ),
        polygons_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"article_id": ["p:en:1:1"], "full_text": ["body"]}),
        articles_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"polygon_id": ["p:way:bad"], "article_id": ["p:en:1:1"]}),
        links_dir / "a.parquet",
    )
    with pytest.raises(CoverageMapError) as excinfo:
        aggregate_geographic_text_coverage(tmp_path / "processed")
    assert "p:way:bad" in str(excinfo.value)
    assert "longitude" in str(excinfo.value).lower() or "lon" in str(excinfo.value).lower()


# --- Caption honours configured threshold -----------------------------


def test_render_caption_reflects_non_default_threshold(tmp_path: Path) -> None:
    """The low-sample caption must cite the configured threshold, not hardcode 20."""
    cells = _cell_fixture()
    out = tmp_path / "coverage.png"

    result = render_geographic_text_coverage(cells, out, min_polygons_per_cell=42)

    assert out.exists()
    # The render call must surface the caption text so callers can introspect
    # or audit it. The caption must cite the configured threshold.
    caption = getattr(result, "caption", None)
    if caption is None:  # backwards-compat with the older Path-only return
        caption = _extract_caption_from_render(out)
    assert caption is not None
    assert "fewer than 42 polygons" in caption
    assert "fewer than 20 polygons" not in caption


def _extract_caption_from_render(rendered_path: Path) -> str | None:
    """Best-effort caption extraction if the renderer does not return it."""
    try:
        with Image.open(rendered_path) as img:  # noqa: F821  (Pillow optional)
            return img.info.get("caption")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - Pillow is optional
        return None


def test_colorbar_ticks_are_formatted_as_percentages(tmp_path: Path) -> None:
    """The colorbar must format its ticks as percentages (0% .. 100%) even though the
    underlying normalization is [0, 1]."""
    import matplotlib
    from matplotlib.ticker import FuncFormatter

    matplotlib.use("Agg")

    captured: dict[str, Any] = {}

    real_savefig = matplotlib.figure.Figure.savefig

    def capture_savefig(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The colorbar lives in its own axes; capture the second axes'
        # y-axis major formatter when it is configured as a percentage
        # formatter.
        if len(self.axes) >= 2:
            colorbar_axes = self.axes[1]
            captured["formatter"] = colorbar_axes.yaxis.get_major_formatter()
        return real_savefig(self, *args, **kwargs)

    matplotlib.figure.Figure.savefig = capture_savefig  # type: ignore[assignment]
    try:
        render_geographic_text_coverage(_cell_fixture(), tmp_path / "second.png")
    finally:
        matplotlib.figure.Figure.savefig = real_savefig  # type: ignore[assignment]

    assert "formatter" in captured, "Colorbar axes must be present on the rendered figure"
    formatter = captured["formatter"]
    assert isinstance(formatter, FuncFormatter), (
        "Colorbar must use a FuncFormatter for percentage ticks"
    )

    func = getattr(formatter, "func", None)
    assert callable(func), "FuncFormatter must expose its underlying function"

    sample_value = 0.5
    label = func(sample_value, None)
    assert isinstance(label, str)
    assert label.endswith("%")
    assert "50" in label
    # The formatter must round-trip the [0, 1] range to percentage labels.
    for value, expected in ((0.0, "0%"), (0.25, "25%"), (1.0, "100%")):
        actual = func(value, None)
        assert actual == expected or actual.startswith(expected.rstrip("%")), (
            f"value={value} produced {actual!r}, expected {expected!r}"
        )

    # The colormap normalization must remain in [0, 1] so aggregation
    # semantics are preserved.
    from osm_polygon_wikidata_only.hf.geographic_text_coverage import (
        _VMAX,
        _VMIN,
    )

    assert _VMIN == 0.0
    assert _VMAX == 1.0


# --- CLI integration: propagate CoverageMapError, do not submit -------


def test_enqueue_upload_propagates_coverage_map_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CoverageMapError from visualization must propagate; no upload may be submitted."""
    import osm_polygon_wikidata_only.cli.commands as commands_mod

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    uploads: list[list[tuple[Path, str]]] = []

    class _StubProcessResult:
        polygons_path = tmp_path / "polygons.parquet"
        articles_path = tmp_path / "articles.parquet"
        polygon_articles_path = tmp_path / "links.parquet"
        manifest_path = tmp_path / "manifest.json"
        polygon_count = 0
        article_count = 0
        link_count = 0

        def __init__(self) -> None:
            self.manifest_entry = {"source_pbf": "fixture.osm.pbf"}
            self.stage_timings_s: dict[str, float] = {}

    for path in (
        _StubProcessResult.polygons_path,
        _StubProcessResult.articles_path,
        _StubProcessResult.polygon_articles_path,
        _StubProcessResult.manifest_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")

    def boom(_data_root: object, destination: Path) -> Path:
        raise CoverageMapError("polygons parquet missing required columns: ['lat']")

    def fake_write_readme(*_args: object, **_kwargs: object) -> None:
        return None

    def submit_spy(files: list[tuple[Path, str]], message: str) -> None:
        uploads.append(list(files))

    monkeypatch.setattr(commands_mod, "_generate_geographic_text_coverage_snapshot", boom)
    monkeypatch.setattr(commands_mod, "_write_readme_snapshot", fake_write_readme)
    monkeypatch.setattr(commands_mod, "ensure_world_land", lambda cache_dir: None)
    monkeypatch.setattr(commands_mod, "generate_coverage_map", lambda *_a, **_k: None)

    upload_queue = type(
        "_Q",
        (),
        {"submit": staticmethod(submit_spy), "close_and_wait": staticmethod(lambda: [])},
    )()

    with pytest.raises(CoverageMapError, match=r"missing required columns"):
        commands_mod._enqueue_core_upload(
            upload_queue,
            data_root=data_root,
            repo_id="org/dataset",
            commit_message="Update PBF fixture.osm.pbf",
            result=_StubProcessResult(),  # type: ignore[arg-type]
        )
    assert uploads == [], "Upload must not be submitted when visualization generation fails."


def test_sync_upload_files_propagates_coverage_map_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CoverageMapError in the sync-dir core path must propagate before any file list is built."""
    import osm_polygon_wikidata_only.cli.commands as commands_mod

    data_root = DataRoot(tmp_path)
    data_root.ensure()

    augmentation_paths = [tmp_path / f"aug-{idx}" for idx in range(5)]
    for path in augmentation_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    aug_manifest = tmp_path / "aug-manifest"
    aug_manifest.parent.mkdir(parents=True, exist_ok=True)
    aug_manifest.write_text("{}", encoding="utf-8")
    augmentation = AugmentationResult(*augmentation_paths, aug_manifest, {})

    core_paths = {
        "polygons_path": tmp_path / "core.parquet",
        "articles_path": tmp_path / "articles.parquet",
        "polygon_articles_path": tmp_path / "links.parquet",
    }
    for path in core_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    data_root.processed_manifests.joinpath("processed_pbfs.json").write_text("{}", encoding="utf-8")

    class _Core:
        pass

    core = _Core()
    core.polygons_path = core_paths["polygons_path"]
    core.articles_path = core_paths["articles_path"]
    core.polygon_articles_path = core_paths["polygon_articles_path"]

    def boom(_data_root: object, destination: Path) -> Path:
        raise CoverageMapError("polygons directory does not exist: /missing")

    monkeypatch.setattr(commands_mod, "_generate_geographic_text_coverage_snapshot", boom)

    with pytest.raises(CoverageMapError, match=r"polygons directory"):
        commands_mod._sync_upload_files(
            data_root, "org/dataset", "monaco-latest", augmentation, core
        )
