"""Row-group streaming scan of the published polygon table.

The scan reads ``<processed>/polygons/*.parquet`` in sorted order and
nothing else: no sidecar table, no raw PBF, and no network call. Only
the four columns the statistics need are read
(:data:`~.validation.REQUIRED_COLUMNS`), and the geometry strings are
decoded one record batch at a time so peak memory stays bounded by the
batch size rather than by the dataset size. The per-row scalars kept
across batches are small fixed-width arrays.

Determinism
-----------
Files are visited in sorted order, samples are concatenated in that
order, and every published float is rounded by :mod:`.summaries`. The
same inputs therefore always produce the same
:class:`~.models.PolygonGeometryStats`, and an unchanged dataset
produces a byte-identical report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .decoding import BboxSample, decode_bbox, decode_geometry
from .models import (
    STATS_CONTRACT_VERSION,
    ExtentSummary,
    PolygonGeometryStats,
    ShapeSummary,
    SourceAreaSummary,
)
from .summaries import (
    build_area_histogram,
    build_area_summary,
    build_distribution,
    build_source_summary,
    rounded,
)
from .validation import (
    POLYGON_SUBDIR,
    REQUIRED_COLUMNS,
    load_manifest_polygon_counts,
    select_manifest_files,
    validate_polygon_schema,
    validate_row_counts,
)

# One record batch of geometry strings is the scan's peak allocation.
BATCH_ROWS = 4096

# Equatorial radius used by the extraction code's equirectangular
# projection. Imported as a literal so this module keeps the hf layer's
# dependency on the domain package at zero.
EARTH_RADIUS_M = 6_378_137.0


@dataclass(slots=True)
class _Accumulator:
    """Mutable per-scan sample buffers. Never leaves this module."""

    areas: list[np.ndarray] = field(default_factory=list)
    source_areas: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    vertices: list[np.ndarray] = field(default_factory=list)
    rings: list[np.ndarray] = field(default_factory=list)
    components: list[np.ndarray] = field(default_factory=list)
    width_deg: list[np.ndarray] = field(default_factory=list)
    height_deg: list[np.ndarray] = field(default_factory=list)
    width_m: list[np.ndarray] = field(default_factory=list)
    height_m: list[np.ndarray] = field(default_factory=list)
    polygon_count: int = 0
    area_null_count: int = 0
    polygon_type_count: int = 0
    multipolygon_type_count: int = 0
    geometry_unreadable_count: int = 0
    with_holes_count: int = 0
    total_holes: int = 0
    bbox_unreadable_count: int = 0
    wider_than_180_count: int = 0
    pole_touching_count: int = 0
    min_lon: float = float("inf")
    min_lat: float = float("inf")
    max_lon: float = float("-inf")
    max_lat: float = float("-inf")


def compute_polygon_geometry_stats(processed_dir: Path) -> PolygonGeometryStats:
    """Compute the surface and geometry statistics of the polygon table.

    Reads every valid row of every published polygon Parquet file under
    ``processed_dir``: no sampling, no truncation, and no other table.
    When a processed manifest exists it defines which files the dataset
    publishes. Raises
    :class:`~.validation.PolygonStatsInputError` when the input is not
    the polygon table, when a manifest-listed file is missing, or when a
    file's row count drifted from the manifest.
    """
    polygons_dir = processed_dir / POLYGON_SUBDIR
    manifest_counts = load_manifest_polygon_counts(processed_dir)
    paths = select_manifest_files(
        paths=sorted(polygons_dir.glob("*.parquet")) if polygons_dir.is_dir() else [],
        manifest_counts=manifest_counts,
        polygons_dir=polygons_dir,
    )
    accumulator = _Accumulator()
    scanned_counts = {path.stem: _accumulate_file(accumulator, path) for path in paths}
    validate_row_counts(
        manifest_counts=manifest_counts,
        scanned_counts=scanned_counts,
        polygons_dir=polygons_dir,
    )
    return _build_stats(accumulator, file_count=len(paths))


def _accumulate_file(accumulator: _Accumulator, path: Path) -> int:
    """Scan one polygon file batch by batch and return its row count."""
    rows = 0
    with pq.ParquetFile(path) as parquet_file:
        validate_polygon_schema(path, parquet_file.schema_arrow)
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_ROWS, columns=list(REQUIRED_COLUMNS)
        ):
            _accumulate_batch(accumulator, batch)
            rows += batch.num_rows
    return rows


def _accumulate_batch(accumulator: _Accumulator, batch: pa.RecordBatch) -> None:
    accumulator.polygon_count += batch.num_rows
    _accumulate_areas(accumulator, batch)
    _accumulate_geometry(accumulator, batch.column("geometry").to_pylist())
    _accumulate_bboxes(accumulator, batch.column("bbox").to_pylist())


def _accumulate_areas(accumulator: _Accumulator, batch: pa.RecordBatch) -> None:
    column = batch.column("area_m2")
    accumulator.area_null_count += column.null_count
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
    finite = values[np.isfinite(values)]
    accumulator.areas.append(finite)
    for source, source_values in _areas_by_source(batch.column("source_pbf"), values):
        accumulator.source_areas[source].append(source_values)


def _areas_by_source(sources: pa.Array, values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Split one batch's areas by ``source_pbf`` without a per-row loop."""
    source_values = np.asarray(sources.to_pylist(), dtype=object)
    grouped: list[tuple[str, np.ndarray]] = []
    for source in sorted({str(value) for value in source_values if value is not None}):
        selected = values[source_values == source]
        grouped.append((source, selected[np.isfinite(selected)]))
    return grouped


def _accumulate_geometry(accumulator: _Accumulator, raw_values: list[object]) -> None:
    samples = [decode_geometry(raw) for raw in raw_values]
    readable = [sample for sample in samples if sample is not None]
    accumulator.geometry_unreadable_count += len(samples) - len(readable)
    accumulator.multipolygon_type_count += sum(1 for s in readable if s.is_multipolygon)
    accumulator.polygon_type_count += sum(1 for s in readable if not s.is_multipolygon)
    accumulator.with_holes_count += sum(1 for s in readable if s.holes)
    accumulator.total_holes += sum(s.holes for s in readable)
    accumulator.vertices.append(_int_array([s.vertices for s in readable]))
    accumulator.rings.append(_int_array([s.rings for s in readable]))
    accumulator.components.append(_int_array([s.components for s in readable]))


def _accumulate_bboxes(accumulator: _Accumulator, raw_values: list[object]) -> None:
    samples = [decode_bbox(raw) for raw in raw_values]
    readable = [sample for sample in samples if sample is not None]
    accumulator.bbox_unreadable_count += len(samples) - len(readable)
    accumulator.wider_than_180_count += sum(1 for s in readable if s.width_deg > 180.0)
    accumulator.pole_touching_count += sum(1 for s in readable if _touches_pole(s))
    _accumulate_extent(accumulator, readable)
    accumulator.width_deg.append(_float_array([s.width_deg for s in readable]))
    accumulator.height_deg.append(_float_array([s.height_deg for s in readable]))
    accumulator.width_m.append(_float_array([_width_m(s) for s in readable]))
    accumulator.height_m.append(_float_array([_height_m(s) for s in readable]))


def _accumulate_extent(accumulator: _Accumulator, readable: list[BboxSample]) -> None:
    for sample in readable:
        accumulator.min_lon = min(accumulator.min_lon, sample.min_lon)
        accumulator.min_lat = min(accumulator.min_lat, sample.min_lat)
        accumulator.max_lon = max(accumulator.max_lon, sample.max_lon)
        accumulator.max_lat = max(accumulator.max_lat, sample.max_lat)


def _touches_pole(sample: BboxSample) -> bool:
    return sample.max_lat >= 90.0 or sample.min_lat <= -90.0


def _width_m(sample: BboxSample) -> float:
    """Equirectangular longitude span in metres at the bbox's mean latitude."""
    return EARTH_RADIUS_M * np.radians(sample.width_deg) * np.cos(np.radians(sample.mean_lat))


def _height_m(sample: BboxSample) -> float:
    """Equirectangular latitude span in metres."""
    return EARTH_RADIUS_M * np.radians(sample.height_deg)


def _int_array(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _float_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _concatenate(chunks: list[np.ndarray], dtype: type) -> np.ndarray:
    if not chunks:
        return np.asarray([], dtype=dtype)
    return np.concatenate(chunks)


def _build_stats(accumulator: _Accumulator, *, file_count: int) -> PolygonGeometryStats:
    areas = _concatenate(accumulator.areas, np.float64)
    return PolygonGeometryStats(
        contract_version=STATS_CONTRACT_VERSION,
        file_count=file_count,
        polygon_count=accumulator.polygon_count,
        area=build_area_summary(areas, null_count=accumulator.area_null_count),
        area_histogram=build_area_histogram(areas),
        shape=_build_shape(accumulator),
        extent=_build_extent(accumulator),
        per_source=_build_per_source(accumulator),
    )


def _build_shape(accumulator: _Accumulator) -> ShapeSummary:
    vertices = _concatenate(accumulator.vertices, np.int64)
    rings = _concatenate(accumulator.rings, np.int64)
    components = _concatenate(accumulator.components, np.int64)
    return ShapeSummary(
        polygon_count=accumulator.polygon_type_count,
        multipolygon_count=accumulator.multipolygon_type_count,
        unreadable_count=accumulator.geometry_unreadable_count,
        with_holes_count=accumulator.with_holes_count,
        total_rings=int(rings.sum()),
        total_holes=accumulator.total_holes,
        total_vertices=int(vertices.sum()),
        vertices=build_distribution(vertices),
        rings=build_distribution(rings),
        components=build_distribution(components),
    )


def _build_extent(accumulator: _Accumulator) -> ExtentSummary:
    bounds = _extent_bounds(accumulator)
    return ExtentSummary(
        dataset_min_lon=bounds[0],
        dataset_min_lat=bounds[1],
        dataset_max_lon=bounds[2],
        dataset_max_lat=bounds[3],
        width_deg=build_distribution(_concatenate(accumulator.width_deg, np.float64)),
        height_deg=build_distribution(_concatenate(accumulator.height_deg, np.float64)),
        width_m=build_distribution(_concatenate(accumulator.width_m, np.float64)),
        height_m=build_distribution(_concatenate(accumulator.height_m, np.float64)),
        wider_than_180_deg_count=accumulator.wider_than_180_count,
        pole_touching_count=accumulator.pole_touching_count,
        unreadable_count=accumulator.bbox_unreadable_count,
    )


def _extent_bounds(accumulator: _Accumulator) -> tuple[float, float, float, float]:
    """Return the dataset envelope, or four zeros when no bbox was read."""
    if accumulator.min_lon == float("inf"):
        return (0.0, 0.0, 0.0, 0.0)
    return (
        rounded(accumulator.min_lon),
        rounded(accumulator.min_lat),
        rounded(accumulator.max_lon),
        rounded(accumulator.max_lat),
    )


def _build_per_source(accumulator: _Accumulator) -> tuple[SourceAreaSummary, ...]:
    return tuple(
        build_source_summary(source, _concatenate(accumulator.source_areas[source], np.float64))
        for source in sorted(accumulator.source_areas)
    )


__all__ = ["BATCH_ROWS", "compute_polygon_geometry_stats"]
