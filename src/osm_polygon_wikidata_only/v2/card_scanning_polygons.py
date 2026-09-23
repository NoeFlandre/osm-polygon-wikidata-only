"""Polygon identity and source scans used by V2 card metrics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.hf._geographic.polygon_identities import (
    PolygonIndex,
    load_unique_polygon_records,
)
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.card_models import PolygonMetrics as _PolygonMetrics


def load_polygon_index(paths: Iterable[Path]) -> PolygonIndex:
    """Load canonical polygon identities while retaining incomplete-test compatibility."""
    materialized = tuple(paths)
    if not materialized:
        return PolygonIndex(records={}, by_polygon_id={})
    if any("polygon_id" not in pq.read_schema(path).names for path in materialized):
        return PolygonIndex(records={}, by_polygon_id={})
    return load_unique_polygon_records(materialized)


def polygon_source_sets(paths: Iterable[Path], identities: set[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for path in paths:
        values.update(polygon_source_file(path, identities))
    return values


def polygon_source_file(path: Path, identities: set[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    with open_parquet(path) as parquet_file:
        if not {"polygon_id", "discovery_sources"}.issubset(parquet_file.schema_arrow.names):
            return values
        for batch in iter_record_batches(
            parquet_file, columns=["polygon_id", "discovery_sources"], batch_size=65_536
        ):
            merge_polygon_sources(values, batch, identities, path)
    return values


def merge_polygon_sources(
    values: dict[str, set[str]],
    batch: Any,
    identities: set[str],
    path: Path,
) -> None:
    for identity, raw_sources in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        if identity not in identities:
            continue
        parsed = parse_source_list(raw_sources, identity, path, "discovery_sources")
        values[str(identity)] = set(parsed)


def parse_source_list(
    raw_sources: Any,
    identity: Any,
    path: Path,
    field: str,
) -> list[str]:
    try:
        parsed = json_loads(str(raw_sources or "[]"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for polygon {identity!r} in {path}") from exc
    return validated_source_list(parsed, identity, field)


def validated_source_list(parsed: Any, identity: Any, field: str) -> list[str]:
    if not isinstance(parsed, list):
        raise ValueError(f"Invalid {field} for polygon {identity!r}")
    if not all(isinstance(source, str) for source in parsed):
        raise ValueError(f"Invalid {field} for polygon {identity!r}")
    return parsed


def polygon_ids_with_link_source(paths: Iterable[Path], source: str) -> set[str]:
    values: set[str] = set()
    for path in paths:
        values.update(link_source_file(path, source))
    return values


def link_source_file(path: Path, source: str) -> set[str]:
    values: set[str] = set()
    with open_parquet(path) as parquet_file:
        if not {"polygon_id", "link_sources"}.issubset(parquet_file.schema_arrow.names):
            return values
        for batch in iter_record_batches(
            parquet_file, columns=["polygon_id", "link_sources"], batch_size=65_536
        ):
            merge_link_sources(values, batch, source, path)
    return values


def merge_link_sources(values: set[str], batch: Any, source: str, path: Path) -> None:
    for identity, raw_sources in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        parsed = parse_source_list(raw_sources, identity, path, "link_sources")
        if identity and source in parsed:
            values.add(str(identity))


def osm_polygon_identity(osm_type: Any, osm_id: Any) -> tuple[str, int] | None:
    if osm_type in (None, "") or osm_id in (None, ""):
        return None
    try:
        return str(osm_type), int(osm_id)
    except (TypeError, ValueError):
        return None


def scan_polygon_metrics(paths: Iterable[Path]) -> _PolygonMetrics:
    """Collect polygon identities, QIDs, and source counts in one pass."""
    metrics = _PolygonMetrics(set(), set())
    for path in paths:
        scan_polygon_file(path, metrics)
    return metrics


def scan_polygon_file(path: Path, metrics: _PolygonMetrics) -> None:
    with open_parquet(path) as parquet_file:
        metadata = parquet_file.metadata
        metrics.polygon_row_count += 0 if metadata is None else int(metadata.num_rows)
        names = set(parquet_file.schema_arrow.names)
        columns = polygon_columns(names)
        if not columns:
            return
        scan_polygon_batches(parquet_file, columns=columns, metrics=metrics)


def polygon_columns(names: set[str]) -> list[str]:
    return [column for column in ("polygon_id", "wikidata", "has_wikidata") if column in names]


def batch_column(
    batch: Any,
    positions: dict[str, int],
    column: str | None,
) -> list[Any] | None:
    if column is None:
        return None
    return batch.column(positions[column]).to_pylist()


def batch_value(values: list[Any] | None, index: int) -> Any:
    return None if values is None else values[index]


def scan_polygon_batches(
    parquet_file: pq.ParquetFile,
    *,
    columns: list[str],
    metrics: _PolygonMetrics,
) -> None:
    positions = {column: index for index, column in enumerate(columns)}
    for batch in iter_record_batches(parquet_file, columns=columns, batch_size=65_536):
        scan_polygon_batch(batch, positions=positions, metrics=metrics)


def scan_polygon_batch(
    batch: Any,
    *,
    positions: dict[str, int],
    metrics: _PolygonMetrics,
) -> None:
    polygon_ids = batch_column(
        batch,
        positions,
        "polygon_id" if "polygon_id" in positions else None,
    )
    wikidata = batch_column(batch, positions, "wikidata" if "wikidata" in positions else None)
    has_wikidata = batch_column(
        batch,
        positions,
        "has_wikidata" if "has_wikidata" in positions else None,
    )
    for index in range(batch.num_rows):
        record_polygon_row(
            metrics,
            polygon_id=batch_value(polygon_ids, index),
            wikidata=batch_value(wikidata, index),
            has_wikidata=batch_value(has_wikidata, index),
        )


def record_polygon_row(
    metrics: _PolygonMetrics,
    *,
    polygon_id: Any,
    wikidata: Any,
    has_wikidata: Any,
) -> None:
    if polygon_id:
        polygon_identity = str(polygon_id)
        metrics.polygon_ids.add(polygon_identity)
    if wikidata:
        metrics.qids.update(qids_from_osm_tag(str(wikidata)))
    if has_wikidata is False:
        metrics.wikipedia_tag_only += 1
