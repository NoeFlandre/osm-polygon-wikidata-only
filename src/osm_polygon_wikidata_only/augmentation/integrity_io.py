"""Parquet readers shared by shard integrity checks."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _record_polygon_wikidata(
    mapping: dict[str, str],
    duplicates: set[str],
    polygon_id: str,
    wikidata: str,
) -> None:
    """Record one polygon identity and flag conflicting Wikidata values."""
    if polygon_id in mapping and mapping[polygon_id] != wikidata:
        duplicates.add(polygon_id)
        return
    mapping[polygon_id] = wikidata


def read_polygon_wikidata_map(polygons_path: Path) -> dict[str, str]:
    """Read canonical polygon QIDs, rejecting conflicting polygon identities."""
    if not polygons_path.is_file():
        raise FileNotFoundError(f"Polygons parquet missing: {polygons_path}")
    table: pa.Table = pq.read_table(polygons_path, columns=["polygon_id", "wikidata"])
    mapping: dict[str, str] = {}
    duplicates: set[str] = set()
    for row in zip(
        table.column("polygon_id").to_pylist(),
        table.column("wikidata").to_pylist(),
        strict=True,
    ):
        polygon_id, wikidata = row
        _record_polygon_wikidata(
            mapping,
            duplicates,
            str(polygon_id),
            str(wikidata) if wikidata is not None else "",
        )
    if duplicates:
        sorted_ids = ", ".join(sorted(duplicates)[:5])
        raise ValueError(
            f"Polygons parquet has conflicting wikidata for polygon_id(s): {sorted_ids}"
        )
    return mapping


def read_polygon_wikidata_set(polygons_path: Path) -> set[str]:
    """Return the set of distinct Wikidata QIDs in the polygons parquet."""
    if not polygons_path.is_file():
        raise FileNotFoundError(f"Polygons parquet missing: {polygons_path}")
    table: pa.Table = pq.read_table(polygons_path, columns=["wikidata"])
    return {str(value) for value in table.column("wikidata").to_pylist() if value}


def read_table_required(path: Path, *, label: str, columns: tuple[str, ...]) -> pa.Table:
    """Read a required parquet table with a stable missing-file diagnostic."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} parquet missing: {path}")
    return pq.read_table(path, columns=list(columns))
