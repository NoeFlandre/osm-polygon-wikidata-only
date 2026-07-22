"""Schema-checked storage helpers for Wikidata recovery transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.config.paths import DataRoot

from .models import RecoveryRepairError


def region_paths(data_root: DataRoot, stem: str) -> dict[str, Path]:
    """Return every local artifact participating in a region repair."""
    return {
        "polygons": data_root.processed_polygons / f"{stem}.parquet",
        "links": data_root.processed_links / f"{stem}.parquet",
        "documents": data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
        "sections": data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet",
        "facts": data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
        "processed_manifest": data_root.processed_manifests / "processed_pbfs.json",
        "augmentation_manifest": data_root.processed
        / "augmentation"
        / "manifests"
        / "augmentation_manifest.json",
    }


def read_table(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
    """Read a Parquet artifact only when its full schema matches."""
    if not path.is_file():
        raise RecoveryRepairError(f"Recovery input is missing: {path}")
    actual: pa.Schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    if not actual.equals(schema, check_metadata=True):
        raise RecoveryRepairError(f"Recovery input schema mismatch: {path}")
    table: pa.Table = pq.read_table(path)  # type: ignore[no-untyped-call]
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def write_table(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    schema: pa.Schema,
) -> None:
    """Write rows using the supplied canonical column order and schema."""
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    table = pa.Table.from_pylist(normalized, schema=schema)
    pq.write_table(table, path, compression="snappy")  # type: ignore[no-untyped-call]


__all__ = ["read_table", "region_paths", "write_table"]
