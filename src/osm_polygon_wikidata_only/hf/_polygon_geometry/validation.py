"""Input validation: the published polygon table is the only accepted input.

The statistics must never be computed from a sidecar table, from a
half-written directory, or from a polygon table that no longer matches
the processed manifest. This module owns those three refusals:

* :func:`validate_polygon_schema` rejects a Parquet file that does not
  carry the canonical polygon columns, so pointing the scanner at
  ``wikipedia/documents`` or any other table fails loudly instead of
  silently producing empty statistics.
* :func:`select_manifest_files` makes the manifest the definition of
  the published polygon table: a listed file that is missing on disk is
  a stale manifest and raises, and a file the manifest does not list is
  not part of the published dataset, so it is skipped with a warning
  rather than silently folded into the statistics.
* :func:`validate_row_counts` rejects a scanned file whose row count
  drifted from the manifest's ``polygon_count``, which is the signature
  of a stale or partially rewritten artifact.
* Every refusal raises :class:`PolygonStatsInputError` with the
  offending path named, so an operator can act on the message.

The manifest cross-check runs only when the manifest exists. A data
root that has never completed a region has no manifest yet, and an
empty polygon directory is a legitimate empty result rather than an
error.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from osm_polygon_wikidata_only.domain.schema import polygon_schema

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.polygon_geometry_stats")

MANIFEST_RELATIVE_PATH = "manifests/processed_pbfs.json"
POLYGON_SUBDIR = "polygons"

# Columns read by the scanner. Pruned reads never touch anything else.
REQUIRED_COLUMNS: tuple[str, ...] = ("source_pbf", "area_m2", "bbox", "geometry")

# Columns that identify the canonical polygon table. A Parquet file
# missing any of them is a different table and is refused.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "polygon_id",
    "source_pbf",
    "osm_type",
    "osm_id",
    "bbox",
    "geometry",
    "area_m2",
    "area_km2",
)


class PolygonStatsInputError(ValueError):
    """Raised when the scanner is handed something other than the polygon table."""


def validate_polygon_schema(path: Path, schema: pa.Schema) -> None:
    """Refuse a Parquet file that is not the canonical polygon table."""
    _refuse_missing_columns(path, schema)
    _refuse_non_canonical_types(path, schema)


def _refuse_missing_columns(path: Path, schema: pa.Schema) -> None:
    missing = tuple(column for column in IDENTITY_COLUMNS if column not in schema.names)
    if missing:
        raise PolygonStatsInputError(
            f"{path} is not the polygon table: missing columns {', '.join(missing)}"
        )


def _refuse_non_canonical_types(path: Path, schema: pa.Schema) -> None:
    expected = polygon_schema()
    mismatched = tuple(
        f"{column} (expected {expected.field(column).type}, got {schema.field(column).type})"
        for column in IDENTITY_COLUMNS
        if schema.field(column).type != expected.field(column).type
    )
    if mismatched:
        raise PolygonStatsInputError(
            f"{path} is not the polygon table: non-canonical column types: {', '.join(mismatched)}"
        )


def load_manifest_polygon_counts(processed_dir: Path) -> dict[str, int] | None:
    """Return ``{stem: polygon_count}`` from the manifest, or ``None``.

    ``None`` means the processed root carries no manifest yet, which is
    the documented "nothing has been published from here" case.
    """
    manifest = processed_dir / MANIFEST_RELATIVE_PATH
    if not manifest.is_file():
        return None
    return _decode_manifest(manifest)


def select_manifest_files(
    *,
    paths: list[Path],
    manifest_counts: Mapping[str, int] | None,
    polygons_dir: Path,
) -> list[Path]:
    """Return the polygon files the manifest publishes, in sorted order.

    With no manifest every discovered file is scanned. With a manifest,
    a listed-but-missing file raises and an unlisted file is skipped
    with a warning: publication uploads manifest-listed regions only, so
    an unlisted local file is not part of the published dataset.
    """
    if manifest_counts is None:
        return paths
    by_stem = {path.stem: path for path in paths}
    _refuse_missing(manifest_counts, by_stem, polygons_dir)
    _warn_unlisted(manifest_counts, by_stem, polygons_dir)
    return [by_stem[stem] for stem in sorted(manifest_counts)]


def validate_row_counts(
    *,
    manifest_counts: Mapping[str, int] | None,
    scanned_counts: Mapping[str, int],
    polygons_dir: Path,
) -> None:
    """Refuse a scanned file whose row count drifted from the manifest."""
    if manifest_counts is None:
        return
    drifted = _drifted_stems(manifest_counts, scanned_counts)
    if not drifted:
        return
    detail = ", ".join(
        f"{stem} (manifest {manifest_counts[stem]}, file {scanned_counts[stem]})"
        for stem in drifted
    )
    raise PolygonStatsInputError(f"{polygons_dir} row counts drifted from the manifest: {detail}")


def _drifted_stems(
    manifest_counts: Mapping[str, int], scanned_counts: Mapping[str, int]
) -> list[str]:
    """Return the stems whose scanned row count differs from the manifest."""
    return sorted(stem for stem, rows in scanned_counts.items() if manifest_counts[stem] != rows)


def _refuse_missing(
    manifest_counts: Mapping[str, int],
    by_stem: Mapping[str, Path],
    polygons_dir: Path,
) -> None:
    missing = sorted(set(manifest_counts) - set(by_stem))
    if missing:
        raise PolygonStatsInputError(
            f"{polygons_dir} is missing manifest-listed polygon files: {', '.join(missing)}"
        )


def _warn_unlisted(
    manifest_counts: Mapping[str, int],
    by_stem: Mapping[str, Path],
    polygons_dir: Path,
) -> None:
    unlisted = sorted(set(by_stem) - set(manifest_counts))
    if unlisted:
        LOGGER.warning(
            "Skipping polygon files absent from the manifest in %s: %s",
            polygons_dir,
            ", ".join(unlisted),
        )


def _decode_manifest(manifest: Path) -> dict[str, int]:
    try:
        payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolygonStatsInputError(
            f"Unreadable processed manifest: {manifest}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PolygonStatsInputError(f"Processed manifest is not a JSON object: {manifest}")
    return dict(_manifest_entries(manifest, payload))


def _manifest_entries(manifest: Path, payload: dict[str, Any]) -> list[tuple[str, int]]:
    return [_manifest_entry(manifest, entry) for entry in payload.values()]


def _manifest_entry(manifest: Path, entry: Any) -> tuple[str, int]:
    if not isinstance(entry, dict):
        raise PolygonStatsInputError(f"Processed manifest holds a non-object entry: {manifest}")
    polygons_path = entry.get("polygons_path")
    polygon_count = entry.get("polygon_count")
    if not isinstance(polygons_path, str) or not isinstance(polygon_count, int):
        raise PolygonStatsInputError(
            f"Processed manifest entry lacks polygons_path/polygon_count: {manifest}"
        )
    return _canonical_polygons_stem(manifest, polygons_path), polygon_count


def _canonical_polygons_stem(manifest: Path, polygons_path: str) -> str:
    """Return the stem of a ``polygons/<stem>.parquet`` manifest path."""
    relative_path = Path(polygons_path)
    expected_path = Path(POLYGON_SUBDIR) / relative_path.name
    if relative_path != expected_path or relative_path.suffix != ".parquet":
        raise PolygonStatsInputError(
            f"Processed manifest entry has non-canonical polygons_path "
            f"{polygons_path!r}: expected polygons/<stem>.parquet: {manifest}"
        )
    return relative_path.stem


__all__ = [
    "IDENTITY_COLUMNS",
    "MANIFEST_RELATIVE_PATH",
    "POLYGON_SUBDIR",
    "REQUIRED_COLUMNS",
    "PolygonStatsInputError",
    "load_manifest_polygon_counts",
    "select_manifest_files",
    "validate_polygon_schema",
    "validate_row_counts",
]
