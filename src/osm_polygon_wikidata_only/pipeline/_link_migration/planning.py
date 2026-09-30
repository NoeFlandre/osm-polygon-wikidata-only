"""Read-only link migration classification and planning services."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.domain.polygon_document_links import (
    CANONICAL_COLUMNS as _CANONICAL_COLUMNS,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import POLYGON_ARTICLE_COLUMNS
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    qids_from_osm_tag as _qids_from_osm_tag,
)
from osm_polygon_wikidata_only.pipeline._link_migration.conversion import (
    build_canonical_rows as _build_canonical_rows,
)
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    MigrationPlan,
    StemClassification,
    StemPlan,
)
from osm_polygon_wikidata_only.pipeline._link_migration.transaction import (
    file_content_hash as _file_content_hash,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema classification
# ---------------------------------------------------------------------------


def classify_stem_schema(columns: list[str] | tuple[str, ...]) -> str:
    """Return one of ``legacy``, ``canonical``.

    Strict column-name match is a necessary but not sufficient
    condition for ``canonical``: the calling code is expected to also
    verify the table's full schema against
    :func:`polygon_document_link_schema` with
    ``check_metadata=True``. The strict-schema check
    :func:`is_canonical_table_schema` is the authoritative comparison.

    Raises :class:`ValueError` for any other schema.
    """
    cols = tuple(columns)
    if cols == POLYGON_ARTICLE_COLUMNS:
        return "legacy"
    if cols == _CANONICAL_COLUMNS:
        return StemClassification.CANONICAL.value
    raise ValueError(
        f"Schema is neither legacy nor canonical: {list(cols)[:6]}... (got {len(cols)} columns)"
    )


def is_canonical_table_schema(table: pa.Table) -> bool:
    """Return True when *table*'s schema equals the canonical schema
    exactly (order, types, nullability, field metadata)."""
    return bool(table.schema.equals(polygon_document_link_schema(), check_metadata=True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_stem(stem: str) -> bool:
    if not stem or stem in {".", ".."}:
        return False
    return "/" not in stem and "\\" not in stem


def _read_table(path: Path) -> pa.Table:
    return pq.read_table(path)


def _table_columns(path: Path) -> tuple[str, ...]:
    """Read only Parquet footer metadata needed to classify a link table."""
    return tuple(pq.read_schema(path).names)


def _blocked_stem(
    stem: str,
    reason: str,
    fingerprints: tuple[str, str, str],
) -> StemPlan:
    """Build the common blocked classification result."""
    return StemPlan(
        stem=stem,
        classification=StemClassification.BLOCKED,
        reason=reason,
        polygons_fingerprint=fingerprints[0],
        links_fingerprint=fingerprints[1],
        documents_fingerprint=fingerprints[2],
        row_count=0,
        canonical_digest=None,
    )


def _stem_paths(stem: str, processed_dir: Path) -> tuple[Path, Path, Path]:
    """Return the polygon, link, and document paths for one stem."""
    return (
        processed_dir / "polygons" / f"{stem}.parquet",
        processed_dir / "polygon_articles" / f"{stem}.parquet",
        processed_dir / "wikipedia" / "documents" / f"{stem}.parquet",
    )


def _link_classification(path: Path) -> tuple[str | None, str | None]:
    """Classify a link table, returning a reason when it cannot be read."""
    try:
        columns = _table_columns(path)
    except (OSError, pa.ArrowInvalid) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Could not read polygon_articles schema at %s: %s", path, detail)
        return None, f"polygon_articles file unreadable: {detail}"
    try:
        return classify_stem_schema(columns), None
    except ValueError as exc:
        return None, f"unrecognised schema: {exc}"


class _UnreadableStemInputError(Exception):
    """A stem input file exists but cannot be decoded as Parquet."""


def _read_stem_table(path: Path, label: str) -> pa.Table:
    """Read one stem input, converting decode failures into a blocking reason."""
    try:
        return _read_table(path)
    except (OSError, pa.ArrowInvalid) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Could not read %s data at %s: %s", label, path, detail)
        raise _UnreadableStemInputError(f"{label} file unreadable: {detail}") from exc


def _canonical_stem_plan(
    stem: str,
    links_path: Path,
    fingerprints: tuple[str, str, str],
) -> StemPlan:
    """Validate and describe an already canonical link table."""
    try:
        links_table = _read_stem_table(links_path, "polygon_articles")
    except _UnreadableStemInputError as exc:
        return _blocked_stem(stem, str(exc), fingerprints)
    if not is_canonical_table_schema(links_table):
        return _blocked_stem(
            stem,
            "link table columns match canonical but schema differs (types or metadata)",
            fingerprints,
        )
    return StemPlan(
        stem=stem,
        classification=StemClassification.CANONICAL,
        reason="",
        polygons_fingerprint=fingerprints[0],
        links_fingerprint=fingerprints[1],
        documents_fingerprint=fingerprints[2],
        row_count=links_table.num_rows,
        canonical_digest=_file_content_hash(links_path),
    )


def _read_legacy_inputs(
    polygons_path: Path,
    links_path: Path,
    docs_path: Path,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Read the legacy link, polygon and document tables needed for conversion."""
    legacy_table = _read_stem_table(links_path, "polygon_articles")
    polygons_table = _read_stem_table(polygons_path, "polygons")
    if not docs_path.is_file():
        raise _UnreadableStemInputError("legacy schema requires wikipedia/documents/<stem>.parquet")
    docs_table = _read_stem_table(docs_path, "wikipedia documents")
    return legacy_table, polygons_table, docs_table


def _legacy_stem_plan(
    stem: str,
    polygons_path: Path,
    links_path: Path,
    docs_path: Path,
    fingerprints: tuple[str, str, str],
) -> StemPlan:
    """Convert a legacy table in memory and describe its planned result."""
    try:
        legacy_table, polygons_table, docs_table = _read_legacy_inputs(
            polygons_path, links_path, docs_path
        )
    except _UnreadableStemInputError as exc:
        return _blocked_stem(stem, str(exc), fingerprints)
    try:
        canonical_rows = _build_canonical_rows(stem, legacy_table, polygons_table, docs_table)
    except Exception as exc:  # noqa: BLE001 -- any conversion failure blocks the stem instead of aborting
        return _blocked_stem(stem, f"legacy conversion failed: {exc}", fingerprints)
    canonical_table = pa.Table.from_pylist(canonical_rows, schema=polygon_document_link_schema())
    return StemPlan(
        stem=stem,
        classification=StemClassification.MIGRATABLE,
        reason="",
        polygons_fingerprint=fingerprints[0],
        links_fingerprint=fingerprints[1],
        documents_fingerprint=fingerprints[2],
        row_count=canonical_table.num_rows,
        canonical_digest=_table_digest(canonical_table),
    )


def _classify_existing_stem(
    stem: str,
    polygons_path: Path,
    links_path: Path,
    docs_path: Path,
    fingerprints: tuple[str, str, str],
) -> StemPlan:
    """Classify a stem whose polygon and link files both exist."""
    classification, reason = _link_classification(links_path)
    if classification is None:
        return _blocked_stem(stem, reason or "polygon_articles file unreadable", fingerprints)
    if classification == StemClassification.CANONICAL.value:
        return _canonical_stem_plan(stem, links_path, fingerprints)
    return _legacy_stem_plan(stem, polygons_path, links_path, docs_path, fingerprints)


# ---------------------------------------------------------------------------
# Per-stem classification
# ---------------------------------------------------------------------------


def classify_stem(stem: str, processed_dir: Path) -> StemPlan:
    polygons_path, links_path, docs_path = _stem_paths(stem, processed_dir)
    fingerprints: tuple[str, str, str] = (
        _file_content_hash(polygons_path),
        _file_content_hash(links_path),
        _file_content_hash(docs_path),
    )
    if not polygons_path.is_file():
        return _blocked_stem(stem, "polygons file missing", fingerprints)
    if not links_path.is_file():
        return _blocked_stem(stem, "polygon_articles file missing", fingerprints)
    return _classify_existing_stem(
        stem,
        polygons_path,
        links_path,
        docs_path,
        fingerprints,
    )


def _table_digest(table: pa.Table) -> str:
    hasher = hashlib.sha256()
    for batch in table.to_batches():
        hasher.update(batch.serialize().to_pybytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_link_migration_normalization_rejections(plan: MigrationPlan) -> list[dict[str, Any]]:
    """Return the deterministic list of rejection records for the
    legacy ``polygon_articles`` rows that fail QID membership.

    A legacy row is rejected when its resolved wikidata QID is NOT
    in the polygon's resolved QID set. The cumulative ledger (after
    apply) must include every record returned here.

    The function is read-only; it does not modify any file.
    """
    rejections: list[dict[str, Any]] = []
    for sp in plan.stems:
        rejections.extend(
            plan_link_migration_normalization_rejections_for_stem(plan.processed_dir, sp)
        )
    rejections.sort(key=lambda r: (r["shard"], r["source_table"], r["identifier"], r["wikidata"]))
    return rejections


def plan_link_migration_normalization_rejections_for_stem(
    processed_dir: Path, sp: StemPlan
) -> list[dict[str, Any]]:
    """Per-stem rejection records for invalid legacy Wikipedia
    polygon↔article relationships. Empty when the stem is not
    MIGRATABLE or the source files are missing.
    """
    if sp.classification != StemClassification.MIGRATABLE:
        return []
    tables = _read_legacy_rejection_tables(processed_dir, sp)
    if tables is None:
        return []
    polygons_table, links_table = tables
    polygon_qids = _polygon_qids_by_id(polygons_table)
    rejections = _legacy_rejection_records(sp.stem, links_table, polygon_qids)
    rejections.sort(key=lambda r: (r["identifier"], r["wikidata"]))
    return rejections


def _read_legacy_rejection_tables(
    processed_dir: Path,
    stem_plan: StemPlan,
) -> tuple[pa.Table, pa.Table] | None:
    """Read only the legacy columns needed for rejection planning."""
    polygons_path, links_path, _ = _stem_paths(stem_plan.stem, processed_dir)
    if not polygons_path.is_file() or not links_path.is_file():
        return None
    try:
        polygons_table = pq.read_table(
            polygons_path,
            columns=["polygon_id", "wikidata"],
        )
        links_table = pq.read_table(
            links_path,
            columns=["polygon_id", "wikidata", "article_id"],
        )
    except (KeyError, pa.ArrowInvalid):
        return None
    return polygons_table, links_table


def _polygon_qids_by_id(polygons_table: pa.Table) -> dict[str, set[str]]:
    """Resolve each polygon's OSM wikidata tag to a QID set."""
    polygon_qids: dict[str, set[str]] = {}
    for row in polygons_table.to_pylist():
        qids = _qids_from_osm_tag(str(row.get("wikidata", "")))
        polygon_qids.setdefault(str(row["polygon_id"]), set()).update(qids)
    return polygon_qids


def _legacy_rejection_records(
    stem: str,
    links_table: pa.Table,
    polygon_qids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Build invalid polygon/article relationship records."""
    rejections: list[dict[str, Any]] = []
    for row in links_table.to_pylist():
        polygon_id = str(row["polygon_id"])
        row_qid = str(row["wikidata"])
        if row_qid and row_qid not in polygon_qids.get(polygon_id, set()):
            rejections.append(
                {
                    "shard": stem,
                    "source_table": "polygon_articles",
                    "identifier": str(row["article_id"]),
                    "wikidata": row_qid,
                    "expected": None,
                    "reason": "wikidata_not_in_polygon_qids",
                    "cascaded_sections": 0,
                }
            )
    return rejections


def plan_link_migration(
    processed_dir: Path,
    stems: set[str] | None = None,
) -> MigrationPlan:
    """Read-only planning stage.

    * Discovered stems = union of ``polygon_articles/*.parquet`` stems,
      ``wikipedia/documents/*.parquet`` stems, and ``polygons/*.parquet``
      stems.
    * Classifies each stem as ``legacy``, ``canonical`` or ``BLOCKED``.
    * For ``legacy`` stems, builds and validates the canonical row set
      without writing any file.
    * Empty stems are a normal no-op: returns an empty plan.
    """
    _validate_requested_stems(stems)
    discovered = _discover_stems(processed_dir)
    if stems is not None:
        discovered.intersection_update(stems)
    stems_data = tuple(classify_stem(stem, processed_dir) for stem in sorted(discovered))
    return MigrationPlan(processed_dir=processed_dir, stems=stems_data)


def _validate_requested_stems(stems: set[str] | None) -> None:
    """Reject path-like stem names before touching the filesystem."""
    if stems is None:
        return
    for stem in stems:
        if not _is_valid_stem(stem):
            raise ValueError(f"Invalid stem name: {stem!r}")


def _discover_stems(processed_dir: Path) -> set[str]:
    """Discover shard stems from all supported processed tables."""
    discovered: set[str] = set()
    for sub in ("polygons", "polygon_articles", "wikipedia/documents"):
        sub_path = processed_dir / sub
        if sub_path.is_dir():
            discovered.update(path.stem for path in sub_path.glob("*.parquet"))
    return discovered
