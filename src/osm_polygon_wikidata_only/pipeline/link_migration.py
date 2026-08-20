"""Link migration: legacy ``polygon_articles/<stem>.parquet`` -> canonical.

The link migration converts the legacy 11-column ``polygon_articles``
table (column set: ``POLYGON_ARTICLE_COLUMNS``) into the canonical 11
columns declared by :mod:`osm_polygon_wikidata_only.domain.polygon_document_links`.

Two public stages:

* :func:`plan_link_migration` -- read-only planning and preflight.
* :func:`apply_link_migration` -- transactional apply.

The apply stage uses a private ordered journaled transaction helper
(``_commit_ordered_replacements``) that enforces manifest-last ordering
and roll-forward recovery on interruption. The helper is deliberately
NOT exposed at module-level.

Classifications:

* ``legacy``    -- the link parquet has the legacy column set.
* ``canonical`` -- the link parquet has the canonical column set.
* ``BLOCKED``   -- any other schema (legacy-but-not-recognised, mixed,
  unreadable, missing a required reference table).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.orchestrator import sidecar_paths
from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
    LEDGER_CONTRACT_VERSION,
    IntegrityPlan,
    RejectionRecord,
    load_ledger,
    merge_records,
    plan_integrity_normalization,
)
from osm_polygon_wikidata_only.augmentation.steps import (
    CONTRACT_VERSION as _AUGMENTATION_CONTRACT_VERSION,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    CANONICAL_COLUMNS as _CANONICAL_COLUMNS,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    LINK_CONTRACT_VERSION as _LINK_CONTRACT_VERSION,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    build_polygon_document_links,
    polygon_document_link_schema,
    validate_polygon_document_links,
)
from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    qids_from_osm_tag as _qids_from_osm_tag,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_parquet
from osm_polygon_wikidata_only.pipeline._link_migration.conversion import (
    build_canonical_rows as _build_canonical_rows,
)
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    MigrationPlan,
    StemClassification,
    StemPlan,
)
from osm_polygon_wikidata_only.pipeline._link_migration.transaction import (
    commit_ordered_replacements as _commit_ordered_replacements,
)
from osm_polygon_wikidata_only.utils.time import utc_now_iso as _utc_now_iso

_LINK_TRANSACTION_VERSION = "link-migration-transaction-v1"


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


def _file_content_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    atomic_write_parquet(path, table)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(raw_tmp)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _polygon_qid_set(polygons_table: pa.Table) -> set[str]:
    """Return the union of QIDs in the polygon's ``wikidata`` column.

    The polygon's ``wikidata`` is the OSM-tag value; this delegates to
    the canonical :func:`qids_from_osm_tag` parser. An invalid tag
    value contributes no QIDs and is surfaced downstream as a BLOCKED
    reason.
    """
    qids: set[str] = set()
    for raw in polygons_table.column("wikidata").to_pylist():
        qids.update(_qids_from_osm_tag(str(raw)))
    return qids


def _read_table(path: Path) -> pa.Table:
    return pq.read_table(path)  # type: ignore[no-untyped-call]


def _read_table_safely(path: Path) -> pa.Table | None:
    try:
        return _read_table(path)
    except Exception:
        return None


def _table_columns(path: Path) -> tuple[str, ...] | None:
    table = _read_table_safely(path)
    if table is None:
        return None
    return tuple(table.schema.names)


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
    columns = _table_columns(path)
    if columns is None:
        return None, "polygon_articles file unreadable"
    try:
        return classify_stem_schema(columns), None
    except ValueError as exc:
        return None, f"unrecognised schema: {exc}"


def _canonical_stem_plan(
    stem: str,
    links_path: Path,
    fingerprints: tuple[str, str, str],
) -> StemPlan:
    """Validate and describe an already canonical link table."""
    links_table = _read_table(links_path)
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


def _legacy_stem_plan(
    stem: str,
    polygons_path: Path,
    links_path: Path,
    docs_path: Path,
    fingerprints: tuple[str, str, str],
) -> StemPlan:
    """Convert a legacy table in memory and describe its planned result."""
    legacy_table = _read_table(links_path)
    polygons_table = _read_table(polygons_path)
    if not docs_path.is_file():
        return _blocked_stem(
            stem,
            "legacy schema requires wikipedia/documents/<stem>.parquet",
            fingerprints,
        )
    docs_table = _read_table(docs_path)
    try:
        canonical_rows = _build_canonical_rows(stem, legacy_table, polygons_table, docs_table)
    except Exception as exc:
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


def _classify_stem(stem: str, processed_dir: Path) -> StemPlan:
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
        polygons_table = pq.read_table(  # type: ignore[no-untyped-call]
            polygons_path,
            columns=["polygon_id", "wikidata"],
        )
        links_table = pq.read_table(  # type: ignore[no-untyped-call]
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
    stems_data = tuple(_classify_stem(stem, processed_dir) for stem in sorted(discovered))
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


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StemApplyInputs:
    """Immutable source tables and paths used by one stem transaction."""

    stem_plan: StemPlan
    links_path: Path
    polygons_path: Path
    docs_path: Path
    voyage_documents_path: Path
    voyage_sections_path: Path
    legacy_table: pa.Table
    polygons_table: pa.Table
    docs_table: pa.Table
    data_root: DataRoot


@dataclass(frozen=True, slots=True)
class _StemApplyContext:
    """Derived canonical data and integrity plan for one stem."""

    inputs: _StemApplyInputs
    integrity_plan: IntegrityPlan | None
    canonical_table: pa.Table


def _apply_replacements(
    processed_dir: Path,
    replacements: list[tuple[Path, Path]],
    *,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Apply an already prepared replacement set through the journal."""
    directory = processed_dir / ".link_migration_journal"
    _commit_ordered_replacements(
        directory,
        stem="__replacements__",
        replacements=replacements,
        _crash_hook=_crash_hook,
    )


def _validate_stem_fingerprints(processed_dir: Path, stem_plan: StemPlan) -> None:
    """Reject a plan whose source files changed before the write phase."""
    current = _classify_stem(stem_plan.stem, processed_dir)
    fingerprints = (
        current.polygons_fingerprint,
        current.links_fingerprint,
        current.documents_fingerprint,
    )
    planned = (
        stem_plan.polygons_fingerprint,
        stem_plan.links_fingerprint,
        stem_plan.documents_fingerprint,
    )
    if fingerprints != planned:
        raise RuntimeError(
            f"Link migration stem {stem_plan.stem!r}: source file changed after planning"
        )


def _load_stem_apply_inputs(processed_dir: Path, stem_plan: StemPlan) -> _StemApplyInputs:
    """Load the immutable source tables needed by a stem transaction."""
    links_path = processed_dir / "polygon_articles" / f"{stem_plan.stem}.parquet"
    polygons_path = processed_dir / "polygons" / f"{stem_plan.stem}.parquet"
    docs_path = processed_dir / "wikipedia" / "documents" / f"{stem_plan.stem}.parquet"
    voyage_documents_path = processed_dir / "wikivoyage" / "documents" / f"{stem_plan.stem}.parquet"
    voyage_sections_path = processed_dir / "wikivoyage" / "sections" / f"{stem_plan.stem}.parquet"
    return _StemApplyInputs(
        stem_plan=stem_plan,
        links_path=links_path,
        polygons_path=polygons_path,
        docs_path=docs_path,
        voyage_documents_path=voyage_documents_path,
        voyage_sections_path=voyage_sections_path,
        legacy_table=_read_table(links_path),
        polygons_table=_read_table(polygons_path),
        docs_table=_read_table(docs_path),
        data_root=DataRoot(processed_dir.parent),
    )


def _build_stem_context(inputs: _StemApplyInputs) -> _StemApplyContext:
    """Derive canonical links and the reject-only Wikivoyage plan."""
    wikipedia_rows = _build_canonical_rows(
        inputs.stem_plan.stem,
        inputs.legacy_table,
        inputs.polygons_table,
        inputs.docs_table,
    )
    integrity_plan = (
        plan_integrity_normalization(inputs.data_root, inputs.stem_plan.stem)
        if inputs.voyage_documents_path.is_file()
        else None
    )
    retained_documents = integrity_plan.retained_documents if integrity_plan is not None else []
    voyage_rows = build_polygon_document_links(
        inputs.polygons_table.to_pylist(),
        wikivoyage_documents=retained_documents,
    )
    canonical_rows = validate_polygon_document_links([*wikipedia_rows, *voyage_rows])
    canonical_table = pa.Table.from_pylist(
        canonical_rows,
        schema=polygon_document_link_schema(),
    )
    return _StemApplyContext(
        inputs=inputs,
        integrity_plan=integrity_plan,
        canonical_table=canonical_table,
    )


def _stage_canonical_link(
    staged_dir: Path,
    links_path: Path,
    canonical_table: pa.Table,
) -> Path:
    """Stage the canonical polygon/article parquet artifact."""
    staged_target = staged_dir / links_path.name
    _atomic_write_parquet(staged_target, canonical_table)
    return staged_target


def _source_pbf_for_stem(inputs: _StemApplyInputs) -> str:
    """Return the unique source PBF recorded by a polygon shard."""
    source_pbf_set = {
        str(value) for value in inputs.polygons_table.column("source_pbf").to_pylist() if value
    }
    if len(source_pbf_set) != 1:
        raise RuntimeError(
            f"Link migration stem {inputs.stem_plan.stem!r}: polygons table has "
            f"{len(source_pbf_set)} distinct source_pbf values; expected exactly 1"
        )
    return next(iter(source_pbf_set))


def _load_json_object(path: Path, error: str) -> dict[str, Any]:
    """Load a JSON object, preserving a precise corruption error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{error}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{error} must be a JSON object")
    return dict(payload)


def _load_processed_entries(path: Path) -> dict[str, dict[str, Any]]:
    """Read processed PBF entries, refusing malformed existing state."""
    if not path.is_file():
        return {}
    payload = _load_json_object(path, "processed_pbfs.json")
    return payload


def _new_processed_entry(
    inputs: _StemApplyInputs,
    source_pbf: str,
) -> dict[str, Any]:
    """Reconstruct a missing processed-manifest entry from polygon rows."""
    regions = {str(row.get("region") or "") for row in inputs.polygons_table.to_pylist()}
    if len(regions) != 1 or not next(iter(regions)):
        raise ValueError(
            f"Cannot reconstruct missing manifest entry for {source_pbf!r}: "
            "polygon region is not unique"
        )
    return {
        "source_pbf": source_pbf,
        "region": next(iter(regions)),
        "polygons_path": f"polygons/{inputs.stem_plan.stem}.parquet",
        "wikipedia_documents_path": (f"wikipedia/documents/{inputs.stem_plan.stem}.parquet"),
        "polygon_articles_path": f"polygon_articles/{inputs.stem_plan.stem}.parquet",
        "extraction_version": "link-migration",
        "processed_at": _utc_now_iso(),
    }


def _updated_processed_entry(
    entries: dict[str, dict[str, Any]],
    inputs: _StemApplyInputs,
    source_pbf: str,
    link_count: int,
) -> dict[str, dict[str, Any]]:
    """Merge one link-schema update into processed-manifest entries."""
    existing_entry = entries.get(source_pbf, {})
    if not isinstance(existing_entry, dict):
        raise ValueError(f"processed_pbfs.json entry for {source_pbf!r} must be an object")
    if not existing_entry:
        existing_entry = _new_processed_entry(inputs, source_pbf)
    updated_entry = dict(existing_entry)
    updated_entry["link_schema_version"] = _LINK_CONTRACT_VERSION
    updated_entry["link_count"] = link_count
    entries[source_pbf] = updated_entry
    return entries


def _stage_processed_manifest(
    processed_dir: Path,
    staged_dir: Path,
    context: _StemApplyContext,
) -> Path:
    """Stage the targeted processed-manifest merge."""
    path = processed_dir / "manifests" / "processed_pbfs.json"
    entries = _load_processed_entries(path)
    source_pbf = _source_pbf_for_stem(context.inputs)
    entries = _updated_processed_entry(
        entries,
        context.inputs,
        source_pbf,
        context.canonical_table.num_rows,
    )
    staged_path = staged_dir / path.name
    _atomic_write_json(staged_path, entries)
    return staged_path


def _load_augmentation_manifest(path: Path) -> dict[str, Any]:
    """Load the augmentation manifest or return an empty merge base."""
    if not path.is_file():
        return {}
    return _load_json_object(path, "augmentation_manifest.json")


def _augmentation_entry(
    data_root: DataRoot,
    stem: str,
    previous_entry: dict[str, Any],
    context: _StemApplyContext,
    staged_target: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    """Build the deterministic augmentation-manifest entry for a stem."""
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    wiki_docs_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    counts = dict(previous_entry.get("counts", {}))
    counts["polygon_articles"] = context.canonical_table.num_rows
    if context.integrity_plan is not None:
        counts["wikivoyage_documents"] = len(context.integrity_plan.retained_documents)
        counts["wikivoyage_sections"] = len(context.integrity_plan.retained_sections)
    return {
        **previous_entry,
        "contract_version": _AUGMENTATION_CONTRACT_VERSION,
        "core_hashes": {
            str(polygons_path): _file_content_hash(polygons_path),
            str(wiki_docs_path): _file_content_hash(wiki_docs_path),
        },
        "paths": [str(path.relative_to(processed_dir)) for path in sidecar_paths(data_root, stem)],
        "counts": counts,
        "completed_at": previous_entry.get("completed_at", _utc_now_iso()),
        "link_schema_version": _LINK_CONTRACT_VERSION,
        "link_artifact_sha256": _file_content_hash(staged_target),
    }


def _stage_augmentation_manifest(
    processed_dir: Path,
    staged_dir: Path,
    context: _StemApplyContext,
    staged_target: Path,
) -> Path:
    """Stage the targeted augmentation-manifest merge."""
    path = processed_dir / "augmentation" / "manifests" / "augmentation_manifest.json"
    manifest = _load_augmentation_manifest(path)
    previous = manifest.get(context.inputs.stem_plan.stem)
    previous_entry = dict(previous) if isinstance(previous, dict) else {}
    manifest[context.inputs.stem_plan.stem] = _augmentation_entry(
        context.inputs.data_root,
        context.inputs.stem_plan.stem,
        previous_entry,
        context,
        staged_target,
        processed_dir,
    )
    staged_path = staged_dir / path.name
    _atomic_write_json(staged_path, manifest)
    return staged_path


def _load_pending_publications(path: Path) -> dict[str, Any]:
    """Load or initialize the pending-publications envelope."""
    if not path.is_file():
        return {"contract_version": "pending-publications-v1", "stems": []}
    payload = _load_json_object(path, "pending_migration_publications.json")
    if payload.get("contract_version") != "pending-publications-v1":
        raise ValueError("Unsupported pending-publications contract")
    return payload


def _updated_pending_publications(
    payload: dict[str, Any],
    stem: str,
    link_artifact_sha256: str,
) -> dict[str, Any]:
    """Add a stem and link fingerprint to the pending envelope."""
    pending_stems = set(payload.get("stems", []))
    pending_stems.add(stem)
    marker = payload.get("metadata_refresh", {})
    marker_stems = set(marker.get("stems", [])) if isinstance(marker, dict) else set()
    marker_hashes = dict(marker.get("fingerprint_hashes", {})) if isinstance(marker, dict) else {}
    marker_stems.add(stem)
    marker_hashes[stem] = link_artifact_sha256
    payload["stems"] = sorted(pending_stems)
    payload["metadata_refresh"] = {
        "stems": sorted(marker_stems),
        "fingerprint_hashes": {key: marker_hashes[key] for key in sorted(marker_stems)},
    }
    return payload


def _stage_pending_publication(
    processed_dir: Path,
    staged_dir: Path,
    stem: str,
    link_artifact_sha256: str,
) -> Path:
    """Stage the durable pending-publication envelope update."""
    path = processed_dir / "manifests" / "pending_migration_publications.json"
    payload = _load_pending_publications(path)
    payload = _updated_pending_publications(payload, stem, link_artifact_sha256)
    staged_path = staged_dir / path.name
    _atomic_write_json(staged_path, payload)
    return staged_path


def _rejection_records(
    wiki_rejections: list[dict[str, Any]],
    integrity_plan: IntegrityPlan | None,
) -> list[RejectionRecord]:
    """Convert planned rejection dictionaries into validated records."""
    records = [
        RejectionRecord(
            shard=row["shard"],
            source_table=row["source_table"],
            identifier=row["identifier"],
            wikidata=row["wikidata"],
            expected=row["expected"],
            reason=row["reason"],
            cascaded_sections=row["cascaded_sections"],
        )
        for row in wiki_rejections
    ]
    if integrity_plan is not None:
        records.extend(integrity_plan.rejections)
    return records


def _stage_rejection_ledger(
    processed_dir: Path,
    staged_dir: Path,
    wiki_rejections: list[dict[str, Any]],
    integrity_plan: IntegrityPlan | None,
) -> Path:
    """Stage the cumulative reject-only integrity ledger."""
    path = processed_dir / "integrity" / "rejection_ledger.json"
    existing_records = load_ledger(path) if path.is_file() else []
    merged_records = merge_records(
        [*existing_records, *_rejection_records(wiki_rejections, integrity_plan)]
    )
    staged_path = staged_dir / path.name
    _atomic_write_json(
        staged_path,
        {
            "contract_version": LEDGER_CONTRACT_VERSION,
            "records": [record.to_dict() for record in merged_records],
        },
    )
    return staged_path


def _stage_retained_voyage_table(
    staged_dir: Path,
    target: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Stage one normalized Wikivoyage table using its original schema."""
    staged_path = staged_dir / target.name
    _atomic_write_parquet(
        staged_path,
        pa.Table.from_pylist(rows, schema=pq.read_schema(target)),  # type: ignore[no-untyped-call]
    )
    return staged_path


def _stage_voyage_replacements(
    staged_dir: Path,
    context: _StemApplyContext,
) -> list[tuple[Path, Path]]:
    """Stage only Wikivoyage tables whose row counts changed."""
    integrity_plan = context.integrity_plan
    if integrity_plan is None:
        return []
    inputs = context.inputs
    replacements: list[tuple[Path, Path]] = []
    if (
        len(integrity_plan.retained_documents)
        != pq.read_metadata(inputs.voyage_documents_path).num_rows  # type: ignore[no-untyped-call]
    ):
        staged = _stage_retained_voyage_table(
            staged_dir,
            inputs.voyage_documents_path,
            integrity_plan.retained_documents,
        )
        replacements.append((inputs.voyage_documents_path, staged))
    if inputs.voyage_sections_path.is_file() and (
        len(integrity_plan.retained_sections)
        != pq.read_metadata(inputs.voyage_sections_path).num_rows  # type: ignore[no-untyped-call]
    ):
        staged = _stage_retained_voyage_table(
            staged_dir,
            inputs.voyage_sections_path,
            integrity_plan.retained_sections,
        )
        replacements.append((inputs.voyage_sections_path, staged))
    return replacements


def _stage_stem_replacements(
    processed_dir: Path,
    context: _StemApplyContext,
    wiki_rejections: list[dict[str, Any]],
) -> list[tuple[Path, Path]]:
    """Stage every artifact for one ordered stem transaction."""
    inputs = context.inputs
    staged_dir = processed_dir / ".link_migration_staging" / inputs.stem_plan.stem
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_target = _stage_canonical_link(staged_dir, inputs.links_path, context.canonical_table)
    staged_manifest = _stage_processed_manifest(processed_dir, staged_dir, context)
    staged_augmentation = _stage_augmentation_manifest(
        processed_dir,
        staged_dir,
        context,
        staged_target,
    )
    link_artifact_sha256 = _file_content_hash(staged_target)
    staged_pending = _stage_pending_publication(
        processed_dir,
        staged_dir,
        inputs.stem_plan.stem,
        link_artifact_sha256,
    )
    staged_ledger = _stage_rejection_ledger(
        processed_dir,
        staged_dir,
        wiki_rejections,
        context.integrity_plan,
    )
    replacements = [(inputs.links_path, staged_target)]
    replacements.extend(_stage_voyage_replacements(staged_dir, context))
    replacements.extend(
        [
            (processed_dir / "integrity" / "rejection_ledger.json", staged_ledger),
            (processed_dir / "manifests" / "processed_pbfs.json", staged_manifest),
            (
                processed_dir / "augmentation" / "manifests" / "augmentation_manifest.json",
                staged_augmentation,
            ),
            (
                processed_dir / "manifests" / "pending_migration_publications.json",
                staged_pending,
            ),
        ]
    )
    return replacements


def _apply_migratable_stem(
    processed_dir: Path,
    stem_plan: StemPlan,
    wiki_rejections: list[dict[str, Any]],
    *,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Stage and commit one unchanged-source stem transaction."""
    _validate_stem_fingerprints(processed_dir, stem_plan)
    inputs = _load_stem_apply_inputs(processed_dir, stem_plan)
    context = _build_stem_context(inputs)
    replacements = _stage_stem_replacements(processed_dir, context, wiki_rejections)
    journal_dir = processed_dir / ".link_migration_journal" / stem_plan.stem
    _commit_ordered_replacements(
        journal_dir,
        stem=stem_plan.stem,
        replacements=replacements,
        _crash_hook=_crash_hook,
    )
    staged_dir = processed_dir / ".link_migration_staging" / stem_plan.stem
    with suppress(OSError):
        staged_dir.rmdir()


def _ensure_migration_plan_safe(plan: MigrationPlan) -> None:
    """Raise when any discovered stem is unsafe to migrate."""
    if plan.is_safe_to_apply:
        return
    blocked = [s.stem for s in plan.stems if s.classification == StemClassification.BLOCKED]
    raise ValueError(f"Link migration plan contains blocked stems: {blocked}")


def _wiki_rejections_by_stem(plan: MigrationPlan) -> dict[str, list[dict[str, Any]]]:
    """Compute legacy rejection records before canonical replacement."""
    return {
        stem_plan.stem: plan_link_migration_normalization_rejections_for_stem(
            plan.processed_dir,
            stem_plan,
        )
        for stem_plan in plan.stems
        if stem_plan.classification == StemClassification.MIGRATABLE
    }


def _apply_migratable_stems(
    processed_dir: Path,
    plan: MigrationPlan,
    wiki_rejections: dict[str, list[dict[str, Any]]],
    *,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Apply every migratable stem while skipping canonical stems."""
    for stem_plan in plan.stems:
        if stem_plan.classification != StemClassification.MIGRATABLE:
            continue
        _apply_migratable_stem(
            processed_dir,
            stem_plan,
            wiki_rejections.get(stem_plan.stem, []),
            _crash_hook=_crash_hook,
        )


def apply_link_migration(
    processed_dir: Path,
    *,
    stems: set[str] | None = None,
    replacements: list[tuple[Path, Path]] | None = None,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Apply stage.

    * When ``replacements`` is ``None``, plans the migration and applies
      every legacy stem atomically. Each stem gets its own journaled
      transaction.
    * When ``replacements`` is supplied, runs the same ordered journaled
      transaction directly (used by tests via the public boundary).
    """
    if replacements is not None:
        _apply_replacements(processed_dir, replacements, _crash_hook=_crash_hook)
        return

    plan = plan_link_migration(processed_dir, stems=stems)
    _ensure_migration_plan_safe(plan)
    _apply_migratable_stems(
        processed_dir,
        plan,
        _wiki_rejections_by_stem(plan),
        _crash_hook=_crash_hook,
    )


__all__ = [
    "MigrationPlan",
    "StemClassification",
    "StemPlan",
    "apply_link_migration",
    "classify_stem_schema",
    "plan_link_migration",
]
