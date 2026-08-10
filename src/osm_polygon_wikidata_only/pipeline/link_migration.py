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
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.orchestrator import sidecar_paths
from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
    LEDGER_CONTRACT_VERSION,
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


# ---------------------------------------------------------------------------
# Per-stem classification
# ---------------------------------------------------------------------------


def _classify_stem(stem: str, processed_dir: Path) -> StemPlan:
    polygons_path = processed_dir / "polygons" / f"{stem}.parquet"
    links_path = processed_dir / "polygon_articles" / f"{stem}.parquet"
    docs_path = processed_dir / "wikipedia" / "documents" / f"{stem}.parquet"

    polygons_hash = _file_content_hash(polygons_path)
    links_hash = _file_content_hash(links_path)
    documents_hash = _file_content_hash(docs_path)

    if not polygons_path.is_file():
        return StemPlan(
            stem=stem,
            classification=StemClassification.BLOCKED,
            reason="polygons file missing",
            polygons_fingerprint=polygons_hash,
            links_fingerprint=links_hash,
            documents_fingerprint=documents_hash,
            row_count=0,
            canonical_digest=None,
        )

    if not links_path.is_file():
        return StemPlan(
            stem=stem,
            classification=StemClassification.BLOCKED,
            reason="polygon_articles file missing",
            polygons_fingerprint=polygons_hash,
            links_fingerprint=links_hash,
            documents_fingerprint=documents_hash,
            row_count=0,
            canonical_digest=None,
        )

    columns = _table_columns(links_path)
    if columns is None:
        return StemPlan(
            stem=stem,
            classification=StemClassification.BLOCKED,
            reason="polygon_articles file unreadable",
            polygons_fingerprint=polygons_hash,
            links_fingerprint=links_hash,
            documents_fingerprint=documents_hash,
            row_count=0,
            canonical_digest=None,
        )

    try:
        classification = classify_stem_schema(columns)
    except ValueError as exc:
        return StemPlan(
            stem=stem,
            classification=StemClassification.BLOCKED,
            reason=f"unrecognised schema: {exc}",
            polygons_fingerprint=polygons_hash,
            links_fingerprint=links_hash,
            documents_fingerprint=documents_hash,
            row_count=0,
            canonical_digest=None,
        )
    if classification == StemClassification.CANONICAL.value:
        # Even when the column names match, a lookalike schema (wrong
        # types, extra metadata, metadata-less, etc.) is BLOCKED.
        links_table = _read_table(links_path)
        if not is_canonical_table_schema(links_table):
            return StemPlan(
                stem=stem,
                classification=StemClassification.BLOCKED,
                reason="link table columns match canonical but schema differs (types or metadata)",
                polygons_fingerprint=polygons_hash,
                links_fingerprint=links_hash,
                documents_fingerprint=documents_hash,
                row_count=0,
                canonical_digest=None,
            )
        return StemPlan(
            stem=stem,
            classification=StemClassification.CANONICAL,
            reason="",
            polygons_fingerprint=polygons_hash,
            links_fingerprint=links_hash,
            documents_fingerprint=documents_hash,
            row_count=links_table.num_rows,
            canonical_digest=_file_content_hash(links_path),
        )

    # classification == "legacy" (MIGRATABLE)
    legacy_table = _read_table(links_path)
    polygons_table = _read_table(polygons_path)
    if not docs_path.is_file():
        return StemPlan(
            stem=stem,
            classification=StemClassification.BLOCKED,
            reason="legacy schema requires wikipedia/documents/<stem>.parquet",
            polygons_fingerprint=polygons_hash,
            links_fingerprint=links_hash,
            documents_fingerprint="",
            row_count=0,
            canonical_digest=None,
        )
    docs_table = _read_table(docs_path)

    try:
        canonical_rows = _build_canonical_rows(stem, legacy_table, polygons_table, docs_table)
    except Exception as exc:
        return StemPlan(
            stem=stem,
            classification=StemClassification.BLOCKED,
            reason=f"legacy conversion failed: {exc}",
            polygons_fingerprint=polygons_hash,
            links_fingerprint=links_hash,
            documents_fingerprint=documents_hash,
            row_count=0,
            canonical_digest=None,
        )

    canonical_table = pa.Table.from_pylist(canonical_rows, schema=polygon_document_link_schema())
    return StemPlan(
        stem=stem,
        classification=StemClassification.MIGRATABLE,
        reason="",
        polygons_fingerprint=polygons_hash,
        links_fingerprint=links_hash,
        documents_fingerprint=documents_hash,
        row_count=canonical_table.num_rows,
        canonical_digest=_table_digest(canonical_table),
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
    polygons_path = processed_dir / "polygons" / f"{sp.stem}.parquet"
    links_path = processed_dir / "polygon_articles" / f"{sp.stem}.parquet"
    if not polygons_path.is_file() or not links_path.is_file():
        return []
    # Defensive: only read columns that exist in the LEGACY schema.
    # The apply stage overwrites polygon_articles with the canonical
    # schema, so callers MUST invoke this BEFORE the apply.
    try:
        polygons_table = pq.read_table(  # type: ignore[no-untyped-call]
            polygons_path, columns=["polygon_id", "wikidata"]
        )
        links_table = pq.read_table(  # type: ignore[no-untyped-call]
            links_path, columns=["polygon_id", "wikidata", "article_id"]
        )
    except (KeyError, pa.ArrowInvalid):
        # The polygon_articles file already has the canonical schema
        # (apply has run). No additional rejections to record.
        return []
    polygon_qids: dict[str, set[str]] = {}
    for row in polygons_table.to_pylist():
        qids = _qids_from_osm_tag(str(row.get("wikidata", "")))
        polygon_qids.setdefault(str(row["polygon_id"]), set()).update(qids)
    rejections: list[dict[str, Any]] = []
    for row in links_table.to_pylist():
        polygon_id = str(row["polygon_id"])
        row_qid = str(row["wikidata"])
        polygon_set = polygon_qids.get(polygon_id, set())
        if row_qid and row_qid not in polygon_set:
            rejections.append(
                {
                    "shard": sp.stem,
                    "source_table": "polygon_articles",
                    "identifier": str(row["article_id"]),
                    "wikidata": row_qid,
                    "expected": None,
                    "reason": "wikidata_not_in_polygon_qids",
                    "cascaded_sections": 0,
                }
            )
    rejections.sort(key=lambda r: (r["identifier"], r["wikidata"]))
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
    if stems is not None:
        for stem in stems:
            if not _is_valid_stem(stem):
                raise ValueError(f"Invalid stem name: {stem!r}")

    discovered: set[str] = set()
    for sub in ("polygons", "polygon_articles", "wikipedia/documents"):
        sub_path = processed_dir / sub
        if sub_path.is_dir():
            discovered.update(p.stem for p in sub_path.glob("*.parquet"))

    if stems is not None:
        discovered = {s for s in discovered if s in stems}

    stems_data: list[StemPlan] = []
    for stem in sorted(discovered):
        stems_data.append(_classify_stem(stem, processed_dir))

    return MigrationPlan(processed_dir=processed_dir, stems=tuple(stems_data))


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


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
        directory = processed_dir / ".link_migration_journal"
        _commit_ordered_replacements(
            directory,
            stem="__replacements__",
            replacements=replacements,
            _crash_hook=_crash_hook,
        )
        return

    plan = plan_link_migration(processed_dir, stems=stems)
    if not plan.is_safe_to_apply:
        blocked = [s.stem for s in plan.stems if s.classification == StemClassification.BLOCKED]
        raise ValueError(f"Link migration plan contains blocked stems: {blocked}")

    # Compute Wikipedia rejection records BEFORE the apply overwrites
    # the legacy polygon_articles schema with the canonical schema.
    wiki_rejections_by_stem = {
        sp.stem: plan_link_migration_normalization_rejections_for_stem(plan.processed_dir, sp)
        for sp in plan.stems
        if sp.classification == StemClassification.MIGRATABLE
    }

    for sp in plan.stems:
        if sp.classification == StemClassification.CANONICAL:
            continue
        if sp.classification != StemClassification.MIGRATABLE:
            continue

        # Re-validate: do not trust the plan; require an un-tampered
        # source set immediately before writes.
        current = _classify_stem(sp.stem, processed_dir)
        if (
            current.polygons_fingerprint != sp.polygons_fingerprint
            or current.links_fingerprint != sp.links_fingerprint
            or current.documents_fingerprint != sp.documents_fingerprint
        ):
            raise RuntimeError(
                f"Link migration stem {sp.stem!r}: source file changed after planning"
            )

        # Rebuild canonical rows (pure) and stage them beside the
        # canonical target.
        links_path = processed_dir / "polygon_articles" / f"{sp.stem}.parquet"
        polygons_path = processed_dir / "polygons" / f"{sp.stem}.parquet"
        docs_path = processed_dir / "wikipedia" / "documents" / f"{sp.stem}.parquet"
        legacy_table = _read_table(links_path)
        polygons_table = _read_table(polygons_path)
        docs_table = _read_table(docs_path)
        wikipedia_rows = _build_canonical_rows(sp.stem, legacy_table, polygons_table, docs_table)

        # Normalize existing Wikivoyage sidecars before deriving their
        # links. This is network-free and reject-only: invalid
        # relationships are omitted and recorded, never rewritten.
        data_root = DataRoot(processed_dir.parent)
        voyage_documents_path = processed_dir / "wikivoyage" / "documents" / f"{sp.stem}.parquet"
        voyage_sections_path = processed_dir / "wikivoyage" / "sections" / f"{sp.stem}.parquet"
        integrity_plan = (
            plan_integrity_normalization(data_root, sp.stem)
            if voyage_documents_path.is_file()
            else None
        )
        retained_voyage_documents = (
            integrity_plan.retained_documents if integrity_plan is not None else []
        )
        voyage_rows = build_polygon_document_links(
            polygons_table.to_pylist(),
            wikivoyage_documents=retained_voyage_documents,
        )
        canonical_rows = validate_polygon_document_links([*wikipedia_rows, *voyage_rows])
        canonical_table = pa.Table.from_pylist(
            canonical_rows, schema=polygon_document_link_schema()
        )

        # Stage the canonical parquet and the processed-manifest
        # update side by side. The migration no longer writes a
        # secondary ``manifests/link_manifest.json`` -- the approved
        # design updates the EXISTING ``processed_pbfs.json`` in
        # place, keyed by ``source_pbf``.
        staged_dir = processed_dir / ".link_migration_staging" / sp.stem
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_target = staged_dir / links_path.name
        _atomic_write_parquet(staged_target, canonical_table)

        # The source PBF for this stem comes from the polygons table.
        # Every polygon row in this shard shares the same source_pbf.
        source_pbf_set = set()
        for row in polygons_table.column("source_pbf").to_pylist():
            if row:
                source_pbf_set.add(str(row))
        if len(source_pbf_set) != 1:
            raise RuntimeError(
                f"Link migration stem {sp.stem!r}: polygons table has "
                f"{len(source_pbf_set)} distinct source_pbf values; expected exactly 1"
            )
        source_pbf = next(iter(source_pbf_set))

        # Build the staged processed manifest update. Merge with the
        # existing processed_pbfs.json so unrelated stems/entries are
        # preserved. A MALFORMED existing manifest is treated as a
        # fatal error -- never silently replace corruption.
        processed_manifest_path = processed_dir / "manifests" / "processed_pbfs.json"
        merged_entries: dict[str, dict[str, Any]] = {}
        if processed_manifest_path.is_file():
            try:
                existing_payload = json.loads(processed_manifest_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed processed_pbfs.json -- refusing to migrate: {exc}"
                ) from exc
            if not isinstance(existing_payload, dict):
                raise ValueError("processed_pbfs.json is not a JSON object -- refusing to migrate")
            merged_entries = dict(existing_payload)
        existing_entry = merged_entries.get(source_pbf, {})
        if not isinstance(existing_entry, dict):
            raise ValueError(f"processed_pbfs.json entry for {source_pbf!r} must be an object")
        if not existing_entry:
            polygon_rows = polygons_table.to_pylist()
            regions = {str(row.get("region") or "") for row in polygon_rows}
            if len(regions) != 1 or not next(iter(regions)):
                raise ValueError(
                    f"Cannot reconstruct missing manifest entry for {source_pbf!r}: "
                    "polygon region is not unique"
                )
            existing_entry = {
                "source_pbf": source_pbf,
                "region": next(iter(regions)),
                "polygons_path": f"polygons/{sp.stem}.parquet",
                "wikipedia_documents_path": (f"wikipedia/documents/{sp.stem}.parquet"),
                "polygon_articles_path": f"polygon_articles/{sp.stem}.parquet",
                "extraction_version": "link-migration",
                "processed_at": _utc_now_iso(),
            }
        updated_entry = dict(existing_entry)
        updated_entry["link_schema_version"] = _LINK_CONTRACT_VERSION
        updated_entry["link_count"] = canonical_table.num_rows
        merged_entries[source_pbf] = updated_entry
        staged_manifest = staged_dir / processed_manifest_path.name
        _atomic_write_json(
            staged_manifest,
            merged_entries,
        )

        # Stage the augmentation manifest by targeted merge. It is a
        # transaction replacement, not an after-step.
        polygons_path = data_root.processed_polygons / f"{sp.stem}.parquet"
        wiki_docs_path = data_root.processed / "wikipedia" / "documents" / f"{sp.stem}.parquet"
        core_hashes = {
            str(polygons_path): _file_content_hash(polygons_path),
            str(wiki_docs_path): _file_content_hash(wiki_docs_path),
        }
        link_artifact_sha256 = _file_content_hash(staged_target)
        augmentation_manifest_path = (
            processed_dir / "augmentation" / "manifests" / "augmentation_manifest.json"
        )
        augmentation_manifest: dict[str, Any] = {}
        if augmentation_manifest_path.is_file():
            raw_augmentation = json.loads(augmentation_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw_augmentation, dict):
                raise ValueError("augmentation_manifest.json must be a JSON object")
            augmentation_manifest = dict(raw_augmentation)
        previous_augmentation = augmentation_manifest.get(sp.stem)
        previous_entry = (
            dict(previous_augmentation) if isinstance(previous_augmentation, dict) else {}
        )
        counts = dict(previous_entry.get("counts", {}))
        counts["polygon_articles"] = canonical_table.num_rows
        if integrity_plan is not None:
            counts["wikivoyage_documents"] = len(integrity_plan.retained_documents)
            counts["wikivoyage_sections"] = len(integrity_plan.retained_sections)
        augmentation_entry = {
            **previous_entry,
            "contract_version": _AUGMENTATION_CONTRACT_VERSION,
            "core_hashes": core_hashes,
            "paths": [
                str(path.relative_to(processed_dir)) for path in sidecar_paths(data_root, sp.stem)
            ],
            "counts": counts,
            "completed_at": previous_entry.get("completed_at", _utc_now_iso()),
            "link_schema_version": _LINK_CONTRACT_VERSION,
            "link_artifact_sha256": link_artifact_sha256,
        }
        augmentation_manifest[sp.stem] = augmentation_entry
        staged_augmentation_manifest = staged_dir / "augmentation_manifest.json"
        _atomic_write_json(staged_augmentation_manifest, augmentation_manifest)

        # Stage the durable publication envelope, preserving unrelated
        # fields and accumulating the repository-level metadata marker.
        pending_path = processed_dir / "manifests" / "pending_migration_publications.json"
        pending_payload: dict[str, Any]
        if pending_path.is_file():
            raw_pending = json.loads(pending_path.read_text(encoding="utf-8"))
            if not isinstance(raw_pending, dict):
                raise ValueError("pending_migration_publications.json must be a JSON object")
            pending_payload = dict(raw_pending)
        else:
            pending_payload = {
                "contract_version": "pending-publications-v1",
                "stems": [],
            }
        if pending_payload.get("contract_version") != "pending-publications-v1":
            raise ValueError("Unsupported pending-publications contract")
        pending_stems = set(pending_payload.get("stems", []))
        pending_stems.add(sp.stem)
        marker = pending_payload.get("metadata_refresh", {})
        marker_stems = set(marker.get("stems", [])) if isinstance(marker, dict) else set()
        marker_hashes = (
            dict(marker.get("fingerprint_hashes", {})) if isinstance(marker, dict) else {}
        )
        marker_stems.add(sp.stem)
        marker_hashes[sp.stem] = link_artifact_sha256
        pending_payload["stems"] = sorted(pending_stems)
        pending_payload["metadata_refresh"] = {
            "stems": sorted(marker_stems),
            "fingerprint_hashes": {stem: marker_hashes[stem] for stem in sorted(marker_stems)},
        }
        staged_pending = staged_dir / pending_path.name
        _atomic_write_json(staged_pending, pending_payload)

        # Stage the cumulative rejection ledger. Both invalid legacy
        # Wikipedia relationships and invalid Wikivoyage documents are
        # retained as durable evidence.
        wiki_rejections = wiki_rejections_by_stem.get(sp.stem, [])
        new_records = [
            RejectionRecord(
                shard=r["shard"],
                source_table=r["source_table"],
                identifier=r["identifier"],
                wikidata=r["wikidata"],
                expected=r["expected"],
                reason=r["reason"],
                cascaded_sections=r["cascaded_sections"],
            )
            for r in wiki_rejections
        ]
        if integrity_plan is not None:
            new_records.extend(integrity_plan.rejections)
        ledger_path = processed_dir / "integrity" / "rejection_ledger.json"
        existing_records = load_ledger(ledger_path) if ledger_path.is_file() else []
        merged_records = merge_records([*existing_records, *new_records])
        staged_ledger = staged_dir / ledger_path.name
        _atomic_write_json(
            staged_ledger,
            {
                "contract_version": LEDGER_CONTRACT_VERSION,
                "records": [record.to_dict() for record in merged_records],
            },
        )

        replacements = [(links_path, staged_target)]
        if (
            integrity_plan is not None
            and len(integrity_plan.retained_documents)
            != pq.read_metadata(voyage_documents_path).num_rows  # type: ignore[no-untyped-call]
        ):
            staged_voyage_documents = staged_dir / "wikivoyage_documents.parquet"
            _atomic_write_parquet(
                staged_voyage_documents,
                pa.Table.from_pylist(
                    integrity_plan.retained_documents,
                    schema=pq.read_schema(voyage_documents_path),  # type: ignore[no-untyped-call]
                ),
            )
            replacements.append((voyage_documents_path, staged_voyage_documents))
        if (
            integrity_plan is not None
            and voyage_sections_path.is_file()
            and len(integrity_plan.retained_sections)
            != pq.read_metadata(voyage_sections_path).num_rows  # type: ignore[no-untyped-call]
        ):
            staged_voyage_sections = staged_dir / "wikivoyage_sections.parquet"
            _atomic_write_parquet(
                staged_voyage_sections,
                pa.Table.from_pylist(
                    integrity_plan.retained_sections,
                    schema=pq.read_schema(voyage_sections_path),  # type: ignore[no-untyped-call]
                ),
            )
            replacements.append((voyage_sections_path, staged_voyage_sections))
        replacements.extend(
            [
                (ledger_path, staged_ledger),
                (processed_manifest_path, staged_manifest),
                (augmentation_manifest_path, staged_augmentation_manifest),
                (pending_path, staged_pending),
            ]
        )
        journal_dir = processed_dir / ".link_migration_journal" / sp.stem
        _commit_ordered_replacements(
            journal_dir,
            stem=sp.stem,
            replacements=replacements,
            _crash_hook=_crash_hook,
        )

        # Remove the staging directory (the staged files have moved
        # into place via os.replace/shutil.move).
        with suppress(OSError):
            staged_dir.rmdir()


__all__ = [
    "MigrationPlan",
    "StemClassification",
    "StemPlan",
    "apply_link_migration",
    "classify_stem_schema",
    "plan_link_migration",
]
