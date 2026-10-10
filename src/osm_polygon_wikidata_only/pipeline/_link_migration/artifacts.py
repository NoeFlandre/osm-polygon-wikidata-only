"""Stage link artifacts and manifests for one migration transaction."""

from __future__ import annotations

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
)
from osm_polygon_wikidata_only.augmentation.steps import (
    CONTRACT_VERSION as _AUGMENTATION_CONTRACT_VERSION,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    LINK_CONTRACT_VERSION as _LINK_CONTRACT_VERSION,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_parquet
from osm_polygon_wikidata_only.io.json_files import read_json
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    StemApplyContext,
    StemApplyInputs,
)
from osm_polygon_wikidata_only.pipeline._link_migration.transaction import (
    atomic_write_journal_json,
    file_content_hash,
)
from osm_polygon_wikidata_only.utils.time import utc_now_iso


def _stage_canonical_link(
    processed_dir: Path,
    staged_dir: Path,
    links_path: Path,
    canonical_table: pa.Table,
) -> Path:
    """Stage the canonical polygon/article parquet artifact."""
    staged_target = _staged_artifact_path(processed_dir, staged_dir, links_path)
    atomic_write_parquet(staged_target, canonical_table)
    return staged_target


def _staged_artifact_path(processed_dir: Path, staged_dir: Path, target: Path) -> Path:
    """Mirror the processed-tree path so equal basenames cannot collide."""
    return staged_dir / target.relative_to(processed_dir)


def source_pbf_for_stem(inputs: StemApplyInputs) -> str:
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
    payload = read_json(path, on_malformed=lambda exc: ValueError(f"{error}: {exc}"))
    if not isinstance(payload, dict):
        raise ValueError(f"{error} must be a JSON object")
    return dict(payload)


def _load_processed_entries(path: Path) -> dict[str, dict[str, Any]]:
    """Read processed PBF entries, refusing malformed existing state."""
    if not path.is_file():
        return {}
    return _load_json_object(path, "processed_pbfs.json")


def new_processed_entry(
    inputs: StemApplyInputs,
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
        "processed_at": utc_now_iso(),
    }


def _updated_processed_entry(
    entries: dict[str, dict[str, Any]],
    inputs: StemApplyInputs,
    source_pbf: str,
    link_count: int,
) -> dict[str, dict[str, Any]]:
    """Merge one link-schema update into processed-manifest entries."""
    existing_entry = entries.get(source_pbf, {})
    if not isinstance(existing_entry, dict):
        raise ValueError(f"processed_pbfs.json entry for {source_pbf!r} must be an object")
    if not existing_entry:
        existing_entry = new_processed_entry(inputs, source_pbf)
    updated_entry = dict(existing_entry)
    updated_entry["link_schema_version"] = _LINK_CONTRACT_VERSION
    updated_entry["link_count"] = link_count
    entries[source_pbf] = updated_entry
    return entries


def _stage_processed_manifest(
    processed_dir: Path,
    staged_dir: Path,
    context: StemApplyContext,
) -> Path:
    """Stage the targeted processed-manifest merge."""
    path = processed_dir / "manifests" / "processed_pbfs.json"
    entries = _load_processed_entries(path)
    source_pbf = source_pbf_for_stem(context.inputs)
    entries = _updated_processed_entry(
        entries,
        context.inputs,
        source_pbf,
        context.canonical_table.num_rows,
    )
    staged_path = _staged_artifact_path(processed_dir, staged_dir, path)
    atomic_write_journal_json(staged_path, entries)
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
    context: StemApplyContext,
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
            str(polygons_path): file_content_hash(polygons_path),
            str(wiki_docs_path): file_content_hash(wiki_docs_path),
        },
        "paths": [str(path.relative_to(processed_dir)) for path in sidecar_paths(data_root, stem)],
        "counts": counts,
        "completed_at": previous_entry.get("completed_at", utc_now_iso()),
        "link_schema_version": _LINK_CONTRACT_VERSION,
        "link_artifact_sha256": file_content_hash(staged_target),
    }


def _stage_augmentation_manifest(
    processed_dir: Path,
    staged_dir: Path,
    context: StemApplyContext,
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
    staged_path = _staged_artifact_path(processed_dir, staged_dir, path)
    atomic_write_journal_json(staged_path, manifest)
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
    staged_path = _staged_artifact_path(processed_dir, staged_dir, path)
    atomic_write_journal_json(staged_path, payload)
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
    staged_path = _staged_artifact_path(processed_dir, staged_dir, path)
    atomic_write_journal_json(
        staged_path,
        {
            "contract_version": LEDGER_CONTRACT_VERSION,
            "records": [record.to_dict() for record in merged_records],
        },
    )
    return staged_path


def _stage_retained_voyage_table(
    processed_dir: Path,
    staged_dir: Path,
    target: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Stage one normalized Wikivoyage table using its original schema."""
    staged_path = _staged_artifact_path(processed_dir, staged_dir, target)
    atomic_write_parquet(
        staged_path,
        pa.Table.from_pylist(rows, schema=pq.read_schema(target)),
    )
    return staged_path


def _stage_voyage_replacements(
    processed_dir: Path,
    staged_dir: Path,
    context: StemApplyContext,
) -> list[tuple[Path, Path]]:
    """Stage only Wikivoyage tables whose row counts changed."""
    integrity_plan = context.integrity_plan
    if integrity_plan is None:
        return []
    inputs = context.inputs
    replacements: list[tuple[Path, Path]] = []
    if (
        len(integrity_plan.retained_documents)
        != pq.read_metadata(inputs.voyage_documents_path).num_rows
    ):
        staged = _stage_retained_voyage_table(
            processed_dir,
            staged_dir,
            inputs.voyage_documents_path,
            integrity_plan.retained_documents,
        )
        replacements.append((inputs.voyage_documents_path, staged))
    if inputs.voyage_sections_path.is_file() and (
        len(integrity_plan.retained_sections)
        != pq.read_metadata(inputs.voyage_sections_path).num_rows
    ):
        staged = _stage_retained_voyage_table(
            processed_dir,
            staged_dir,
            inputs.voyage_sections_path,
            integrity_plan.retained_sections,
        )
        replacements.append((inputs.voyage_sections_path, staged))
    return replacements


def stage_stem_replacements(
    processed_dir: Path,
    context: StemApplyContext,
    wiki_rejections: list[dict[str, Any]],
) -> list[tuple[Path, Path]]:
    """Stage every artifact for one ordered stem transaction."""
    inputs = context.inputs
    staged_dir = processed_dir / ".link_migration_staging" / inputs.stem_plan.stem
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_target = _stage_canonical_link(
        processed_dir, staged_dir, inputs.links_path, context.canonical_table
    )
    staged_manifest = _stage_processed_manifest(processed_dir, staged_dir, context)
    staged_augmentation = _stage_augmentation_manifest(
        processed_dir,
        staged_dir,
        context,
        staged_target,
    )
    link_artifact_sha256 = file_content_hash(staged_target)
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
    replacements.extend(_stage_voyage_replacements(processed_dir, staged_dir, context))
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


__all__ = ["new_processed_entry", "source_pbf_for_stem", "stage_stem_replacements"]
