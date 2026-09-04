"""Fail-closed audit and staging for whole-file containment retirement."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_replacement, atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps

from .containment_policy import (
    CONTAINMENT_RULES,
    TABLE_CONTRACTS,
    ContainmentRule,
    TableContract,
    validate_stem,
)


@dataclass(frozen=True, slots=True)
class TableAudit:
    subdir: str
    child_rows: int
    missing_from_parent: int
    parent_duplicate_identities: int
    child_duplicate_identities: int


@dataclass(frozen=True, slots=True)
class ChildAudit:
    stem: str
    tables: tuple[TableAudit, ...]


@dataclass(frozen=True, slots=True)
class RuleAudit:
    parent: str
    children: tuple[ChildAudit, ...]
    blockers: tuple[str, ...]

    @property
    def safe_to_stage(self) -> bool:
        return not self.blockers


@dataclass(frozen=True, slots=True)
class StagedRule:
    parent: str
    children: tuple[str, ...]
    artifacts: tuple[tuple[str, Path], ...]

    def artifact(self, subdir: str) -> Path:
        """Return a staged artifact path by canonical sub-directory."""
        for candidate, path in self.artifacts:
            if candidate == subdir:
                return path
        raise KeyError(subdir)


@dataclass(frozen=True, slots=True)
class PreparedRule:
    parent: str
    children: tuple[str, ...]


RETIREMENT_FILENAME = "containment_retirements.json"
RETIREMENT_CONTRACT_VERSION = "contained-region-v1"


def _identity_set(path: Path, contract: TableContract) -> tuple[set[tuple[Any, ...]], int]:
    table = pq.read_table(path, columns=list(contract.identity_columns))  # type: ignore[no-untyped-call]
    rows = table.to_pylist()
    identities = {tuple(row[column] for column in contract.identity_columns) for row in rows}
    return identities, len(rows) - len(identities)


def _contract_paths(
    processed_dir: Path,
    contract: TableContract,
    parent: str,
    child: str,
) -> tuple[Path, Path]:
    """Return live parent and child paths for one table contract."""
    return (
        processed_dir / contract.subdir / f"{parent}.parquet",
        processed_dir / contract.subdir / f"{child}.parquet",
    )


def _duplicate_blockers(
    child: str,
    subdir: str,
    parent_duplicates: int,
    child_duplicates: int,
) -> list[str]:
    """Describe duplicate identity blockers for one audited table."""
    blockers: list[str] = []
    if parent_duplicates:
        blockers.append(f"{child}: {subdir} parent has {parent_duplicates} duplicate identities")
    if child_duplicates:
        blockers.append(f"{child}: {subdir} child has {child_duplicates} duplicate identities")
    return blockers


def _audit_present_contract(
    processed_dir: Path,
    contract: TableContract,
    parent: str,
    child: str,
    parent_path: Path,
    child_path: Path,
) -> tuple[TableAudit, list[str]]:
    """Audit one contract whose parent and child files are present."""
    try:
        parent_schema = pq.read_schema(parent_path)  # type: ignore[no-untyped-call]
        child_schema = pq.read_schema(child_path)  # type: ignore[no-untyped-call]
        if not parent_schema.equals(child_schema, check_metadata=True):
            return (
                TableAudit(contract.subdir, 0, 0, 0, 0),
                [f"{child}: schema mismatch for {contract.subdir}"],
            )
        parent_ids, parent_duplicates = _identity_set(parent_path, contract)
        child_ids, child_duplicates = _identity_set(child_path, contract)
    except Exception as error:
        return (
            TableAudit(contract.subdir, 0, 0, 0, 0),
            [f"{child}: unreadable {contract.subdir}: {type(error).__name__}"],
        )
    audit = TableAudit(
        contract.subdir,
        len(child_ids),
        len(child_ids - parent_ids),
        parent_duplicates,
        child_duplicates,
    )
    return audit, _duplicate_blockers(
        child,
        contract.subdir,
        parent_duplicates,
        child_duplicates,
    )


def _audit_contract(
    processed_dir: Path,
    contract: TableContract,
    parent: str,
    child: str,
) -> tuple[TableAudit, list[str]]:
    """Audit one table contract and return its findings and blockers."""
    parent_path, child_path = _contract_paths(processed_dir, contract, parent, child)
    missing_paths = [path for path in (parent_path, child_path) if not path.is_file()]
    if missing_paths:
        blockers = [
            f"{child}: missing file {path.relative_to(processed_dir)}" for path in missing_paths
        ]
        return TableAudit(contract.subdir, 0, 0, 0, 0), blockers
    return _audit_present_contract(
        processed_dir,
        contract,
        parent,
        child,
        parent_path,
        child_path,
    )


def _audit_child(
    processed_dir: Path,
    parent: str,
    child: str,
) -> tuple[ChildAudit, list[str]]:
    """Audit every supported table for one child."""
    table_audits: list[TableAudit] = []
    blockers: list[str] = []
    for contract in TABLE_CONTRACTS:
        table_audit, contract_blockers = _audit_contract(
            processed_dir,
            contract,
            parent,
            child,
        )
        table_audits.append(table_audit)
        blockers.extend(contract_blockers)
    return ChildAudit(child, tuple(table_audits)), blockers


def audit_rule(processed_dir: Path, rule: ContainmentRule) -> RuleAudit:
    """Audit a rule without mutating files; any uncertainty blocks staging."""
    parent = validate_stem(rule.parent)
    children: list[ChildAudit] = []
    blockers: list[str] = []
    for child_value in sorted(rule.children):
        child, child_blockers = _audit_child(processed_dir, parent, validate_stem(child_value))
        children.append(child)
        blockers.extend(child_blockers)
    return RuleAudit(parent, tuple(children), tuple(sorted(blockers)))


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    with atomic_replacement(path) as temporary:
        pq.write_table(table, temporary)  # type: ignore[no-untyped-call]


def _identity(row: dict[str, Any], contract: TableContract) -> tuple[Any, ...]:
    return tuple(row[column] for column in contract.identity_columns)


def _remap_link(
    row: dict[str, Any],
    *,
    parent_stem: str,
    parent_polygons: dict[tuple[Any, Any], dict[str, Any]],
) -> dict[str, Any]:
    if "polygon_id" not in row:
        return row
    polygon = parent_polygons[(row["osm_type"], row["osm_id"])]
    remapped = dict(row)
    remapped["polygon_id"] = polygon["polygon_id"]
    if "source_pbf" in remapped:
        remapped["source_pbf"] = polygon.get("source_pbf", f"{parent_stem}.osm.pbf")
    if "region" in remapped:
        remapped["region"] = polygon.get("region", parent_stem.removesuffix("-latest"))
    return remapped


def _canonical_polygon_row(
    parent: dict[str, Any] | None, child: dict[str, Any], *, parent_stem: str
) -> dict[str, Any]:
    """Keep the newest OSM snapshot while retaining canonical parent provenance."""
    newest = (
        child
        if parent is None or child.get("extracted_at", "") > parent.get("extracted_at", "")
        else parent
    )
    canonical = dict(newest)
    for field in ("polygon_id", "region", "source_pbf"):
        if field in canonical:
            canonical[field] = _canonical_provenance_value(
                field,
                parent,
                child,
                parent_stem,
            )
    return canonical


def _canonical_provenance_value(
    field: str,
    parent: dict[str, Any] | None,
    child: dict[str, Any],
    parent_stem: str,
) -> Any:
    """Return one canonical parent-provenance field."""
    if field == "polygon_id":
        return _polygon_provenance(parent, child, parent_stem)
    if field == "region":
        return _parent_or_default(parent, "region", parent_stem.removesuffix("-latest"))
    return _parent_or_default(parent, "source_pbf", f"{parent_stem}.osm.pbf")


def _polygon_provenance(
    parent: dict[str, Any] | None,
    child: dict[str, Any],
    parent_stem: str,
) -> str:
    """Return the canonical polygon identity."""
    if parent is not None:
        return str(parent["polygon_id"])
    return f"{parent_stem}:{child['osm_type']}:{child['osm_id']}"


def _parent_or_default(
    parent: dict[str, Any] | None,
    field: str,
    default: str,
) -> str:
    """Read a parent provenance field with its canonical fallback."""
    if parent is None:
        return default
    return str(parent.get(field, default))


def _merge_polygon_rows(
    processed_dir: Path,
    parent_stem: str,
    children: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[tuple[Any, Any], dict[str, Any]]]:
    """Merge child polygon snapshots into canonical parent rows."""
    parent_path = processed_dir / "polygons" / f"{parent_stem}.parquet"
    polygon_rows = pq.read_table(parent_path).to_pylist()  # type: ignore[no-untyped-call]
    polygon_positions = {
        (row["osm_type"], row["osm_id"]): position for position, row in enumerate(polygon_rows)
    }
    for child in children:
        child_path = processed_dir / "polygons" / f"{child}.parquet"
        _merge_child_polygon_rows(
            polygon_rows,
            polygon_positions,
            pq.read_table(child_path).to_pylist(),  # type: ignore[no-untyped-call]
            parent_stem,
        )
    parent_polygons = {(row["osm_type"], row["osm_id"]): row for row in polygon_rows}
    return polygon_rows, parent_polygons


def _merge_child_polygon_rows(
    polygon_rows: list[dict[str, Any]],
    polygon_positions: dict[tuple[Any, Any], int],
    child_rows: list[dict[str, Any]],
    parent_stem: str,
) -> None:
    """Merge one child's polygon snapshots into the parent rows."""
    for child_row in child_rows:
        key = (child_row["osm_type"], child_row["osm_id"])
        position = polygon_positions.get(key)
        if position is None:
            polygon_positions[key] = len(polygon_rows)
            polygon_rows.append(_canonical_polygon_row(None, child_row, parent_stem=parent_stem))
            continue
        polygon_rows[position] = _canonical_polygon_row(
            polygon_rows[position],
            child_row,
            parent_stem=parent_stem,
        )


def _append_child_rows(
    rows: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    child_rows: list[dict[str, Any]],
    contract: TableContract,
    parent_stem: str,
    parent_polygons: dict[tuple[Any, Any], dict[str, Any]],
) -> None:
    """Append unseen child rows, remapping polygon article provenance."""
    for candidate in child_rows:
        key = _identity(candidate, contract)
        if key in seen:
            continue
        if contract.subdir == "polygon_articles":
            candidate = _remap_link(
                candidate,
                parent_stem=parent_stem,
                parent_polygons=parent_polygons,
            )
        rows.append(candidate)
        seen.add(key)


def _merged_contract_rows(
    processed_dir: Path,
    parent_stem: str,
    children: tuple[str, ...],
    contract: TableContract,
    polygon_rows: list[dict[str, Any]],
    parent_polygons: dict[tuple[Any, Any], dict[str, Any]],
) -> tuple[list[dict[str, Any]], pa.Schema]:
    """Merge one containment table and return rows plus its parent schema."""
    parent_path = processed_dir / contract.subdir / f"{parent_stem}.parquet"
    parent = pq.read_table(parent_path)  # type: ignore[no-untyped-call]
    rows = polygon_rows if contract.subdir == "polygons" else parent.to_pylist()
    seen = {_identity(row, contract) for row in rows}
    for child in children:
        child_path = processed_dir / contract.subdir / f"{child}.parquet"
        child_rows = pq.read_table(child_path).to_pylist()  # type: ignore[no-untyped-call]
        _append_child_rows(
            rows,
            seen,
            child_rows,
            contract,
            parent_stem,
            parent_polygons,
        )
    return rows, parent.schema


def _stage_contract_artifact(
    processed_dir: Path,
    cache_dir: Path,
    parent_stem: str,
    children: tuple[str, ...],
    contract: TableContract,
    polygon_rows: list[dict[str, Any]],
    parent_polygons: dict[tuple[Any, Any], dict[str, Any]],
) -> tuple[str, Path]:
    """Merge and stage one containment table."""
    rows, schema = _merged_contract_rows(
        processed_dir,
        parent_stem,
        children,
        contract,
        polygon_rows,
        parent_polygons,
    )
    staged_table = pa.Table.from_pylist(rows, schema=schema)
    target = cache_dir / parent_stem / contract.subdir / f"{parent_stem}.parquet"
    _atomic_write_parquet(target, staged_table)
    return contract.subdir, target


def stage_rule(processed_dir: Path, cache_dir: Path, audit: RuleAudit) -> StagedRule:
    """Stage lossless canonical parent tables without modifying originals."""
    if not audit.safe_to_stage:
        raise ValueError(
            f"Containment rule {audit.parent!r} is not safe to stage: {audit.blockers}"
        )
    parent_stem = validate_stem(audit.parent)
    children = tuple(child.stem for child in audit.children)
    polygon_rows, parent_polygons = _merge_polygon_rows(processed_dir, parent_stem, children)
    artifacts = [
        _stage_contract_artifact(
            processed_dir,
            cache_dir,
            parent_stem,
            children,
            contract,
            polygon_rows,
            parent_polygons,
        )
        for contract in TABLE_CONTRACTS
    ]
    return StagedRule(parent_stem, children, tuple(artifacts))


def _retirement_path(processed_dir: Path) -> Path:
    return processed_dir / "manifests" / RETIREMENT_FILENAME


def _load_retirement_payload(processed_dir: Path) -> dict[str, Any]:
    path = _retirement_path(processed_dir)
    if not path.is_file():
        return {"contract_version": RETIREMENT_CONTRACT_VERSION, "retired": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_version") != RETIREMENT_CONTRACT_VERSION:
        raise ValueError("Unsupported containment retirement contract version")
    if not isinstance(payload.get("retired"), dict):
        raise ValueError("Malformed containment retirement manifest")
    return cast(dict[str, Any], payload)


def _parquet_row_count(path: Path) -> int:
    """Read only Parquet metadata when updating manifest row counts."""
    metadata = pq.read_metadata(path)  # type: ignore[no-untyped-call]
    return cast(int, metadata.num_rows)


def _canonical_manifest_stats(staged: StagedRule) -> dict[str, Any]:
    """Recompute the existing processed-manifest statistics from staged tables."""
    polygons = pq.read_table(staged.artifact("polygons")).to_pylist()  # type: ignore[no-untyped-call]
    documents = pq.read_table(  # type: ignore[no-untyped-call]
        staged.artifact("wikipedia/documents"),
        columns=["language", "article_length_chars"],
    ).to_pylist()
    return {
        **_polygon_manifest_stats(polygons),
        **_document_manifest_stats(documents),
    }


def _polygon_manifest_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate manifest values derived from polygon rows."""
    return {
        "polygon_count": len(rows),
        "unique_wikidata_count": len({row["wikidata"] for row in rows if row["wikidata"]}),
        "rows_with_wikipedia": sum(bool(row["has_wikipedia"]) for row in rows),
        "rows_with_full_text": sum(bool(row["text_available"]) for row in rows),
        "area_bucket_counts": _area_bucket_counts(rows),
        "top_tag_keys": _top_tag_keys(rows),
    }


def _area_bucket_counts(rows: list[dict[str, Any]]) -> dict[Any, int]:
    """Count polygon rows by their existing area bucket."""
    return dict(Counter(row["area_bucket"] for row in rows))


def _top_tag_keys(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count valid serialized tag keys, ignoring malformed rows."""
    tag_keys: Counter[str] = Counter()
    for row in rows:
        try:
            tag_keys.update(json.loads(row["tag_keys"]))
        except (TypeError, ValueError):
            continue
    return dict(tag_keys.most_common(50))


def _document_manifest_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate manifest values derived from Wikipedia document rows."""
    languages = sorted({row["language"] for row in rows})
    return {
        "article_count": len(rows),
        "language_count": len(languages),
        "languages": languages,
        "total_full_text_chars": sum(row["article_length_chars"] for row in rows),
    }


def load_retired_children(processed_dir: Path) -> frozenset[str]:
    """Load durable child exclusions from the local retirement manifest."""
    return frozenset(_load_retirement_payload(processed_dir)["retired"])


def load_retired_parent_children(processed_dir: Path) -> dict[str, tuple[str, ...]]:
    """Return durable retirements grouped by retained parent."""
    grouped: dict[str, list[str]] = {}
    for child, entry in _load_retirement_payload(processed_dir)["retired"].items():
        parent = entry.get("parent") if isinstance(entry, dict) else None
        if not isinstance(parent, str):
            raise ValueError(f"Malformed containment retirement entry for {child!r}")
        grouped.setdefault(parent, []).append(child)
    return {parent: tuple(sorted(children)) for parent, children in sorted(grouped.items())}


def _copy_once(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def _install_file(source: Path, target: Path) -> None:
    """Install ``source`` at ``target``, preserving the source's metadata."""
    with atomic_replacement(target) as temporary:
        shutil.copy2(source, temporary)


def _remove_active_children(processed_dir: Path, children: tuple[str, ...]) -> None:
    for child in children:
        for contract in TABLE_CONTRACTS:
            (processed_dir / contract.subdir / f"{child}.parquet").unlink(missing_ok=True)


def _quarantine_and_install(
    processed_dir: Path,
    quarantine: Path,
    staged: StagedRule,
    children: tuple[str, ...],
) -> None:
    """Quarantine live inputs and install staged parent artifacts."""
    for contract in TABLE_CONTRACTS:
        parent_live = processed_dir / contract.subdir / f"{staged.parent}.parquet"
        _copy_once(
            parent_live,
            quarantine / "_parents" / staged.parent / contract.subdir / parent_live.name,
        )
        for child in children:
            child_live = processed_dir / contract.subdir / f"{child}.parquet"
            _copy_once(child_live, quarantine / child / contract.subdir / child_live.name)
        _install_file(staged.artifact(contract.subdir), parent_live)


def _persist_prepared_rule(
    processed_dir: Path,
    staged: StagedRule,
    children: tuple[str, ...],
) -> None:
    """Record a completed local containment preparation."""
    payload = _load_retirement_payload(processed_dir)
    for child in children:
        payload["retired"][child] = {"parent": staged.parent, "status": "prepared"}
    atomic_write_text(_retirement_path(processed_dir), dumps(payload) + "\n")


def _update_pipeline_manifests(processed_dir: Path, staged: StagedRule) -> None:
    _update_processed_manifest(processed_dir, staged)
    _update_augmentation_manifest(processed_dir, staged)


def _update_processed_manifest(processed_dir: Path, staged: StagedRule) -> None:
    processed_manifest = processed_dir / "manifests" / "processed_pbfs.json"
    if not processed_manifest.is_file():
        return
    payload = json.loads(processed_manifest.read_text(encoding="utf-8"))
    for child in staged.children:
        payload.pop(f"{child}.osm.pbf", None)
    parent_entry = payload.get(f"{staged.parent}.osm.pbf")
    if isinstance(parent_entry, dict):
        parent_entry.update(_canonical_manifest_stats(staged))
    atomic_write_text(processed_manifest, dumps(payload) + "\n")


def _update_augmentation_manifest(processed_dir: Path, staged: StagedRule) -> None:
    augmentation_manifest = (
        processed_dir / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if not augmentation_manifest.is_file():
        return
    from osm_polygon_wikidata_only.augmentation.steps import sha256_file

    payload = json.loads(augmentation_manifest.read_text(encoding="utf-8"))
    for child in staged.children:
        payload.pop(child, None)
    parent_entry = payload.get(staged.parent)
    if isinstance(parent_entry, dict):
        parent_entry["counts"] = {
            "wikipedia_documents": _parquet_row_count(staged.artifact("wikipedia/documents")),
            "wikipedia_sections": _parquet_row_count(staged.artifact("wikipedia/sections")),
            "wikivoyage_documents": _parquet_row_count(staged.artifact("wikivoyage/documents")),
            "wikivoyage_sections": _parquet_row_count(staged.artifact("wikivoyage/sections")),
            "wikidata_facts": _parquet_row_count(staged.artifact("wikidata/facts")),
        }
        live_polygons = processed_dir / "polygons" / f"{staged.parent}.parquet"
        live_documents = processed_dir / "wikipedia" / "documents" / f"{staged.parent}.parquet"
        parent_entry["core_hashes"] = {
            str(live_polygons): sha256_file(staged.artifact("polygons")),
            str(live_documents): sha256_file(staged.artifact("wikipedia/documents")),
        }
    atomic_write_text(augmentation_manifest, dumps(payload) + "\n")


def prepare_local_rule(data_root: Path, audit: RuleAudit) -> PreparedRule:
    """Quarantine originals, install canonical parents, and persist exclusion."""
    processed_dir = data_root / "processed"
    children = tuple(child.stem for child in audit.children)
    prepared = PreparedRule(audit.parent, children)
    retired = load_retired_children(processed_dir)
    if set(children).issubset(retired):
        _remove_active_children(processed_dir, children)
        return prepared
    staged = stage_rule(processed_dir, data_root / "cache" / "containment_retirement", audit)
    quarantine = data_root / "quarantine" / "containment-v1"
    _quarantine_and_install(processed_dir, quarantine, staged, children)
    _update_pipeline_manifests(processed_dir, staged)
    _persist_prepared_rule(processed_dir, staged, children)
    _remove_active_children(processed_dir, children)
    return prepared


def _pending_rule(rule: ContainmentRule, retired: frozenset[str]) -> ContainmentRule | None:
    """Return the still-active part of a policy rule, if any."""
    pending_children = tuple(child for child in rule.children if child not in retired)
    return ContainmentRule(rule.parent, pending_children) if pending_children else None


def _has_required_files(processed_dir: Path, rule: ContainmentRule) -> bool:
    """Return whether every table file needed to audit ``rule`` exists."""
    stems = (rule.parent, *rule.children)
    return all(
        (processed_dir / contract.subdir / f"{stem}.parquet").is_file()
        for contract in TABLE_CONTRACTS
        for stem in stems
    )


def _prepare_audited_rule(
    data_root: Path, audit: RuleAudit, *, dry_run: bool
) -> PreparedRule | None:
    """Apply one safe audit unless preparation was explicitly disabled."""
    return None if dry_run else prepare_local_rule(data_root, audit)


def _record_audited_rule(
    data_root: Path,
    audit: RuleAudit,
    *,
    dry_run: bool,
    prepared: list[PreparedRule],
    blocked: list[RuleAudit],
) -> None:
    """Record one audit outcome and prepare it when safe and requested."""
    if not audit.safe_to_stage:
        blocked.append(audit)
        return
    result = _prepare_audited_rule(data_root, audit, dry_run=dry_run)
    if result is not None:
        prepared.append(result)


def prepare_safe_rules(
    data_root: Path, *, dry_run: bool
) -> tuple[tuple[PreparedRule, ...], tuple[RuleAudit, ...]]:
    """Audit known rules and prepare only those proven polygon-lossless."""
    processed_dir = data_root / "processed"
    retired = load_retired_children(processed_dir)
    prepared: list[PreparedRule] = []
    blocked: list[RuleAudit] = []
    for rule in CONTAINMENT_RULES:
        scoped = _pending_rule(rule, retired)
        if scoped is None:
            continue
        if not _has_required_files(processed_dir, scoped):
            continue
        audit = audit_rule(processed_dir, scoped)
        _record_audited_rule(
            data_root,
            audit,
            dry_run=dry_run,
            prepared=prepared,
            blocked=blocked,
        )
    return tuple(prepared), tuple(blocked)


__all__ = [
    "RETIREMENT_CONTRACT_VERSION",
    "RETIREMENT_FILENAME",
    "ChildAudit",
    "PreparedRule",
    "RuleAudit",
    "StagedRule",
    "TableAudit",
    "audit_rule",
    "load_retired_children",
    "load_retired_parent_children",
    "prepare_local_rule",
    "prepare_safe_rules",
    "stage_rule",
]
