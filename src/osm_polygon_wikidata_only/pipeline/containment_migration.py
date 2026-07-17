"""Fail-closed audit and staging for whole-file containment retirement."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
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


def audit_rule(processed_dir: Path, rule: ContainmentRule) -> RuleAudit:
    """Audit a rule without mutating files; any uncertainty blocks staging."""
    parent = validate_stem(rule.parent)
    children: list[ChildAudit] = []
    blockers: list[str] = []
    for child_value in sorted(rule.children):
        child = validate_stem(child_value)
        table_audits: list[TableAudit] = []
        for contract in TABLE_CONTRACTS:
            parent_path = processed_dir / contract.subdir / f"{parent}.parquet"
            child_path = processed_dir / contract.subdir / f"{child}.parquet"
            missing_paths = [path for path in (parent_path, child_path) if not path.is_file()]
            if missing_paths:
                blockers.extend(
                    f"{child}: missing file {path.relative_to(processed_dir)}"
                    for path in missing_paths
                )
                table_audits.append(TableAudit(contract.subdir, 0, 0, 0, 0))
                continue
            try:
                parent_schema = pq.read_schema(parent_path)  # type: ignore[no-untyped-call]
                child_schema = pq.read_schema(child_path)  # type: ignore[no-untyped-call]
                if not parent_schema.equals(child_schema, check_metadata=True):
                    blockers.append(f"{child}: schema mismatch for {contract.subdir}")
                    table_audits.append(TableAudit(contract.subdir, 0, 0, 0, 0))
                    continue
                parent_ids, parent_duplicates = _identity_set(parent_path, contract)
                child_ids, child_duplicates = _identity_set(child_path, contract)
            except Exception as error:
                blockers.append(f"{child}: unreadable {contract.subdir}: {type(error).__name__}")
                table_audits.append(TableAudit(contract.subdir, 0, 0, 0, 0))
                continue
            missing = child_ids - parent_ids
            table_audits.append(
                TableAudit(
                    contract.subdir,
                    len(child_ids),
                    len(missing),
                    parent_duplicates,
                    child_duplicates,
                )
            )
            if parent_duplicates:
                blockers.append(
                    f"{child}: {contract.subdir} parent has {parent_duplicates} duplicate identities"
                )
            if child_duplicates:
                blockers.append(
                    f"{child}: {contract.subdir} child has {child_duplicates} duplicate identities"
                )
            if contract.subdir == "polygons" and missing:
                blockers.append(f"{child}: polygons missing {len(missing)} identity from parent")
        children.append(ChildAudit(child, tuple(table_audits)))
    return RuleAudit(parent, tuple(children), tuple(sorted(blockers)))


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary)  # type: ignore[no-untyped-call]
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    parent: dict[str, Any], child: dict[str, Any], *, parent_stem: str
) -> dict[str, Any]:
    """Keep the newest OSM snapshot while retaining canonical parent provenance."""
    newest = child if child.get("extracted_at", "") > parent.get("extracted_at", "") else parent
    canonical = dict(newest)
    if "polygon_id" in canonical:
        canonical["polygon_id"] = parent["polygon_id"]
    if "region" in canonical:
        canonical["region"] = parent.get("region", parent_stem.removesuffix("-latest"))
    if "source_pbf" in canonical:
        canonical["source_pbf"] = parent.get("source_pbf", f"{parent_stem}.osm.pbf")
    return canonical


def stage_rule(processed_dir: Path, cache_dir: Path, audit: RuleAudit) -> StagedRule:
    """Stage lossless canonical parent tables without modifying originals."""
    if not audit.safe_to_stage:
        raise ValueError(
            f"Containment rule {audit.parent!r} is not safe to stage: {audit.blockers}"
        )
    parent_stem = validate_stem(audit.parent)
    children = tuple(child.stem for child in audit.children)
    polygon_path = processed_dir / "polygons" / f"{parent_stem}.parquet"
    polygon_rows = pq.read_table(polygon_path).to_pylist()  # type: ignore[no-untyped-call]
    polygon_positions = {
        (row["osm_type"], row["osm_id"]): position for position, row in enumerate(polygon_rows)
    }
    for child in children:
        child_path = processed_dir / "polygons" / f"{child}.parquet"
        for child_row in pq.read_table(child_path).to_pylist():  # type: ignore[no-untyped-call]
            key = (child_row["osm_type"], child_row["osm_id"])
            position = polygon_positions[key]
            polygon_rows[position] = _canonical_polygon_row(
                polygon_rows[position], child_row, parent_stem=parent_stem
            )
    parent_polygons = {(row["osm_type"], row["osm_id"]): row for row in polygon_rows}
    artifacts: list[tuple[str, Path]] = []
    for contract in TABLE_CONTRACTS:
        parent_path = processed_dir / contract.subdir / f"{parent_stem}.parquet"
        parent = pq.read_table(parent_path)  # type: ignore[no-untyped-call]
        rows = polygon_rows if contract.subdir == "polygons" else parent.to_pylist()
        seen = {_identity(row, contract) for row in rows}
        for child in children:
            child_path = processed_dir / contract.subdir / f"{child}.parquet"
            child_rows = pq.read_table(child_path).to_pylist()  # type: ignore[no-untyped-call]
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
        staged_table = pa.Table.from_pylist(rows, schema=parent.schema)
        target = cache_dir / parent_stem / contract.subdir / f"{parent_stem}.parquet"
        _atomic_write_parquet(target, staged_table)
        artifacts.append((contract.subdir, target))
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
    area_buckets = Counter(row["area_bucket"] for row in polygons)
    tag_keys: Counter[str] = Counter()
    for row in polygons:
        try:
            tag_keys.update(json.loads(row["tag_keys"]))
        except (TypeError, ValueError):
            continue
    languages = sorted({row["language"] for row in documents})
    return {
        "polygon_count": len(polygons),
        "unique_wikidata_count": len({row["wikidata"] for row in polygons if row["wikidata"]}),
        "article_count": len(documents),
        "language_count": len(languages),
        "languages": languages,
        "rows_with_wikipedia": sum(bool(row["has_wikipedia"]) for row in polygons),
        "rows_with_full_text": sum(bool(row["text_available"]) for row in polygons),
        "total_full_text_chars": sum(row["article_length_chars"] for row in documents),
        "area_bucket_counts": dict(area_buckets),
        "top_tag_keys": dict(tag_keys.most_common(50)),
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
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_active_children(processed_dir: Path, children: tuple[str, ...]) -> None:
    for child in children:
        for contract in TABLE_CONTRACTS:
            (processed_dir / contract.subdir / f"{child}.parquet").unlink(missing_ok=True)


def _update_pipeline_manifests(processed_dir: Path, staged: StagedRule) -> None:
    processed_manifest = processed_dir / "manifests" / "processed_pbfs.json"
    if processed_manifest.is_file():
        payload = json.loads(processed_manifest.read_text(encoding="utf-8"))
        for child in staged.children:
            payload.pop(f"{child}.osm.pbf", None)
        parent_entry = payload.get(f"{staged.parent}.osm.pbf")
        if isinstance(parent_entry, dict):
            parent_entry.update(_canonical_manifest_stats(staged))
        atomic_write_text(processed_manifest, dumps(payload) + "\n")

    augmentation_manifest = (
        processed_dir / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if augmentation_manifest.is_file():
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
    _update_pipeline_manifests(processed_dir, staged)
    payload = _load_retirement_payload(processed_dir)
    for child in children:
        payload["retired"][child] = {"parent": staged.parent, "status": "prepared"}
    atomic_write_text(_retirement_path(processed_dir), dumps(payload) + "\n")
    _remove_active_children(processed_dir, children)
    return prepared


def prepare_safe_rules(
    data_root: Path, *, dry_run: bool
) -> tuple[tuple[PreparedRule, ...], tuple[RuleAudit, ...]]:
    """Audit known rules and prepare only those proven polygon-lossless."""
    processed_dir = data_root / "processed"
    retired = load_retired_children(processed_dir)
    prepared: list[PreparedRule] = []
    blocked: list[RuleAudit] = []
    for rule in CONTAINMENT_RULES:
        pending_children = tuple(child for child in rule.children if child not in retired)
        if not pending_children:
            continue
        scoped = ContainmentRule(rule.parent, pending_children)
        expected = [
            processed_dir / contract.subdir / f"{stem}.parquet"
            for contract in TABLE_CONTRACTS
            for stem in (scoped.parent, *scoped.children)
        ]
        if not all(path.is_file() for path in expected):
            continue
        audit = audit_rule(processed_dir, scoped)
        if not audit.safe_to_stage:
            blocked.append(audit)
            continue
        if not dry_run:
            prepared.append(prepare_local_rule(data_root, audit))
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
