"""Fail-closed audit of whole-file containment retirement candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .containment_policy import (
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


def _identity_set(path: Path, contract: TableContract) -> tuple[set[tuple[Any, ...]], int]:
    table = pq.read_table(path, columns=list(contract.identity_columns))
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
    processed_dir: Path,  # noqa: ARG001 -- audit hook signature shared with the absent-contract path
    contract: TableContract,
    parent: str,  # noqa: ARG001 -- audit hook signature shared with the absent-contract path
    child: str,
    parent_path: Path,
    child_path: Path,
) -> tuple[TableAudit, list[str]]:
    """Audit one contract whose parent and child files are present."""
    try:
        parent_schema = pq.read_schema(parent_path)
        child_schema = pq.read_schema(child_path)
        if not parent_schema.equals(child_schema, check_metadata=True):
            return (
                TableAudit(contract.subdir, 0, 0, 0, 0),
                [f"{child}: schema mismatch for {contract.subdir}"],
            )
        parent_ids, parent_duplicates = _identity_set(parent_path, contract)
        child_ids, child_duplicates = _identity_set(child_path, contract)
    except Exception as error:  # noqa: BLE001 -- audit reports any unreadable table as a finding
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
