"""Cumulative rejection ledger for the augmentation pipeline.

The rejection ledger is a deterministic, append-only record of every
row dropped during integrity normalization. It is committed last in
the integrity transaction so the cumulative ledger file's mtime is
later than every per-shard artifact.

Two explicit stages:

* :func:`plan_integrity_normalization` -- read-only planning.
* :func:`apply_integrity_normalization` -- transactional apply.

The plan returns an :class:`IntegrityPlan` whose ``retained_documents``
and ``retained_sections`` are the kept rows and whose ``rejections``
is the deterministic list of rejected records. The apply stage
journal-commits the kept parquet tables, the per-shard audit, and the
cumulative ledger in a single ordered transaction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.integrity import (
    _atomic_overwrite_parquet,
)
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    section_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    qids_from_osm_tag as _qids_from_osm_tag,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text

LEDGER_CONTRACT_VERSION = "rejection-ledger-v1"
LEDGER_FILENAME = "rejection_ledger.json"

SUPPORTED_SOURCE_TABLES: frozenset[str] = frozenset({"polygon_articles", "wikivoyage_documents"})


def supported_source_tables() -> frozenset[str]:
    return SUPPORTED_SOURCE_TABLES


def _is_valid_qid(value: str | None) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return bool(_qids_from_osm_tag(value))


# ---------------------------------------------------------------------------
# RejectionRecord (re-export with strict validation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    """Strict, validated rejection record.

    Full identity is ``(shard, source_table, identifier, wikidata, expected)``.
    """

    shard: str
    source_table: str
    identifier: str
    wikidata: str
    expected: str | None
    reason: str
    cascaded_sections: int = 0

    def __post_init__(self) -> None:
        _validate_rejection_strings(self.shard, self.identifier, self.reason)
        _validate_source_table(self.source_table)
        _validate_rejection_qids(self.wikidata, self.expected)
        _validate_cascaded_sections(self.cascaded_sections)

    @property
    def identity(self) -> tuple[str, str, str, str, str | None]:
        return (self.shard, self.source_table, self.identifier, self.wikidata, self.expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "source_table": self.source_table,
            "identifier": self.identifier,
            "wikidata": self.wikidata,
            "expected": self.expected,
            "reason": self.reason,
            "cascaded_sections": self.cascaded_sections,
        }


def _validate_rejection_strings(shard: object, identifier: object, reason: object) -> None:
    _require_nonempty_string(shard, "shard")
    _require_nonempty_string(identifier, "identifier")
    _require_nonempty_string(reason, "reason")


def _require_nonempty_string(value: object, name: str) -> None:
    if isinstance(value, str) and value:
        return
    raise ValueError(f"Rejection {name} must be a non-empty string, got {value!r}")


def _validate_source_table(source_table: str) -> None:
    if source_table not in SUPPORTED_SOURCE_TABLES:
        raise ValueError(
            f"Unsupported source_table {source_table!r}; "
            f"must be one of {sorted(SUPPORTED_SOURCE_TABLES)}"
        )


def _validate_rejection_qids(wikidata: str, expected: str | None) -> None:
    if not _is_valid_qid(wikidata):
        raise ValueError(f"Rejection observed wikidata must be a valid QID, got {wikidata!r}")
    if expected is not None and not _is_valid_qid(expected):
        raise ValueError(
            f"Rejection expected wikidata must be a valid QID or None, got {expected!r}"
        )


def _validate_cascaded_sections(cascaded_sections: object) -> None:
    if not isinstance(cascaded_sections, int) or cascaded_sections < 0:
        raise ValueError(
            f"Rejection cascaded_sections must be a non-negative int, got {cascaded_sections!r}"
        )


# ---------------------------------------------------------------------------
# Merging and persistence
# ---------------------------------------------------------------------------


def merge_records(records: Iterable[RejectionRecord]) -> list[RejectionRecord]:
    """Merge a sequence of records.

    Same full identity -> one record (max cascaded_sections).
    Different observed QID -> separate records.
    """
    by_identity: dict[tuple[str, str, str, str, str | None], RejectionRecord] = {}
    for record in records:
        existing = by_identity.get(record.identity)
        if existing is None:
            by_identity[record.identity] = record
        else:
            merged = RejectionRecord(
                shard=existing.shard,
                source_table=existing.source_table,
                identifier=existing.identifier,
                wikidata=existing.wikidata,
                expected=existing.expected,
                reason=existing.reason,
                cascaded_sections=max(existing.cascaded_sections, record.cascaded_sections),
            )
            by_identity[record.identity] = merged
    return sorted(
        by_identity.values(),
        key=lambda record: (record.shard, record.source_table, record.identifier, record.wikidata),
    )


def save_ledger(path: Path, records: Iterable[RejectionRecord]) -> None:
    """Save a deterministic cumulative ledger file.

    Idempotent: a no-op save (identical input) does NOT rewrite the
    file (preserves the file's mtime and content hash).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records,
        key=lambda record: (record.shard, record.source_table, record.identifier, record.wikidata),
    )
    payload = {
        "contract_version": LEDGER_CONTRACT_VERSION,
        "records": [record.to_dict() for record in ordered],
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == payload:
            return
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_ledger(path: Path) -> list[RejectionRecord]:
    """Load a cumulative ledger file. Returns an empty list if the file
    is absent."""
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("contract_version") != LEDGER_CONTRACT_VERSION:
        raise ValueError(f"Invalid ledger contract version: {raw.get('contract_version')!r}")
    out: list[RejectionRecord] = []
    for entry in raw.get("records", []):
        out.append(
            RejectionRecord(
                shard=entry["shard"],
                source_table=entry["source_table"],
                identifier=entry["identifier"],
                wikidata=entry["wikidata"],
                expected=entry.get("expected"),
                reason=entry["reason"],
                cascaded_sections=int(entry.get("cascaded_sections", 0)),
            )
        )
    return out


def merge_ledger_files(sources: Iterable[Path], target: Path) -> None:
    """Merge per-stem ledger files into one cumulative ledger.

    The merge preserves every entry from every source -- neither stem
    is erased. Records sharing the same full identity (shard,
    source_table, identifier, wikidata, expected) are deduplicated by
    keeping the maximum ``cascaded_sections``.
    """
    records: list[RejectionRecord] = []
    for source in sources:
        if not source.is_file():
            continue
        for record in load_ledger(source):
            records.append(record)
    merged = merge_records(records)
    save_ledger(target, merged)


# ---------------------------------------------------------------------------
# Plan/Apply for integrity normalization
# ---------------------------------------------------------------------------


@dataclass
class IntegrityPlan:
    """Read-only result of :func:`plan_integrity_normalization`.

    Attributes
    ----------
    stem:
        The shard stem.
    data_root:
        The DataRoot whose processed/ subtree the plan was derived from.
        The apply stage uses this to write the per-shard artifacts and
        the cumulative ledger.
    retained_documents:
        Wikivoyage documents kept (without rejected entries).
    retained_sections:
        Wikivoyage sections kept (cascaded sections dropped).
    rejections:
        Deterministic list of rejection records.
    """

    stem: str
    data_root: DataRoot
    retained_documents: list[dict[str, Any]]
    retained_sections: list[dict[str, Any]]
    rejections: list[RejectionRecord] = field(default_factory=list)


def plan_integrity_normalization(data_root: DataRoot, stem: str) -> IntegrityPlan:
    """Pure read-only planning stage.

    Reads polygons, wikivoyage documents, and wikivoyage sections.
    Returns a deterministic plan with the retained tables and the
    rejection records. Does NOT modify any file on disk.
    """
    valid_qids, documents_rows, sections_rows = _load_normalization_rows(data_root, stem)
    retained_documents, rejected_document_ids, rejections = _retain_documents(
        stem, valid_qids, documents_rows
    )
    retained_sections, cascades_by_document = _retain_sections(rejected_document_ids, sections_rows)
    rejections = _attach_cascade_counts(rejections, cascades_by_document)

    rejections_sorted = merge_records(rejections)
    return IntegrityPlan(
        stem=stem,
        data_root=data_root,
        retained_documents=retained_documents,
        retained_sections=retained_sections,
        rejections=rejections_sorted,
    )


def _load_normalization_rows(
    data_root: DataRoot,
    stem: str,
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    if not polygons_path.is_file():
        raise FileNotFoundError(f"Polygons parquet missing: {polygons_path}")
    polygons_table = pq.read_table(polygons_path, columns=["wikidata"])  # type: ignore[no-untyped-call]
    valid_qids: set[str] = set()
    for raw in polygons_table.column("wikidata").to_pylist():
        valid_qids.update(_qids_from_osm_tag(str(raw)))

    documents_path = data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    sections_path = data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    if not documents_path.is_file():
        raise FileNotFoundError(f"wikivoyage/documents parquet missing: {documents_path}")
    documents_rows = pq.read_table(documents_path, columns=list(DOCUMENT_COLUMNS)).to_pylist()  # type: ignore[no-untyped-call]
    sections_rows = _read_sections(sections_path)
    return valid_qids, documents_rows, sections_rows


def _read_sections(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pq.read_table(path, columns=list(SECTION_COLUMNS)).to_pylist()  # type: ignore[no-untyped-call]


def _retain_documents(
    stem: str,
    valid_qids: set[str],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str], list[RejectionRecord]]:
    retained: list[dict[str, Any]] = []
    rejected_ids: set[str] = set()
    rejections: list[RejectionRecord] = []
    for row in rows:
        document_id = str(row.get("document_id", ""))
        wikidata = str(row.get("wikidata", ""))
        if wikidata in valid_qids:
            retained.append({col: row.get(col) for col in DOCUMENT_COLUMNS})
        else:
            rejected_ids.add(document_id)
            rejections.append(
                RejectionRecord(
                    shard=stem,
                    source_table="wikivoyage_documents",
                    identifier=document_id,
                    wikidata=wikidata,
                    expected=None,
                    reason="wikidata_absent_from_polygons",
                )
            )
    return retained, rejected_ids, rejections


def _retain_sections(
    rejected_ids: set[str],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    retained: list[dict[str, Any]] = []
    cascades: dict[str, int] = {}
    for row in rows:
        document_id = str(row.get("document_id", ""))
        if document_id in rejected_ids:
            cascades[document_id] = cascades.get(document_id, 0) + 1
        else:
            retained.append({col: row.get(col) for col in SECTION_COLUMNS})
    return retained, cascades


def _attach_cascade_counts(
    rejections: list[RejectionRecord],
    cascades: dict[str, int],
) -> list[RejectionRecord]:
    return [
        RejectionRecord(
            shard=record.shard,
            source_table=record.source_table,
            identifier=record.identifier,
            wikidata=record.wikidata,
            expected=record.expected,
            reason=record.reason,
            cascaded_sections=cascades.get(record.identifier, 0),
        )
        for record in rejections
    ]


def apply_integrity_normalization(plan: IntegrityPlan) -> None:
    """Transactional apply stage.

    Writes the retained documents and sections parquets, then merges
    the per-stem rejections into the cumulative ledger file. The
    cumulative ledger is the LAST write in the transaction.

    Roll-forward safety: the cumulative ledger merge includes the
    existing ledger file (if any), so prior rejection history is
    preserved across runs.
    """
    data_root = plan.data_root
    _write_retained_table(
        data_root.processed / "wikivoyage" / "documents" / f"{plan.stem}.parquet",
        plan.retained_documents,
        DOCUMENT_COLUMNS,
        document_schema(),
    )
    _write_retained_table(
        data_root.processed / "wikivoyage" / "sections" / f"{plan.stem}.parquet",
        plan.retained_sections,
        SECTION_COLUMNS,
        section_schema(),
    )
    _write_integrity_ledgers(data_root, plan.stem, plan.rejections)


def _write_retained_table(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    schema: pa.Schema,
) -> None:
    table = (
        pa.Table.from_pylist(rows, schema=schema)
        if rows
        else pa.table({column: [] for column in columns}, schema=schema)
    )
    _atomic_overwrite_parquet(path, table)


def _write_integrity_ledgers(
    data_root: DataRoot,
    stem: str,
    rejections: list[RejectionRecord],
) -> None:
    ledger_path = data_root.processed / "integrity" / LEDGER_FILENAME
    stem_ledger_path = data_root.processed / "integrity" / f"{stem}.json"
    save_ledger(stem_ledger_path, rejections)
    sources: list[Path] = [stem_ledger_path]
    if ledger_path.is_file():
        sources.append(ledger_path)
    merge_ledger_files(sources, ledger_path)


__all__ = [
    "LEDGER_CONTRACT_VERSION",
    "LEDGER_FILENAME",
    "SUPPORTED_SOURCE_TABLES",
    "IntegrityPlan",
    "RejectionRecord",
    "apply_integrity_normalization",
    "load_ledger",
    "merge_ledger_files",
    "merge_records",
    "plan_integrity_normalization",
    "save_ledger",
    "supported_source_tables",
]
