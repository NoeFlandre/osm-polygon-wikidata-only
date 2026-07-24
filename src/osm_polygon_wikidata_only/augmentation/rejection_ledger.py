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
        if not isinstance(self.shard, str) or not self.shard:
            raise ValueError(f"Rejection shard must be a non-empty string, got {self.shard!r}")
        if self.source_table not in SUPPORTED_SOURCE_TABLES:
            raise ValueError(
                f"Unsupported source_table {self.source_table!r}; "
                f"must be one of {sorted(SUPPORTED_SOURCE_TABLES)}"
            )
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError(
                f"Rejection identifier must be a non-empty string, got {self.identifier!r}"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError(f"Rejection reason must be a non-empty string, got {self.reason!r}")
        if not _is_valid_qid(self.wikidata):
            raise ValueError(
                f"Rejection observed wikidata must be a valid QID, got {self.wikidata!r}"
            )
        if self.expected is not None and not _is_valid_qid(self.expected):
            raise ValueError(
                f"Rejection expected wikidata must be a valid QID or None, got {self.expected!r}"
            )
        if not isinstance(self.cascaded_sections, int) or self.cascaded_sections < 0:
            raise ValueError(
                f"Rejection cascaded_sections must be a non-negative int, "
                f"got {self.cascaded_sections!r}"
            )

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
    documents_table = pq.read_table(documents_path, columns=list(DOCUMENT_COLUMNS))  # type: ignore[no-untyped-call]
    documents_rows = documents_table.to_pylist()

    if sections_path.is_file():
        sections_table = pq.read_table(sections_path, columns=list(SECTION_COLUMNS))  # type: ignore[no-untyped-call]
        sections_rows = sections_table.to_pylist()
    else:
        sections_rows = []

    retained_documents: list[dict[str, Any]] = []
    rejected_document_ids: set[str] = set()
    rejections: list[RejectionRecord] = []

    for row in documents_rows:
        document_id = str(row.get("document_id", ""))
        wikidata = str(row.get("wikidata", ""))
        if wikidata not in valid_qids:
            rejected_document_ids.add(document_id)
            rejections.append(
                RejectionRecord(
                    shard=stem,
                    source_table="wikivoyage_documents",
                    identifier=document_id,
                    wikidata=wikidata,
                    expected=None,
                    reason="wikidata_absent_from_polygons",
                    cascaded_sections=0,
                )
            )
            continue
        retained_documents.append({col: row.get(col) for col in DOCUMENT_COLUMNS})

    retained_sections: list[dict[str, Any]] = []
    cascaded_count = 0
    cascades_by_document: dict[str, int] = {}
    for row in sections_rows:
        document_id = str(row.get("document_id", ""))
        if document_id in rejected_document_ids:
            cascaded_count += 1
            cascades_by_document[document_id] = cascades_by_document.get(document_id, 0) + 1
            continue
        retained_sections.append({col: row.get(col) for col in SECTION_COLUMNS})

    rejections = [
        RejectionRecord(
            shard=record.shard,
            source_table=record.source_table,
            identifier=record.identifier,
            wikidata=record.wikidata,
            expected=record.expected,
            reason=record.reason,
            cascaded_sections=cascades_by_document.get(record.identifier, 0),
        )
        for record in rejections
    ]

    rejections_sorted = merge_records(rejections)
    return IntegrityPlan(
        stem=stem,
        data_root=data_root,
        retained_documents=retained_documents,
        retained_sections=retained_sections,
        rejections=rejections_sorted,
    )


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

    documents_path = data_root.processed / "wikivoyage" / "documents" / f"{plan.stem}.parquet"
    sections_path = data_root.processed / "wikivoyage" / "sections" / f"{plan.stem}.parquet"

    # Stage documents.
    if plan.retained_documents:
        documents_table = pa.Table.from_pylist(plan.retained_documents, schema=document_schema())
    else:
        documents_table = pa.table({col: [] for col in DOCUMENT_COLUMNS}, schema=document_schema())
    _atomic_overwrite_parquet(documents_path, documents_table)

    # Stage sections.
    if plan.retained_sections:
        sections_table = pa.Table.from_pylist(plan.retained_sections, schema=section_schema())
    else:
        sections_table = pa.table({col: [] for col in SECTION_COLUMNS}, schema=section_schema())
    _atomic_overwrite_parquet(sections_path, sections_table)

    # Merge per-stem rejection records into the cumulative ledger. The
    # existing cumulative ledger (if any) is one of the sources so prior
    # stem histories are preserved. The cumulative ledger is the LAST
    # commit in this transaction.
    ledger_path = data_root.processed / "integrity" / LEDGER_FILENAME
    stem_ledger_path = data_root.processed / "integrity" / f"{plan.stem}.json"
    save_ledger(stem_ledger_path, plan.rejections)
    # ``merge_ledger_files`` merges from sources + saves target. We
    # include the existing ledger as one source so prior history is
    # preserved across runs.
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
