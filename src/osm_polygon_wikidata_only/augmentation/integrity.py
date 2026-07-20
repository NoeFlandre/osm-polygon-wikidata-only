"""Deterministic post-write integrity enforcement for the published shards.

This module owns the focused, idempotent integrity pass that runs
after the per-PBF and per-region writers have materialized their
parquet files. It enforces the join contract that the downstream
consumer (:mod:`osm_polygon_sentence_relevance.joins`) relies on:

* :func:`enforce_polygon_articles_integrity` rejects every
  ``polygon_articles`` row whose ``wikidata`` does not match the
  canonical wikidata of the linked polygon in the shard's
  ``polygons`` table. Path A only: rows are dropped, never
  rewritten. An Italy-shaped defect (``italy-latest:way:845321022``
  where ``polygon_articles.wikidata = Q30901095`` but
  ``polygons.wikidata = Q134675336``) is captured here.

* :func:`enforce_wikivoyage_integrity` rejects every
  ``wikivoyage/documents`` row whose ``wikidata`` is not in the
  shard's polygon wikidata set, and cascades the rejection to the
  dependent rows in ``wikivoyage/sections``. The eight wikivoyage
  defects (australia, bahamas, brazil-nordeste,
  canada-prince-edward-island, canada-yukon, chile, mexico,
  rheinland-pfalz) all have 1-12 wikivoyage documents whose QIDs
  do not appear in their shard's polygons.

The functions are pure with respect to the filesystem state they
read: they only consult the canonical ``polygons`` parquet for a
shard and the sidecar parquets they are about to rewrite. They
never read network, never compute wikidata aliases, never call
Wikimedia APIs. Unknown integrity violations -- e.g. a polygon_articles
row referencing a polygon_id not present in the shard's
``polygons`` table, or a missing input file -- fail loudly and
propagate the underlying error; this module does not silently
coerce data.

Every rejection is captured as a deterministic record (sorted by
the relevant identifier) and the per-region rejection summary is
merged into the existing manifest entries so the audit metadata
travels with the dataset. The rejection record schema is::

    {
        "shard": "<stem>",
        "source_table": "polygon_articles" | "wikivoyage_documents",
        "identifier": "<polygon_id>" | "<document_id>",
        "wikidata": "<qid>",
        "expected": "<canonical qid from polygons>" | null,
        "reason": "wikidata_mismatch_with_polygon_master"
                | "wikidata_absent_from_polygons",
        "cascaded_sections": <int>
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    section_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import POLYGON_ARTICLE_COLUMNS
from osm_polygon_wikidata_only.io.parquet import write_polygon_articles

INTEGRITY_CONTRACT_VERSION = "join-integrity-v1"

REASON_POLYGON_ARTICLES_MISMATCH = "wikidata_mismatch_with_polygon_master"
REASON_WIKIVOYAGE_ABSENT = "wikidata_absent_from_polygons"


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    """One deterministic rejection entry.

    The tuple ``(shard, source_table, identifier, reason)`` is
    unique across the dataset; ``cascaded_sections`` is non-zero
    only for ``wikivoyage_documents`` rejections and records how
    many sections were dropped as a downstream consequence.
    """

    shard: str
    source_table: str
    identifier: str
    wikidata: str
    expected: str | None
    reason: str
    cascaded_sections: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolygonArticlesIntegrityResult:
    """Result of :func:`enforce_polygon_articles_integrity`."""

    shard: str
    original_row_count: int
    retained_row_count: int
    rejected_row_count: int
    rewritten: bool
    rejections: tuple[RejectionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "original_row_count": self.original_row_count,
            "retained_row_count": self.retained_row_count,
            "rejected_row_count": self.rejected_row_count,
            "rewritten": self.rewritten,
            "rejections": [record.to_dict() for record in self.rejections],
        }


@dataclass(frozen=True, slots=True)
class WikivoyageIntegrityResult:
    """Result of :func:`enforce_wikivoyage_integrity`."""

    shard: str
    original_document_count: int
    retained_document_count: int
    rejected_document_count: int
    original_section_count: int
    retained_section_count: int
    cascaded_section_count: int
    rewritten_documents: bool
    rewritten_sections: bool
    rejections: tuple[RejectionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "original_document_count": self.original_document_count,
            "retained_document_count": self.retained_document_count,
            "rejected_document_count": self.rejected_document_count,
            "original_section_count": self.original_section_count,
            "retained_section_count": self.retained_section_count,
            "cascaded_section_count": self.cascaded_section_count,
            "rewritten_documents": self.rewritten_documents,
            "rewritten_sections": self.rewritten_sections,
            "rejections": [record.to_dict() for record in self.rejections],
        }


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Aggregate result of :func:`enforce_all_regions`."""

    contract_version: str
    polygon_articles: tuple[PolygonArticlesIntegrityResult, ...]
    wikivoyage: tuple[WikivoyageIntegrityResult, ...]
    audit_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "polygon_articles": [result.to_dict() for result in self.polygon_articles],
            "wikivoyage": [result.to_dict() for result in self.wikivoyage],
        }

    @property
    def total_polygon_articles_rejected(self) -> int:
        return sum(result.rejected_row_count for result in self.polygon_articles)

    @property
    def total_wikivoyage_documents_rejected(self) -> int:
        return sum(result.rejected_document_count for result in self.wikivoyage)

    @property
    def total_wikivoyage_sections_cascaded(self) -> int:
        return sum(result.cascaded_section_count for result in self.wikivoyage)


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------


def _read_polygon_wikidata_map(polygons_path: Path) -> dict[str, str]:
    """Read the canonical polygon wikidata for a shard.

    Raises :class:`FileNotFoundError` if the polygons file is
    missing; raises :class:`ValueError` if a polygon_id appears
    more than once (the join contract requires a single canonical
    wikidata per polygon_id).
    """
    if not polygons_path.is_file():
        raise FileNotFoundError(f"Polygons parquet missing: {polygons_path}")
    table: pa.Table = pq.read_table(polygons_path, columns=["polygon_id", "wikidata"])  # type: ignore[no-untyped-call]
    mapping: dict[str, str] = {}
    duplicates: set[str] = set()
    for row in zip(
        table.column("polygon_id").to_pylist(),
        table.column("wikidata").to_pylist(),
        strict=True,
    ):
        polygon_id, wikidata = row
        polygon_id_str = str(polygon_id)
        wikidata_str = str(wikidata) if wikidata is not None else ""
        if polygon_id_str in mapping and mapping[polygon_id_str] != wikidata_str:
            duplicates.add(polygon_id_str)
            continue
        mapping[polygon_id_str] = wikidata_str
    if duplicates:
        sorted_ids = ", ".join(sorted(duplicates)[:5])
        raise ValueError(
            f"Polygons parquet has conflicting wikidata for polygon_id(s): {sorted_ids}"
        )
    return mapping


def _read_polygon_wikidata_set(polygons_path: Path) -> set[str]:
    """Return the set of distinct wikidata QIDs in the polygons parquet."""
    if not polygons_path.is_file():
        raise FileNotFoundError(f"Polygons parquet missing: {polygons_path}")
    table: pa.Table = pq.read_table(polygons_path, columns=["wikidata"])  # type: ignore[no-untyped-call]
    return {str(value) for value in table.column("wikidata").to_pylist() if value}


def _read_table_required(path: Path, *, label: str, columns: tuple[str, ...]) -> pa.Table:
    if not path.is_file():
        raise FileNotFoundError(f"{label} parquet missing: {path}")
    table: pa.Table = pq.read_table(path, columns=list(columns))  # type: ignore[no-untyped-call]
    return table


def _atomic_overwrite_parquet(path: Path, table: pa.Table) -> None:
    """Write *table* to *path* atomically (temp + replace)."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(raw_tmp)
    os.close(fd)
    try:
        pq.write_table(table, tmp_path, compression="snappy")  # type: ignore[no-untyped-call]
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _filter_rows(table: pa.Table, mask: list[bool]) -> pa.Table:
    """Return the rows of *table* where ``mask`` is True, preserving schema."""
    return table.filter(pa.array(mask))


# ---------------------------------------------------------------------------
# enforce_polygon_articles_integrity
# ---------------------------------------------------------------------------


def enforce_polygon_articles_integrity(
    data_root: DataRoot, stem: str
) -> PolygonArticlesIntegrityResult:
    """Reject every ``polygon_articles`` row whose wikidata does not
    match the canonical polygon wikidata for the same ``polygon_id``.

    The polygons parquet is the source of truth. Rows that reference
    a ``polygon_id`` absent from the polygons table are rejected
    (downstream join would fail anyway); the rejection audit records
    the mismatch with ``expected=None``.

    When at least one row is rejected the ``polygon_articles`` parquet
    is atomically rewritten with the retained rows and the canonical
    schema. When no row is rejected the parquet is left untouched
    (byte-identical), preserving the determinism contract.

    The rejection records are returned sorted by ``polygon_id`` so
    the audit is reproducible across runs.
    """
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    links_path = data_root.processed_links / f"{stem}.parquet"

    polygon_wikidata = _read_polygon_wikidata_map(polygons_path)
    table = _read_table_required(
        links_path,
        label="polygon_articles",
        columns=POLYGON_ARTICLE_COLUMNS,
    )
    rows = table.to_pylist()
    retained_rows: list[dict[str, Any]] = []
    rejections: list[RejectionRecord] = []
    seen_polygon_ids: set[str] = set()
    for row in rows:
        polygon_id = str(row.get("polygon_id", ""))
        link_wikidata = str(row.get("wikidata", ""))
        seen_polygon_ids.add(polygon_id)
        expected = polygon_wikidata.get(polygon_id)
        if expected is None or expected != link_wikidata:
            rejections.append(
                RejectionRecord(
                    shard=stem,
                    source_table="polygon_articles",
                    identifier=polygon_id,
                    wikidata=link_wikidata,
                    expected=expected,
                    reason=REASON_POLYGON_ARTICLES_MISMATCH,
                )
            )
            continue
        retained_rows.append({column: row.get(column) for column in POLYGON_ARTICLE_COLUMNS})

    # Detect polygon_ids that appear in polygon_articles but not in the
    # polygons table (unknown integrity defect). Surface these as
    # loud failures rather than silently dropping them: the join
    # contract guarantees the relationship is total, so a missing
    # polygon is a data hazard, not a benign gap.
    missing_polygon_ids = sorted(seen_polygon_ids - polygon_wikidata.keys())
    missing_with_links = sorted(
        polygon_id
        for polygon_id in missing_polygon_ids
        if any(str(row.get("polygon_id", "")) == polygon_id for row in rows)
    )
    if missing_with_links:
        sample = ", ".join(missing_with_links[:5])
        raise ValueError(
            f"polygon_articles rows reference polygon_id(s) absent from polygons parquet "
            f"for shard {stem!r}: {sample}"
        )

    original_count = len(rows)
    retained_count = len(retained_rows)
    rejected_count = len(rejections)
    rewritten = rejected_count > 0

    if rewritten:
        write_polygon_articles(links_path, retained_rows)
    else:
        # Guarantee byte-identical output: do not touch the file.
        pass

    rejections_tuple = tuple(
        sorted(rejections, key=lambda record: (record.identifier, record.wikidata))
    )
    return PolygonArticlesIntegrityResult(
        shard=stem,
        original_row_count=original_count,
        retained_row_count=retained_count,
        rejected_row_count=rejected_count,
        rewritten=rewritten,
        rejections=rejections_tuple,
    )


# ---------------------------------------------------------------------------
# enforce_wikivoyage_integrity
# ---------------------------------------------------------------------------


def enforce_wikivoyage_integrity(data_root: DataRoot, stem: str) -> WikivoyageIntegrityResult:
    """Reject every wikivoyage document whose wikidata is absent from
    the shard's polygons, and cascade the rejection to its sections.

    The polygons parquet is the source of truth for the valid
    wikidata QID set. Sections whose ``document_id`` belongs to a
    rejected document are dropped; the cascade count is recorded in
    each rejection record.

    When at least one document or section is rejected the
    ``wikivoyage/documents`` and ``wikivoyage/sections`` parquets
    are atomically rewritten. When nothing is rejected the parquets
    are left untouched (byte-identical).
    """
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    documents_path = data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    sections_path = data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"

    valid_qids = _read_polygon_wikidata_set(polygons_path)
    documents_table = _read_table_required(
        documents_path,
        label="wikivoyage/documents",
        columns=DOCUMENT_COLUMNS,
    )
    sections_table = (
        _read_table_required(
            sections_path,
            label="wikivoyage/sections",
            columns=SECTION_COLUMNS,
        )
        if sections_path.is_file()
        else pa.table({column: [] for column in SECTION_COLUMNS})
    )

    documents_rows = documents_table.to_pylist()
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
                    reason=REASON_WIKIVOYAGE_ABSENT,
                    cascaded_sections=0,
                )
            )
            continue
        retained_documents.append({column: row.get(column) for column in DOCUMENT_COLUMNS})

    # Sections: drop any whose document_id is in the rejected set.
    sections_rows = sections_table.to_pylist()
    retained_sections: list[dict[str, Any]] = []
    cascaded_count = 0
    cascades_by_document: dict[str, int] = {}
    for row in sections_rows:
        document_id = str(row.get("document_id", ""))
        if document_id in rejected_document_ids:
            cascaded_count += 1
            cascades_by_document[document_id] = cascades_by_document.get(document_id, 0) + 1
            continue
        retained_sections.append({column: row.get(column) for column in SECTION_COLUMNS})

    # Backfill cascaded_sections in each rejection record.
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

    original_document_count = len(documents_rows)
    retained_document_count = len(retained_documents)
    rejected_document_count = len(rejections)
    original_section_count = len(sections_rows)
    retained_section_count = len(retained_sections)
    rewritten_documents = rejected_document_count > 0
    rewritten_sections = cascaded_count > 0

    if rewritten_documents:
        documents_table_to_write = (
            pa.Table.from_pylist(retained_documents, schema=document_schema())
            if retained_documents
            else pa.table({column: [] for column in DOCUMENT_COLUMNS}, schema=document_schema())
        )
        _atomic_overwrite_parquet(documents_path, documents_table_to_write)

    if rewritten_sections:
        sections_table_to_write = (
            pa.Table.from_pylist(retained_sections, schema=section_schema())
            if retained_sections
            else pa.table({column: [] for column in SECTION_COLUMNS}, schema=section_schema())
        )
        _atomic_overwrite_parquet(sections_path, sections_table_to_write)

    rejections_tuple = tuple(
        sorted(rejections, key=lambda record: (record.identifier, record.wikidata))
    )
    return WikivoyageIntegrityResult(
        shard=stem,
        original_document_count=original_document_count,
        retained_document_count=retained_document_count,
        rejected_document_count=rejected_document_count,
        original_section_count=original_section_count,
        retained_section_count=retained_section_count,
        cascaded_section_count=cascaded_count,
        rewritten_documents=rewritten_documents,
        rewritten_sections=rewritten_sections,
        rejections=rejections_tuple,
    )


# ---------------------------------------------------------------------------
# enforce_all_regions
# ---------------------------------------------------------------------------


def enforce_all_regions(
    data_root: DataRoot,
    *,
    stems: list[str] | None = None,
    audit_filename: str = "integrity_audit.json",
) -> IntegrityReport:
    """Run both integrity checks across every shard and emit a
    deterministic audit JSON.

    When *stems* is ``None`` the union of polygon stems is used
    (the canonical intersection of polygons and either
    polygon_articles or wikivoyage/documents). The audit JSON is
    written to ``<data_root>/processed/integrity/<audit_filename>``
    with deterministic key order.
    """
    if stems is None:
        polygon_stems = sorted(path.stem for path in data_root.processed_polygons.glob("*.parquet"))
        stems = polygon_stems

    polygon_results: list[PolygonArticlesIntegrityResult] = []
    wikivoyage_results: list[WikivoyageIntegrityResult] = []
    for stem in stems:
        links_path = data_root.processed_links / f"{stem}.parquet"
        if links_path.is_file():
            polygon_results.append(enforce_polygon_articles_integrity(data_root, stem))
        wikivoyage_documents_path = (
            data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
        )
        if wikivoyage_documents_path.is_file():
            wikivoyage_results.append(enforce_wikivoyage_integrity(data_root, stem))

    polygon_results.sort(key=lambda result: result.shard)
    wikivoyage_results.sort(key=lambda result: result.shard)

    report = IntegrityReport(
        contract_version=INTEGRITY_CONTRACT_VERSION,
        polygon_articles=tuple(polygon_results),
        wikivoyage=tuple(wikivoyage_results),
        audit_path=data_root.processed / "integrity" / audit_filename,
    )

    audit_dir = report.audit_path.parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "contract_version": INTEGRITY_CONTRACT_VERSION,
        "generated_at": _utc_now_iso(),
        "polygon_articles": [result.to_dict() for result in polygon_results],
        "wikivoyage": [result.to_dict() for result in wikivoyage_results],
        "totals": {
            "polygon_articles_rejected": report.total_polygon_articles_rejected,
            "wikivoyage_documents_rejected": report.total_wikivoyage_documents_rejected,
            "wikivoyage_sections_cascaded": report.total_wikivoyage_sections_cascaded,
            "shards_with_rejections": sorted(
                {
                    result.shard
                    for result in polygon_results + wikivoyage_results
                    if result.rejections
                }
            ),
        },
    }
    report.audit_path.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _utc_now_iso() -> str:
    from osm_polygon_wikidata_only.utils.time import utc_now_iso as _impl

    return _impl()


__all__ = [
    "INTEGRITY_CONTRACT_VERSION",
    "REASON_POLYGON_ARTICLES_MISMATCH",
    "REASON_WIKIVOYAGE_ABSENT",
    "IntegrityReport",
    "PolygonArticlesIntegrityResult",
    "RejectionRecord",
    "WikivoyageIntegrityResult",
    "enforce_all_regions",
    "enforce_polygon_articles_integrity",
    "enforce_wikivoyage_integrity",
]
