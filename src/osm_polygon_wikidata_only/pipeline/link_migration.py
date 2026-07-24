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
import shutil
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
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
from osm_polygon_wikidata_only.utils.time import utc_now_iso as _utc_now_iso

_LINK_TRANSACTION_VERSION = "link-migration-transaction-v1"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class StemClassification(StrEnum):
    MIGRATABLE = "migratable"
    CANONICAL = "canonical"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class StemPlan:
    """Per-stem migration plan entry."""

    stem: str
    classification: StemClassification
    reason: str
    polygons_fingerprint: str
    links_fingerprint: str
    documents_fingerprint: str
    row_count: int
    canonical_digest: str | None


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """Immutable read-only migration plan."""

    processed_dir: Path
    stems: tuple[StemPlan, ...]

    @property
    def is_safe_to_apply(self) -> bool:
        return all(s.classification != StemClassification.BLOCKED for s in self.stems)


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


def _build_canonical_rows(
    stem: str,
    legacy_table: pa.Table,
    polygons_table: pa.Table,
    docs_table: pa.Table,
) -> list[dict[str, Any]]:
    """Build canonical-polygon-document-link rows from a legacy stem.

    Lossless conversion: every canonical row corresponds to exactly one
    distinct legacy ``(polygon_id, article_id)`` tuple. The polygon
    and document fields are joined **per legacy row**, never via a
    QID-wide Cartesian product across polygons x documents.

    Per-polygon QID membership: the resolved document's QID must belong
    to the SPECIFIC polygon's parsed multi-QID tag (not the region's
    union). Legacy row QID / page_id / revision_id / language fields
    are validated against the resolved document and polygon.

    Duplicate legacy rows for the same ``(polygon_id, article_id)``:
    byte-identical rows collapse to one; conflicting rows
    (different QID, page_id, revision_id, or language) block the
    stem.
    """
    polygons_by_id: dict[str, dict[str, Any]] = {
        str(row["polygon_id"]): row for row in polygons_table.to_pylist()
    }
    docs_by_article_id: dict[str, list[dict[str, Any]]] = {}
    for doc in docs_table.to_pylist():
        docs_by_article_id.setdefault(str(doc["article_id"]), []).append(doc)

    # Pre-compute each polygon's resolved QID set.
    polygon_qids: dict[str, set[str]] = {}
    for pid, prow in polygons_by_id.items():
        polygon_qids[pid] = set(_qids_from_osm_tag(str(prow.get("wikidata", ""))))

    legacy_rows = legacy_table.to_pylist()
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    seen_legacy: dict[tuple[str, str], dict[str, Any]] = {}
    for legacy_row in legacy_rows:
        polygon_id = str(legacy_row.get("polygon_id", ""))
        article_id = str(legacy_row.get("article_id", ""))
        legacy_wikidata = str(legacy_row.get("wikidata", ""))
        legacy_page_id = int(legacy_row.get("page_id", 0))
        legacy_revision_id = int(legacy_row.get("revision_id", 0))
        legacy_language = str(legacy_row.get("language", ""))
        identity = (polygon_id, article_id)
        normalized_legacy = {column: legacy_row.get(column) for column in POLYGON_ARTICLE_COLUMNS}
        if identity in seen_legacy:
            if seen_legacy[identity] != normalized_legacy:
                raise ValueError(
                    f"conflicting duplicate legacy rows for (polygon_id={polygon_id!r}, "
                    f"article_id={article_id!r}); cannot collapse"
                )
            continue
        seen_legacy[identity] = normalized_legacy

        if polygon_id not in polygons_by_id:
            raise ValueError(
                f"legacy polygon_id={polygon_id!r} is not present in polygons/{stem}.parquet"
            )
        matching_docs = docs_by_article_id.get(article_id)
        if not matching_docs:
            raise ValueError(
                f"legacy article_id={article_id!r} for polygon_id={polygon_id!r} "
                f"has no matching wikipedia/documents/{stem}.parquet row"
            )
        if len(matching_docs) > 1:
            raise ValueError(
                f"legacy article_id={article_id!r} for polygon_id={polygon_id!r} "
                f"is ambiguous: {len(matching_docs)} matching documents"
            )
        doc = matching_docs[0]

        # Per-polygon QID membership: the document's QID must be in
        # this specific polygon's resolved QID set.
        doc_qid = str(doc["wikidata"])
        polygon_set = polygon_qids.get(polygon_id, set())
        if doc_qid and doc_qid not in polygon_set:
            # Reject-only normalization: the invalid relationship is
            # recorded by
            # ``plan_link_migration_normalization_rejections_for_stem``
            # and omitted from the canonical table. Never rewrite a
            # QID to make the relationship appear valid.
            continue

        # Validate legacy row fields against the resolved document.
        if legacy_wikidata and legacy_wikidata != doc_qid:
            raise ValueError(
                f"legacy row (polygon_id={polygon_id!r}, article_id={article_id!r}) "
                f"wikidata={legacy_wikidata!r} conflicts with document wikidata={doc_qid!r}"
            )
        if legacy_page_id and legacy_page_id != int(doc.get("page_id", 0)):
            raise ValueError(
                f"legacy row (polygon_id={polygon_id!r}, article_id={article_id!r}) "
                f"page_id={legacy_page_id} conflicts with document page_id="
                f"{int(doc.get('page_id', 0))}"
            )
        if legacy_revision_id and legacy_revision_id != int(doc.get("revision_id", 0)):
            raise ValueError(
                f"legacy row (polygon_id={polygon_id!r}, article_id={article_id!r}) "
                f"revision_id={legacy_revision_id} conflicts with document revision_id="
                f"{int(doc.get('revision_id', 0))}"
            )
        if legacy_language and legacy_language != str(doc.get("language", "")):
            raise ValueError(
                f"legacy row (polygon_id={polygon_id!r}, article_id={article_id!r}) "
                f"language={legacy_language!r} conflicts with document language="
                f"{str(doc.get('language', ''))!r}"
            )

        canonical_row = {
            "polygon_id": polygon_id,
            "document_id": str(doc["document_id"]),
            "project": "wikipedia",
            "wikidata": doc_qid,
            "language": str(doc.get("language", "")),
            "source_pbf": str(polygons_by_id[polygon_id].get("source_pbf", "")),
            "region": str(polygons_by_id[polygon_id].get("region", "")),
            "osm_type": str(polygons_by_id[polygon_id].get("osm_type", "")),
            "osm_id": int(polygons_by_id[polygon_id].get("osm_id", 0)),
            "page_id": int(doc.get("page_id", 0)),
            "revision_id": int(doc.get("revision_id", 0)),
        }
        seen[identity] = canonical_row
    return validate_polygon_document_links(seen.values())


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
# Ordered journaled transaction primitive (private)
# ---------------------------------------------------------------------------


@dataclass
class _TransactionEntry:
    target: Path
    staged: Path
    backup: Path | None
    existed: bool
    original_hash: str
    staged_hash: str


def _commit_ordered_replacements(
    directory: Path,
    stem: str,
    replacements: list[tuple[Path, Path]],
    *,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Public entry: ordered journaled replacement (private to the module).

    Manifests come last -- any ``*.json`` target is moved to the end of
    the replacement list, regardless of input order. The journal is
    stamped at every phase. On exception, the helper rolls back and
    cleans up.
    """
    if not replacements:
        return
    targets = [target for target, _ in replacements]
    if len(set(targets)) != len(targets):
        raise ValueError("Link migration transaction contains duplicate targets")

    # Sort: non-JSON replacements first, JSON manifest last.
    def _kind(p: Path) -> int:
        return 1 if p.suffix == ".json" else 0

    ordered = sorted(replacements, key=lambda item: (_kind(item[0]), str(item[0])))

    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "journal.json"
    if journal_path.exists():
        # Recovery: replay the journal. For each entry, if the target
        # is already at the staged hash, the entry was already applied
        # before the crash; otherwise, apply it now.
        _recover_directory(directory, stem)
        return

    entries: list[_TransactionEntry] = []
    for target, staged in ordered:
        if not staged.is_file():
            raise FileNotFoundError(f"Staged link migration file is missing: {staged}")
        backup = directory / f"{target.name}.backup"
        existed = target.is_file()
        original_hash = ""
        if existed:
            shutil.copyfile(target, backup)
            original_hash = _file_content_hash(target)
        entries.append(
            _TransactionEntry(
                target=target,
                staged=staged,
                backup=backup if existed else None,
                existed=existed,
                original_hash=original_hash,
                staged_hash=_file_content_hash(staged),
            )
        )

    journal: dict[str, Any] = {
        "contract_version": _LINK_TRANSACTION_VERSION,
        "stem": stem,
        "phase": "prepared",
        "entries": [
            {
                "target": str(entry.target),
                "staged": str(entry.staged),
                "backup": str(entry.backup) if entry.backup else "",
                "existed": entry.existed,
                "original_hash": entry.original_hash,
                "staged_hash": entry.staged_hash,
            }
            for entry in entries
        ],
    }
    _atomic_write_json(journal_path, journal)

    try:
        for index, entry in enumerate(entries):
            _apply_single(entry)
            if _crash_hook is not None:
                _crash_hook(index, entry.target)
            # Post-hook verification: the hook may have corrupted the
            # target. If so, roll back this entry and re-raise.
            if _file_content_hash(entry.target) != entry.staged_hash:
                raise RuntimeError(f"Link migration post-hook hash mismatch for {entry.target}")
        journal["phase"] = "committed"
        _atomic_write_json(journal_path, journal)
    except BaseException:
        # Distinguish two cases:
        # 1. Validation-time failure (raise on index 0, before any
        #    progress): full rollback; no partial state preserved.
        # 2. Mid-flight crash (raise on index > 0): roll forward;
        #    leave the journal AND every staged file so a subsequent
        #    call can replay. We must NOT delete staged files here --
        #    roll-forward recovery needs them on disk.
        if index == 0:
            _rollback_entries(entries)
            journal["phase"] = "rolled_back"
            _atomic_write_json(journal_path, journal)
            _cleanup(directory)
        else:
            journal["phase"] = "interrupted"
            journal["interrupted_at_index"] = int(index)
            _atomic_write_json(journal_path, journal)
            # Intentionally do NOT unlink staged files. The next call
            # to ``_commit_ordered_replacements`` will see the journal
            # and replay from where the crash interrupted.
        raise

    _cleanup(directory)


def _apply_single(entry: _TransactionEntry) -> None:
    target = entry.target
    staged = entry.staged
    staged_hash = entry.staged_hash
    if target.is_file() and _file_content_hash(target) == staged_hash:
        return
    if not staged.is_file() or _file_content_hash(staged) != staged_hash:
        raise RuntimeError(f"Link migration staged file is unavailable: {staged}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        os.replace(staged, target)
    else:
        shutil.move(str(staged), str(target))
    if _file_content_hash(target) != staged_hash:
        raise RuntimeError(f"Link migration verification failed: {target}")


def _rollback_entries(entries: list[_TransactionEntry]) -> None:
    for entry in reversed(entries):
        if entry.existed and entry.backup is not None:
            if not entry.backup.is_file():
                raise RuntimeError(f"Link migration backup is unavailable: {entry.backup}")
            os.replace(entry.backup, entry.target)
            if _file_content_hash(entry.target) != entry.original_hash:
                raise RuntimeError(f"Link migration rollback verification failed: {entry.target}")
        else:
            if entry.target.is_file():
                entry.target.unlink()


def _cleanup(directory: Path) -> None:
    for entry in directory.iterdir():
        if entry.is_file():
            entry.unlink()
    with suppress(OSError):
        directory.rmdir()


def _recover_directory(directory: Path, stem: str) -> None:
    journal_path = directory / "journal.json"
    if not journal_path.is_file():
        return
    raw = json.loads(journal_path.read_text(encoding="utf-8"))
    if raw.get("contract_version") != _LINK_TRANSACTION_VERSION:
        raise RuntimeError(f"Invalid link migration journal: {journal_path}")
    if raw.get("stem") != stem:
        raise RuntimeError(f"Link migration journal stem mismatch: {raw.get('stem')!r} vs {stem!r}")
    entries = raw.get("entries", [])
    for entry in entries:
        target = Path(entry["target"])
        staged = Path(entry["staged"])
        backup = Path(entry["backup"]) if entry.get("backup") else None
        existed = bool(entry["existed"])
        staged_hash = entry["staged_hash"]
        if target.is_file() and _file_content_hash(target) == staged_hash:
            # Already applied; verify and continue.
            continue
        if not staged.is_file() or _file_content_hash(staged) != staged_hash:
            raise RuntimeError(f"Link migration recovery: staged file unavailable: {staged}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if existed and backup is not None and backup.is_file():
            backup.unlink()
        if target.is_file():
            os.replace(staged, target)
        else:
            shutil.move(str(staged), str(target))
        if _file_content_hash(target) != staged_hash:
            raise RuntimeError(f"Link migration recovery verification failed: {target}")
    _cleanup(directory)


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
