"""Safe, offline, lossless Wikipedia-document backfill engine.

Two-stage migration for upgrading legacy ``articles/`` data to the canonical
32-column ``wikipedia/documents/`` format:

1. :func:`plan_migration` — read-only deterministic planning and preflight.
2. :func:`apply_migration` — explicit apply stage with atomic writes.

Only ``wikipedia/documents/<stem>.parquet`` is ever written.
``articles/``, ``wikipedia/sections/``, ``polygons/``,
``polygon_articles/``, manifests, caches, README, and all other sidecars
are never modified.

The engine performs no network calls and never imports from ``tests/``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import document_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WikipediaDocumentConversionError,
    build_wikipedia_document_table,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import article_schema

__all__ = [
    "ApplyResult",
    "MigrationError",
    "MigrationOperation",
    "MigrationPlan",
    "StemPlan",
    "apply_migration",
    "plan_migration",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MigrationOperation(StrEnum):
    """Classification of what action a stem requires."""

    CREATE_MISSING = "create_missing"
    UPGRADE_LEGACY = "upgrade_legacy"
    ALREADY_CANONICAL = "already_canonical"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class StemPlan:
    """Per-stem migration plan entry.

    Attributes
    ----------
    stem:
        Article stem name (filename without ``.parquet``).
    operation:
        What action this stem requires.
    reason:
        Empty for non-blocked operations. Descriptive error message for
        blocked stems, naming the problem without leaking absolute paths.
    article_hash:
        SHA-256 content hash of the article file at planning time.
        Empty string when no article file was found.
    document_hash:
        SHA-256 content hash of the document file at planning time.
        ``None`` when no document file existed.
    row_count:
        Number of canonical document rows, or zero for blocked stems.
    canonical_digest:
        Deterministic digest of the canonical schema and record batches.
        ``None`` for blocked stems. No dataset rows are retained in the plan.
    """

    stem: str
    operation: MigrationOperation
    reason: str
    article_hash: str
    document_hash: str | None
    row_count: int
    canonical_digest: str | None


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """Immutable, validated migration plan.

    Built by :func:`plan_migration`. Passed to :func:`apply_migration`.
    """

    processed_dir: Path
    stems: tuple[StemPlan, ...]

    @property
    def is_safe_to_apply(self) -> bool:
        """True when no stems are blocked."""
        return all(s.operation != MigrationOperation.BLOCKED for s in self.stems)

    @property
    def blocked_stems(self) -> tuple[str, ...]:
        """Stems classified as blocked."""
        return tuple(s.stem for s in self.stems if s.operation == MigrationOperation.BLOCKED)


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Deterministic result of applying a migration plan."""

    planned: int
    created: int
    upgraded: int
    skipped: int
    blocked: int
    created_stems: tuple[str, ...]
    upgraded_stems: tuple[str, ...]
    skipped_stems: tuple[str, ...]
    blocked_stems: tuple[str, ...]


class MigrationError(Exception):
    """Raised when migration planning or application fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _file_content_hash(path: Path) -> str:
    """Compute SHA-256 of file bytes."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _table_digest(table: pa.Table) -> str:
    """Digest a table deterministically without retaining serialized rows."""
    hasher = hashlib.sha256(table.schema.serialize().to_pybytes())
    for batch in table.to_batches(max_chunksize=65_536):
        hasher.update(batch.serialize().to_pybytes())
    return hasher.hexdigest()


def _blocked_plan(
    stem: str,
    reason: str,
    *,
    article_hash: str = "",
    document_hash: str | None = None,
) -> StemPlan:
    return StemPlan(
        stem=stem,
        operation=MigrationOperation.BLOCKED,
        reason=reason,
        article_hash=article_hash,
        document_hash=document_hash,
        row_count=0,
        canonical_digest=None,
    )


def _ready_plan(
    stem: str,
    operation: MigrationOperation,
    canonical_table: pa.Table,
    *,
    article_hash: str,
    document_hash: str | None,
) -> StemPlan:
    return StemPlan(
        stem=stem,
        operation=operation,
        reason="",
        article_hash=article_hash,
        document_hash=document_hash,
        row_count=canonical_table.num_rows,
        canonical_digest=_table_digest(canonical_table),
    )


def _discover_all_stems(processed_dir: Path) -> list[str]:
    """Discover the deterministic union of article and document stems."""
    articles_dir = processed_dir / "articles"
    docs_dir = processed_dir / "wikipedia" / "documents"
    stems: set[str] = set()
    if articles_dir.is_dir():
        stems.update(p.stem for p in articles_dir.glob("*.parquet"))
    if docs_dir.is_dir():
        stems.update(p.stem for p in docs_dir.glob("*.parquet"))
    return sorted(stems)


def _read_article_table(path: Path, stem: str) -> pa.Table:
    """Read and strictly validate an article parquet file."""
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise MigrationError(
            f"Stem '{stem}': unreadable article file ({type(exc).__name__})"
        ) from exc

    expected = article_schema()
    if table.schema != expected:
        raise MigrationError(f"Stem '{stem}': article schema does not match article_schema()")
    return table


def _check_shared_values(
    legacy_table: pa.Table,
    canonical_table: pa.Table,
    stem: str,
) -> None:
    """Verify all shared column values match, keyed by document_id.

    Raises MigrationError on row-count mismatch, identity set mismatch,
    duplicate identities, or any shared-value conflict.
    """
    legacy_ids: list[str] = legacy_table.column("document_id").to_pylist()
    canonical_ids: list[str] = canonical_table.column("document_id").to_pylist()

    # Row count check
    if len(legacy_ids) != len(canonical_ids):
        raise MigrationError(
            f"Stem '{stem}': row count mismatch "
            f"(document has {len(legacy_ids)}, canonical has {len(canonical_ids)})"
        )

    legacy_id_set = set(legacy_ids)
    canonical_id_set = set(canonical_ids)

    # Duplicate document_id in existing document
    if len(legacy_id_set) != len(legacy_ids):
        raise MigrationError(f"Stem '{stem}': duplicate document_id in existing document")

    # Identity set mismatch
    if legacy_id_set != canonical_id_set:
        diff = sorted(legacy_id_set ^ canonical_id_set)
        raise MigrationError(
            f"Stem '{stem}': document_id set mismatch (symmetric difference: {diff})"
        )

    # Shared columns excluding document_id (the join key)
    legacy_cols = set(legacy_table.schema.names)
    canonical_cols = set(canonical_table.schema.names)
    shared_cols = sorted((legacy_cols & canonical_cols) - {"document_id"})

    # Build lookup dictionaries keyed by document_id
    select_cols = [*shared_cols, "document_id"]
    legacy_rows_list = legacy_table.select(select_cols).to_pylist()
    canonical_rows_list = canonical_table.select(select_cols).to_pylist()

    legacy_by_id: dict[str, dict[str, Any]] = {r["document_id"]: r for r in legacy_rows_list}
    canonical_by_id: dict[str, dict[str, Any]] = {r["document_id"]: r for r in canonical_rows_list}

    for doc_id in sorted(canonical_id_set):
        l_row = legacy_by_id[doc_id]
        c_row = canonical_by_id[doc_id]
        for col in shared_cols:
            lv = l_row[col]
            cv = c_row[col]
            if lv != cv:
                raise MigrationError(
                    f"Stem '{stem}': shared-value conflict for document_id "
                    f"'{doc_id}' in column '{col}'"
                )


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    """Write a Parquet file atomically via temp file and os.replace."""
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


def _validate_stem_path(stem: str, docs_dir: Path) -> Path:
    """Validate a stem name and return the safe target path.

    Rejects empty stems, path separators, ``..``, and any stem whose
    resolved target escapes the documents directory.
    """
    if not stem or stem in (".", ".."):
        raise MigrationError(f"Invalid stem name: '{stem}'")
    if "/" in stem or "\\" in stem:
        raise MigrationError(f"Stem '{stem}': must not contain path separators")

    target = docs_dir / f"{stem}.parquet"
    resolved_target = target.resolve()
    resolved_docs = docs_dir.resolve()

    try:
        resolved_target.relative_to(resolved_docs)
    except ValueError:
        raise MigrationError(f"Stem '{stem}': target path escapes documents directory") from None

    return target


def _classify_stem(stem: str, processed_dir: Path) -> StemPlan:
    """Classify a single stem and build its plan entry."""
    article_path = processed_dir / "articles" / f"{stem}.parquet"
    doc_path = processed_dir / "wikipedia" / "documents" / f"{stem}.parquet"

    has_article = article_path.is_file()
    has_doc = doc_path.is_file()

    # Document without article → BLOCKED
    if has_doc and not has_article:
        try:
            doc_hash = _file_content_hash(doc_path)
        except OSError as exc:
            return _blocked_plan(stem, f"unreadable document file ({type(exc).__name__})")
        return _blocked_plan(
            stem,
            "document exists without corresponding article",
            document_hash=doc_hash,
        )

    # No article and no document (shouldn't happen with union discovery)
    if not has_article:
        return _blocked_plan(stem, "no article file found")

    # Read and validate article
    try:
        article_table = _read_article_table(article_path, stem)
    except MigrationError as exc:
        return _blocked_plan(stem, str(exc))

    try:
        article_hash = _file_content_hash(article_path)
    except OSError as exc:
        return _blocked_plan(stem, f"unreadable article file ({type(exc).__name__})")

    # Build canonical table from validated article
    try:
        canonical_table = build_wikipedia_document_table(article_table)
    except WikipediaDocumentConversionError as exc:
        return _blocked_plan(
            stem,
            f"article conversion failed: {exc}",
            article_hash=article_hash,
        )

    # No existing document → CREATE_MISSING
    if not has_doc:
        return _ready_plan(
            stem,
            MigrationOperation.CREATE_MISSING,
            canonical_table,
            article_hash=article_hash,
            document_hash=None,
        )

    # Read existing document
    try:
        doc_hash = _file_content_hash(doc_path)
    except OSError as exc:
        return _blocked_plan(
            stem,
            f"unreadable document file ({type(exc).__name__})",
            article_hash=article_hash,
        )
    try:
        doc_table = pq.read_table(doc_path)  # type: ignore[no-untyped-call]
    except Exception as exc:
        return _blocked_plan(
            stem,
            f"unreadable document file ({type(exc).__name__})",
            article_hash=article_hash,
            document_hash=doc_hash,
        )

    canonical_schema = wikipedia_document_schema()
    legacy_schema = document_schema()

    # Check exact canonical schema (names, types, order, metadata)
    if doc_table.schema.equals(canonical_schema, check_metadata=True):
        if doc_table.equals(canonical_table, check_metadata=True):
            return _ready_plan(
                stem,
                MigrationOperation.ALREADY_CANONICAL,
                canonical_table,
                article_hash=article_hash,
                document_hash=doc_hash,
            )
        return _blocked_plan(
            stem,
            "document has canonical schema but content differs",
            article_hash=article_hash,
            document_hash=doc_hash,
        )

    # Check exact legacy schema (names, types, order, metadata)
    if doc_table.schema.equals(legacy_schema, check_metadata=True):
        try:
            _check_shared_values(doc_table, canonical_table, stem)
        except MigrationError as exc:
            return _blocked_plan(
                stem,
                str(exc),
                article_hash=article_hash,
                document_hash=doc_hash,
            )
        return _ready_plan(
            stem,
            MigrationOperation.UPGRADE_LEGACY,
            canonical_table,
            article_hash=article_hash,
            document_hash=doc_hash,
        )

    # Unexpected schema
    return _blocked_plan(
        stem,
        f"unexpected document schema ({len(doc_table.schema)} columns)",
        article_hash=article_hash,
        document_hash=doc_hash,
    )


# ---------------------------------------------------------------------------
# Apply-time revalidation
# ---------------------------------------------------------------------------

_APPLY_WRITE = "write"
_APPLY_SKIP = "skip"


def _validate_documents_root(processed_dir: Path, docs_dir: Path) -> None:
    """Reject a documents directory resolving outside the processed root."""
    try:
        docs_dir.resolve().relative_to(processed_dir.resolve())
    except ValueError:
        raise MigrationError("Wikipedia documents directory escapes processed directory") from None


def _transition_action(planned: StemPlan, current: StemPlan) -> str:
    """Return the safe action for a freshly re-planned stem."""
    if planned == current:
        if current.operation == MigrationOperation.ALREADY_CANONICAL:
            return _APPLY_SKIP
        return _APPLY_WRITE

    became_canonical = (
        planned.operation in {MigrationOperation.CREATE_MISSING, MigrationOperation.UPGRADE_LEGACY}
        and current.operation == MigrationOperation.ALREADY_CANONICAL
        and planned.article_hash == current.article_hash
        and planned.row_count == current.row_count
        and planned.canonical_digest == current.canonical_digest
    )
    if became_canonical:
        return _APPLY_SKIP

    if planned.article_hash != current.article_hash:
        raise MigrationError(f"Stem '{planned.stem}': article file changed after planning")
    if current.operation == MigrationOperation.BLOCKED and "unreadable" in current.reason:
        raise MigrationError(f"Stem '{planned.stem}': {current.reason}")
    if planned.operation == MigrationOperation.CREATE_MISSING and current.document_hash is not None:
        raise MigrationError(f"Stem '{planned.stem}': conflicting target appeared after planning")
    if planned.document_hash != current.document_hash:
        raise MigrationError(f"Stem '{planned.stem}': document file changed after planning")

    raise MigrationError(f"Stem '{planned.stem}': migration plan changed after validation")


def _rebuild_table_for_write(sp: StemPlan, processed_dir: Path, target: Path) -> pa.Table:
    """Rebuild and verify one canonical table immediately before replacement."""
    article_path = processed_dir / "articles" / f"{sp.stem}.parquet"
    try:
        if _file_content_hash(article_path) != sp.article_hash:
            raise MigrationError(f"Stem '{sp.stem}': article file changed before write")
    except OSError as exc:
        raise MigrationError(
            f"Stem '{sp.stem}': article file unreadable before write ({type(exc).__name__})"
        ) from exc

    if sp.operation == MigrationOperation.CREATE_MISSING:
        if target.exists():
            raise MigrationError(f"Stem '{sp.stem}': target appeared before write")
    elif sp.operation == MigrationOperation.UPGRADE_LEGACY:
        if not target.is_file():
            raise MigrationError(f"Stem '{sp.stem}': document disappeared before write")
        try:
            current_hash = _file_content_hash(target)
        except OSError as exc:
            raise MigrationError(
                f"Stem '{sp.stem}': document unreadable before write ({type(exc).__name__})"
            ) from exc
        if current_hash != sp.document_hash:
            raise MigrationError(f"Stem '{sp.stem}': document changed before write")

    table = build_wikipedia_document_table(_read_article_table(article_path, sp.stem))
    if table.num_rows != sp.row_count or _table_digest(table) != sp.canonical_digest:
        raise MigrationError(f"Stem '{sp.stem}': canonical output changed before write")
    return table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_migration(processed_dir: Path) -> MigrationPlan:
    """Read-only planning stage.

    Discovers the deterministic union of article and Wikipedia-document stems,
    classifies each, validates data, and builds canonical tables.
    Makes no filesystem modifications.

    A document stem lacking its required article source produces a BLOCKED
    entry with a clear reason.

    Parameters
    ----------
    processed_dir:
        Path to the ``processed/`` directory containing ``articles/``,
        ``wikipedia/documents/``, and other dataset tables.

    Returns
    -------
    MigrationPlan
        Immutable, validated plan with per-stem classifications.
    """
    stems_data: list[StemPlan] = []
    for stem in _discover_all_stems(processed_dir):
        stems_data.append(_classify_stem(stem, processed_dir))

    return MigrationPlan(
        processed_dir=processed_dir,
        stems=tuple(stems_data),
    )


def apply_migration(plan: MigrationPlan) -> ApplyResult:
    """Apply stage.

    Accepts a validated immutable plan and writes only
    ``wikipedia/documents/<stem>.parquet`` using atomic writes.

    Before writing any stem, the apply stage performs a complete read-only
    revalidation of every stem against current filesystem state.  If any
    stem's article or document file has changed since planning, the entire
    apply aborts with zero writes.

    Parameters
    ----------
    plan:
        A validated :class:`MigrationPlan` from :func:`plan_migration`.

    Returns
    -------
    ApplyResult
        Typed deterministic result with counts and affected stems.

    Raises
    ------
    MigrationError
        If the plan contains blocked stems or any stem fails revalidation.
    """
    if not plan.is_safe_to_apply:
        blocked = list(plan.blocked_stems)
        raise MigrationError(
            f"Plan is not safe to apply: {len(blocked)} blocked stem(s): {blocked}"
        )

    processed_dir = plan.processed_dir
    docs_dir = processed_dir / "wikipedia" / "documents"
    _validate_documents_root(processed_dir, docs_dir)

    # Phase 1: Validate all stem paths before any I/O
    safe_targets: dict[str, Path] = {}
    for sp in plan.stems:
        safe_targets[sp.stem] = _validate_stem_path(sp.stem, docs_dir)

    # Phase 2: Re-plan all stems before any writes (zero writes on failure).
    # The caller-supplied plan contains metadata only and is never a source
    # of rows written to disk.
    current_plan = plan_migration(processed_dir)
    if tuple(sp.stem for sp in current_plan.stems) != tuple(sp.stem for sp in plan.stems):
        raise MigrationError("Migration plan stem set changed after validation")

    actions: list[tuple[str, str]] = []
    for planned, current in zip(plan.stems, current_plan.stems, strict=True):
        action = _transition_action(planned, current)
        actions.append((current.stem, action))

    # Phase 3: Execute writes
    created_stems: list[str] = []
    upgraded_stems: list[str] = []
    skipped_stems: list[str] = []

    for sp, (_stem, action) in zip(current_plan.stems, actions, strict=True):
        if action == _APPLY_SKIP:
            skipped_stems.append(sp.stem)
            continue

        canonical_table = _rebuild_table_for_write(sp, processed_dir, safe_targets[sp.stem])
        _atomic_write_parquet(safe_targets[sp.stem], canonical_table)

        if sp.operation == MigrationOperation.CREATE_MISSING:
            created_stems.append(sp.stem)
        else:
            upgraded_stems.append(sp.stem)

    return ApplyResult(
        planned=len(plan.stems),
        created=len(created_stems),
        upgraded=len(upgraded_stems),
        skipped=len(skipped_stems),
        blocked=0,
        created_stems=tuple(created_stems),
        upgraded_stems=tuple(upgraded_stems),
        skipped_stems=tuple(skipped_stems),
        blocked_stems=(),
    )
