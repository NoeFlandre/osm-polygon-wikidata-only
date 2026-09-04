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
from osm_polygon_wikidata_only.io.atomic import atomic_write_parquet

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
    shared_cols, legacy_by_id, canonical_by_id = _shared_comparison_inputs(
        legacy_table, canonical_table, stem
    )
    for doc_id in sorted(canonical_by_id):
        _validate_shared_row(
            stem,
            doc_id,
            shared_cols,
            legacy_by_id[doc_id],
            canonical_by_id[doc_id],
        )


def _shared_comparison_inputs(
    legacy_table: pa.Table,
    canonical_table: pa.Table,
    stem: str,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate shared identities and build row lookups for comparison."""
    legacy_ids: list[str] = legacy_table.column("document_id").to_pylist()
    canonical_ids: list[str] = canonical_table.column("document_id").to_pylist()
    _validate_shared_identities(legacy_ids, canonical_ids, stem)
    shared_cols = sorted(
        (set(legacy_table.schema.names) & set(canonical_table.schema.names)) - {"document_id"}
    )
    select_cols = [*shared_cols, "document_id"]
    legacy_rows = legacy_table.select(select_cols).to_pylist()
    canonical_rows = canonical_table.select(select_cols).to_pylist()
    return (
        shared_cols,
        {row["document_id"]: row for row in legacy_rows},
        {row["document_id"]: row for row in canonical_rows},
    )


def _validate_shared_identities(
    legacy_ids: list[str],
    canonical_ids: list[str],
    stem: str,
) -> None:
    """Validate row counts, uniqueness, and document identity sets."""
    if len(legacy_ids) != len(canonical_ids):
        raise MigrationError(
            f"Stem '{stem}': row count mismatch "
            f"(document has {len(legacy_ids)}, canonical has {len(canonical_ids)})"
        )
    legacy_id_set = set(legacy_ids)
    canonical_id_set = set(canonical_ids)
    if len(legacy_id_set) != len(legacy_ids):
        raise MigrationError(f"Stem '{stem}': duplicate document_id in existing document")
    if legacy_id_set != canonical_id_set:
        diff = sorted(legacy_id_set ^ canonical_id_set)
        raise MigrationError(
            f"Stem '{stem}': document_id set mismatch (symmetric difference: {diff})"
        )


def _validate_shared_row(
    stem: str,
    document_id: str,
    shared_cols: list[str],
    legacy_row: dict[str, Any],
    canonical_row: dict[str, Any],
) -> None:
    """Reject the first shared-column value conflict for one document."""
    for column in shared_cols:
        if legacy_row[column] != canonical_row[column]:
            raise MigrationError(
                f"Stem '{stem}': shared-value conflict for document_id "
                f"'{document_id}' in column '{column}'"
            )


def _assert_canonical_preserves_legacy(
    canonical_table: pa.Table,
    legacy_canonical_table: pa.Table,
    stem: str,
) -> None:
    """Require every converted legacy row to exist unchanged.

    The canonical table may contain additional documents discovered
    after the legacy article table was written.
    """
    canonical_rows = canonical_table.to_pylist()
    canonical_by_id = {str(row["document_id"]): row for row in canonical_rows}
    if len(canonical_by_id) != len(canonical_rows):
        raise MigrationError(f"Stem '{stem}': duplicate document_id in canonical document")
    for expected_row in legacy_canonical_table.to_pylist():
        document_id = str(expected_row["document_id"])
        if canonical_by_id.get(document_id) != expected_row:
            raise MigrationError(
                f"Stem '{stem}': canonical documents do not preserve "
                f"legacy document '{document_id}'"
            )


def _validate_stem_path(stem: str, docs_dir: Path) -> Path:
    """Validate a stem name and return the safe target path.

    Rejects empty stems, path separators, ``..``, and any stem whose
    resolved target escapes the documents directory.
    """
    _validate_stem_name(stem)
    target = docs_dir / f"{stem}.parquet"
    _validate_target_inside_docs(stem, target, docs_dir)
    return target


def _validate_stem_name(stem: str) -> None:
    """Reject empty, parent, or separator-containing stem names."""
    if not stem or stem in (".", ".."):
        raise MigrationError(f"Invalid stem name: '{stem}'")
    if "/" in stem or "\\" in stem:
        raise MigrationError(f"Stem '{stem}': must not contain path separators")


def _validate_target_inside_docs(stem: str, target: Path, docs_dir: Path) -> None:
    """Reject a resolved target that escapes the documents directory."""
    resolved_target = target.resolve()
    resolved_docs = docs_dir.resolve()
    try:
        resolved_target.relative_to(resolved_docs)
    except ValueError:
        raise MigrationError(f"Stem '{stem}': target path escapes documents directory") from None


def _missing_stem_plan(
    stem: str,
    article_path: Path,
    doc_path: Path,
) -> StemPlan | None:
    """Classify stems whose source or target file is absent."""
    has_article = article_path.is_file()
    has_doc = doc_path.is_file()
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
    if not has_article:
        return _blocked_plan(stem, "no article file found")
    return None


def _article_plan_inputs(
    stem: str,
    article_path: Path,
) -> tuple[str, pa.Table] | StemPlan:
    """Read, hash, and convert one article source for planning."""
    try:
        article_table = _read_article_table(article_path, stem)
    except MigrationError as exc:
        return _blocked_plan(stem, str(exc))
    try:
        article_hash = _file_content_hash(article_path)
    except OSError as exc:
        return _blocked_plan(stem, f"unreadable article file ({type(exc).__name__})")
    try:
        canonical_table = build_wikipedia_document_table(article_table)
    except WikipediaDocumentConversionError as exc:
        return _blocked_plan(
            stem,
            f"article conversion failed: {exc}",
            article_hash=article_hash,
        )
    return article_hash, canonical_table


def _canonical_document_plan(
    stem: str,
    document_table: pa.Table,
    canonical_table: pa.Table,
    *,
    article_hash: str,
    document_hash: str,
) -> StemPlan:
    """Classify an existing canonical document after preservation checks."""
    try:
        _assert_canonical_preserves_legacy(document_table, canonical_table, stem)
    except MigrationError as exc:
        return _blocked_plan(
            stem,
            str(exc),
            article_hash=article_hash,
            document_hash=document_hash,
        )
    return _ready_plan(
        stem,
        MigrationOperation.ALREADY_CANONICAL,
        document_table,
        article_hash=article_hash,
        document_hash=document_hash,
    )


def _legacy_document_plan(
    stem: str,
    document_table: pa.Table,
    canonical_table: pa.Table,
    *,
    article_hash: str,
    document_hash: str,
) -> StemPlan:
    """Classify an existing legacy document after shared-value checks."""
    try:
        _check_shared_values(document_table, canonical_table, stem)
    except MigrationError as exc:
        return _blocked_plan(
            stem,
            str(exc),
            article_hash=article_hash,
            document_hash=document_hash,
        )
    return _ready_plan(
        stem,
        MigrationOperation.UPGRADE_LEGACY,
        canonical_table,
        article_hash=article_hash,
        document_hash=document_hash,
    )


def _existing_document_plan(
    stem: str,
    doc_path: Path,
    canonical_table: pa.Table,
    *,
    article_hash: str,
) -> StemPlan:
    """Classify a document file that already exists for a stem."""
    try:
        doc_hash = _file_content_hash(doc_path)
    except OSError as exc:
        return _blocked_plan(
            stem,
            f"unreadable document file ({type(exc).__name__})",
            article_hash=article_hash,
        )
    try:
        document_table = pq.read_table(doc_path)  # type: ignore[no-untyped-call]
    except Exception as exc:
        return _blocked_plan(
            stem,
            f"unreadable document file ({type(exc).__name__})",
            article_hash=article_hash,
            document_hash=doc_hash,
        )
    if document_table.schema.equals(wikipedia_document_schema(), check_metadata=True):
        return _canonical_document_plan(
            stem,
            document_table,
            canonical_table,
            article_hash=article_hash,
            document_hash=doc_hash,
        )
    if document_table.schema.equals(document_schema(), check_metadata=True):
        return _legacy_document_plan(
            stem,
            document_table,
            canonical_table,
            article_hash=article_hash,
            document_hash=doc_hash,
        )
    return _blocked_plan(
        stem,
        f"unexpected document schema ({len(document_table.schema)} columns)",
        article_hash=article_hash,
        document_hash=doc_hash,
    )


def _classify_stem(stem: str, processed_dir: Path) -> StemPlan:
    """Classify a single stem and build its plan entry."""
    article_path = processed_dir / "articles" / f"{stem}.parquet"
    doc_path = processed_dir / "wikipedia" / "documents" / f"{stem}.parquet"
    missing_plan = _missing_stem_plan(stem, article_path, doc_path)
    if missing_plan is not None:
        return missing_plan
    article_inputs = _article_plan_inputs(stem, article_path)
    if isinstance(article_inputs, StemPlan):
        return article_inputs
    article_hash, canonical_table = article_inputs
    if not doc_path.is_file():
        return _ready_plan(
            stem,
            MigrationOperation.CREATE_MISSING,
            canonical_table,
            article_hash=article_hash,
            document_hash=None,
        )
    return _existing_document_plan(
        stem,
        doc_path,
        canonical_table,
        article_hash=article_hash,
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
    if _became_canonical(planned, current):
        return _APPLY_SKIP
    _raise_transition_conflict(planned, current)
    raise AssertionError("unreachable")


def _became_canonical(planned: StemPlan, current: StemPlan) -> bool:
    """Return whether another process safely completed this stem."""
    return (
        planned.operation in {MigrationOperation.CREATE_MISSING, MigrationOperation.UPGRADE_LEGACY}
        and current.operation == MigrationOperation.ALREADY_CANONICAL
        and planned.article_hash == current.article_hash
        and planned.row_count == current.row_count
        and planned.canonical_digest == current.canonical_digest
    )


def _raise_transition_conflict(planned: StemPlan, current: StemPlan) -> None:
    """Raise the first deterministic conflict found after re-planning."""
    checks = (
        (
            planned.article_hash != current.article_hash,
            f"Stem '{planned.stem}': article file changed after planning",
        ),
        (
            current.operation == MigrationOperation.BLOCKED and "unreadable" in current.reason,
            f"Stem '{planned.stem}': {current.reason}",
        ),
        (
            planned.operation == MigrationOperation.CREATE_MISSING
            and current.document_hash is not None,
            f"Stem '{planned.stem}': conflicting target appeared after planning",
        ),
        (
            planned.document_hash != current.document_hash,
            f"Stem '{planned.stem}': document file changed after planning",
        ),
    )
    for changed, message in checks:
        if changed:
            raise MigrationError(message)

    raise MigrationError(f"Stem '{planned.stem}': migration plan changed after validation")


def _rebuild_table_for_write(sp: StemPlan, processed_dir: Path, target: Path) -> pa.Table:
    """Rebuild and verify one canonical table immediately before replacement."""
    article_path = processed_dir / "articles" / f"{sp.stem}.parquet"
    _validate_article_before_write(sp, article_path)
    _validate_target_before_write(sp, target)
    table = build_wikipedia_document_table(_read_article_table(article_path, sp.stem))
    _validate_canonical_output(sp, table)
    return table


def _validate_article_before_write(sp: StemPlan, article_path: Path) -> None:
    """Verify the article source has not changed since planning."""
    try:
        if _file_content_hash(article_path) != sp.article_hash:
            raise MigrationError(f"Stem '{sp.stem}': article file changed before write")
    except OSError as exc:
        raise MigrationError(
            f"Stem '{sp.stem}': article file unreadable before write ({type(exc).__name__})"
        ) from exc


def _validate_target_before_write(sp: StemPlan, target: Path) -> None:
    """Verify the planned target state before rebuilding its table."""
    if sp.operation == MigrationOperation.CREATE_MISSING:
        if target.exists():
            raise MigrationError(f"Stem '{sp.stem}': target appeared before write")
    elif sp.operation == MigrationOperation.UPGRADE_LEGACY:
        _validate_upgrade_target(sp, target)


def _validate_upgrade_target(sp: StemPlan, target: Path) -> None:
    """Verify an existing legacy document has not changed."""
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


def _validate_canonical_output(sp: StemPlan, table: pa.Table) -> None:
    """Verify rebuilt canonical rows match the immutable plan metadata."""
    if table.num_rows != sp.row_count or _table_digest(table) != sp.canonical_digest:
        raise MigrationError(f"Stem '{sp.stem}': canonical output changed before write")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_migration(processed_dir: Path, stems: set[str] | None = None) -> MigrationPlan:
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
    stems:
        Optional set of specific stems to scope/restrict the migration plan to.

    Returns
    -------
    MigrationPlan
        Immutable, validated plan with per-stem classifications.
    """
    stems_data: list[StemPlan] = []
    discovered = _discover_all_stems(processed_dir)
    if stems is not None:
        discovered = [s for s in discovered if s in stems]
    for stem in discovered:
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
    processed_dir = plan.processed_dir
    docs_dir = processed_dir / "wikipedia" / "documents"
    safe_targets = _validate_apply_inputs(plan, processed_dir, docs_dir)
    current_plan, actions = _revalidated_actions(plan, processed_dir)
    created_stems, upgraded_stems, skipped_stems = _execute_actions(
        current_plan,
        actions,
        processed_dir,
        safe_targets,
    )

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


def _validate_apply_inputs(
    plan: MigrationPlan,
    processed_dir: Path,
    docs_dir: Path,
) -> dict[str, Path]:
    """Validate plan safety and all target paths before any writes."""
    if not plan.is_safe_to_apply:
        blocked = list(plan.blocked_stems)
        raise MigrationError(
            f"Plan is not safe to apply: {len(blocked)} blocked stem(s): {blocked}"
        )
    _validate_documents_root(processed_dir, docs_dir)
    return {
        stem_plan.stem: _validate_stem_path(stem_plan.stem, docs_dir) for stem_plan in plan.stems
    }


def _revalidated_actions(
    plan: MigrationPlan,
    processed_dir: Path,
) -> tuple[MigrationPlan, list[tuple[str, str]]]:
    """Re-plan every stem and compute safe actions without writing."""
    current_plan = plan_migration(
        processed_dir,
        stems={stem_plan.stem for stem_plan in plan.stems},
    )
    _ensure_plan_stems_match(plan, current_plan)
    return current_plan, _transition_actions(plan, current_plan)


def _ensure_plan_stems_match(plan: MigrationPlan, current_plan: MigrationPlan) -> None:
    """Reject a changed stem set during apply-time revalidation."""
    planned_stems = tuple(stem_plan.stem for stem_plan in plan.stems)
    current_stems = tuple(stem_plan.stem for stem_plan in current_plan.stems)
    if current_stems != planned_stems:
        raise MigrationError("Migration plan stem set changed after validation")


def _transition_actions(
    plan: MigrationPlan,
    current_plan: MigrationPlan,
) -> list[tuple[str, str]]:
    """Compute one safe action for each revalidated stem pair."""
    return [
        (current.stem, _transition_action(planned, current))
        for planned, current in zip(plan.stems, current_plan.stems, strict=True)
    ]


def _execute_actions(
    current_plan: MigrationPlan,
    actions: list[tuple[str, str]],
    processed_dir: Path,
    safe_targets: dict[str, Path],
) -> tuple[list[str], list[str], list[str]]:
    """Execute validated migration actions and collect affected stems."""
    created_stems: list[str] = []
    upgraded_stems: list[str] = []
    skipped_stems: list[str] = []
    for stem_plan, (_stem, action) in zip(current_plan.stems, actions, strict=True):
        if action == _APPLY_SKIP:
            skipped_stems.append(stem_plan.stem)
            continue
        target = safe_targets[stem_plan.stem]
        canonical_table = _rebuild_table_for_write(stem_plan, processed_dir, target)
        atomic_write_parquet(target, canonical_table)
        _record_applied_stem(stem_plan, created_stems, upgraded_stems)
    return created_stems, upgraded_stems, skipped_stems


def _record_applied_stem(
    stem_plan: StemPlan,
    created_stems: list[str],
    upgraded_stems: list[str],
) -> None:
    """Append one written stem to its operation-specific result list."""
    if stem_plan.operation == MigrationOperation.CREATE_MISSING:
        created_stems.append(stem_plan.stem)
    else:
        upgraded_stems.append(stem_plan.stem)
