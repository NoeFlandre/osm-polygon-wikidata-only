"""Apply validated Wikipedia document migration plans atomically."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    build_wikipedia_document_table,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_parquet

from .models import ApplyResult, MigrationError, MigrationOperation, MigrationPlan, StemPlan
from .planning import (
    file_content_hash,
    plan_migration,
    read_article_table,
    table_digest,
    validate_stem_path,
)

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
    table = build_wikipedia_document_table(read_article_table(article_path, sp.stem))
    _validate_canonical_output(sp, table)
    return table


def _validate_article_before_write(sp: StemPlan, article_path: Path) -> None:
    """Verify the article source has not changed since planning."""
    try:
        if file_content_hash(article_path) != sp.article_hash:
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
        current_hash = file_content_hash(target)
    except OSError as exc:
        raise MigrationError(
            f"Stem '{sp.stem}': document unreadable before write ({type(exc).__name__})"
        ) from exc
    if current_hash != sp.document_hash:
        raise MigrationError(f"Stem '{sp.stem}': document changed before write")


def _validate_canonical_output(sp: StemPlan, table: pa.Table) -> None:
    """Verify rebuilt canonical rows match the immutable plan metadata."""
    if table.num_rows != sp.row_count or table_digest(table) != sp.canonical_digest:
        raise MigrationError(f"Stem '{sp.stem}': canonical output changed before write")


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
        stem_plan.stem: validate_stem_path(stem_plan.stem, docs_dir) for stem_plan in plan.stems
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
