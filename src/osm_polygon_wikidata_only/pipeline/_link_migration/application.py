"""Transactional apply services for link migration."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
    plan_integrity_normalization,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    build_polygon_document_links,
    polygon_document_link_schema,
    validate_polygon_document_links,
)
from osm_polygon_wikidata_only.pipeline._link_migration.artifacts import stage_stem_replacements
from osm_polygon_wikidata_only.pipeline._link_migration.conversion import (
    build_canonical_rows as _build_canonical_rows,
)
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    MigrationPlan,
    StemApplyContext,
    StemApplyInputs,
    StemClassification,
    StemPlan,
)
from osm_polygon_wikidata_only.pipeline._link_migration.planning import (
    classify_stem,
    plan_link_migration,
    plan_link_migration_normalization_rejections_for_stem,
)
from osm_polygon_wikidata_only.pipeline._link_migration.transaction import (
    commit_ordered_replacements as _commit_ordered_replacements,
)

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _apply_replacements(
    processed_dir: Path,
    replacements: list[tuple[Path, Path]],
    *,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Apply an already prepared replacement set through the journal."""
    directory = processed_dir / ".link_migration_journal"
    _commit_ordered_replacements(
        directory,
        stem="__replacements__",
        replacements=replacements,
        data_root=processed_dir,
        _crash_hook=_crash_hook,
    )


def _validate_stem_fingerprints(processed_dir: Path, stem_plan: StemPlan) -> None:
    """Reject a plan whose source files changed before the write phase."""
    current = classify_stem(stem_plan.stem, processed_dir)
    fingerprints = (
        current.polygons_fingerprint,
        current.links_fingerprint,
        current.documents_fingerprint,
    )
    planned = (
        stem_plan.polygons_fingerprint,
        stem_plan.links_fingerprint,
        stem_plan.documents_fingerprint,
    )
    if fingerprints != planned:
        raise RuntimeError(
            f"Link migration stem {stem_plan.stem!r}: source file changed after planning"
        )


def _load_stem_apply_inputs(processed_dir: Path, stem_plan: StemPlan) -> StemApplyInputs:
    """Load the immutable source tables needed by a stem transaction."""
    links_path = processed_dir / "polygon_articles" / f"{stem_plan.stem}.parquet"
    polygons_path = processed_dir / "polygons" / f"{stem_plan.stem}.parquet"
    docs_path = processed_dir / "wikipedia" / "documents" / f"{stem_plan.stem}.parquet"
    voyage_documents_path = processed_dir / "wikivoyage" / "documents" / f"{stem_plan.stem}.parquet"
    voyage_sections_path = processed_dir / "wikivoyage" / "sections" / f"{stem_plan.stem}.parquet"
    return StemApplyInputs(
        stem_plan=stem_plan,
        links_path=links_path,
        polygons_path=polygons_path,
        docs_path=docs_path,
        voyage_documents_path=voyage_documents_path,
        voyage_sections_path=voyage_sections_path,
        legacy_table=pq.read_table(links_path),
        polygons_table=pq.read_table(polygons_path),
        docs_table=pq.read_table(docs_path),
        data_root=DataRoot(processed_dir.parent),
    )


def _build_stem_context(inputs: StemApplyInputs) -> StemApplyContext:
    """Derive canonical links and the reject-only Wikivoyage plan."""
    wikipedia_rows = _build_canonical_rows(
        inputs.stem_plan.stem,
        inputs.legacy_table,
        inputs.polygons_table,
        inputs.docs_table,
    )
    integrity_plan = (
        plan_integrity_normalization(inputs.data_root, inputs.stem_plan.stem)
        if inputs.voyage_documents_path.is_file()
        else None
    )
    retained_documents = integrity_plan.retained_documents if integrity_plan is not None else []
    voyage_rows = build_polygon_document_links(
        inputs.polygons_table.to_pylist(),
        wikivoyage_documents=retained_documents,
    )
    canonical_rows = validate_polygon_document_links([*wikipedia_rows, *voyage_rows])
    canonical_table = pa.Table.from_pylist(
        canonical_rows,
        schema=polygon_document_link_schema(),
    )
    return StemApplyContext(
        inputs=inputs,
        integrity_plan=integrity_plan,
        canonical_table=canonical_table,
    )


def _apply_migratable_stem(
    processed_dir: Path,
    stem_plan: StemPlan,
    wiki_rejections: list[dict[str, Any]],
    *,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Stage and commit one unchanged-source stem transaction."""
    _validate_stem_fingerprints(processed_dir, stem_plan)
    inputs = _load_stem_apply_inputs(processed_dir, stem_plan)
    context = _build_stem_context(inputs)
    replacements = stage_stem_replacements(processed_dir, context, wiki_rejections)
    journal_dir = processed_dir / ".link_migration_journal" / stem_plan.stem
    _commit_ordered_replacements(
        journal_dir,
        stem=stem_plan.stem,
        replacements=replacements,
        data_root=processed_dir,
        _crash_hook=_crash_hook,
    )
    staged_dir = processed_dir / ".link_migration_staging" / stem_plan.stem
    with suppress(OSError):
        shutil.rmtree(staged_dir)


def _ensure_migration_plan_safe(plan: MigrationPlan) -> None:
    """Raise when any discovered stem is unsafe to migrate."""
    if plan.is_safe_to_apply:
        return
    blocked = [s.stem for s in plan.stems if s.classification == StemClassification.BLOCKED]
    raise ValueError(f"Link migration plan contains blocked stems: {blocked}")


def _wiki_rejections_by_stem(plan: MigrationPlan) -> dict[str, list[dict[str, Any]]]:
    """Compute legacy rejection records before canonical replacement."""
    return {
        stem_plan.stem: plan_link_migration_normalization_rejections_for_stem(
            plan.processed_dir,
            stem_plan,
        )
        for stem_plan in plan.stems
        if stem_plan.classification == StemClassification.MIGRATABLE
    }


def _apply_migratable_stems(
    processed_dir: Path,
    plan: MigrationPlan,
    wiki_rejections: dict[str, list[dict[str, Any]]],
    *,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Apply every migratable stem while skipping canonical stems."""
    for stem_plan in plan.stems:
        if stem_plan.classification != StemClassification.MIGRATABLE:
            continue
        _apply_migratable_stem(
            processed_dir,
            stem_plan,
            wiki_rejections.get(stem_plan.stem, []),
            _crash_hook=_crash_hook,
        )


def apply_link_migration(
    processed_dir: Path,
    *,
    stems: set[str] | None = None,
    replacements: list[tuple[Path, Path]] | None = None,
    plan: MigrationPlan | None = None,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Apply stage.

    * When ``replacements`` is ``None``, plans the migration and applies
      every legacy stem atomically. Each stem gets its own journaled
      transaction.
    * When ``plan`` is supplied (from :func:`plan_link_migration` for the
      same ``processed_dir``), it is applied instead of planning again;
      ``stems`` is then ignored. Per-stem source fingerprints are still
      re-validated before any file is replaced.
    * When ``replacements`` is supplied, runs the same ordered journaled
      transaction directly (used by tests via the public boundary).
    """
    if replacements is not None:
        _apply_replacements(processed_dir, replacements, _crash_hook=_crash_hook)
        return

    if plan is None:
        plan = plan_link_migration(processed_dir, stems=stems)
    _ensure_migration_plan_safe(plan)
    _apply_migratable_stems(
        processed_dir,
        plan,
        _wiki_rejections_by_stem(plan),
        _crash_hook=_crash_hook,
    )
