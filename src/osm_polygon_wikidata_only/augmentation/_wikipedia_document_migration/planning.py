"""Read-only classification and planning for Wikipedia document migration."""

from __future__ import annotations

import hashlib
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
from osm_polygon_wikidata_only.io.hashing import sha256_file_uncached

from .models import MigrationError, MigrationOperation, MigrationPlan, StemPlan


def _file_content_hash(path: Path) -> str:
    """Compute SHA-256 of file bytes, bypassing the digest cache."""
    return sha256_file_uncached(path)


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
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
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
        document_table = pq.read_table(doc_path)
    except (OSError, pa.ArrowException) as exc:
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
    discovered = _discover_all_stems(processed_dir)
    if stems is not None:
        discovered = [s for s in discovered if s in stems]
    stems_data = [_classify_stem(stem, processed_dir) for stem in discovered]

    return MigrationPlan(
        processed_dir=processed_dir,
        stems=tuple(stems_data),
    )


# Named collaborators used by the apply service across the planning boundary.
file_content_hash = _file_content_hash
read_article_table = _read_article_table
table_digest = _table_digest
validate_stem_path = _validate_stem_path
assert_canonical_preserves_legacy = _assert_canonical_preserves_legacy
