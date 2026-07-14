"""Tests for the Wikipedia-document backfill migration engine (Phase 3).

Covers the two-stage migration (planning + apply) including:
- read-only planning, determinism, classification
- creation, upgrade, skip, and blocked operations
- atomicity, idempotency, and isolation guarantees
- error handling and fail-closed semantics
- Phase 1 audit compatibility after successful migration
"""

from __future__ import annotations

import hashlib
import socket
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.models import document_from_article_row
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema as legacy_document_schema,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    MigrationError,
    MigrationOperation,
    MigrationPlan,
    StemPlan,
    apply_migration,
    plan_migration,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    build_wikipedia_document_table,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import ARTICLE_COLUMNS, article_schema

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_article_row(
    wikidata: str = "Q235",
    language: str = "en",
    page_id: int = 100,
    revision_id: int = 200,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a complete 30-column article row with deterministic defaults."""
    article_id = f"{wikidata}:{language}:{page_id}:{revision_id}"
    row: dict[str, Any] = {
        "article_id": article_id,
        "wikidata": wikidata,
        "language": language,
        "site": f"{language}wiki",
        "title": f"Article {wikidata}",
        "url": f"https://{language}.wikipedia.org/wiki/Article_{wikidata}",
        "page_id": page_id,
        "revision_id": revision_id,
        "revision_timestamp": "2026-01-15T10:30:00Z",
        "retrieved_at": "2026-07-14T12:00:00Z",
        "wikidata_label": f"Label {wikidata}",
        "wikidata_description": f"Description {wikidata}",
        "wikidata_aliases": '["Alias1"]',
        "lead_text": f"Lead text for {wikidata}.",
        "extract": f"Extract for {wikidata}.",
        "full_text": f"Full text for {wikidata}.",
        "full_text_format": "plain_text",
        "article_length_chars": 20,
        "article_length_words": 4,
        "article_length_tokens_estimate": 5,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "[]",
        "license": "CC BY-SA",
        "attribution": "Wikipedia contributors",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": hashlib.sha256(f"Full text for {wikidata}.".encode()).hexdigest(),
    }
    row.update(overrides)
    return row


def _make_article_table(rows: list[dict[str, Any]] | None = None) -> pa.Table:
    if rows is None:
        rows = [_make_article_row()]
    return pa.Table.from_pylist(rows, schema=article_schema())


def _make_legacy_document_table(article_rows: list[dict[str, Any]]) -> pa.Table:
    """Build a legacy 23-column document table from article rows."""
    doc_rows = [document_from_article_row(r).to_dict() for r in article_rows]
    return pa.Table.from_pylist(doc_rows, schema=legacy_document_schema())


def _make_canonical_document_table(article_rows: list[dict[str, Any]]) -> pa.Table:
    """Build a canonical 32-column document table via the Phase 2 converter."""
    return build_wikipedia_document_table(_make_article_table(article_rows))


def _make_section_table(article_row: dict[str, Any] | None = None) -> pa.Table:
    """Build a minimal section table matching one article row."""
    if article_row is None:
        article_row = _make_article_row()
    doc_id = (
        f"{article_row['wikidata']}:wikipedia:{article_row['language']}:"
        f"{article_row['page_id']}:{article_row['revision_id']}"
    )
    row = {
        "section_id": f"{doc_id}:0",
        "document_id": doc_id,
        "article_id": article_row["article_id"],
        "wikidata": article_row["wikidata"],
        "project": "wikipedia",
        "language": article_row["language"],
        "site": article_row["site"],
        "page_id": article_row["page_id"],
        "revision_id": article_row["revision_id"],
        "section_index": 0,
        "heading": "Overview",
        "anchor": "Overview",
        "level": 1,
        "parent_section_id": "",
        "section_path": "Overview",
        "text": article_row["full_text"],
        "text_length_chars": article_row["article_length_chars"],
        "text_length_words": article_row["article_length_words"],
        "text_length_tokens_estimate": article_row["article_length_tokens_estimate"],
        "content_hash": article_row["content_hash"],
        "license": article_row["license"],
        "attribution": article_row["attribution"],
    }
    return pa.Table.from_pylist([row], schema=section_schema())


def _build_processed_dir(
    tmp_path: Path,
    *,
    articles: dict[str, list[dict[str, Any]]] | None = None,
    documents: dict[str, pa.Table] | None = None,
    sections: dict[str, pa.Table] | None = None,
    extras: dict[str, bytes] | None = None,
) -> Path:
    """Build a synthetic processed/ directory with the given files.

    Returns the processed/ directory path.
    """
    processed = tmp_path / "processed"
    for d in [
        processed / "articles",
        processed / "wikipedia" / "documents",
        processed / "wikipedia" / "sections",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    if articles:
        for stem, rows in articles.items():
            table = pa.Table.from_pylist(rows, schema=article_schema())
            pq.write_table(table, processed / "articles" / f"{stem}.parquet", compression="snappy")

    if documents:
        for stem, table in documents.items():
            pq.write_table(
                table,
                processed / "wikipedia" / "documents" / f"{stem}.parquet",
                compression="snappy",
            )

    if sections:
        for stem, table in sections.items():
            pq.write_table(
                table,
                processed / "wikipedia" / "sections" / f"{stem}.parquet",
                compression="snappy",
            )

    if extras:
        for rel_path, content in extras.items():
            full_path = processed / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(content)

    return processed


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _snapshot_files(processed: Path) -> dict[str, dict[str, Any]]:
    """Capture relative path, hash, size, and mtime for every file under processed/."""
    snapshot: dict[str, dict[str, Any]] = {}
    for p in sorted(processed.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(processed))
            stat = p.stat()
            snapshot[rel] = {
                "hash": _file_sha256(p),
                "size": stat.st_size,
                "mtime": stat.st_mtime_ns,
            }
    return snapshot


# ===========================================================================
# Planning stage tests
# ===========================================================================


class TestPlanningReadOnly:
    """The planning stage must not modify any files."""

    def test_planning_is_read_only(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
            sections={"stem-a": _make_section_table(row)},
            extras={"README.md": b"hello"},
        )
        before = _snapshot_files(processed)
        plan_migration(processed)
        after = _snapshot_files(processed)
        assert before == after


class TestPlanningDeterminism:
    def test_deterministic_stem_and_operation_ordering(self, tmp_path: Path) -> None:
        row_a = _make_article_row(wikidata="Q235")
        row_b = _make_article_row(wikidata="Q236")
        row_c = _make_article_row(wikidata="Q237")
        processed = _build_processed_dir(
            tmp_path,
            articles={
                "stem-c": [row_c],
                "stem-a": [row_a],
                "stem-b": [row_b],
            },
        )
        plan = plan_migration(processed)
        stems = [s.stem for s in plan.stems]
        assert stems == ["stem-a", "stem-b", "stem-c"]
        operations = [s.operation for s in plan.stems]
        plan2 = plan_migration(processed)
        operations2 = [s.operation for s in plan2.stems]
        assert operations == operations2


class TestPlanningClassification:
    def test_create_missing_for_articles_only(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        plan = plan_migration(processed)
        assert len(plan.stems) == 1
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.CREATE_MISSING
        assert sp.canonical_digest is not None
        assert sp.reason == ""

    def test_upgrade_from_legacy_23_column(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.UPGRADE_LEGACY
        assert sp.canonical_digest is not None

    def test_already_canonical_identical(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_canonical_document_table([row])},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.ALREADY_CANONICAL

    def test_mixed_operations(self, tmp_path: Path) -> None:
        row_a = _make_article_row(wikidata="Q235")
        row_b = _make_article_row(wikidata="Q236")
        row_c = _make_article_row(wikidata="Q237")
        processed = _build_processed_dir(
            tmp_path,
            articles={
                "stem-create": [row_a],
                "stem-upgrade": [row_b],
                "stem-skip": [row_c],
            },
            documents={
                "stem-upgrade": _make_legacy_document_table([row_b]),
                "stem-skip": _make_canonical_document_table([row_c]),
            },
        )
        plan = plan_migration(processed)
        ops = {s.stem: s.operation for s in plan.stems}
        assert ops["stem-create"] == MigrationOperation.CREATE_MISSING
        assert ops["stem-upgrade"] == MigrationOperation.UPGRADE_LEGACY
        assert ops["stem-skip"] == MigrationOperation.ALREADY_CANONICAL
        assert plan.is_safe_to_apply


class TestPlanningValidation:
    def test_preserves_all_30_article_columns(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        plan = plan_migration(processed)
        apply_migration(plan)
        table = pq.read_table(processed / "wikipedia" / "documents" / "stem-a.parquet")
        assert len(table.schema.names) == 32
        assert set(ARTICLE_COLUMNS).issubset(set(table.schema.names))

    def test_deterministic_document_id_and_project(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        plan = plan_migration(processed)
        apply_migration(plan)
        table = pq.read_table(processed / "wikipedia" / "documents" / "stem-a.parquet")
        doc_ids = table.column("document_id").to_pylist()
        projects = table.column("project").to_pylist()
        assert doc_ids == ["Q235:wikipedia:en:100:200"]
        assert all(p == "wikipedia" for p in projects)

    def test_exact_canonical_schema_and_metadata(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        plan = plan_migration(processed)
        apply_migration(plan)
        table = pq.read_table(processed / "wikipedia" / "documents" / "stem-a.parquet")
        canonical_schema = wikipedia_document_schema()
        assert tuple(table.schema.names) == tuple(canonical_schema.names)
        for i, expected_field in enumerate(canonical_schema):
            actual_field = table.schema.field(i)
            assert actual_field.type == expected_field.type
            assert actual_field.metadata == expected_field.metadata

    def test_identity_based_comparison_different_row_order(self, tmp_path: Path) -> None:
        """Legacy document with different row order still validates correctly."""
        row_a = _make_article_row(wikidata="Q235")
        row_b = _make_article_row(wikidata="Q236")
        # Article rows in order [A, B], but legacy document rows in order [B, A]
        legacy_rows = [document_from_article_row(r).to_dict() for r in [row_b, row_a]]
        legacy_table = pa.Table.from_pylist(legacy_rows, schema=legacy_document_schema())
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row_a, row_b]},
            documents={"stem-a": legacy_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.UPGRADE_LEGACY


class TestPlanningBlockers:
    def test_malformed_article_schema_blocks(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        # Overwrite the article with a wrong schema (missing column)
        row = _make_article_row()
        del row["content_hash"]

        fields = [f for f in article_schema() if f.name != "content_hash"]
        bad_table = pa.Table.from_pylist([row], schema=pa.schema(fields))
        pq.write_table(bad_table, processed / "articles" / "stem-a.parquet", compression="snappy")
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED
        assert "stem-a" in sp.reason or sp.stem == "stem-a"
        assert sp.canonical_digest is None
        assert not plan.is_safe_to_apply

    def test_unreadable_article_file_blocks(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        # Overwrite article with garbage
        (processed / "articles" / "stem-a.parquet").write_bytes(b"NOT PARQUET")
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED
        assert "unreadable" in sp.reason.lower()

    def test_unreadable_document_file_blocks(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
            documents={"stem-a": _make_legacy_document_table([_make_article_row()])},
        )
        # Overwrite document with garbage
        (processed / "wikipedia" / "documents" / "stem-a.parquet").write_bytes(b"NOT PARQUET")
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED
        assert "unreadable" in sp.reason.lower()

    def test_shared_value_conflict_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        legacy_table = _make_legacy_document_table([row])
        # Corrupt a shared value in the legacy document
        col_idx = legacy_table.schema.get_field_index("title")
        modified = legacy_table.set_column(
            col_idx, "title", pa.array(["WRONG TITLE"], type=pa.string())
        )
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": modified},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED
        assert "conflict" in sp.reason.lower() or "mismatch" in sp.reason.lower()

    def test_duplicate_document_identity_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        # Create a legacy document with a duplicate document_id
        doc_row = document_from_article_row(row).to_dict()
        legacy_table = pa.Table.from_pylist([doc_row, doc_row], schema=legacy_document_schema())
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": legacy_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED
        assert "duplicate" in sp.reason.lower() or "mismatch" in sp.reason.lower()

    def test_row_count_mismatch_blocks(self, tmp_path: Path) -> None:
        row_a = _make_article_row(wikidata="Q235")
        row_b = _make_article_row(wikidata="Q236")
        # Article has 2 rows but legacy document has 1
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row_a, row_b]},
            documents={"stem-a": _make_legacy_document_table([row_a])},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED
        assert "mismatch" in sp.reason.lower() or "count" in sp.reason.lower()

    def test_unexpected_document_schema_blocks(self, tmp_path: Path) -> None:
        # Create a document with completely wrong columns
        bad_schema = pa.schema([pa.field("foo", pa.string()), pa.field("bar", pa.int64())])
        bad_table = pa.table({"foo": ["x"], "bar": [1]}, schema=bad_schema)
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
            documents={"stem-a": bad_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED
        assert "unexpected" in sp.reason.lower()

    def test_canonical_schema_content_mismatch_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        canonical_table = _make_canonical_document_table([row])
        # Modify the data in the canonical table
        col_idx = canonical_table.schema.get_field_index("title")
        modified = canonical_table.set_column(
            col_idx, "title", pa.array(["WRONG"], type=pa.string())
        )
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": modified},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED


# ===========================================================================
# Apply stage tests
# ===========================================================================


class TestApplyCreate:
    def test_create_missing_writes_canonical_document(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
        )
        plan = plan_migration(processed)
        result = apply_migration(plan)
        assert result.created == 1
        assert result.upgraded == 0
        assert result.skipped == 0
        assert result.blocked == 0
        assert result.created_stems == ("stem-a",)
        # Verify written file
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"
        assert doc_path.exists()
        written = pq.read_table(doc_path)
        expected = build_wikipedia_document_table(_make_article_table([row]))
        assert written.equals(expected)
        assert tuple(written.schema.names) == tuple(wikipedia_document_schema().names)


class TestApplyUpgrade:
    def test_upgrade_legacy_replaces_document(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        plan = plan_migration(processed)
        result = apply_migration(plan)
        assert result.upgraded == 1
        assert result.created == 0
        assert result.upgraded_stems == ("stem-a",)
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"
        written = pq.read_table(doc_path)
        assert len(written.schema.names) == 32
        expected = build_wikipedia_document_table(_make_article_table([row]))
        assert written.equals(expected)


class TestApplyIdempotency:
    def test_apply_twice_is_idempotent(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        plan = plan_migration(processed)
        result1 = apply_migration(plan)
        assert result1.upgraded == 1
        hash1 = _file_sha256(processed / "wikipedia" / "documents" / "stem-a.parquet")
        # Apply the SAME plan again
        result2 = apply_migration(plan)
        assert result2.upgraded == 0
        assert result2.skipped == 1
        hash2 = _file_sha256(processed / "wikipedia" / "documents" / "stem-a.parquet")
        assert hash1 == hash2

    def test_skip_already_canonical_no_mtime_change(self, tmp_path: Path) -> None:
        row = _make_article_row()
        canonical_doc = _make_canonical_document_table([row])
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": canonical_doc},
        )
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"
        mtime_before = doc_path.stat().st_mtime_ns
        hash_before = _file_sha256(doc_path)
        plan = plan_migration(processed)
        result = apply_migration(plan)
        assert result.skipped == 1
        assert result.created == 0
        assert result.upgraded == 0
        mtime_after = doc_path.stat().st_mtime_ns
        hash_after = _file_sha256(doc_path)
        assert hash_before == hash_after
        assert mtime_before == mtime_after


class TestApplyIsolation:
    def test_sections_remain_byte_identical(self, tmp_path: Path) -> None:
        row = _make_article_row()
        section_table = _make_section_table(row)
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
            sections={"stem-a": section_table},
        )
        sec_path = processed / "wikipedia" / "sections" / "stem-a.parquet"
        hash_before = _file_sha256(sec_path)
        plan = plan_migration(processed)
        apply_migration(plan)
        hash_after = _file_sha256(sec_path)
        assert hash_before == hash_after

    def test_articles_remain_byte_identical(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        art_path = processed / "articles" / "stem-a.parquet"
        hash_before = _file_sha256(art_path)
        plan = plan_migration(processed)
        apply_migration(plan)
        hash_after = _file_sha256(art_path)
        assert hash_before == hash_after

    def test_unrelated_files_untouched(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
            extras={
                "README.md": b"important readme",
                "manifests/data.json": b'{"key": "value"}',
            },
        )
        before = _snapshot_files(processed)
        plan = plan_migration(processed)
        apply_migration(plan)
        after = _snapshot_files(processed)
        # Only the document file should have changed
        changed = {k for k in after if k in before and before[k]["hash"] != after[k]["hash"]}
        assert changed == {"wikipedia/documents/stem-a.parquet"}
        # Unrelated files must be byte-identical
        for name in ["README.md", "manifests/data.json"]:
            assert before[name]["hash"] == after[name]["hash"]


class TestApplySafety:
    def test_atomic_write_failure_preserves_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"
        original_hash = _file_sha256(doc_path)
        original_mtime = doc_path.stat().st_mtime_ns

        plan = plan_migration(processed)

        # Inject failure when writing to temp files
        original_write_table = pq.write_table

        def failing_write_table(table: pa.Table, path: Any, **kwargs: Any) -> Any:
            if isinstance(path, (str, Path)) and ".tmp" in str(path):
                raise OSError("Injected write failure")
            return original_write_table(table, path, **kwargs)

        monkeypatch.setattr(pq, "write_table", failing_write_table)

        with pytest.raises((MigrationError, OSError)):
            apply_migration(plan)

        # Original must be intact
        assert _file_sha256(doc_path) == original_hash
        assert doc_path.stat().st_mtime_ns == original_mtime

        # No temp files remaining
        docs_dir = processed / "wikipedia" / "documents"
        temp_files = [p for p in docs_dir.iterdir() if p.name.startswith(".")]
        assert len(temp_files) == 0

    def test_blocked_stem_prevents_apply(self, tmp_path: Path) -> None:
        row_good = _make_article_row(wikidata="Q235")
        processed = _build_processed_dir(
            tmp_path,
            articles={
                "stem-good": [row_good],
                "stem-bad": [_make_article_row(wikidata="Q236")],
            },
        )
        # Make stem-bad article unreadable
        (processed / "articles" / "stem-bad.parquet").write_bytes(b"GARBAGE")
        plan = plan_migration(processed)
        assert not plan.is_safe_to_apply
        assert "stem-bad" in plan.blocked_stems
        with pytest.raises(MigrationError, match="not safe to apply"):
            apply_migration(plan)

    def test_no_network_access(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
        )

        def blocking_socket(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Network access blocked")

        monkeypatch.setattr(socket, "socket", blocking_socket)

        plan = plan_migration(processed)
        result = apply_migration(plan)
        assert result.created == 1


class TestApplyResult:
    def test_result_counts_and_stems(self, tmp_path: Path) -> None:
        row_a = _make_article_row(wikidata="Q235")
        row_b = _make_article_row(wikidata="Q236")
        row_c = _make_article_row(wikidata="Q237")
        processed = _build_processed_dir(
            tmp_path,
            articles={
                "stem-create": [row_a],
                "stem-upgrade": [row_b],
                "stem-skip": [row_c],
            },
            documents={
                "stem-upgrade": _make_legacy_document_table([row_b]),
                "stem-skip": _make_canonical_document_table([row_c]),
            },
        )
        plan = plan_migration(processed)
        result = apply_migration(plan)
        assert result.planned == 3
        assert result.created == 1
        assert result.upgraded == 1
        assert result.skipped == 1
        assert result.blocked == 0
        assert result.created_stems == ("stem-create",)
        assert result.upgraded_stems == ("stem-upgrade",)
        assert result.skipped_stems == ("stem-skip",)
        assert result.blocked_stems == ()


# ===========================================================================
# Phase 1 audit compatibility
# ===========================================================================


class TestAuditCompatibility:
    def test_audit_shows_both_equivalent_after_migration(self, tmp_path: Path) -> None:
        from tests.migration.audit import run_audit

        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
            sections={"stem-a": _make_section_table(row)},
        )

        # Before migration: Phase 1 audit sees legacy document
        report_before = run_audit(tmp_path)
        assert report_before["per_stem"]["stem-a"]["state"] == "both_needing_schema_upgrade"

        # Run migration
        plan = plan_migration(processed)
        assert plan.is_safe_to_apply
        apply_migration(plan)

        # After migration: Phase 1 audit should see both_equivalent
        report_after = run_audit(tmp_path)
        assert report_after["per_stem"]["stem-a"]["state"] == "both_equivalent"
        assert report_after["per_stem"]["stem-a"]["discrepancies"] == []


# ===========================================================================
# Stale-plan detection tests
# ===========================================================================


class TestStalePlanDetection:
    """Apply must revalidate every stem against filesystem state before writing."""

    def test_article_changed_between_plan_and_apply(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        plan = plan_migration(processed)
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"
        original_doc_hash = _file_sha256(doc_path)

        # Modify article after planning
        modified_row = _make_article_row(wikidata="Q999")
        pq.write_table(
            _make_article_table([modified_row]),
            processed / "articles" / "stem-a.parquet",
            compression="snappy",
        )

        with pytest.raises(MigrationError, match=r"article.*changed"):
            apply_migration(plan)

        # Zero writes: document unchanged
        assert _file_sha256(doc_path) == original_doc_hash

    def test_legacy_document_changed_between_plan_and_apply(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        plan = plan_migration(processed)
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"

        # Overwrite document with different legacy content
        modified_row = _make_article_row(wikidata="Q236")
        modified_doc = _make_legacy_document_table([modified_row])
        pq.write_table(modified_doc, doc_path, compression="snappy")
        modified_hash = _file_sha256(doc_path)

        with pytest.raises(MigrationError, match=r"document.*changed"):
            apply_migration(plan)

        # Zero writes: file still has our modification, not canonical
        assert _file_sha256(doc_path) == modified_hash
        written = pq.read_table(doc_path)
        assert len(written.schema.names) == 23

    def test_unreadable_target_introduced_between_plan_and_apply(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        plan = plan_migration(processed)
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"

        # Corrupt the document
        doc_path.write_bytes(b"CORRUPT")

        with pytest.raises(MigrationError, match="unreadable"):
            apply_migration(plan)

        # Never overwritten
        assert doc_path.read_bytes() == b"CORRUPT"

    def test_conflicting_target_after_create_missing(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
        )
        plan = plan_migration(processed)
        assert plan.stems[0].operation == MigrationOperation.CREATE_MISSING

        # Write a conflicting (legacy) file after planning
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"
        pq.write_table(_make_legacy_document_table([row]), doc_path, compression="snappy")
        conflicting_hash = _file_sha256(doc_path)

        with pytest.raises(MigrationError, match="conflicting"):
            apply_migration(plan)

        assert _file_sha256(doc_path) == conflicting_hash

    def test_identical_canonical_after_create_missing_is_skipped(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
        )
        plan = plan_migration(processed)
        assert plan.stems[0].operation == MigrationOperation.CREATE_MISSING

        # Write identical canonical after planning
        doc_path = processed / "wikipedia" / "documents" / "stem-a.parquet"
        canonical = _make_canonical_document_table([row])
        pq.write_table(canonical, doc_path, compression="snappy")

        result = apply_migration(plan)
        assert result.created == 0
        assert result.skipped == 1

    def test_stale_later_stem_causes_zero_writes(self, tmp_path: Path) -> None:
        row_a = _make_article_row(wikidata="Q235")
        row_b = _make_article_row(wikidata="Q236")
        processed = _build_processed_dir(
            tmp_path,
            articles={
                "stem-a": [row_a],
                "stem-b": [row_b],
            },
        )
        plan = plan_migration(processed)
        doc_a = processed / "wikipedia" / "documents" / "stem-a.parquet"

        # Modify stem-b article after planning (stems are sorted: a, b)
        modified_b = _make_article_row(wikidata="Q999")
        pq.write_table(
            _make_article_table([modified_b]),
            processed / "articles" / "stem-b.parquet",
            compression="snappy",
        )

        with pytest.raises(MigrationError, match="stem-b"):
            apply_migration(plan)

        # stem-a must NOT have been written
        assert not doc_a.exists()


# ===========================================================================
# Exact schema metadata tests
# ===========================================================================


def _canonical_schema_no_metadata() -> pa.Schema:
    """Canonical column names and types with all field metadata stripped."""
    return pa.schema([pa.field(f.name, f.type) for f in wikipedia_document_schema()])


def _canonical_schema_wrong_metadata() -> pa.Schema:
    """Canonical schema with every field's description set to a wrong value."""
    return pa.schema(
        [
            pa.field(f.name, f.type, metadata={b"description": b"WRONG"})
            for f in wikipedia_document_schema()
        ]
    )


def _canonical_schema_extra_metadata() -> pa.Schema:
    """Canonical schema with correct metadata plus an extra key on each field."""
    fields = []
    for f in wikipedia_document_schema():
        meta = dict(f.metadata or {})
        meta[b"extra_key"] = b"extra_val"
        fields.append(pa.field(f.name, f.type, metadata=meta))
    return pa.schema(fields)


class TestExactSchemaMetadata:
    """Canonical-looking schemas with metadata differences must be BLOCKED."""

    def test_canonical_schema_missing_metadata_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        canonical_data = _make_canonical_document_table([row])
        bad_table = pa.Table.from_pylist(
            canonical_data.to_pylist(), schema=_canonical_schema_no_metadata()
        )
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": bad_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED

    def test_canonical_schema_incorrect_metadata_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        canonical_data = _make_canonical_document_table([row])
        bad_table = pa.Table.from_pylist(
            canonical_data.to_pylist(), schema=_canonical_schema_wrong_metadata()
        )
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": bad_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED

    def test_canonical_schema_extra_metadata_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        canonical_data = _make_canonical_document_table([row])
        bad_table = pa.Table.from_pylist(
            canonical_data.to_pylist(), schema=_canonical_schema_extra_metadata()
        )
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": bad_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED


# ===========================================================================
# Strict legacy schema acceptance tests
# ===========================================================================


def _legacy_schema_reordered() -> pa.Schema:
    """Legacy document schema with fields reversed."""
    fields = list(legacy_document_schema())
    return pa.schema(list(reversed(fields)))


def _legacy_schema_with_metadata() -> pa.Schema:
    """Legacy schema with unexpected metadata added to each field."""
    return pa.schema(
        [
            pa.field(f.name, f.type, metadata={b"description": b"should not be here"})
            for f in legacy_document_schema()
        ]
    )


def _partial_upgrade_schema() -> pa.Schema:
    """23 legacy columns plus 5 of the 9 canonical-only upgrade columns."""
    canonical = wikipedia_document_schema()
    legacy = legacy_document_schema()
    canonical_only = [c for c in canonical.names if c not in DOCUMENT_COLUMNS]
    partial_names = list(DOCUMENT_COLUMNS) + canonical_only[:5]
    fields = []
    for name in partial_names:
        if name in legacy.names:
            fields.append(legacy.field(legacy.get_field_index(name)))
        else:
            fields.append(canonical.field(canonical.get_field_index(name)))
    return pa.schema(fields)


class TestStrictLegacySchema:
    """Only exact legacy or exact canonical schemas are accepted."""

    def test_legacy_schema_reordered_fields_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        legacy_data = _make_legacy_document_table([row])
        bad_table = pa.Table.from_pylist(legacy_data.to_pylist(), schema=_legacy_schema_reordered())
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": bad_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED

    def test_legacy_schema_with_extra_metadata_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        legacy_data = _make_legacy_document_table([row])
        bad_table = pa.Table.from_pylist(
            legacy_data.to_pylist(), schema=_legacy_schema_with_metadata()
        )
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": bad_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED

    def test_partial_upgrade_schema_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        partial_schema = _partial_upgrade_schema()
        int_cols = {
            "page_id",
            "revision_id",
            "article_length_chars",
            "article_length_words",
            "article_length_tokens_estimate",
        }
        partial_row: dict[str, Any] = {}
        for name in partial_schema.names:
            partial_row[name] = 0 if name in int_cols else ""
        partial_row["document_id"] = "Q235:wikipedia:en:100:200"
        partial_row["article_id"] = "Q235:en:100:200"
        partial_row["page_id"] = 100
        partial_row["revision_id"] = 200
        bad_table = pa.Table.from_pylist([partial_row], schema=partial_schema)
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [row]},
            documents={"stem-a": bad_table},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.operation == MigrationOperation.BLOCKED


# ===========================================================================
# Document-without-article detection tests
# ===========================================================================


class TestDocumentWithoutArticle:
    """A document stem lacking its article source must be BLOCKED."""

    def test_document_only_stem_blocks(self, tmp_path: Path) -> None:
        row = _make_article_row()
        processed = _build_processed_dir(
            tmp_path,
            documents={"stem-a": _make_legacy_document_table([row])},
        )
        plan = plan_migration(processed)
        sp = plan.stems[0]
        assert sp.stem == "stem-a"
        assert sp.operation == MigrationOperation.BLOCKED
        assert "article" in sp.reason.lower()
        assert not plan.is_safe_to_apply


# ===========================================================================
# Path-traversal defense tests
# ===========================================================================


class TestPathTraversal:
    """A malicious stem cannot escape the documents directory."""

    def test_malicious_stem_cannot_escape(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        malicious_plan = MigrationPlan(
            processed_dir=processed,
            stems=(
                StemPlan(
                    stem="../../../etc/passwd",
                    operation=MigrationOperation.CREATE_MISSING,
                    reason="",
                    article_hash="dummy",
                    document_hash=None,
                    row_count=1,
                    canonical_digest="dummy",
                ),
            ),
        )
        with pytest.raises(MigrationError, match=r"separator|escape|invalid"):
            apply_migration(malicious_plan)

    def test_empty_stem_rejected(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        malicious_plan = MigrationPlan(
            processed_dir=processed,
            stems=(
                StemPlan(
                    stem="",
                    operation=MigrationOperation.CREATE_MISSING,
                    reason="",
                    article_hash="dummy",
                    document_hash=None,
                    row_count=1,
                    canonical_digest="dummy",
                ),
            ),
        )
        with pytest.raises(MigrationError):
            apply_migration(malicious_plan)

    def test_dotdot_stem_rejected(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        malicious_plan = MigrationPlan(
            processed_dir=processed,
            stems=(
                StemPlan(
                    stem="..",
                    operation=MigrationOperation.CREATE_MISSING,
                    reason="",
                    article_hash="dummy",
                    document_hash=None,
                    row_count=1,
                    canonical_digest="dummy",
                ),
            ),
        )
        with pytest.raises(MigrationError):
            apply_migration(malicious_plan)


class TestLightweightValidatedPlans:
    """Plans contain metadata only and cannot inject caller-supplied rows."""

    def test_plan_stores_no_arrow_tables_or_rows(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        plan = plan_migration(processed)
        assert "canonical_table" not in {field.name for field in fields(StemPlan)}
        for stem_plan in plan.stems:
            assert all(not isinstance(value, pa.Table) for value in vars_for_slots(stem_plan))

    def test_forged_digest_is_rejected_before_writing(self, tmp_path: Path) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        plan = plan_migration(processed)
        forged = replace(
            plan,
            stems=(replace(plan.stems[0], canonical_digest="0" * 64),),
        )
        with pytest.raises(MigrationError, match=r"validated|changed|plan"):
            apply_migration(forged)
        assert not (processed / "wikipedia" / "documents" / "stem-a.parquet").exists()

    def test_symlinked_documents_directory_cannot_escape_processed_root(
        self, tmp_path: Path
    ) -> None:
        processed = _build_processed_dir(
            tmp_path,
            articles={"stem-a": [_make_article_row()]},
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        wikipedia = processed / "wikipedia"
        wikipedia.mkdir(exist_ok=True)
        (wikipedia / "documents").rmdir()
        (wikipedia / "documents").symlink_to(outside, target_is_directory=True)
        plan = plan_migration(processed)
        with pytest.raises(MigrationError, match=r"outside|escape|directory"):
            apply_migration(plan)
        assert list(outside.iterdir()) == []


def vars_for_slots(value: object) -> tuple[object, ...]:
    """Return dataclass values without requiring ``__dict__``."""
    return tuple(getattr(value, field.name) for field in fields(value))
