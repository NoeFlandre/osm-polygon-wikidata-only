"""Phase 2 / Group D: durable rejection ledger + plan/apply integrity split.

Red tests for ``osm_polygon_wikidata_only.augmentation.rejection_ledger``.

Strict ledger-row validation, full stable identity that includes
observed and expected QID, two-stem merge without erasing either, and
no-op second runs that preserve ledger hash + mtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest


def _import_module():
    try:
        from osm_polygon_wikidata_only.augmentation import rejection_ledger as mod
    except ImportError as exc:
        pytest.fail(
            "Expected osm_polygon_wikidata_only.augmentation.rejection_ledger to exist "
            f"(Phase 2 group D: cumulative rejection ledger); got ImportError: {exc}"
        )
    return mod


def _pyarrow():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    return pa


VALID_QID = re.compile(r"^Q\d+$")


def _valid_qid(value: str | None) -> bool:
    if value is None:
        return True  # allowed for expected
    return bool(VALID_QID.match(value))


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exposes_ledger_api() -> None:
    mod = _import_module()
    for name in (
        "merge_records",
        "merge_ledger_files",
        "load_ledger",
        "save_ledger",
        "RejectionRecord",
        "LEDGER_CONTRACT_VERSION",
        "supported_source_tables",
    ):
        assert hasattr(mod, name), f"Missing ledger API: rejection_ledger.{name}"


# ---------------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------------


def _base_record(**overrides) -> dict:
    base = {
        "shard": "monaco-latest",
        "source_table": "polygon_articles",
        "identifier": "monaco-latest:relation:1",
        "wikidata": "Q1",
        "expected": "Q2",
        "reason": "wikidata_mismatch_with_polygon_master",
        "cascaded_sections": 0,
    }
    base.update(overrides)
    return base


def test_full_identity_includes_observed_and_expected_qid() -> None:
    """The full stable identity is (shard, source_table, identifier, wikidata, expected)."""
    mod = _import_module()
    rec_a = mod.RejectionRecord(**_base_record())
    rec_b = mod.RejectionRecord(**_base_record(expected="Q3"))
    assert rec_a.identity != rec_b.identity, (
        "Two records that differ only in expected QID must have distinct identities"
    )


def test_merge_dedups_same_full_identity() -> None:
    """Two records with the same full identity must merge into one."""
    mod = _import_module()
    rec_a = mod.RejectionRecord(**_base_record())
    rec_b = mod.RejectionRecord(**_base_record())
    merged = mod.merge_records([rec_a, rec_b])
    assert len(merged) == 1, f"Expected 1 record after merging duplicates, got {len(merged)}"


def test_merge_keeps_record_with_different_observed_qid_separate() -> None:
    """Two records with different observed QIDs are NOT the same identity."""
    mod = _import_module()
    rec_a = mod.RejectionRecord(**_base_record(wikidata="Q1"))
    rec_b = mod.RejectionRecord(**_base_record(wikidata="Q99"))
    merged = mod.merge_records([rec_a, rec_b])
    assert len(merged) == 2, f"Expected 2 records for different observed QIDs, got {len(merged)}"


def test_merge_uses_max_cascaded_sections_for_same_identity() -> None:
    """Same identity, different cascaded_sections -> keep the max."""
    mod = _import_module()
    rec_a = mod.RejectionRecord(**_base_record(cascaded_sections=2))
    rec_b = mod.RejectionRecord(**_base_record(cascaded_sections=5))
    merged = mod.merge_records([rec_a, rec_b])
    assert len(merged) == 1
    assert merged[0].cascaded_sections == 5, (
        f"Expected max cascaded_sections=5, got {merged[0].cascaded_sections}"
    )


def test_validate_rejects_unsupported_source_table() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.RejectionRecord(**_base_record(source_table="not_a_real_table"))


def test_validate_rejects_empty_shard() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.RejectionRecord(**_base_record(shard=""))


def test_validate_rejects_empty_identifier() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.RejectionRecord(**_base_record(identifier=""))


def test_validate_rejects_empty_reason() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.RejectionRecord(**_base_record(reason=""))


def test_validate_rejects_invalid_observed_qid() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.RejectionRecord(**_base_record(wikidata="not-a-qid"))


def test_validate_accepts_null_expected_qid() -> None:
    mod = _import_module()
    rec = mod.RejectionRecord(**_base_record(expected=None))
    assert rec.expected is None


def test_validate_rejects_invalid_expected_qid() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.RejectionRecord(**_base_record(expected="not-a-qid"))


def test_validate_rejects_negative_cascaded_sections() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.RejectionRecord(**_base_record(cascaded_sections=-1))


def test_ledger_has_no_timestamps() -> None:
    """The ledger record and ledger file must not contain any timestamp fields."""
    mod = _import_module()
    rec = mod.RejectionRecord(**_base_record())
    payload = rec.to_dict()
    for forbidden in ("requested_at", "created_at", "updated_at", "timestamp"):
        assert forbidden not in payload, (
            f"Rejection ledger must not contain timestamp field {forbidden!r}; got {payload}"
        )


# ---------------------------------------------------------------------------
# Two-stem merge without erasing either
# ---------------------------------------------------------------------------


def test_two_stem_merge_preserves_both_stems_records(tmp_path: Path) -> None:
    """Merge two stems' ledgers into one cumulative ledger.

    Neither stem's entries must be erased.
    """
    mod = _import_module()
    rec_a = mod.RejectionRecord(**_base_record(shard="monaco-latest"))
    rec_b = mod.RejectionRecord(**_base_record(shard="italy-latest"))
    save_ledger_a = tmp_path / "a.json"
    save_ledger_b = tmp_path / "b.json"
    mod.save_ledger(save_ledger_a, [rec_a])
    mod.save_ledger(save_ledger_b, [rec_b])
    merged_path = tmp_path / "merged.json"
    mod.merge_ledger_files([save_ledger_a, save_ledger_b], merged_path)
    merged = mod.load_ledger(merged_path)
    shards = sorted(record.shard for record in merged)
    assert shards == ["italy-latest", "monaco-latest"], (
        f"Both stems must remain in the merged ledger, got {shards}"
    )


def test_two_stem_merge_aggregate_last_word(tmp_path: Path) -> None:
    """The aggregate current-pass audit and cumulative ledger must be the
    LAST writes in the integrity transaction (verified by mtime).

    This is a structural test: after a planned integrity apply, the
    cumulative ledger file's mtime must be later than the per-stem
    documents and sections files. (Test uses tmp_path; no real-data.)"""
    mod = _import_module()
    # Seed two stems' worth of fake wiki documents (so the aggregate
    # contains at least one merged record).
    rec_a = mod.RejectionRecord(**_base_record(shard="monaco-latest"))
    rec_b = mod.RejectionRecord(**_base_record(shard="italy-latest"))
    save_ledger_a = tmp_path / "a.json"
    save_ledger_b = tmp_path / "b.json"
    mod.save_ledger(save_ledger_a, [rec_a])
    mod.save_ledger(save_ledger_b, [rec_b])
    merged_path = tmp_path / "merged.json"
    import os
    import time

    # Touch the inputs to an older mtime.
    older = time.time_ns() - 10_000_000
    os.utime(save_ledger_a, ns=(older, older))
    os.utime(save_ledger_b, ns=(older, older))
    mod.merge_ledger_files([save_ledger_a, save_ledger_b], merged_path)
    assert merged_path.stat().st_mtime_ns > save_ledger_a.stat().st_mtime_ns, (
        "Cumulative ledger must be written AFTER (later mtime than) the per-stem ledgers"
    )


# ---------------------------------------------------------------------------
# No-op second pass: ledger hash + mtime stable
# ---------------------------------------------------------------------------


def test_ledger_survives_noop_second_run_byte_for_byte(tmp_path: Path) -> None:
    """A no-op second pass must NOT rewrite the cumulative ledger."""
    mod = _import_module()
    rec = mod.RejectionRecord(**_base_record())
    path = tmp_path / "ledger.json"
    mod.save_ledger(path, [rec])
    first_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    import os

    # Bump mtime backwards so any rewrite would be visible.
    older = path.stat().st_mtime_ns - 1_000_000
    os.utime(path, ns=(older, older))
    # A second save with the same record must not change the file.
    mod.save_ledger(path, [rec])
    second_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first_hash == second_hash, (
        f"Identical ledger must produce byte-identical file; first={first_hash}, second={second_hash}"
    )
    assert path.stat().st_mtime_ns == older, (
        "Idempotent save must preserve the file's mtime (no rewrite)"
    )


def test_ledger_orders_entries_deterministically(tmp_path: Path) -> None:
    """The ledger must serialize entries in a fixed order."""
    mod = _import_module()
    rec_a = mod.RejectionRecord(**_base_record(shard="monaco-latest", identifier="monaco:1"))
    rec_b = mod.RejectionRecord(**_base_record(shard="italy-latest", identifier="italy:1"))
    rec_c = mod.RejectionRecord(**_base_record(shard="monaco-latest", identifier="monaco:2"))
    path = tmp_path / "ledger.json"
    mod.save_ledger(path, [rec_b, rec_a, rec_c])
    raw = json.loads(path.read_text())
    identifiers = [entry["identifier"] for entry in raw["records"]]
    assert identifiers == sorted(identifiers), (
        f"Ledger entries must be sorted deterministically, got {identifiers}"
    )


def test_ledger_records_serialize_expected_none_as_null() -> None:
    """expected=None must serialize as JSON null, not the string 'None'."""
    mod = _import_module()
    rec = mod.RejectionRecord(**_base_record(expected=None))
    payload = rec.to_dict()
    assert payload["expected"] is None
    raw = json.dumps(payload)
    assert '"expected": null' in raw, f"Expected null in JSON, got {raw}"


def test_ledger_serialize_record_with_revision_id(tmp_path: Path) -> None:
    """Round-tripping a record must preserve the full identity."""
    mod = _import_module()
    rec = mod.RejectionRecord(**_base_record())
    path = tmp_path / "ledger.json"
    mod.save_ledger(path, [rec])
    loaded = mod.load_ledger(path)
    assert len(loaded) == 1
    assert loaded[0].identity == rec.identity


# ---------------------------------------------------------------------------
# Plan/apply split for integrity (Group D public surface)
# ---------------------------------------------------------------------------


def test_plan_integrity_normalization_is_read_only(tmp_path: Path) -> None:
    """``plan_integrity_normalization`` must NOT modify any file on disk."""
    try:
        from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
            plan_integrity_normalization,
        )
    except ImportError:
        pytest.fail(
            "rejection_ledger.plan_integrity_normalization must exist "
            "(Phase 2 group D: pure planning stage)"
        )

    pa = _pyarrow()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "monaco-latest"

    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    polygons_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.table({"polygon_id": [f"{stem}:relation:1"], "wikidata": ["Q1"]}),
        polygons_path,
        compression="snappy",
    )

    voyage_docs_path = data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    voyage_docs_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.table(
            {
                "document_id": ["Q1:wikivoyage:en:1:1"],
                "article_id": ["Q1:en:1:1"],
                "wikidata": ["Q1"],
                "project": ["wikivoyage"],
                "language": ["en"],
                "site": ["enwikivoyage"],
                "title": ["T"],
                "url": ["u"],
                "page_id": [1],
                "revision_id": [1],
                "revision_timestamp": [""],
                "retrieved_at": [""],
                "full_text": ["x"],
                "full_text_format": ["plain_text"],
                "article_length_chars": [1],
                "article_length_words": [1],
                "article_length_tokens_estimate": [1],
                "license": [""],
                "attribution": [""],
                "source_api": [""],
                "fetch_status": ["ok"],
                "fetch_error": [""],
                "content_hash": [""],
            }
        ),
        voyage_docs_path,
        compression="snappy",
    )

    voyage_sections_path = data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    voyage_sections_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.table(
            {
                "section_id": ["s1"],
                "document_id": ["Q1:wikivoyage:en:1:1"],
                "article_id": ["Q1:en:1:1"],
                "wikidata": ["Q1"],
                "project": ["wikivoyage"],
                "language": ["en"],
                "site": ["enwikivoyage"],
                "page_id": [1],
                "revision_id": [1],
                "section_index": [0],
                "heading": [""],
                "anchor": [""],
                "level": [0],
                "parent_section_id": [""],
                "section_path": ["[]"],
                "text": ["kept"],
                "text_length_chars": [4],
                "text_length_words": [1],
                "text_length_tokens_estimate": [1],
                "content_hash": [""],
                "license": [""],
                "attribution": [""],
            }
        ),
        voyage_sections_path,
        compression="snappy",
    )

    # Snapshot input files.
    polys_before = polygons_path.read_bytes()
    docs_before = voyage_docs_path.read_bytes()
    sections_before = voyage_sections_path.read_bytes()

    plan = plan_integrity_normalization(data_root, stem)

    # Inputs must be unchanged.
    assert polygons_path.read_bytes() == polys_before, "polygons file must not be touched by plan"
    assert voyage_docs_path.read_bytes() == docs_before, (
        "wikivoyage/documents must not be touched by plan"
    )
    assert voyage_sections_path.read_bytes() == sections_before, (
        "wikivoyage/sections must not be touched by plan"
    )

    # Plan must contain a retained documents table and a rejection list.
    assert hasattr(plan, "retained_documents"), "IntegrityPlan must have retained_documents"
    assert hasattr(plan, "retained_sections"), "IntegrityPlan must have retained_sections"
    assert hasattr(plan, "rejections"), "IntegrityPlan must have rejections"


def test_plan_then_apply_commit_writes_all_outputs_atomically(tmp_path: Path) -> None:
    """plan_integrity_normalization + apply_integrity_normalization is a
    transactional apply: documents, sections, and ledger are committed
    together. The ledger file is the LAST write (verified by mtime)."""
    try:
        from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
            apply_integrity_normalization,
            plan_integrity_normalization,
        )
    except ImportError:
        pytest.fail(
            "rejection_ledger.apply_integrity_normalization must exist "
            "(Phase 2 group D: transactional apply stage)"
        )

    pa = _pyarrow()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "monaco-latest"

    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    polygons_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.table({"polygon_id": [f"{stem}:relation:1"], "wikidata": ["Q1"]}),
        polygons_path,
        compression="snappy",
    )

    voyage_docs_path = data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    voyage_docs_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.table(
            {
                "document_id": ["Q99:wikivoyage:en:2:2"],  # Q99 is absent from polygons
                "article_id": ["Q99:en:2:2"],
                "wikidata": ["Q99"],
                "project": ["wikivoyage"],
                "language": ["en"],
                "site": ["enwikivoyage"],
                "title": ["T"],
                "url": ["u"],
                "page_id": [2],
                "revision_id": [2],
                "revision_timestamp": [""],
                "retrieved_at": [""],
                "full_text": ["x"],
                "full_text_format": ["plain_text"],
                "article_length_chars": [1],
                "article_length_words": [1],
                "article_length_tokens_estimate": [1],
                "license": [""],
                "attribution": [""],
                "source_api": [""],
                "fetch_status": ["ok"],
                "fetch_error": [""],
                "content_hash": [""],
            }
        ),
        voyage_docs_path,
        compression="snappy",
    )

    voyage_sections_path = data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    voyage_sections_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.table(
            {
                "section_id": ["s1"],
                "document_id": ["Q99:wikivoyage:en:2:2"],
                "article_id": ["Q99:en:2:2"],
                "wikidata": ["Q99"],
                "project": ["wikivoyage"],
                "language": ["en"],
                "site": ["enwikivoyage"],
                "page_id": [2],
                "revision_id": [2],
                "section_index": [0],
                "heading": [""],
                "anchor": [""],
                "level": [0],
                "parent_section_id": [""],
                "section_path": ["[]"],
                "text": ["cascaded"],
                "text_length_chars": [8],
                "text_length_words": [1],
                "text_length_tokens_estimate": [1],
                "content_hash": [""],
                "license": [""],
                "attribution": [""],
            }
        ),
        voyage_sections_path,
        compression="snappy",
    )

    plan = plan_integrity_normalization(data_root, stem)
    assert len(plan.rejections) == 1, (
        f"Q99 must be rejected (absent from polygons), got {len(plan.rejections)} rejections"
    )

    apply_integrity_normalization(plan)

    # Q99 must be gone from documents.
    docs = pa.parquet.read_table(voyage_docs_path).to_pylist()
    assert all(doc["wikidata"] != "Q99" for doc in docs), (
        f"Q99 must be dropped from documents, got {docs}"
    )
    # The cascaded section must be gone.
    sections = pa.parquet.read_table(voyage_sections_path).to_pylist()
    assert all(sec["document_id"] != "Q99:wikivoyage:en:2:2" for sec in sections), (
        f"Cascaded section must be dropped, got {sections}"
    )
    # The cumulative ledger must be the last commit.
    ledger_path = data_root.processed / "integrity" / "rejection_ledger.json"
    assert ledger_path.is_file(), "Cumulative ledger must be written by apply"
    assert ledger_path.stat().st_mtime_ns >= voyage_docs_path.stat().st_mtime_ns, (
        "Ledger must be written AFTER (or at the same time as) the documents file"
    )
