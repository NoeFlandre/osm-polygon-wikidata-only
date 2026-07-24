"""Phase 2.5 / Defect 6: Rejection normalization must be transactional.

The current ``apply_integrity_normalization`` sequentially overwrites
documents, sections, per-stem ledger and cumulative ledger with NO
journal. A crash between writes leaves the cumulative ledger without
the latest rejections.

The apply stage must be staged and recoverable: interruption after
documents, after sections, after per-stem ledger and after cumulative
ledger must allow a fresh process to resume safely without data loss.

Test crash injection after every boundary; verify a fresh process
converges without losing rejection history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
    LEDGER_FILENAME,
)
from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema
from osm_polygon_wikidata_only.config.paths import DataRoot


def _import_module():
    from osm_polygon_wikidata_only.augmentation import rejection_ledger as mod

    return mod


def _write_documents(path: Path, rows: list[dict]) -> None:
    if rows:
        table = pa.Table.from_pylist(rows, schema=document_schema())
    else:
        # Build a "no rows" placeholder with all required columns.
        placeholder_cols = {col: [None] for col in document_schema().names}
        table = pa.Table.from_pylist([placeholder_cols], schema=document_schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_sections(path: Path, rows: list[dict]) -> None:
    if rows:
        table = pa.Table.from_pylist(rows, schema=section_schema())
    else:
        table = pa.Table.from_pylist(
            [
                {
                    "section_id": "_placeholder",
                    "document_id": "_placeholder",
                    "article_id": "_placeholder",
                    "wikidata": "_placeholder",
                    "project": "_placeholder",
                    "language": "_placeholder",
                    "site": "_placeholder",
                    "page_id": 0,
                    "revision_id": 0,
                    "section_index": 0,
                    "heading": "",
                    "anchor": "",
                    "level": 0,
                    "parent_section_id": "",
                    "section_path": "[]",
                    "text": "",
                    "text_length_chars": 0,
                    "text_length_words": 0,
                    "text_length_tokens_estimate": 0,
                    "content_hash": "",
                    "license": "",
                    "attribution": "",
                }
            ],
            schema=section_schema(),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_polygons(data_root: Path, stem: str, qids: list[str]) -> None:
    path = data_root / "processed" / "polygons" / f"{stem}.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "polygon_id": [f"{stem}:way:{i}" for i in range(len(qids))],
                "wikidata": qids,
                "source_pbf": [f"{stem}.osm.pbf" for _ in qids],
                "region": ["r" for _ in qids],
            }
        ),
        path,
    )


def _sample_doc_row(document_id: str, qid: str) -> dict:
    return {
        "document_id": document_id,
        "article_id": f"{qid}:en:1:1",
        "wikidata": qid,
        "project": "wikivoyage",
        "language": "en",
        "site": "enwikivoyage",
        "title": "T",
        "url": "https://en.wikivoyage.org/wiki/T",
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "2026-07-24T00:00:00Z",
        "retrieved_at": "2026-07-24T00:00:00Z",
        "wikidata_label": "L",
        "wikidata_description": "D",
        "wikidata_aliases": "",
        "lead_text": "",
        "extract": "",
        "full_text": "",
        "full_text_format": "plain_text",
        "article_length_chars": 0,
        "article_length_words": 0,
        "article_length_tokens_estimate": 0,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "",
        "license": "CC-BY-SA",
        "attribution": "A",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "h",
    }


def _fresh_data_root(tmp_path: Path) -> DataRoot:
    return DataRoot(tmp_path)


def _seed(data_root: DataRoot, stem: str) -> None:
    _write_polygons(data_root.path, stem, ["Q1"])
    _write_documents(
        data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
        [_sample_doc_row("Q99:wikivoyage:en:2:2", "Q99")],
    )
    _write_sections(data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet", [])


# ---------------------------------------------------------------------------
# Crash injection after documents write
# ---------------------------------------------------------------------------


def test_crash_after_documents_write_resumes(tmp_path: Path) -> None:
    """A crash after the documents parquet write but before the
    cumulative ledger write must allow a fresh process to resume
    safely. After resume, the cumulative ledger must include the
    rejected Q99 record.
    """
    mod = _import_module()
    data_root = _fresh_data_root(tmp_path)
    data_root.ensure()
    stem = "alpha-latest"
    _seed(data_root, stem)

    plan = mod.plan_integrity_normalization(data_root, stem)
    assert len(plan.rejections) == 1

    # Patch save_ledger to crash on first call (cumulative ledger
    # write).
    real_save = mod.save_ledger

    def _crash_save(*args, **kwargs):
        raise RuntimeError("simulated crash before cumulative ledger commit")

    mod.save_ledger = _crash_save
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            mod.apply_integrity_normalization(plan)
    finally:
        mod.save_ledger = real_save

    # Documents may have been written; cumulative ledger NOT.
    # A fresh apply with the same plan must complete the transaction.
    mod.apply_integrity_normalization(plan)

    ledger_path = data_root.processed / "integrity" / LEDGER_FILENAME
    payload = json.loads(ledger_path.read_text())
    shards = {r["shard"] for r in payload["records"]}
    assert stem in shards, f"After resume, cumulative ledger must include {stem!r}; got {shards}"


# ---------------------------------------------------------------------------
# Crash injection after sections write
# ---------------------------------------------------------------------------


def test_crash_after_sections_write_resumes(tmp_path: Path) -> None:
    mod = _import_module()
    data_root = _fresh_data_root(tmp_path)
    data_root.ensure()
    stem = "alpha-latest"
    _seed(data_root, stem)

    plan = mod.plan_integrity_normalization(data_root, stem)
    real_save = mod.save_ledger

    call_count = {"n": 0}

    def _crash_after_sections(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash after sections write")
        return real_save(*args, **kwargs)

    mod.save_ledger = _crash_after_sections
    try:
        with pytest.raises(RuntimeError, match="simulated crash after sections"):
            mod.apply_integrity_normalization(plan)
    finally:
        mod.save_ledger = real_save

    mod.apply_integrity_normalization(plan)

    ledger_path = data_root.processed / "integrity" / LEDGER_FILENAME
    payload = json.loads(ledger_path.read_text())
    shards = {r["shard"] for r in payload["records"]}
    assert stem in shards


# ---------------------------------------------------------------------------
# Crash injection after per-stem ledger write but before cumulative
# ---------------------------------------------------------------------------


def test_crash_after_per_stem_ledger_resumes(tmp_path: Path) -> None:
    mod = _import_module()
    data_root = _fresh_data_root(tmp_path)
    data_root.ensure()
    stem = "alpha-latest"
    _seed(data_root, stem)

    plan = mod.plan_integrity_normalization(data_root, stem)
    real_save = mod.save_ledger

    # First save_ledger is the per-stem ledger (we want to crash on
    # the SECOND call -- the cumulative ledger merge).
    call_count = {"n": 0}

    def _crash_after_per_stem(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_save(*args, **kwargs)
        raise RuntimeError("simulated crash after per-stem ledger")

    mod.save_ledger = _crash_after_per_stem
    try:
        with pytest.raises(RuntimeError, match="simulated crash after per-stem"):
            mod.apply_integrity_normalization(plan)
    finally:
        mod.save_ledger = real_save

    mod.apply_integrity_normalization(plan)

    ledger_path = data_root.processed / "integrity" / LEDGER_FILENAME
    payload = json.loads(ledger_path.read_text())
    shards = {r["shard"] for r in payload["records"]}
    assert stem in shards


# ---------------------------------------------------------------------------
# Fresh process converges after crash
# ---------------------------------------------------------------------------


def test_fresh_process_resume_after_crash_converges(tmp_path: Path) -> None:
    """After a crash mid-transaction, a brand-new process that does
    NOT share state with the failed one must still converge by
    re-running the plan against the source artifacts that have NOT
    yet been normalized.
    """
    mod = _import_module()
    data_root = _fresh_data_root(tmp_path)
    data_root.ensure()
    stem = "alpha-latest"
    _seed(data_root, stem)

    plan = mod.plan_integrity_normalization(data_root, stem)
    real_save = mod.save_ledger

    def _always_crash(*args, **kwargs):
        raise RuntimeError("crash")

    mod.save_ledger = _always_crash
    try:
        with pytest.raises(RuntimeError):
            mod.apply_integrity_normalization(plan)
    finally:
        mod.save_ledger = real_save

    # Documents/sections may have been overwritten by the failed
    # apply. Re-seed the SOURCE artifacts and re-apply with the
    # original plan to simulate a fresh process restart.
    _seed(data_root, stem)
    mod.apply_integrity_normalization(plan)

    ledger_path = data_root.processed / "integrity" / LEDGER_FILENAME
    payload = json.loads(ledger_path.read_text())
    shards = {r["shard"] for r in payload["records"]}
    assert stem in shards
