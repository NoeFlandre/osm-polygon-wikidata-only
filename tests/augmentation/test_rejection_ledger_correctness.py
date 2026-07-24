"""Phase 2 / Amendment 6: Rejection ledger correctness.

The cumulative ledger at ``processed/integrity/rejection_ledger.json``
must merge (not replace) across runs. Each run must preserve prior
rejection history deterministically. The apply stage must be
journaled/recoverable: a crash before the ledger commit must not lose
prior rejections, and a crash after the documents/sections commit
but before the ledger commit must still allow roll-forward.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _import_module():
    from osm_polygon_wikidata_only.augmentation import rejection_ledger as mod

    return mod


def _write_documents(path: Path, rows: list[dict]) -> None:
    from osm_polygon_wikidata_only.augmentation.schema import document_schema

    if rows:
        table = pa.Table.from_pylist(rows, schema=document_schema())
    else:
        # Provide a minimal "stub" row matching every required column to
        # the empty schema (will be filtered by apply stage).
        table = pa.Table.from_pylist(
            [
                {
                    col: None
                    for col in [
                        "document_id",
                        "article_id",
                        "wikidata",
                        "project",
                        "language",
                        "site",
                        "page_id",
                        "revision_id",
                        "revision_timestamp",
                        "retrieved_at",
                        "wikidata_label",
                        "wikidata_description",
                        "wikidata_aliases",
                        "lead_text",
                        "extract",
                        "full_text",
                        "full_text_format",
                        "article_length_chars",
                        "article_length_words",
                        "article_length_tokens_estimate",
                        "thumbnail_url",
                        "categories",
                        "license",
                        "attribution",
                        "source_api",
                        "fetch_status",
                        "fetch_error",
                        "content_hash",
                    ]
                }
            ],
            schema=document_schema(),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_sections(path: Path, rows: list[dict]) -> None:
    from osm_polygon_wikidata_only.augmentation.schema import section_schema

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
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"polygon_id": f"{stem}:way:{i}", "wikidata": qid} for i, qid in enumerate(qids)]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "polygon_id": [r["polygon_id"] for r in rows],
                "wikidata": [r["wikidata"] for r in rows],
            }
        ),
        path,
    )


def _sample_doc_row(document_id: str, qid: str, language: str = "en") -> dict:
    return {
        "document_id": document_id,
        "article_id": f"{qid}:{language}:1:1",
        "wikidata": qid,
        "project": "wikivoyage",
        "language": language,
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


def test_apply_merges_with_existing_cumulative_ledger(tmp_path: Path) -> None:
    """A second apply for stem B must NOT erase prior rejections for stem A."""
    mod = _import_module()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(tmp_path)
    dr.ensure()

    stem_a = "alpha-latest"
    stem_b = "beta-latest"

    _write_polygons(tmp_path, stem_a, ["Q1"])
    _write_documents(
        dr.processed / "wikivoyage" / "documents" / f"{stem_a}.parquet",
        [_sample_doc_row("Q99:wikivoyage:en:2:2", "Q99")],
    )
    _write_sections(dr.processed / "wikivoyage" / "sections" / f"{stem_a}.parquet", [])

    plan_a = mod.plan_integrity_normalization(dr, stem_a)
    mod.apply_integrity_normalization(plan_a)
    ledger_path = dr.processed / "integrity" / "rejection_ledger.json"
    payload_a = json.loads(ledger_path.read_text())
    assert any(r["shard"] == stem_a for r in payload_a["records"])

    # Now stem B with its own rejection.
    _write_polygons(tmp_path, stem_b, ["Q10"])
    _write_documents(
        dr.processed / "wikivoyage" / "documents" / f"{stem_b}.parquet",
        [_sample_doc_row("Q88:wikivoyage:en:2:2", "Q88")],
    )
    _write_sections(dr.processed / "wikivoyage" / "sections" / f"{stem_b}.parquet", [])

    plan_b = mod.plan_integrity_normalization(dr, stem_b)
    mod.apply_integrity_normalization(plan_b)
    payload_b = json.loads(ledger_path.read_text())
    shards_in_ledger = {r["shard"] for r in payload_b["records"]}
    assert stem_a in shards_in_ledger, (
        f"apply(stem_b) erased prior rejection history for stem_a: {payload_b}"
    )
    assert stem_b in shards_in_ledger


def test_apply_is_recoverable_after_mid_flight_crash(tmp_path: Path) -> None:
    """A crash before the cumulative-ledger write must allow a fresh
    apply to complete (idempotent roll-forward)."""
    mod = _import_module()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(tmp_path)
    dr.ensure()
    stem = "alpha-latest"
    _write_polygons(tmp_path, stem, ["Q1"])
    _write_documents(
        dr.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
        [_sample_doc_row("Q99:wikivoyage:en:2:2", "Q99")],
    )
    _write_sections(dr.processed / "wikivoyage" / "sections" / f"{stem}.parquet", [])

    plan = mod.plan_integrity_normalization(dr, stem)
    # Patch save_ledger to crash once on first call, then succeed.
    import osm_polygon_wikidata_only.augmentation.rejection_ledger as rl

    calls = {"count": 0}
    real_save_ledger = rl.save_ledger

    def _crash_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated crash before ledger commit")
        return real_save_ledger(*args, **kwargs)

    rl.save_ledger = _crash_once
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            mod.apply_integrity_normalization(plan)
    finally:
        rl.save_ledger = real_save_ledger

    # The documents/sections may already be on disk; re-running apply
    # with the original plan should reach ledger commit.
    mod.apply_integrity_normalization(plan)
    ledger_path = dr.processed / "integrity" / "rejection_ledger.json"
    payload = json.loads(ledger_path.read_text())
    assert any(r["shard"] == stem for r in payload["records"]), (
        "Re-applying after crash must still commit the cumulative ledger"
    )


def test_rejection_history_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Two identical runs (same plan twice) must produce the same
    ledger bytes -- idempotent re-apply."""
    mod = _import_module()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(tmp_path)
    dr.ensure()
    stem = "alpha-latest"
    _write_polygons(tmp_path, stem, ["Q1"])
    _write_documents(
        dr.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
        [_sample_doc_row("Q99:wikivoyage:en:2:2", "Q99")],
    )
    _write_sections(dr.processed / "wikivoyage" / "sections" / f"{stem}.parquet", [])

    plan = mod.plan_integrity_normalization(dr, stem)
    mod.apply_integrity_normalization(plan)
    first = (dr.processed / "integrity" / "rejection_ledger.json").read_bytes()

    # Run again -- idempotent re-apply must produce the same bytes
    # (apply on a stem whose plan is already applied).
    mod.apply_integrity_normalization(plan)
    second = (dr.processed / "integrity" / "rejection_ledger.json").read_bytes()
    assert first == second, (
        f"Re-applying the same plan must not change the cumulative ledger bytes; "
        f"first={first!r}, second={second!r}"
    )
