"""Audit execution, validation, and determinism checks for legacy tables and sidecars."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    document_schema,
    section_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    article_schema,
    polygon_article_schema,
)
from tests.migration.audit import capture_dataset_fingerprint, run_audit, write_audit_report

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"


def test_wikipedia_storage_audit_on_fixtures(tmp_path: Path) -> None:
    # Run audit on Monaco fixtures
    report = run_audit(FIXTURES_ROOT)

    # 1. Assertions on Monaco fixture stem
    assert "monaco-latest" in report["per_stem"]
    stem_info = report["per_stem"]["monaco-latest"]
    # Monaco documents currently lack canonical columns, so it must be both_needing_schema_upgrade
    assert stem_info["state"] == "both_needing_schema_upgrade"
    assert len(stem_info["discrepancies"]) == 0
    assert stem_info["article_rows"] == 1
    assert stem_info["document_rows"] == 1
    assert stem_info["section_rows"] == 1
    assert stem_info["link_rows"] == 1

    # 2. Assertions on aggregate counts
    counts = report["aggregate_counts"]
    assert counts["article_files"] == 1
    assert counts["wikipedia_document_files"] == 1
    assert counts["wikipedia_section_files"] == 1
    assert counts["polygon_article_link_files"] == 1
    assert counts["article_rows"] == 1
    assert counts["wikipedia_document_rows"] == 1
    assert counts["wikipedia_section_rows"] == 1
    assert counts["polygon_article_link_rows"] == 1
    assert counts["shared_article_document_rows"] == 1
    assert counts["stems_by_state"]["both_needing_schema_upgrade"] == 1
    assert counts["stems_by_state"]["both_equivalent"] == 0
    assert counts["stems_by_state"]["articles_only"] == 0
    assert counts["stems_by_state"]["documents_only"] == 0
    assert counts["stems_by_state"]["conflicting"] == 0
    assert counts["stems_by_state"]["orphaned"] == 0
    assert counts["conflicting_stem_count"] == 0
    assert counts["discrepancy_count"] == 0
    assert counts["unreadable_file_count"] == 0
    assert counts["duplicate_primary_id_count"] == 0
    assert counts["total_unresolved_links"] == 0
    assert counts["total_unresolved_sections"] == 0

    # 3. Schema overlap summary validation (derived from imports)
    summary = report["schema_overlap_summary"]
    assert summary["articles_columns_count"] == len(ARTICLE_COLUMNS)
    assert summary["documents_columns_count"] == len(DOCUMENT_COLUMNS)
    assert summary["shared_columns_count"] == len(set(ARTICLE_COLUMNS) & set(DOCUMENT_COLUMNS))
    assert len(summary["articles_only_columns"]) == len(
        set(ARTICLE_COLUMNS) - set(DOCUMENT_COLUMNS)
    )
    assert len(summary["documents_only_columns"]) == len(
        set(DOCUMENT_COLUMNS) - set(ARTICLE_COLUMNS)
    )

    # 4. Safe to migrate flag
    assert report["safe_to_migrate"] is True
    assert len(report["blocking_reasons"]) == 0

    # 5. Deterministic serialization and double-run check (using isolated tmp_path)
    tmp_report_path = tmp_path / "monaco-fixtures-audit.json"
    write_audit_report(report, tmp_report_path)
    content_1 = tmp_report_path.read_bytes()

    # Run again
    report_2 = run_audit(FIXTURES_ROOT)
    write_audit_report(report_2, tmp_report_path)
    content_2 = tmp_report_path.read_bytes()

    assert content_1 == content_2, "Audit reports are not byte-identical between runs"


def test_wikipedia_storage_audit_on_real_data() -> None:
    opt_in = os.environ.get("RUN_REAL_WIKIPEDIA_STORAGE_AUDIT")
    env_root = os.environ.get("OSM_POLYGON_DATA_ROOT")

    if opt_in != "1" or not env_root:
        pytest.skip(
            "Real-data audit is opt-in and requires both RUN_REAL_WIKIPEDIA_STORAGE_AUDIT=1 "
            "and OSM_POLYGON_DATA_ROOT environment variables to be set."
        )

    data_root_path = Path(env_root).expanduser()
    if not data_root_path.exists() or not data_root_path.is_dir():
        pytest.fail(
            f"OSM_POLYGON_DATA_ROOT path '{data_root_path}' does not exist or is not a directory."
        )

    output_path = Path("/tmp/wikipedia-storage-audit.json")

    # Mutation protection: Capture dataset fingerprint before audit
    fingerprint_before = capture_dataset_fingerprint(data_root_path)

    # Run audit twice to assert determinism and byte-stable output
    report_1 = run_audit(data_root_path)
    write_audit_report(report_1, output_path)
    report_bytes_1 = output_path.read_bytes()

    report_2 = run_audit(data_root_path)
    write_audit_report(report_2, output_path)
    report_bytes_2 = output_path.read_bytes()

    assert report_bytes_1 == report_bytes_2, "Real-data audit reports are not byte-identical"

    # Mutation protection: Capture dataset fingerprint after audit and compare
    fingerprint_after = capture_dataset_fingerprint(data_root_path)
    assert fingerprint_before == fingerprint_after, "Dataset was modified during audit!"

    # Additional validations on report structure
    report = json.loads(report_bytes_1)
    assert "aggregate_counts" in report
    assert "byte_totals" in report
    assert "per_stem" in report
    assert "schema_overlap_summary" in report
    assert "safe_to_migrate" in report
    assert "blocking_reasons" in report

    # Make sure no absolute machine paths or credentials exist in the report
    serialized_text = report_bytes_1.decode("utf-8")
    assert "/Volumes/" not in serialized_text
    assert "/Users/" not in serialized_text
    assert "API_KEY" not in serialized_text
    assert "PASSWORD" not in serialized_text


def test_real_data_audit_skip_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Proves that merely having the conventional path or OSM_POLYGON_DATA_ROOT set does not run the test
    # without RUN_REAL_WIKIPEDIA_STORAGE_AUDIT=1.
    monkeypatch.setenv(
        "OSM_POLYGON_DATA_ROOT", "/Volumes/Seagate M3/projects/osm-polygon-wikidata-only"
    )
    monkeypatch.delenv("RUN_REAL_WIKIPEDIA_STORAGE_AUDIT", raising=False)

    with pytest.raises(pytest.skip.Exception):
        test_wikipedia_storage_audit_on_real_data()


def _setup_tmp_dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    art_dir = processed / "articles"
    doc_dir = processed / "wikipedia" / "documents"
    sec_dir = processed / "wikipedia" / "sections"
    link_dir = processed / "polygon_articles"

    art_dir.mkdir(exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    sec_dir.mkdir(parents=True, exist_ok=True)
    link_dir.mkdir(exist_ok=True)

    return processed, art_dir, doc_dir, sec_dir, link_dir


def _make_dummy_article(
    wikidata: str = "Q123",
    language: str = "en",
    page_id: int = 456,
    revision_id: int = 789,
    full_text: str = "This is full text.",
    content_hash: str = "hash123",
    article_id: str | None = None,
    **kwargs: Any,
) -> dict:
    a_id = article_id or f"{wikidata}:{language}:{page_id}:{revision_id}"
    row = {
        "article_id": a_id,
        "wikidata": wikidata,
        "language": language,
        "site": "enwiki",
        "title": "Monaco",
        "url": "https://en.wikipedia.org/wiki/Monaco",
        "page_id": page_id,
        "revision_id": revision_id,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "wikidata_label": "Monaco",
        "wikidata_description": "Country",
        "wikidata_aliases": "[]",
        "lead_text": "Monaco text",
        "extract": "extract",
        "full_text": full_text,
        "full_text_format": "plain_text",
        "article_length_chars": len(full_text),
        "article_length_words": len(full_text.split()),
        "article_length_tokens_estimate": len(full_text) // 4,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "[]",
        "license": "CC",
        "attribution": "Wikipedia",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": content_hash,
    }
    row.update(kwargs)
    return row


def _make_dummy_document(
    wikidata: str = "Q123",
    language: str = "en",
    page_id: int = 456,
    revision_id: int = 789,
    full_text: str = "This is full text.",
    content_hash: str = "hash123",
    article_id: str | None = None,
    document_id: str | None = None,
    project: str = "wikipedia",
    include_canonical_cols: bool = False,
    **kwargs: Any,
) -> dict:
    a_id = article_id or f"{wikidata}:{language}:{page_id}:{revision_id}"
    d_id = document_id or f"{wikidata}:wikipedia:{language}:{page_id}:{revision_id}"
    row = {
        "document_id": d_id,
        "article_id": a_id,
        "wikidata": wikidata,
        "project": project,
        "language": language,
        "site": "enwiki",
        "title": "Monaco",
        "url": "https://en.wikipedia.org/wiki/Monaco",
        "page_id": page_id,
        "revision_id": revision_id,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "full_text": full_text,
        "full_text_format": "plain_text",
        "article_length_chars": len(full_text),
        "article_length_words": len(full_text.split()),
        "article_length_tokens_estimate": len(full_text) // 4,
        "license": "CC",
        "attribution": "Wikipedia",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": content_hash,
    }
    if include_canonical_cols:
        row.update(
            {
                "wikidata_label": "Monaco",
                "wikidata_description": "Country",
                "wikidata_aliases": "[]",
                "lead_text": "Monaco text",
                "extract": "extract",
                "thumbnail_url": "",
                "thumbnail_width": None,  # genuine match to None
                "thumbnail_height": None,  # genuine match to None
                "categories": "[]",
            }
        )
    row.update(kwargs)
    return row


def _make_dummy_section(
    section_id: str | None = None,
    document_id: str | None = None,
    article_id: str | None = None,
    wikidata: str = "Q123",
    language: str = "en",
    page_id: int = 456,
    revision_id: int = 789,
) -> dict:
    a_id = article_id or f"{wikidata}:{language}:{page_id}:{revision_id}"
    d_id = document_id or f"{wikidata}:wikipedia:{language}:{page_id}:{revision_id}"
    s_id = section_id or f"{d_id}:0"
    return {
        "section_id": s_id,
        "document_id": d_id,
        "article_id": a_id,
        "wikidata": wikidata,
        "project": "wikipedia",
        "language": language,
        "site": "enwiki",
        "page_id": page_id,
        "revision_id": revision_id,
        "section_index": 0,
        "heading": "Heading",
        "anchor": "Anchor",
        "level": 2,
        "parent_section_id": "",
        "section_path": "Path",
        "text": "Section text",
        "text_length_chars": 12,
        "text_length_words": 2,
        "text_length_tokens_estimate": 3,
        "content_hash": "sechash1",
        "license": "CC",
        "attribution": "Wikipedia",
    }


def _make_dummy_link(
    polygon_id: str = "monaco-latest:way:1",
    article_id: str | None = None,
    wikidata: str = "Q123",
    language: str = "en",
) -> dict:
    a_id = article_id or f"{wikidata}:{language}:456:789"
    return {
        "polygon_id": polygon_id,
        "article_id": a_id,
        "wikidata": wikidata,
        "language": language,
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "way",
        "osm_id": 1,
        "page_id": 456,
        "revision_id": 789,
        "is_best_language": True,
    }


def test_orphaned_classification_states(tmp_path: Path) -> None:
    _, art_dir, doc_dir, sec_dir, link_dir = _setup_tmp_dataset(tmp_path)

    # 1. Clean article without documents or sections -> articles_only
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_article()], schema=article_schema()),
        art_dir / "clean-article.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["clean-article"]["state"] == "articles_only"

    # 2. Article + sections but no document -> orphaned
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
        art_dir / "art-sec-no-doc.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_section(wikidata="Q1")], schema=section_schema()),
        sec_dir / "art-sec-no-doc.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["art-sec-no-doc"]["state"] == "orphaned"

    # 3. Link without article or document -> orphaned
    pq.write_table(
        pa.Table.from_pylist(
            [_make_dummy_link(wikidata="Q2", article_id="unresolved")],
            schema=polygon_article_schema(),
        ),
        link_dir / "link-only.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["link-only"]["state"] == "orphaned"

    # 4. Section-only -> orphaned
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_section(wikidata="Q3")], schema=section_schema()),
        sec_dir / "section-only.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["section-only"]["state"] == "orphaned"

    # 5. Document + sections but no source article -> orphaned
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_document(wikidata="Q4")], schema=document_schema()),
        doc_dir / "doc-sec-no-art.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_section(wikidata="Q4")], schema=section_schema()),
        sec_dir / "doc-sec-no-art.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["doc-sec-no-art"]["state"] == "orphaned"
    assert res["safe_to_migrate"] is False


def test_article_link_classification_states(tmp_path: Path) -> None:
    _, art_dir, _, _, link_dir = _setup_tmp_dataset(tmp_path)

    # 1. Article + valid links + no document = articles_only and safe backfill candidate
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
        art_dir / "valid-links.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_link(wikidata="Q1")], schema=polygon_article_schema()),
        link_dir / "valid-links.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["valid-links"]["state"] == "articles_only"
    assert res["safe_to_migrate"] is True

    # 2. Article + unresolved links + no document = unsafe (safe_to_migrate: false)
    _, art_dir_unres, _, _, link_dir_unres = _setup_tmp_dataset(tmp_path / "unresolved")
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
        art_dir_unres / "unres-links.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_make_dummy_link(wikidata="Q2", article_id="Q2:en:456:789")],
            schema=polygon_article_schema(),
        ),
        link_dir_unres / "unres-links.parquet",
    )
    res_unres = run_audit(tmp_path / "unresolved")
    assert res_unres["safe_to_migrate"] is False
    assert res_unres["aggregate_counts"]["total_unresolved_links"] > 0


def test_state_both_equivalent_upgrade(tmp_path: Path) -> None:
    _, art_dir, doc_dir, _, _ = _setup_tmp_dataset(tmp_path)

    # 1. Lack of canonical fields -> both_needing_schema_upgrade
    pq.write_table(
        pa.Table.from_pylist([_make_dummy_article(wikidata="Q10")], schema=article_schema()),
        art_dir / "test-upgrade.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_make_dummy_document(wikidata="Q10", include_canonical_cols=False)],
            schema=document_schema(),
        ),
        doc_dir / "test-upgrade.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["test-upgrade"]["state"] == "both_needing_schema_upgrade"

    # 2. Complete future canonical schema superset -> both_equivalent
    doc_upgraded_schema = pa.schema(
        [
            *document_schema(),
            pa.field("wikidata_label", pa.string()),
            pa.field("wikidata_description", pa.string()),
            pa.field("wikidata_aliases", pa.string()),
            pa.field("lead_text", pa.string()),
            pa.field("extract", pa.string()),
            pa.field("thumbnail_url", pa.string()),
            pa.field("thumbnail_width", pa.int64()),
            pa.field("thumbnail_height", pa.int64()),
            pa.field("categories", pa.string()),
        ]
    )

    pq.write_table(
        pa.Table.from_pylist(
            [_make_dummy_document(wikidata="Q10", include_canonical_cols=True)],
            schema=doc_upgraded_schema,
        ),
        doc_dir / "test-upgrade.parquet",
    )
    res = run_audit(tmp_path)
    assert res["per_stem"]["test-upgrade"]["state"] == "both_equivalent"


@pytest.mark.parametrize(
    "case_name, expected_reason",
    [
        ("case1", "Schema type mismatch"),
        ("case2", "Value mismatch"),
        ("case3", "Value mismatch"),
        ("case4", "Value mismatch"),
        ("case5", "Value mismatch"),
        ("case6", "Value mismatch"),
        ("case7", "Article ID set mismatch"),
        ("case8", "Deterministic document_id mismatch"),
        ("case9", "Project column is not 'wikipedia'"),
        ("case10", "Duplicate article_id values found"),
        ("case11", "Duplicate document_id values found"),
        ("case12", "Duplicate section_id values found"),
        ("case13", "Duplicate polygon-article links found"),
        ("case14", "unresolved article_id 'unresolved'"),
        ("case15", "references unresolved article_id 'unresolved'"),
        ("case16", "references unresolved document_id 'unresolved'"),
        ("case17", "unreadable Parquet"),
        ("case18", "unreadable Parquet"),
        ("case19", "unreadable Parquet"),
        ("case20", "unreadable Parquet"),
        ("case21", "Unknown extra column"),
        ("case22", "Missing column"),
        ("case23", "clean_articles_only"),
        ("case24_label", "Value mismatch.*wikidata_label"),
        ("case25_description", "Value mismatch.*wikidata_description"),
        ("case26_aliases", "Value mismatch.*wikidata_aliases"),
        ("case27_lead", "Value mismatch.*lead_text"),
        ("case28_extract", "Value mismatch.*extract"),
        ("case29_thumb_url", "Value mismatch.*thumbnail_url"),
        ("case30_thumb_width", "Value mismatch.*thumbnail_width"),
        ("case31_thumb_height", "Value mismatch.*thumbnail_height"),
        ("case32_categories", "Value mismatch.*categories"),
    ],
)
def test_synthetic_failure_modes(tmp_path: Path, case_name: str, expected_reason: str) -> None:
    base_dir = tmp_path
    _, art_dir, doc_dir, sec_dir, link_dir = _setup_tmp_dataset(base_dir)

    if case_name == "case1":
        art_wrong_schema = pa.schema(
            [
                pa.field(c, pa.float64() if c == "page_id" else f.type)
                for c, f in zip(ARTICLE_COLUMNS, article_schema())
            ]
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article()], schema=art_wrong_schema),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document()], schema=document_schema()),
            doc_dir / "case.parquet",
        )

    elif case_name == "case2":
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_article(wikidata="Q1", title="")], schema=article_schema()
            ),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_document(wikidata="Q1", title=None)], schema=document_schema()
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case3":
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_article(wikidata="Q1", title="A ")], schema=article_schema()
            ),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_document(wikidata="Q1", title="A")], schema=document_schema()
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case4":
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_article(wikidata="Q1", full_text="text A")], schema=article_schema()
            ),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_document(wikidata="Q1", full_text="text B")], schema=document_schema()
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case5":
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_article(wikidata="Q1", content_hash="hashA")], schema=article_schema()
            ),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_document(wikidata="Q1", content_hash="hashB")],
                schema=document_schema(),
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case6":
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _make_dummy_article(
                        wikidata="Q1",
                        article_id="Q1:en:456:shared",
                        revision_id=100,
                    )
                ],
                schema=article_schema(),
            ),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _make_dummy_document(
                        wikidata="Q1",
                        article_id="Q1:en:456:shared",
                        revision_id=200,
                    )
                ],
                schema=document_schema(),
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case7":
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_article(wikidata="Q1", revision_id=1)], schema=article_schema()
            ),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_document(wikidata="Q1", revision_id=2)], schema=document_schema()
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case8":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_document(wikidata="Q1", document_id="invalid_id")],
                schema=document_schema(),
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case9":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_document(wikidata="Q1", project="wikivoyage")],
                schema=document_schema(),
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case10":
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _make_dummy_article(wikidata="Q1"),
                    _make_dummy_article(wikidata="Q1"),
                ],
                schema=article_schema(),
            ),
            art_dir / "case.parquet",
        )

    elif case_name == "case11":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _make_dummy_document(wikidata="Q1"),
                    _make_dummy_document(wikidata="Q1"),
                ],
                schema=document_schema(),
            ),
            doc_dir / "case.parquet",
        )

    elif case_name == "case12":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document(wikidata="Q1")], schema=document_schema()),
            doc_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _make_dummy_section(wikidata="Q1"),
                    _make_dummy_section(wikidata="Q1"),
                ],
                schema=section_schema(),
            ),
            sec_dir / "case.parquet",
        )

    elif case_name == "case13":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document(wikidata="Q1")], schema=document_schema()),
            doc_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _make_dummy_link(wikidata="Q1"),
                    _make_dummy_link(wikidata="Q1"),
                ],
                schema=polygon_article_schema(),
            ),
            link_dir / "case.parquet",
        )

    elif case_name == "case14":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document(wikidata="Q1")], schema=document_schema()),
            doc_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_link(wikidata="Q100", article_id="unresolved")],
                schema=polygon_article_schema(),
            ),
            link_dir / "case.parquet",
        )

    elif case_name == "case15":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document(wikidata="Q1")], schema=document_schema()),
            doc_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_section(wikidata="Q1", article_id="unresolved")],
                schema=section_schema(),
            ),
            sec_dir / "case.parquet",
        )

    elif case_name == "case16":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document(wikidata="Q1")], schema=document_schema()),
            doc_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [_make_dummy_section(wikidata="Q1", document_id="unresolved")],
                schema=section_schema(),
            ),
            sec_dir / "case.parquet",
        )

    elif case_name == "case17":
        (art_dir / "case.parquet").write_bytes(b"corrupt bytes")

    elif case_name == "case18":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        (doc_dir / "case.parquet").write_bytes(b"corrupt bytes")

    elif case_name == "case19":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document(wikidata="Q1")], schema=document_schema()),
            doc_dir / "case.parquet",
        )
        (sec_dir / "case.parquet").write_bytes(b"corrupt bytes")

    elif case_name == "case20":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_document(wikidata="Q1")], schema=document_schema()),
            doc_dir / "case.parquet",
        )
        (link_dir / "case.parquet").write_bytes(b"corrupt bytes")

    elif case_name == "case21":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        doc_bad_schema = pa.schema(
            [
                *document_schema(),
                pa.field("unknown_column_extra", pa.string()),
            ]
        )
        doc_row = _make_dummy_document(wikidata="Q1", unknown_column_extra="value")
        pq.write_table(
            pa.Table.from_pylist([doc_row], schema=doc_bad_schema),
            doc_dir / "case.parquet",
        )

    elif case_name == "case22":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )
        doc_missing_schema = pa.schema([f for f in document_schema() if f.name != "url"])
        doc_row = _make_dummy_document(wikidata="Q1")
        del doc_row["url"]
        pq.write_table(
            pa.Table.from_pylist([doc_row], schema=doc_missing_schema),
            doc_dir / "case.parquet",
        )

    elif case_name == "case23":
        pq.write_table(
            pa.Table.from_pylist([_make_dummy_article(wikidata="Q1")], schema=article_schema()),
            art_dir / "case.parquet",
        )

    elif (
        case_name.startswith("case24_")
        or case_name.startswith("case25_")
        or case_name.startswith("case26_")
        or case_name.startswith("case27_")
        or case_name.startswith("case28_")
        or case_name.startswith("case29_")
        or case_name.startswith("case30_")
        or case_name.startswith("case31_")
        or case_name.startswith("case32_")
    ):
        col_to_mismatch = case_name.split("_")[1]
        if col_to_mismatch == "thumb":
            col_to_mismatch = "thumbnail_" + case_name.split("_")[2]
        elif col_to_mismatch == "label":
            col_to_mismatch = "wikidata_label"
        elif col_to_mismatch == "description":
            col_to_mismatch = "wikidata_description"
        elif col_to_mismatch == "aliases":
            col_to_mismatch = "wikidata_aliases"
        elif col_to_mismatch == "lead":
            col_to_mismatch = "lead_text"

        art_row = _make_dummy_article(wikidata="Q1")
        doc_upgraded_schema = pa.schema(
            [
                *document_schema(),
                pa.field("wikidata_label", pa.string()),
                pa.field("wikidata_description", pa.string()),
                pa.field("wikidata_aliases", pa.string()),
                pa.field("lead_text", pa.string()),
                pa.field("extract", pa.string()),
                pa.field("thumbnail_url", pa.string()),
                pa.field("thumbnail_width", pa.int64()),
                pa.field("thumbnail_height", pa.int64()),
                pa.field("categories", pa.string()),
            ]
        )
        doc_row = _make_dummy_document(wikidata="Q1", include_canonical_cols=True)

        # Apply mismatch
        if col_to_mismatch == "thumbnail_width":
            art_row["thumbnail_width"] = None
            doc_row["thumbnail_width"] = 0
        elif col_to_mismatch == "thumbnail_height":
            art_row["thumbnail_height"] = None
            doc_row["thumbnail_height"] = 0
        else:
            doc_row[col_to_mismatch] = "mismatched_value"

        pq.write_table(
            pa.Table.from_pylist([art_row], schema=article_schema()),
            art_dir / "case.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([doc_row], schema=doc_upgraded_schema),
            doc_dir / "case.parquet",
        )

    # Run audit
    report = run_audit(base_dir)
    if case_name == "case23":
        assert report["safe_to_migrate"] is True
        assert report["per_stem"]["case"]["state"] == "articles_only"
    else:
        assert report["safe_to_migrate"] is False
        reasons_str = " ".join(report["blocking_reasons"])
        assert re.search(expected_reason, reasons_str) is not None, (
            f"Expected {expected_reason} in {reasons_str}"
        )


def test_no_absolute_paths_in_serialized_report(tmp_path: Path) -> None:
    _, art_dir, _, _, _ = _setup_tmp_dataset(tmp_path)
    (art_dir / "corrupt-file.parquet").write_bytes(b"corrupt bytes")

    report = run_audit(tmp_path)
    assert report["safe_to_migrate"] is False

    serialized = json.dumps(report)
    assert "private" not in serialized
    assert "Volumes" not in serialized
    assert "Users" not in serialized

    # Test sanitize_error directly to confirm paths are replaced
    from tests.migration.audit import sanitize_error

    fake_exc = Exception(f"Failed to open /Users/noeflandre/file.parquet and {tmp_path}/processed")
    sanitized = sanitize_error(fake_exc, tmp_path)
    assert "Users" not in sanitized
    assert "private" not in sanitized
    assert "USER_HOME" in sanitized or "DATA_ROOT" in sanitized


def test_hash_seed_determinism(tmp_path: Path) -> None:
    import subprocess
    import sys

    # Write a dataset that causes failures (e.g. mismatched values) to generate discrepancies
    _, art_dir, doc_dir, _, _ = _setup_tmp_dataset(tmp_path / "dataset")
    pq.write_table(
        pa.Table.from_pylist(
            [_make_dummy_article(wikidata="Q1", title="A")], schema=article_schema()
        ),
        art_dir / "mismatch.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_make_dummy_document(wikidata="Q1", title="B")], schema=document_schema()
        ),
        doc_dir / "mismatch.parquet",
    )

    report_path_1 = tmp_path / "report_1.json"
    report_path_2 = tmp_path / "report_2.json"

    # Run with PYTHONHASHSEED=42
    env1 = os.environ.copy()
    env1["PYTHONHASHSEED"] = "42"
    cmd1 = [
        sys.executable,
        "-c",
        f"from tests.migration.audit import run_audit, write_audit_report; "
        f"from pathlib import Path; "
        f"report = run_audit(Path('{tmp_path / 'dataset'}')); "
        f"write_audit_report(report, Path('{report_path_1}'))",
    ]
    subprocess.run(cmd1, env=env1, check=True)

    # Run with PYTHONHASHSEED=123
    env2 = os.environ.copy()
    env2["PYTHONHASHSEED"] = "123"
    cmd2 = [
        sys.executable,
        "-c",
        f"from tests.migration.audit import run_audit, write_audit_report; "
        f"from pathlib import Path; "
        f"report = run_audit(Path('{tmp_path / 'dataset'}')); "
        f"write_audit_report(report, Path('{report_path_2}'))",
    ]
    subprocess.run(cmd2, env=env2, check=True)

    content_1 = report_path_1.read_bytes()
    content_2 = report_path_2.read_bytes()
    assert content_1 == content_2, (
        "Reports generated under different PYTHONHASHSEEDs are not byte-identical"
    )
