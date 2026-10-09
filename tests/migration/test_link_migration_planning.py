"""Classify link shards as legacy, canonical, or blocked using real Parquet schemas."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pyarrow import ArrowInvalid

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    document_schema,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
    polygon_schema,
)
from osm_polygon_wikidata_only.pipeline import link_migration
from osm_polygon_wikidata_only.pipeline._link_migration import application as link_application
from osm_polygon_wikidata_only.pipeline._link_migration import planning as link_planning
from osm_polygon_wikidata_only.pipeline._link_migration.models import StemClassification, StemPlan

EXPECTED_CANONICAL_COLUMNS: tuple[str, ...] = (
    "polygon_id",
    "document_id",
    "project",
    "wikidata",
    "language",
    "source_pbf",
    "region",
    "osm_type",
    "osm_id",
    "page_id",
    "revision_id",
)


def _pyarrow():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    return pa


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exposes_planner_and_classifier() -> None:
    for name in ("plan_link_migration", "apply_link_migration", "classify_stem_schema"):
        assert hasattr(link_migration, name), f"Missing public API: link_migration.{name}"


# ---------------------------------------------------------------------------
# classify_stem_schema
# ---------------------------------------------------------------------------


def test_classify_stem_schema_recognizes_legacy() -> None:
    classification = link_migration.classify_stem_schema(list(POLYGON_ARTICLE_COLUMNS))
    assert classification == "legacy", (
        f"Expected legacy schema classification for POLYGON_ARTICLE_COLUMNS, got {classification!r}"
    )


def test_classify_stem_schema_recognizes_canonical() -> None:
    classification = link_migration.classify_stem_schema(list(EXPECTED_CANONICAL_COLUMNS))
    assert classification == "canonical", (
        f"Expected canonical schema classification, got {classification!r}"
    )


def test_classify_stem_schema_rejects_mixed_or_unknown() -> None:
    mixed_columns = (*list(POLYGON_ARTICLE_COLUMNS), "junk")
    with pytest.raises(ValueError) as mixed_error:
        link_migration.classify_stem_schema(mixed_columns)
    assert str(mixed_error.value) == (
        "Schema is neither legacy nor canonical: "
        f"{list(mixed_columns)[:6]}... (got {len(mixed_columns)} columns)"
    )

    with pytest.raises(ValueError) as empty_error:
        link_migration.classify_stem_schema([])
    assert str(empty_error.value) == "Schema is neither legacy nor canonical: []... (got 0 columns)"


# ---------------------------------------------------------------------------
# Fixture helpers using real schemas
# ---------------------------------------------------------------------------


def _polygon_row(polygon_id: str, qid: str, *, source_pbf: str, region: str) -> dict:
    return {
        "polygon_id": polygon_id,
        "region": region,
        "source_pbf": source_pbf,
        "osm_type": "relation",
        "osm_id": 1,
        "wikidata": qid,
        "name": "",
        "tags": json.dumps({"wikidata": qid}),
        "tag_keys": json.dumps(["wikidata"]),
        "tag_count": 1,
        "osm_primary_tag": "",
        "centroid": json.dumps({"type": "Point", "coordinates": [0.0, 0.0]}),
        "lat": 0.0,
        "lon": 0.0,
        "bbox": json.dumps([0.0, 0.0, 0.0, 0.0]),
        "geometry": "",
        "area_m2": 0.0,
        "area_km2": 0.0,
        "area_bucket": "0",
        "has_name": False,
        "has_wikidata": True,
        "has_wikipedia": True,
        "wikipedia_language_count": 1,
        "wikipedia_languages": json.dumps(["en"]),
        "wikipedia_article_count": 1,
        "has_english_wikipedia": True,
        "has_french_wikipedia": False,
        "text_available": True,
        "best_language": "en",
        "extraction_version": "test",
        "extracted_at": "2026-01-01T00:00:00Z",
    }


def _legacy_link_row(
    polygon_id: str, article_id: str, qid: str, *, source_pbf: str, region: str
) -> dict:
    return {
        "polygon_id": polygon_id,
        "article_id": article_id,
        "wikidata": qid,
        "language": "en",
        "source_pbf": source_pbf,
        "region": region,
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
        "is_best_language": True,
    }


def _legacy_document_row(document_id: str, article_id: str, qid: str) -> dict:
    """Build a legacy 23-column document row (using DOCUMENT_COLUMNS schema)."""
    return {
        "document_id": document_id,
        "article_id": article_id,
        "wikidata": qid,
        "project": "wikipedia",
        "language": "en",
        "site": "enwiki",
        "title": "T",
        "url": "https://en.wikipedia.org/wiki/T",
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "full_text": "body",
        "full_text_format": "plain_text",
        "article_length_chars": 4,
        "article_length_words": 1,
        "article_length_tokens_estimate": 1,
        "license": "CC-BY-SA",
        "attribution": "Wikipedia contributors",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": hashlib.sha256(b"body").hexdigest(),
    }


def _write_polygons(path: Path, rows: list[dict]) -> None:
    pa = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    pa_table = pa.Table.from_pylist(rows, schema=polygon_schema())
    pa.parquet.write_table(pa_table, path, compression="snappy")


def _write_legacy_links(path: Path, rows: list[dict]) -> None:
    pa = _pyarrow()
    from osm_polygon_wikidata_only.domain.schema import polygon_article_schema

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{col: row.get(col) for col in POLYGON_ARTICLE_COLUMNS} for row in rows]
    pa_table = pa.Table.from_pylist(normalized, schema=polygon_article_schema())
    pa.parquet.write_table(pa_table, path, compression="snappy")


def _write_legacy_documents(path: Path, rows: list[dict]) -> None:
    pa = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{col: row.get(col) for col in DOCUMENT_COLUMNS} for row in rows]
    pa_table = pa.Table.from_pylist(normalized, schema=document_schema())
    pa.parquet.write_table(pa_table, path, compression="snappy")


def _processed_layout(tmp_path: Path) -> dict[str, Path]:
    """Return the canonical processed/ layout used by the planner."""
    return {
        "polygons": tmp_path / "polygons",
        "polygon_articles": tmp_path / "polygon_articles",
        "wiki_docs": tmp_path / "wikipedia" / "documents",
    }


def _seed_stem_links_without_changing_article_identity(
    tmp_path: Path, stem: str, shared_article_id: str
) -> None:
    """Write one stem's polygon and link rows for cross-stem identity cases."""
    layout = _processed_layout(tmp_path)
    _write_polygons(
        layout["polygons"] / f"{stem}.parquet",
        [_polygon_row(f"{stem}:relation:1", "Q1", source_pbf=f"{stem}.osm.pbf", region=stem)],
    )
    _write_legacy_links(
        layout["polygon_articles"] / f"{stem}.parquet",
        [
            _legacy_link_row(
                f"{stem}:relation:1",
                shared_article_id,
                "Q1",
                source_pbf=f"{stem}.osm.pbf",
                region=stem,
            )
        ],
    )


def _seed_full_legacy_stem(
    tmp_path: Path, stem: str, qid: str, *, article_id: str, document_id: str
) -> None:
    """Build a stem that is migratable: legacy links + matching legacy doc."""
    layout = _processed_layout(tmp_path)
    _write_polygons(
        layout["polygons"] / f"{stem}.parquet",
        [_polygon_row(f"{stem}:relation:1", qid, source_pbf=f"{stem}.osm.pbf", region=stem)],
    )
    _write_legacy_links(
        layout["polygon_articles"] / f"{stem}.parquet",
        [
            _legacy_link_row(
                f"{stem}:relation:1",
                article_id,
                qid,
                source_pbf=f"{stem}.osm.pbf",
                region=stem,
            )
        ],
    )
    _write_legacy_documents(
        layout["wiki_docs"] / f"{stem}.parquet",
        [_legacy_document_row(document_id, article_id, qid)],
    )


def _seed_canonical_stem(tmp_path: Path, stem: str = "monaco-latest") -> Path:
    pa = _pyarrow()
    from osm_polygon_wikidata_only.domain.polygon_document_links import (
        polygon_document_link_schema,
    )

    layout = _processed_layout(tmp_path)
    _write_polygons(
        layout["polygons"] / f"{stem}.parquet",
        [_polygon_row(f"{stem}:relation:1", "Q1", source_pbf=f"{stem}.osm.pbf", region=stem)],
    )
    links_path = layout["polygon_articles"] / f"{stem}.parquet"
    links_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "polygon_id": f"{stem}:relation:1",
                "document_id": "Q1:wikipedia:en:1:1",
                "project": "wikipedia",
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": f"{stem}.osm.pbf",
                "region": stem,
                "osm_type": "relation",
                "osm_id": 1,
                "page_id": 1,
                "revision_id": 1,
            }
        ],
        schema=polygon_document_link_schema(),
    )
    pa.parquet.write_table(table, links_path)
    return links_path


def _corrupt_first_data_page(path: Path) -> None:
    """Keep the Parquet footer valid while making row data unreadable."""
    pa = _pyarrow()
    parquet_file = pa.parquet.ParquetFile(path)
    offset = parquet_file.metadata.row_group(0).column(0).data_page_offset
    with path.open("r+b") as stream:
        stream.seek(offset)
        first_byte = stream.read(1)
        assert first_byte
        stream.seek(offset)
        stream.write(bytes([first_byte[0] ^ 0xFF]))


def test_read_legacy_rejection_tables_requests_only_relationship_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = _pyarrow()
    stem = "alpha-latest"
    _seed_full_legacy_stem(
        tmp_path, stem, "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    stem_plan = link_migration.plan_link_migration(tmp_path, stems={stem}).stems[0]
    original_read = link_planning.pq.read_table
    calls: list[tuple[Path, list[str] | None]] = []

    def read_table(path: Path, *args: Any, **kwargs: Any) -> pa.Table:
        columns = kwargs.get("columns")
        calls.append((path, columns if isinstance(columns, list) else None))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(link_planning.pq, "read_table", read_table)

    assert link_planning._read_legacy_rejection_tables(tmp_path, stem_plan) is not None
    assert calls == [
        (tmp_path / "polygons" / f"{stem}.parquet", ["polygon_id", "wikidata"]),
        (
            tmp_path / "polygon_articles" / f"{stem}.parquet",
            ["polygon_id", "wikidata", "article_id"],
        ),
    ]


# ---------------------------------------------------------------------------
# Mutation contracts for migration planning helpers
# ---------------------------------------------------------------------------


def test_read_legacy_rejection_tables_returns_none_when_either_file_is_missing(
    tmp_path: Path,
) -> None:
    stem = "alpha"
    _write_polygons(
        tmp_path / "polygons" / f"{stem}.parquet",
        [_polygon_row("p1", "Q1", source_pbf="alpha.osm.pbf", region="alpha")],
    )
    assert not (tmp_path / "polygon_articles" / f"{stem}.parquet").exists()

    stem_plan = link_planning.StemPlan(
        stem=stem,
        classification=link_planning.StemClassification.MIGRATABLE,
        reason="",
        polygons_fingerprint="p",
        links_fingerprint="l",
        documents_fingerprint="d",
        row_count=0,
        canonical_digest=None,
    )
    assert link_planning._read_legacy_rejection_tables(tmp_path, stem_plan) is None


def test_legacy_rejection_for_unknown_polygon_keeps_empty_qid_fallback() -> None:
    pa = _pyarrow()
    links = pa.Table.from_pylist(
        [{"polygon_id": "missing", "article_id": "Q1:en:1:1", "wikidata": "Q1"}]
    )

    rejections = link_planning._legacy_rejection_records("alpha", links, {})

    assert rejections == [
        {
            "shard": "alpha",
            "source_table": "polygon_articles",
            "identifier": "Q1:en:1:1",
            "wikidata": "Q1",
            "expected": None,
            "reason": "wikidata_not_in_polygon_qids",
            "cascaded_sections": 0,
        }
    ]


def test_canonical_stem_plan_keeps_fingerprints_and_reports_schema_reason(
    tmp_path: Path,
) -> None:
    pa = _pyarrow()
    links_path = tmp_path / "links.parquet"
    canonical = pa.Table.from_pylist([], schema=polygon_document_link_schema())
    pa.parquet.write_table(canonical, links_path)
    fingerprints = ("polygon-hash", "link-hash", "document-hash")

    plan = link_planning._canonical_stem_plan("alpha", links_path, fingerprints)

    assert plan.polygons_fingerprint == "polygon-hash"
    assert plan.links_fingerprint == "link-hash"
    assert plan.documents_fingerprint == "document-hash"
    assert plan.canonical_digest == hashlib.sha256(links_path.read_bytes()).hexdigest()

    wrong_schema = pa.table({name: [] for name in EXPECTED_CANONICAL_COLUMNS})
    pa.parquet.write_table(wrong_schema, links_path)
    blocked = link_planning._canonical_stem_plan("alpha", links_path, fingerprints)
    assert (
        blocked.reason
        == "link table columns match canonical but schema differs (types or metadata)"
    )


def test_classification_warnings_keep_path_and_error_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "links.parquet"
    path.write_bytes(b"invalid parquet")
    caplog.set_level(logging.WARNING)
    classification, reason = link_planning._link_classification(path)
    assert classification is None
    assert reason is not None and reason.startswith(
        "polygon_articles file unreadable: ArrowInvalid:"
    )
    assert (
        caplog.records[-1]
        .getMessage()
        .startswith(f"Could not read polygon_articles schema at {path}: ArrowInvalid:")
    )

    with pytest.raises(link_planning._UnreadableStemInputError):
        link_planning._read_stem_table(path, "polygon_articles")
    assert (
        caplog.records[-1]
        .getMessage()
        .startswith(f"Could not read polygon_articles data at {path}: ArrowInvalid:")
    )


def test_legacy_stem_plan_keeps_the_canonical_schema_for_empty_rows(
    tmp_path: Path,
) -> None:
    pa = _pyarrow()
    stem = "alpha"
    polygons_path = tmp_path / "polygons.parquet"
    links_path = tmp_path / "links.parquet"
    docs_path = tmp_path / "documents.parquet"
    _write_polygons(
        polygons_path,
        [_polygon_row("p1", "Q1", source_pbf="alpha.osm.pbf", region="alpha")],
    )
    _write_legacy_links(links_path, [])
    _write_legacy_documents(docs_path, [])

    plan = link_planning._legacy_stem_plan(
        stem, polygons_path, links_path, docs_path, ("p", "l", "d")
    )

    assert plan.row_count == 0
    expected = pa.Table.from_pylist([], schema=polygon_document_link_schema())
    assert link_planning._table_digest(expected) == plan.canonical_digest


def test_legacy_stem_plan_uses_the_canonical_schema_for_nonempty_rows(
    tmp_path: Path,
) -> None:
    pa = _pyarrow()
    stem = "alpha"
    article_id = "Q1:en:1:1"
    document_id = "Q1:wikipedia:en:1:1"
    polygons_path = tmp_path / "polygons.parquet"
    links_path = tmp_path / "links.parquet"
    docs_path = tmp_path / "documents.parquet"
    _write_polygons(
        polygons_path,
        [_polygon_row("p1", "Q1", source_pbf="alpha.osm.pbf", region="alpha")],
    )
    _write_legacy_links(
        links_path,
        [_legacy_link_row("p1", article_id, "Q1", source_pbf="alpha.osm.pbf", region="alpha")],
    )
    _write_legacy_documents(
        docs_path,
        [_legacy_document_row(document_id, article_id, "Q1")],
    )

    plan = link_planning._legacy_stem_plan(
        stem, polygons_path, links_path, docs_path, ("p", "l", "d")
    )

    legacy, polygons, documents = (
        pa.parquet.read_table(path) for path in (links_path, polygons_path, docs_path)
    )
    rows = link_planning._build_canonical_rows(stem, legacy, polygons, documents)
    expected = pa.Table.from_pylist(rows, schema=polygon_document_link_schema())
    assert plan.row_count == 1
    assert plan.canonical_digest == link_planning._table_digest(expected)


def test_public_planning_reports_an_unreadable_link_file(tmp_path: Path) -> None:
    (tmp_path / "polygons").mkdir()
    (tmp_path / "polygons" / "alpha.parquet").write_bytes(
        b"polygon input is not read by classification"
    )
    links_dir = tmp_path / "polygon_articles"
    links_dir.mkdir()
    (links_dir / "alpha.parquet").write_bytes(b"invalid parquet")

    plan = link_migration.plan_link_migration(tmp_path, stems={"alpha"})

    assert len(plan.stems) == 1
    assert plan.stems[0].classification.value == "BLOCKED"
    assert plan.stems[0].reason.startswith("polygon_articles file unreadable: ArrowInvalid:")


def test_missing_link_shard_reason_is_specific(tmp_path: Path) -> None:
    stem = "alpha-latest"
    _write_polygons(
        tmp_path / "polygons" / f"{stem}.parquet",
        [_polygon_row("p1", "Q1", source_pbf="alpha.osm.pbf", region="alpha")],
    )

    plan = link_planning.classify_stem(stem, tmp_path)

    assert plan.reason == "polygon_articles file missing"


def test_apply_inputs_keep_source_paths_and_stem_diagnostics(tmp_path: Path) -> None:
    stem = "alpha-latest"
    _seed_full_legacy_stem(
        tmp_path,
        stem,
        "Q1",
        article_id="Q1:en:1:1",
        document_id="Q1:wikipedia:en:1:1",
    )
    docs_path = tmp_path / "wikipedia" / "documents" / f"{stem}.parquet"
    _write_legacy_documents(docs_path, [])
    plan = link_planning.classify_stem(stem, tmp_path)

    inputs = link_application._load_stem_apply_inputs(tmp_path, plan)

    assert inputs.links_path == tmp_path / "polygon_articles" / f"{stem}.parquet"
    assert inputs.polygons_path == tmp_path / "polygons" / f"{stem}.parquet"
    assert inputs.docs_path == docs_path
    with pytest.raises(ValueError) as error:
        link_application._build_stem_context(inputs)
    assert f"wikipedia/documents/{stem}.parquet" in str(error.value)


def test_link_migration_stages_replacements_under_a_hidden_per_stem_directory(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _seed_full_legacy_stem(
        processed,
        stem,
        "Q1",
        article_id="Q1:en:1:1",
        document_id="Q1:wikipedia:en:1:1",
    )
    plan = link_migration.plan_link_migration(processed, stems={stem})

    def interrupt_after_second_replacement(index: int, target: Path) -> None:
        if index == 1:
            raise RuntimeError(f"simulated interruption at {target}")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        link_migration.apply_link_migration(
            processed,
            plan=plan,
            _crash_hook=interrupt_after_second_replacement,
        )

    staging_dir = processed / ".link_migration_staging" / stem
    journal_path = processed / ".link_migration_journal" / stem / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    staged_paths = [Path(entry["staged"]) for entry in journal["entries"]]

    assert staging_dir.is_dir()
    assert staged_paths
    assert all(staging_dir.resolve() in path.parents for path in staged_paths)


# ---------------------------------------------------------------------------
# plan_link_migration happy paths
# ---------------------------------------------------------------------------


def test_plan_link_migration_produces_empty_plan_for_no_stems(tmp_path: Path) -> None:
    """Empty stems list is a normal no-op: planner returns an empty plan."""
    plan = link_migration.plan_link_migration(tmp_path, stems=set())
    assert plan.stems == (), f"Empty stems must yield empty plan, got {plan.stems}"
    # And no writes happened.
    assert not (tmp_path / "polygon_articles").exists() or not any(
        (tmp_path / "polygon_articles").glob("*.parquet")
    )


def test_plan_link_migration_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        link_migration.plan_link_migration(tmp_path, stems={"../escape"})


@pytest.mark.parametrize("stem", ["", ".", "..", "..\\escape"])
def test_plan_link_migration_rejects_empty_and_dot_stems(tmp_path: Path, stem: str) -> None:
    with pytest.raises(ValueError, match="Invalid stem name"):
        link_migration.plan_link_migration(tmp_path, stems={stem})


def test_blocked_stem_reports_missing_polygon_and_keeps_input_fingerprints(
    tmp_path: Path,
) -> None:
    stem = "orphan-latest"
    _seed_full_legacy_stem(
        tmp_path,
        stem,
        "Q1",
        article_id="Q1:en:1:1",
        document_id="Q1:wikipedia:en:1:1",
    )
    polygons_path = tmp_path / "polygons" / f"{stem}.parquet"
    links_path = tmp_path / "polygon_articles" / f"{stem}.parquet"
    documents_path = tmp_path / "wikipedia" / "documents" / f"{stem}.parquet"
    polygons_path.unlink()

    plan = link_migration.plan_link_migration(tmp_path, stems={stem})
    stem_plan = plan.stems[0]

    assert stem_plan.classification == link_migration.StemClassification.BLOCKED
    assert stem_plan.reason == "polygons file missing"
    assert stem_plan.polygons_fingerprint == ""
    assert stem_plan.links_fingerprint == hashlib.sha256(links_path.read_bytes()).hexdigest()
    assert (
        stem_plan.documents_fingerprint == hashlib.sha256(documents_path.read_bytes()).hexdigest()
    )
    assert stem_plan.row_count == 0
    assert stem_plan.canonical_digest is None


def test_plan_link_migration_discovers_stems_from_every_supported_table(
    tmp_path: Path,
) -> None:
    pa = _pyarrow()
    source_paths = (
        tmp_path / "polygons" / "polygon-only.parquet",
        tmp_path / "polygon_articles" / "links-only.parquet",
        tmp_path / "wikipedia" / "documents" / "documents-only.parquet",
    )
    for path in source_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        pa.parquet.write_table(pa.table({"placeholder": []}), path)

    plan = link_migration.plan_link_migration(tmp_path)

    assert [stem_plan.stem for stem_plan in plan.stems] == [
        "documents-only",
        "links-only",
        "polygon-only",
    ]
    assert all(
        stem_plan.classification == link_migration.StemClassification.BLOCKED
        for stem_plan in plan.stems
    )


def test_plan_link_migration_classifies_legacy_stem_as_migratable(tmp_path: Path) -> None:
    stem = "monaco-latest"
    _seed_full_legacy_stem(
        tmp_path, stem, "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = link_migration.plan_link_migration(tmp_path, stems={"monaco-latest"})
    stem_plan = plan.stems[0]
    assert stem_plan.classification == link_migration.StemClassification.MIGRATABLE
    assert stem_plan.reason == ""
    assert (
        stem_plan.polygons_fingerprint
        == hashlib.sha256((tmp_path / "polygons" / f"{stem}.parquet").read_bytes()).hexdigest()
    )
    assert (
        stem_plan.links_fingerprint
        == hashlib.sha256(
            (tmp_path / "polygon_articles" / f"{stem}.parquet").read_bytes()
        ).hexdigest()
    )
    assert (
        stem_plan.documents_fingerprint
        == hashlib.sha256(
            (tmp_path / "wikipedia" / "documents" / f"{stem}.parquet").read_bytes()
        ).hexdigest()
    )
    assert stem_plan.row_count == 1
    assert stem_plan.canonical_digest is not None
    assert len(stem_plan.canonical_digest) == 64


def test_legacy_conversion_block_reason_names_the_source_shard(
    tmp_path: Path,
) -> None:
    stem = "monaco-latest"
    _seed_full_legacy_stem(
        tmp_path, stem, "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    links_path = tmp_path / "polygon_articles" / f"{stem}.parquet"
    links = _pyarrow().parquet.read_table(links_path).to_pylist()
    links[0]["polygon_id"] = "missing-polygon"
    from osm_polygon_wikidata_only.domain.schema import polygon_article_schema

    _pyarrow().parquet.write_table(
        _pyarrow().Table.from_pylist(links, schema=polygon_article_schema()), links_path
    )

    plan = link_migration.plan_link_migration(tmp_path, stems={stem})

    assert plan.stems[0].classification == link_migration.StemClassification.BLOCKED
    assert "missing-polygon" in plan.stems[0].reason
    assert f"polygons/{stem}.parquet" in plan.stems[0].reason


@pytest.mark.parametrize(
    ("relative_path", "label"),
    [
        ("polygon_articles", "polygon_articles"),
        ("polygons", "polygons"),
        ("wikipedia/documents", "wikipedia documents"),
    ],
)
def test_unreadable_legacy_source_tables_keep_their_labels(
    tmp_path: Path,
    relative_path: str,
    label: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stem = "monaco-latest"
    _seed_full_legacy_stem(
        tmp_path, stem, "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    source_path = tmp_path / relative_path / f"{stem}.parquet"
    _corrupt_first_data_page(source_path)

    with caplog.at_level("WARNING"):
        plan = link_migration.plan_link_migration(tmp_path, stems={stem})

    assert plan.stems[0].classification == link_migration.StemClassification.BLOCKED
    assert plan.stems[0].reason.startswith(f"{label} file unreadable:")
    assert f"Could not read {label} data at {source_path}:" in caplog.text


def test_legacy_schema_without_a_documents_file_has_a_specific_block_reason(
    tmp_path: Path,
) -> None:
    stem = "monaco-latest"
    _seed_full_legacy_stem(
        tmp_path, stem, "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    (tmp_path / "wikipedia" / "documents" / f"{stem}.parquet").unlink()

    plan = link_migration.plan_link_migration(tmp_path, stems={stem})

    assert plan.stems[0].classification == link_migration.StemClassification.BLOCKED
    assert plan.stems[0].reason == "legacy schema requires wikipedia/documents/<stem>.parquet"


def test_apply_link_migration_refuses_a_plan_with_blocked_stems(tmp_path: Path) -> None:
    pa = _pyarrow()
    layout = _processed_layout(tmp_path)
    layout["polygon_articles"].mkdir(parents=True, exist_ok=True)
    links_path = layout["polygon_articles"] / "monaco-latest.parquet"
    pa.parquet.write_table(pa.table({"polygon_id": ["p"], "article_id": ["x"]}), links_path)
    original = links_path.read_bytes()

    with pytest.raises(ValueError, match=r"blocked stems: \['monaco-latest'\]"):
        link_migration.apply_link_migration(tmp_path, stems={"monaco-latest"})

    assert links_path.read_bytes() == original


def test_plan_link_migration_classifies_mixed_schema_exactly_blocked(
    tmp_path: Path,
) -> None:
    """Mixed schema must be classified exactly BLOCKED, never 'incomplete'."""
    pa = _pyarrow()
    layout = _processed_layout(tmp_path)
    layout["polygons"].mkdir(parents=True, exist_ok=True)
    layout["polygon_articles"].mkdir(parents=True, exist_ok=True)
    # An incomplete schema: only "polygon_id" and "article_id".
    mixed_table = pa.table({"polygon_id": ["p"], "article_id": ["x"]})
    pa.parquet.write_table(mixed_table, layout["polygon_articles"] / "monaco-latest.parquet")
    plan = link_migration.plan_link_migration(tmp_path, stems={"monaco-latest"})
    stem_plans = {s.stem: s for s in plan.stems}
    assert stem_plans["monaco-latest"].classification == "BLOCKED", (
        f"Mixed schema must be classified exactly BLOCKED, got "
        f"{stem_plans['monaco-latest'].classification!r}"
    )


def test_plan_link_migration_reads_canonical_link_table_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    links_path = _seed_canonical_stem(tmp_path)
    original_read_table = link_planning.pq.read_table
    link_table_reads = 0

    def count_link_table_reads(path, *args, **kwargs):
        nonlocal link_table_reads
        if Path(path) == links_path:
            link_table_reads += 1
        return original_read_table(path, *args, **kwargs)

    monkeypatch.setattr(link_planning.pq, "read_table", count_link_table_reads)
    plan = link_migration.plan_link_migration(tmp_path, stems={"monaco-latest"})

    assert plan.stems[0].classification == "canonical"
    assert plan.stems[0].reason == ""
    assert plan.stems[0].row_count == 1
    assert plan.stems[0].canonical_digest == hashlib.sha256(links_path.read_bytes()).hexdigest()
    assert link_table_reads == 1


@pytest.mark.parametrize("schema", ["canonical", "legacy"])
def test_unreadable_data_pages_block_only_their_stem(
    tmp_path: Path,
    schema: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pa = _pyarrow()
    if schema == "canonical":
        broken_links = _seed_canonical_stem(tmp_path, "broken")
    else:
        _seed_full_legacy_stem(
            tmp_path,
            "broken",
            "Q1",
            article_id="Q1:en:1:1",
            document_id="Q1:wikipedia:en:1:1",
        )
        broken_links = _processed_layout(tmp_path)["polygon_articles"] / "broken.parquet"
    _seed_canonical_stem(tmp_path, "healthy")
    _corrupt_first_data_page(broken_links)

    assert pa.parquet.read_schema(broken_links).names
    with pytest.raises((OSError, pa.ArrowInvalid)):
        pa.parquet.read_table(broken_links)

    with caplog.at_level("WARNING"):
        plan = link_migration.plan_link_migration(tmp_path, stems={"broken", "healthy"})
    by_stem = {stem.stem: stem for stem in plan.stems}

    assert by_stem["broken"].classification == "BLOCKED"
    assert "polygon_articles file unreadable" in by_stem["broken"].reason
    assert by_stem["healthy"].classification == "canonical"
    assert "Could not read polygon_articles data at" in caplog.text
    assert str(broken_links) in caplog.text
    assert any(
        error_type in by_stem["broken"].reason for error_type in ("OSError:", "ArrowInvalid:")
    )


def test_plan_link_migration_includes_corrupt_parquet_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = _processed_layout(tmp_path)
    _write_polygons(
        layout["polygons"] / "broken.parquet",
        [_polygon_row("broken:relation:1", "Q1", source_pbf="broken.osm.pbf", region="broken")],
    )
    links_path = layout["polygon_articles"] / "broken.parquet"
    links_path.parent.mkdir(parents=True, exist_ok=True)
    links_path.write_bytes(b"not a parquet file")

    with caplog.at_level("WARNING"):
        plan = link_migration.plan_link_migration(tmp_path, stems={"broken"})

    blocked = plan.stems[0]
    assert blocked.classification == "BLOCKED"
    assert "polygon_articles file unreadable" in blocked.reason
    assert "ArrowInvalid:" in blocked.reason
    assert "ArrowInvalid" in caplog.text
    assert "Could not read polygon_articles schema" in caplog.text


def test_plan_link_migration_propagates_unexpected_schema_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    links_path = _seed_canonical_stem(tmp_path)
    original_read_schema = link_planning.pq.read_schema

    def fail_unexpectedly(path, *args, **kwargs):
        if Path(path) == links_path:
            raise RuntimeError("unexpected schema reader failure")
        return original_read_schema(path, *args, **kwargs)

    monkeypatch.setattr(link_planning.pq, "read_schema", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="unexpected schema reader failure"):
        link_migration.plan_link_migration(tmp_path, stems={"monaco-latest"})


# ---------------------------------------------------------------------------
# Within-stem isolation: the alpha/beta case
# ---------------------------------------------------------------------------


def test_plan_link_migration_within_stem_isolation_alpha_migratable_beta_blocked(
    tmp_path: Path,
) -> None:
    """Alpha has the legacy article + matching alpha doc -> migratable.

    Beta has the SAME legacy article_id but no matching beta doc (only
    alpha matches) -> beta must be BLOCKED, and beta must NOT resolve
    from alpha's documents.
    """
    layout = _processed_layout(tmp_path)
    # Both stems carry the same article_id but different polygon ids.
    shared_article_id = "Q1:en:1:1"
    for stem in ("alpha", "beta"):
        _seed_stem_links_without_changing_article_identity(tmp_path, stem, shared_article_id)
    # Only alpha has a matching legacy document. Beta has no doc file.
    _write_legacy_documents(
        layout["wiki_docs"] / "alpha.parquet",
        [_legacy_document_row("Q1:wikipedia:en:1:1", shared_article_id, "Q1")],
    )
    # Beta's directory exists but has no document file.

    plan = link_migration.plan_link_migration(tmp_path, stems={"alpha", "beta"})
    buckets = {s.stem: s.classification for s in plan.stems}
    assert buckets["alpha"] == "migratable", (
        f"alpha has matching legacy document -> must be migratable, got {buckets['alpha']!r}"
    )
    assert buckets["beta"] == "BLOCKED", (
        f"beta has the same legacy article_id but no matching beta doc -> must be BLOCKED, "
        f"got {buckets['beta']!r}"
    )


def test_plan_link_migration_keys_resolution_within_stem_unique_document_id(
    tmp_path: Path,
) -> None:
    """The same article_id appearing in two stems must map to the SAME
    document_id revision (the document_id is the canonical identity)."""
    layout = _processed_layout(tmp_path)
    shared_article_id = "Q1:en:1:1"
    document_id = "Q1:wikipedia:en:1:1"
    for stem in ("alpha", "beta"):
        _seed_stem_links_without_changing_article_identity(tmp_path, stem, shared_article_id)
        # Both stems have documents with the SAME document_id (and same revision).
        _write_legacy_documents(
            layout["wiki_docs"] / f"{stem}.parquet",
            [_legacy_document_row(document_id, shared_article_id, "Q1")],
        )

    plan = link_migration.plan_link_migration(tmp_path, stems={"alpha", "beta"})
    buckets = {s.stem: s.classification for s in plan.stems}
    assert buckets["alpha"] == "migratable"
    assert buckets["beta"] == "migratable"


# ---------------------------------------------------------------------------
# Stale-plan fingerprint tests
# ---------------------------------------------------------------------------


def test_apply_link_migration_aborts_when_polygons_change_after_planning(
    tmp_path: Path,
) -> None:
    """If the polygons file is mutated after planning, apply must abort."""
    _seed_full_legacy_stem(
        tmp_path, "monaco-latest", "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = link_migration.plan_link_migration(tmp_path, stems={"monaco-latest"})
    # Tamper with the polygons file after planning.
    polygons_path = tmp_path / "polygons" / "monaco-latest.parquet"
    polygons_path.write_bytes(polygons_path.read_bytes() + b"corrupt")
    with pytest.raises(
        RuntimeError,
        match="Link migration stem 'monaco-latest': source file changed after planning",
    ):
        link_migration.apply_link_migration(tmp_path, plan=plan)


def test_apply_link_migration_aborts_when_legacy_links_change_after_planning(
    tmp_path: Path,
) -> None:
    """If the legacy links file is mutated after planning, apply must abort."""
    _seed_full_legacy_stem(
        tmp_path, "monaco-latest", "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = link_migration.plan_link_migration(tmp_path, stems={"monaco-latest"})
    links_path = tmp_path / "polygon_articles" / "monaco-latest.parquet"
    links_path.write_bytes(links_path.read_bytes() + b"corrupt")
    with pytest.raises(
        RuntimeError,
        match="Link migration stem 'monaco-latest': source file changed after planning",
    ):
        link_migration.apply_link_migration(tmp_path, plan=plan)


def test_apply_link_migration_aborts_when_legacy_documents_change_after_planning(
    tmp_path: Path,
) -> None:
    """If the legacy documents file is mutated after planning, apply must abort."""
    _seed_full_legacy_stem(
        tmp_path, "monaco-latest", "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = link_migration.plan_link_migration(tmp_path, stems={"monaco-latest"})
    docs_path = tmp_path / "wikipedia" / "documents" / "monaco-latest.parquet"
    docs_path.write_bytes(docs_path.read_bytes() + b"corrupt")
    with pytest.raises(
        RuntimeError,
        match="Link migration stem 'monaco-latest': source file changed after planning",
    ):
        link_migration.apply_link_migration(tmp_path, plan=plan)


def test_apply_migratable_stems_continues_after_a_canonical_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    _seed_canonical_stem(processed, "alpha-canonical")
    _seed_full_legacy_stem(
        processed,
        "beta-legacy",
        "Q1",
        article_id="Q1:en:1:1",
        document_id="Q1:wikipedia:en:1:1",
    )
    plan = link_migration.plan_link_migration(processed)
    legacy_plan = next(stem for stem in plan.stems if stem.stem == "beta-legacy")
    inputs = link_application._load_stem_apply_inputs(processed, legacy_plan)
    assert inputs.voyage_sections_path == (
        processed / "wikivoyage" / "sections" / "beta-legacy.parquet"
    )

    def fail_cleanup(_path: Path) -> None:
        raise OSError("staging directory is still in use")

    monkeypatch.setattr(link_application.shutil, "rmtree", fail_cleanup)

    link_application._apply_migratable_stems(processed, plan, {})

    canonical_path = processed / "polygon_articles" / "alpha-canonical.parquet"
    migrated_path = processed / "polygon_articles" / "beta-legacy.parquet"
    assert (
        link_migration.classify_stem_schema(_pyarrow().parquet.read_schema(canonical_path).names)
        == "canonical"
    )
    assert (
        link_migration.classify_stem_schema(_pyarrow().parquet.read_schema(migrated_path).names)
        == "canonical"
    )


def test_apply_link_migration_is_idempotent_on_second_run(tmp_path: Path) -> None:
    """Second migration run must be a no-op (canonical shards are skipped)."""
    processed = tmp_path / "processed"
    _seed_full_legacy_stem(
        processed,
        "monaco-latest",
        "Q1",
        article_id="Q1:en:1:1",
        document_id="Q1:wikipedia:en:1:1",
    )
    link_migration.apply_link_migration(processed, stems={"monaco-latest"})
    first_hash = hashlib.sha256(
        (processed / "polygon_articles" / "monaco-latest.parquet").read_bytes()
    ).hexdigest()
    link_migration.apply_link_migration(processed, stems={"monaco-latest"})
    second_hash = hashlib.sha256(
        (processed / "polygon_articles" / "monaco-latest.parquet").read_bytes()
    ).hexdigest()
    assert first_hash == second_hash, "Second migration must be byte-stable for canonical stem"


@pytest.mark.parametrize("error", [KeyError("missing column"), ArrowInvalid("bad schema")])
def test_legacy_rejection_table_reader_returns_none_for_invalid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    stem_plan = StemPlan(
        "region-latest",
        StemClassification.MIGRATABLE,
        "",
        "",
        "",
        "",
        0,
        None,
    )
    for path in link_planning._stem_paths(stem_plan.stem, tmp_path)[:2]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        link_planning.pq, "read_table", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )

    assert link_planning._read_legacy_rejection_tables(tmp_path, stem_plan) is None


def test_link_migration_legacy_rejection_table_reader_handles_missing_files(
    tmp_path: Path,
) -> None:
    plan = StemPlan(
        "region-latest",
        StemClassification.MIGRATABLE,
        "",
        "",
        "",
        "",
        0,
        None,
    )
    assert link_planning._read_legacy_rejection_tables(tmp_path, plan) is None
