"""Phase 2 / Group B: legacy/canonical schema detection and migration planning.

Red tests for ``osm_polygon_wikidata_only.pipeline.link_migration``.

Real schemas used throughout -- ``polygon_schema()``,
``wikipedia_document_schema()`` (the canonical document schema),
``document_schema()`` (the legacy document schema) and the legacy
``polygon_article_schema()``. The planner must classify stems strictly
into ``legacy``, ``canonical`` or ``BLOCKED`` -- never "incomplete".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    document_schema,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
    polygon_schema,
)

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


def _import_module():
    try:
        from osm_polygon_wikidata_only.pipeline import link_migration as mod
    except ImportError as exc:
        pytest.fail(
            "Expected osm_polygon_wikidata_only.pipeline.link_migration to exist "
            f"(Phase 2 group B: schema detection/planning); got ImportError: {exc}"
        )
    return mod


def _pyarrow():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    return pa


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exposes_planner_and_classifier() -> None:
    mod = _import_module()
    for name in ("plan_link_migration", "apply_link_migration", "classify_stem_schema"):
        assert hasattr(mod, name), f"Missing public API: link_migration.{name}"


# ---------------------------------------------------------------------------
# classify_stem_schema
# ---------------------------------------------------------------------------


def test_classify_stem_schema_recognizes_legacy() -> None:
    mod = _import_module()
    classification = mod.classify_stem_schema(list(POLYGON_ARTICLE_COLUMNS))
    assert classification == "legacy", (
        f"Expected legacy schema classification for POLYGON_ARTICLE_COLUMNS, got {classification!r}"
    )


def test_classify_stem_schema_recognizes_canonical() -> None:
    mod = _import_module()
    classification = mod.classify_stem_schema(list(EXPECTED_CANONICAL_COLUMNS))
    assert classification == "canonical", (
        f"Expected canonical schema classification, got {classification!r}"
    )


def test_classify_stem_schema_rejects_mixed_or_unknown() -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.classify_stem_schema((*list(POLYGON_ARTICLE_COLUMNS), "junk"))
    with pytest.raises(ValueError):
        mod.classify_stem_schema([])


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


def _wiki_document_row(
    document_id: str, article_id: str, qid: str, *, revision_id: int = 1
) -> dict:
    """Build a canonical Wikipedia-document row using the canonical schema."""
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
        "revision_id": revision_id,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "wikidata_label": "T",
        "wikidata_description": "",
        "wikidata_aliases": "[]",
        "lead_text": "",
        "extract": "",
        "full_text": "body",
        "full_text_format": "plain_text",
        "article_length_chars": 4,
        "article_length_words": 1,
        "article_length_tokens_estimate": 1,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "[]",
        "license": "CC-BY-SA",
        "attribution": "Wikipedia contributors",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": hashlib.sha256(b"body").hexdigest(),
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


def _write_canonical_documents(path: Path, rows: list[dict]) -> None:
    pa = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    pa_table = pa.Table.from_pylist(rows, schema=wikipedia_document_schema())
    pa.parquet.write_table(pa_table, path, compression="snappy")


def _processed_layout(tmp_path: Path) -> dict[str, Path]:
    """Return the canonical processed/ layout used by the planner."""
    return {
        "polygons": tmp_path / "polygons",
        "polygon_articles": tmp_path / "polygon_articles",
        "wiki_docs": tmp_path / "wikipedia" / "documents",
    }


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


# ---------------------------------------------------------------------------
# plan_link_migration happy paths
# ---------------------------------------------------------------------------


def test_plan_link_migration_produces_empty_plan_for_no_stems(tmp_path: Path) -> None:
    """Empty stems list is a normal no-op: planner returns an empty plan."""
    mod = _import_module()
    plan = mod.plan_link_migration(tmp_path, stems=[])
    assert plan.stems == (), f"Empty stems must yield empty plan, got {plan.stems}"
    # And no writes happened.
    assert not (tmp_path / "polygon_articles").exists() or not any(
        (tmp_path / "polygon_articles").glob("*.parquet")
    )


def test_plan_link_migration_rejects_path_traversal(tmp_path: Path) -> None:
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.plan_link_migration(tmp_path, stems=["../escape"])


def test_plan_link_migration_classifies_legacy_stem_as_migratable(tmp_path: Path) -> None:
    mod = _import_module()
    _seed_full_legacy_stem(
        tmp_path, "monaco-latest", "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = mod.plan_link_migration(tmp_path, stems=["monaco-latest"])
    stem_plans = {s.stem: s for s in plan.stems}
    assert stem_plans["monaco-latest"].classification == "migratable", (
        f"Legacy stem with matching document must be migratable, got "
        f"{stem_plans['monaco-latest'].classification!r}"
    )


def test_plan_link_migration_classifies_mixed_schema_exactly_blocked(
    tmp_path: Path,
) -> None:
    """Mixed schema must be classified exactly BLOCKED, never 'incomplete'."""
    mod = _import_module()
    pa = _pyarrow()
    layout = _processed_layout(tmp_path)
    layout["polygons"].mkdir(parents=True, exist_ok=True)
    layout["polygon_articles"].mkdir(parents=True, exist_ok=True)
    # An incomplete schema: only "polygon_id" and "article_id".
    mixed_table = pa.table({"polygon_id": ["p"], "article_id": ["x"]})
    pa.parquet.write_table(mixed_table, layout["polygon_articles"] / "monaco-latest.parquet")
    plan = mod.plan_link_migration(tmp_path, stems=["monaco-latest"])
    stem_plans = {s.stem: s for s in plan.stems}
    assert stem_plans["monaco-latest"].classification == "BLOCKED", (
        f"Mixed schema must be classified exactly BLOCKED, got "
        f"{stem_plans['monaco-latest'].classification!r}"
    )


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
    mod = _import_module()
    pa = _pyarrow()
    from osm_polygon_wikidata_only.domain.schema import polygon_article_schema

    layout = _processed_layout(tmp_path)
    layout["polygon_articles"].mkdir(parents=True, exist_ok=True)
    # Both stems carry the same article_id but different polygon ids.
    shared_article_id = "Q1:en:1:1"
    for stem in ("alpha", "beta"):
        _write_polygons(
            layout["polygons"] / f"{stem}.parquet",
            [_polygon_row(f"{stem}:relation:1", "Q1", source_pbf=f"{stem}.osm.pbf", region=stem)],
        )
        normalized = [
            {
                "polygon_id": f"{stem}:relation:1",
                "article_id": shared_article_id,
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": f"{stem}.osm.pbf",
                "region": stem,
                "osm_type": "relation",
                "osm_id": 1,
                "page_id": 1,
                "revision_id": 1,
                "is_best_language": True,
            }
        ]
        pa.parquet.write_table(
            pa.Table.from_pylist(normalized, schema=polygon_article_schema()),
            layout["polygon_articles"] / f"{stem}.parquet",
        )
    # Only alpha has a matching legacy document. Beta has no doc file.
    _write_legacy_documents(
        layout["wiki_docs"] / "alpha.parquet",
        [_legacy_document_row("Q1:wikipedia:en:1:1", shared_article_id, "Q1")],
    )
    # Beta's directory exists but has no document file.

    plan = mod.plan_link_migration(tmp_path, stems=["alpha", "beta"])
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
    mod = _import_module()
    pa = _pyarrow()
    from osm_polygon_wikidata_only.domain.schema import polygon_article_schema

    layout = _processed_layout(tmp_path)
    layout["polygon_articles"].mkdir(parents=True, exist_ok=True)
    shared_article_id = "Q1:en:1:1"
    document_id = "Q1:wikipedia:en:1:1"
    for stem in ("alpha", "beta"):
        _write_polygons(
            layout["polygons"] / f"{stem}.parquet",
            [_polygon_row(f"{stem}:relation:1", "Q1", source_pbf=f"{stem}.osm.pbf", region=stem)],
        )
        normalized = [
            {
                "polygon_id": f"{stem}:relation:1",
                "article_id": shared_article_id,
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": f"{stem}.osm.pbf",
                "region": stem,
                "osm_type": "relation",
                "osm_id": 1,
                "page_id": 1,
                "revision_id": 1,
                "is_best_language": True,
            }
        ]
        pa.parquet.write_table(
            pa.Table.from_pylist(normalized, schema=polygon_article_schema()),
            layout["polygon_articles"] / f"{stem}.parquet",
        )
        # Both stems have documents with the SAME document_id (and same revision).
        _write_legacy_documents(
            layout["wiki_docs"] / f"{stem}.parquet",
            [_legacy_document_row(document_id, shared_article_id, "Q1")],
        )

    plan = mod.plan_link_migration(tmp_path, stems=["alpha", "beta"])
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
    mod = _import_module()
    _seed_full_legacy_stem(
        tmp_path, "monaco-latest", "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = mod.plan_link_migration(tmp_path, stems=["monaco-latest"])
    # Tamper with the polygons file after planning.
    polygons_path = tmp_path / "polygons" / "monaco-latest.parquet"
    polygons_path.write_bytes(polygons_path.read_bytes() + b"corrupt")
    with pytest.raises(Exception):
        mod.apply_link_migration(plan)


def test_apply_link_migration_aborts_when_legacy_links_change_after_planning(
    tmp_path: Path,
) -> None:
    """If the legacy links file is mutated after planning, apply must abort."""
    mod = _import_module()
    _seed_full_legacy_stem(
        tmp_path, "monaco-latest", "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = mod.plan_link_migration(tmp_path, stems=["monaco-latest"])
    links_path = tmp_path / "polygon_articles" / "monaco-latest.parquet"
    links_path.write_bytes(links_path.read_bytes() + b"corrupt")
    with pytest.raises(Exception):
        mod.apply_link_migration(plan)


def test_apply_link_migration_aborts_when_legacy_documents_change_after_planning(
    tmp_path: Path,
) -> None:
    """If the legacy documents file is mutated after planning, apply must abort."""
    mod = _import_module()
    _seed_full_legacy_stem(
        tmp_path, "monaco-latest", "Q1", article_id="Q1:en:1:1", document_id="Q1:wikipedia:en:1:1"
    )
    plan = mod.plan_link_migration(tmp_path, stems=["monaco-latest"])
    docs_path = tmp_path / "wikipedia" / "documents" / "monaco-latest.parquet"
    docs_path.write_bytes(docs_path.read_bytes() + b"corrupt")
    with pytest.raises(Exception):
        mod.apply_link_migration(plan)


def test_apply_link_migration_is_idempotent_on_second_run(tmp_path: Path) -> None:
    """Second migration run must be a no-op (canonical shards are skipped)."""
    mod = _import_module()
    processed = tmp_path / "processed"
    _seed_full_legacy_stem(
        processed,
        "monaco-latest",
        "Q1",
        article_id="Q1:en:1:1",
        document_id="Q1:wikipedia:en:1:1",
    )
    mod.apply_link_migration(processed, stems=["monaco-latest"])
    first_hash = hashlib.sha256(
        (processed / "polygon_articles" / "monaco-latest.parquet").read_bytes()
    ).hexdigest()
    mod.apply_link_migration(processed, stems=["monaco-latest"])
    second_hash = hashlib.sha256(
        (processed / "polygon_articles" / "monaco-latest.parquet").read_bytes()
    ).hexdigest()
    assert first_hash == second_hash, "Second migration must be byte-stable for canonical stem"
