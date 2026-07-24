"""Phase 2 / Amendment 1: Lossless legacy-link conversion.

The migration MUST reconstruct every canonical link from the exact
legacy ``(polygon_id, article_id)`` row paired with that stem's unique
within-stem document. A QID-wide Cartesian join across polygons and
documents is FORBIDDEN because it can introduce relationships absent
from the legacy table.

Each canonical row corresponds to ONE legacy ``(polygon_id,
article_id)`` tuple. Byte-identical legacy duplicates collapse. A
legacy ``article_id`` that has no matching ``wikipedia/documents`` row
blocks the stem.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema


def _import_module():
    try:
        from osm_polygon_wikidata_only.pipeline import link_migration as mod
    except ImportError:
        pytest.fail("link_migration module must exist for lossless conversion")
    return mod


def _write_legacy_links(
    processed_dir: Path,
    stem: str,
    rows: list[dict],
) -> None:
    table = pa.Table.from_pylist(rows, schema=polygon_article_schema())
    path = processed_dir / "polygon_articles" / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_polygons(
    processed_dir: Path,
    stem: str,
    rows: list[dict],
) -> None:
    table = pa.Table.from_pylist(rows, schema=polygon_schema())
    path = processed_dir / "polygons" / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_documents(
    processed_dir: Path,
    stem: str,
    rows: list[dict],
) -> None:
    schema = wikipedia_document_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    path = processed_dir / "wikipedia" / "documents" / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _legacy_row(
    polygon_id: str,
    article_id: str,
    wikidata: str,
    page_id: int = 100,
    revision_id: int = 1,
) -> dict:
    return {
        "polygon_id": polygon_id,
        "article_id": article_id,
        "wikidata": wikidata,
        "language": "en",
        "source_pbf": "alpha-latest.osm.pbf",
        "region": "alpha",
        "osm_type": "way",
        "osm_id": 1,
        "page_id": page_id,
        "revision_id": revision_id,
        "is_best_language": True,
    }


def _doc_row(
    article_id: str,
    document_id: str,
    wikidata: str,
    page_id: int,
    revision_id: int,
) -> dict:
    return {
        "document_id": document_id,
        "article_id": article_id,
        "wikidata": wikidata,
        "language": "en",
        "site": "enwiki",
        "title": "T",
        "url": "https://en.wikipedia.org/wiki/T",
        "page_id": page_id,
        "revision_id": revision_id,
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


def _poly_row(polygon_id: str, wikidata: str) -> dict:
    return {
        "polygon_id": polygon_id,
        "region": "alpha",
        "source_pbf": "alpha-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": 1,
        "wikidata": wikidata,
        "name": "",
        "tags": "",
        "tag_keys": "",
        "tag_count": 0,
        "osm_primary_tag": "",
        "centroid": "",
        "lat": 0.0,
        "lon": 0.0,
        "bbox": "",
        "geometry": "",
        "area_m2": 0.0,
        "area_km2": 0.0,
        "area_bucket": "",
        "has_name": False,
        "has_wikidata": True,
        "has_wikipedia": False,
        "wikipedia_language_count": 0,
        "wikipedia_languages": "",
        "wikipedia_article_count": 0,
        "has_english_wikipedia": False,
        "has_french_wikipedia": False,
        "text_available": False,
        "best_language": "en",
        "extraction_version": "test",
        "extracted_at": "2026-07-24T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# 1. No Cartesian join across polygons with the same QID
# ---------------------------------------------------------------------------


def test_no_cartesian_join_when_multiple_polygons_share_qid(tmp_path: Path) -> None:
    """Two polygons share Q1; legacy has only p1->a1; canonical must have
    exactly 1 row, not 4 (the cartesian join over polygons x documents).

    This is the exact bug the amendment targets.
    """
    mod = _import_module()
    stem = "alpha-latest"
    processed = tmp_path / "processed"

    _write_polygons(
        processed,
        stem,
        [_poly_row("p1", "Q1"), _poly_row("p2", "Q1")],
    )
    _write_documents(
        processed,
        stem,
        [
            _doc_row("a1", "Q1:wikipedia:en:100:1", "Q1", 100, 1),
            _doc_row("a2", "Q1:wikipedia:en:200:2", "Q1", 200, 2),
        ],
    )
    # Legacy ONLY contains p1 -> a1. There is NO legacy row for p2.
    _write_legacy_links(
        processed,
        stem,
        [_legacy_row("p1", "a1", "Q1")],
    )

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply, (
        f"Stem should be migratable, plan: {[s.reason for s in plan.stems]}"
    )
    sp = next(s for s in plan.stems if s.stem == stem)
    assert sp.row_count == 1, (
        f"|canonical| must equal |distinct legacy identities| = 1, got {sp.row_count}"
    )
    mod.apply_link_migration(processed)
    canonical = pq.read_table(  # type: ignore[no-untyped-call]
        processed / "polygon_articles" / f"{stem}.parquet"
    ).to_pylist()
    assert len(canonical) == 1, (
        f"Canonical table must have 1 row, got {len(canonical)}: {canonical}"
    )
    assert canonical[0]["polygon_id"] == "p1"
    assert canonical[0]["document_id"] == "Q1:wikipedia:en:100:1"


# ---------------------------------------------------------------------------
# 2. Legacy row identity preserved byte-for-byte
# ---------------------------------------------------------------------------


def test_legacy_row_identity_preserved_byte_for_byte(tmp_path: Path) -> None:
    """Each legacy (polygon_id, article_id) -> exactly one canonical row.

    Two legacy rows with the same identity collapse to one. Two legacy
    rows with different article_id but same polygon_id produce two
    canonical rows.
    """
    mod = _import_module()
    stem = "alpha-latest"
    processed = tmp_path / "processed"

    _write_polygons(processed, stem, [_poly_row("p1", "Q1")])
    _write_documents(
        processed,
        stem,
        [
            _doc_row("a1", "Q1:wikipedia:en:100:1", "Q1", 100, 1),
            _doc_row("a2", "Q1:wikipedia:en:200:2", "Q1", 200, 2),
        ],
    )
    # Two legacy rows with the same identity (p1, a1) -- collapse to one.
    # Plus one distinct row (p1, a2). Legacy page_ids must match the
    # resolved documents (per per-polygon QID membership rules).
    _write_legacy_links(
        processed,
        stem,
        [
            _legacy_row("p1", "a1", "Q1", page_id=100, revision_id=1),
            _legacy_row("p1", "a1", "Q1", page_id=100, revision_id=1),
            _legacy_row("p1", "a2", "Q1", page_id=200, revision_id=2),
        ],
    )

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    mod.apply_link_migration(processed)
    canonical = pq.read_table(  # type: ignore[no-untyped-call]
        processed / "polygon_articles" / f"{stem}.parquet"
    ).to_pylist()
    assert len(canonical) == 2, (
        f"Two distinct legacy (polygon_id, article_id) tuples expected; got {len(canonical)}"
    )
    doc_ids = sorted(r["document_id"] for r in canonical)
    assert doc_ids == ["Q1:wikipedia:en:100:1", "Q1:wikipedia:en:200:2"], (
        f"Canonical must mirror legacy article_ids, got {doc_ids}"
    )


# ---------------------------------------------------------------------------
# 3. Cardinality equals distinct legacy identities
# ---------------------------------------------------------------------------


def test_cardinality_matches_unique_legacy_rows(tmp_path: Path) -> None:
    """Cardinality rule: |canonical| == |distinct legacy (polygon_id, article_id)|."""
    mod = _import_module()
    stem = "alpha-latest"
    processed = tmp_path / "processed"

    _write_polygons(
        processed,
        stem,
        [_poly_row("p1", "Q1"), _poly_row("p2", "Q1")],
    )
    _write_documents(
        processed,
        stem,
        [
            _doc_row("a1", "Q1:wikipedia:en:100:1", "Q1", 100, 1),
            _doc_row("a2", "Q1:wikipedia:en:200:2", "Q1", 200, 2),
        ],
    )
    # Legacy: 4 rows but 3 distinct identities. Legacy page_ids must
    # match the resolved documents (per per-polygon QID membership).
    _write_legacy_links(
        processed,
        stem,
        [
            _legacy_row("p1", "a1", "Q1", page_id=100, revision_id=1),
            _legacy_row("p1", "a1", "Q1", page_id=100, revision_id=1),  # dup
            _legacy_row("p2", "a1", "Q1", page_id=100, revision_id=1),
            _legacy_row("p1", "a2", "Q1", page_id=200, revision_id=2),
        ],
    )

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    sp = next(s for s in plan.stems if s.stem == stem)
    assert sp.row_count == 3, (
        f"plan row_count must equal |distinct legacy identities| = 3, got {sp.row_count}"
    )
    mod.apply_link_migration(processed)
    canonical = pq.read_table(  # type: ignore[no-untyped-call]
        processed / "polygon_articles" / f"{stem}.parquet"
    ).to_pylist()
    assert len(canonical) == 3


# ---------------------------------------------------------------------------
# 4. Missing or ambiguous article_id blocks the stem
# ---------------------------------------------------------------------------


def test_legacy_article_id_missing_from_documents_blocks_stem(tmp_path: Path) -> None:
    """A legacy article_id without a matching document MUST block the stem."""
    mod = _import_module()
    stem = "alpha-latest"
    processed = tmp_path / "processed"

    _write_polygons(processed, stem, [_poly_row("p1", "Q1")])
    _write_documents(
        processed,
        stem,
        [_doc_row("a-EXISTS", "Q1:wikipedia:en:100:1", "Q1", 100, 1)],
    )
    _write_legacy_links(
        processed,
        stem,
        [
            _legacy_row("p1", "a-EXISTS", "Q1"),
            _legacy_row("p1", "a-MISSING", "Q1"),
        ],
    )

    plan = mod.plan_link_migration(processed)
    assert not plan.is_safe_to_apply
    sp = next(s for s in plan.stems if s.stem == stem)
    assert sp.classification == mod.StemClassification.BLOCKED
    assert "a-MISSING" in sp.reason


def test_ambiguous_article_id_blocks_stem(tmp_path: Path) -> None:
    """An article_id mapping to multiple documents blocks the stem."""
    mod = _import_module()
    stem = "alpha-latest"
    processed = tmp_path / "processed"

    _write_polygons(processed, stem, [_poly_row("p1", "Q1")])
    # Same article_id maps to two documents (different page_id).
    _write_documents(
        processed,
        stem,
        [
            _doc_row("a-AMBIG", "Q1:wikipedia:en:100:1", "Q1", 100, 1),
            _doc_row("a-AMBIG", "Q1:wikipedia:en:100:2", "Q1", 100, 2),
        ],
    )
    _write_legacy_links(
        processed,
        stem,
        [_legacy_row("p1", "a-AMBIG", "Q1")],
    )

    plan = mod.plan_link_migration(processed)
    assert not plan.is_safe_to_apply
    sp = next(s for s in plan.stems if s.stem == stem)
    assert sp.classification == mod.StemClassification.BLOCKED
    assert "AMBIG" in sp.reason or "ambig" in sp.reason or "multiple" in sp.reason.lower()
