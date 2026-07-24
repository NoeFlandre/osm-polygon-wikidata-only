"""Phase 2.5 / Defect 8: Per-polygon QID membership and row
field validation.

For every legacy ``polygon_articles`` row:

* The resolved document's QID must belong to that specific polygon's
  parsed multi-QID tag (per-polygon membership, not a region-wide
  union).
* The legacy row's QID, page_id, revision_id and language fields must
  agree with the resolved document.
* Byte-identical duplicate legacy rows may collapse; conflicting
  duplicates must BLOCK the migration.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _import_modules():
    from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
        wikipedia_document_schema,
    )
    from osm_polygon_wikidata_only.domain.schema import polygon_article_schema
    from osm_polygon_wikidata_only.pipeline import link_migration as lm

    return lm, wikipedia_document_schema, polygon_article_schema


def _seed_polygons(path: Path, polygon_id: str, qids: list[str]) -> None:
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "polygon_id": [polygon_id],
                "wikidata": [";".join(qids)],
                "source_pbf": ["test.osm.pbf"],
                "region": ["r"],
            }
        ),
        path,
    )


def _seed_wiki_doc(path: Path, document_id: str, qid: str, page_id: int = 1) -> None:
    _, wikipedia_document_schema, _ = _import_modules()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "document_id": document_id,
                    "article_id": f"{qid}:en:{page_id}:1",
                    "wikidata": qid,
                    "language": "en",
                    "site": "enwiki",
                    "title": "T",
                    "url": f"https://en.wikipedia.org/wiki/{qid}",
                    "page_id": page_id,
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
            ],
            schema=wikipedia_document_schema(),
        ),
        path,
    )


def _seed_legacy(path: Path, polygon_id: str, qid: str, page_id: int = 1) -> None:
    _, _, polygon_article_schema = _import_modules()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "polygon_id": polygon_id,
                    "article_id": f"{qid}:en:{page_id}:1",
                    "wikidata": qid,
                    "language": "en",
                    "source_pbf": "test.osm.pbf",
                    "region": "r",
                    "osm_type": "way",
                    "osm_id": 1,
                    "page_id": page_id,
                    "revision_id": 1,
                    "is_best_language": True,
                }
            ],
            schema=polygon_article_schema(),
        ),
        path,
    )


# ---------------------------------------------------------------------------
# 1. Per-polygon QID membership: reject if the document's QID is not
#    in that polygon's QID set, even if another polygon in the same
#    region does include it.
# ---------------------------------------------------------------------------


def test_qid_membership_is_per_polygon_not_region_wide(tmp_path: Path) -> None:
    """Polygon p1 has Q1 only; polygon p2 has Q1 AND Q2. A legacy
    row referencing Q2 on polygon p1 must be rejected (Q2 is in
    the region's set but not in p1's set).
    """
    lm, _, _ = _import_modules()

    processed = tmp_path / "processed"
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
        "augmentation/manifests",
    ):
        (processed / sub).mkdir(parents=True, exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "polygon_id": ["p1", "p2"],
                "wikidata": ["Q1", "Q1;Q2"],
                "source_pbf": ["test.osm.pbf", "test.osm.pbf"],
                "region": ["r", "r"],
            }
        ),
        processed / "polygons" / "alpha-latest.parquet",
    )
    _seed_legacy(processed / "polygon_articles" / "alpha-latest.parquet", "p1", "Q2")
    _seed_wiki_doc(
        processed / "wikipedia" / "documents" / "alpha-latest.parquet",
        "Q2:wikipedia:en:1:1",
        "Q2",
    )
    (processed / "manifests" / "processed_pbfs.json").write_text(
        '{"test.osm.pbf": {"source_pbf": "test.osm.pbf"}}'
    )
    # Empty sidecars.
    from osm_polygon_wikidata_only.augmentation.schema import section_schema

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikipedia" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(
        pa.table({"_placeholder": []}),
        processed / "wikivoyage" / "documents" / "alpha-latest.parquet",
    )  # type: ignore[no-untyped-call]
    pq.write_table(
        pa.table({"_placeholder": []}), processed / "wikidata" / "facts" / "alpha-latest.parquet"
    )  # type: ignore[no-untyped-call]

    plan = lm.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    rejections = lm.plan_link_migration_normalization_rejections(plan)
    assert len(rejections) == 1
    assert rejections[0]["wikidata"] == "Q2"


# ---------------------------------------------------------------------------
# 2. Conflicting duplicate legacy rows block migration
# ---------------------------------------------------------------------------


def test_conflicting_duplicate_legacy_rows_block_migration(tmp_path: Path) -> None:
    """Two legacy rows for the same (polygon_id, article_id) but with
    different wikidata values must BLOCK the migration -- they cannot
    silently collapse.
    """
    lm, _, polygon_article_schema = _import_modules()

    processed = tmp_path / "processed"
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
        "augmentation/manifests",
    ):
        (processed / sub).mkdir(parents=True, exist_ok=True)
    _seed_polygons(processed / "polygons" / "alpha-latest.parquet", "p1", ["Q1"])
    # Two legacy rows for (p1, "Q1:en:1:1") -- one with wikidata=Q1, one with Q2.
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "polygon_id": "p1",
                    "article_id": "Q1:en:1:1",
                    "wikidata": "Q1",
                    "language": "en",
                    "source_pbf": "test.osm.pbf",
                    "region": "r",
                    "osm_type": "way",
                    "osm_id": 1,
                    "page_id": 1,
                    "revision_id": 1,
                    "is_best_language": True,
                },
                {
                    "polygon_id": "p1",
                    "article_id": "Q1:en:1:1",
                    "wikidata": "Q2",
                    "language": "en",
                    "source_pbf": "test.osm.pbf",
                    "region": "r",
                    "osm_type": "way",
                    "osm_id": 1,
                    "page_id": 1,
                    "revision_id": 1,
                    "is_best_language": True,
                },
            ],
            schema=polygon_article_schema(),
        ),
        processed / "polygon_articles" / "alpha-latest.parquet",
    )
    _seed_wiki_doc(
        processed / "wikipedia" / "documents" / "alpha-latest.parquet",
        "Q1:wikipedia:en:1:1",
        "Q1",
    )
    (processed / "manifests" / "processed_pbfs.json").write_text(
        '{"test.osm.pbf": {"source_pbf": "test.osm.pbf"}}'
    )
    from osm_polygon_wikidata_only.augmentation.schema import section_schema

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikipedia" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(
        pa.table({"_placeholder": []}),
        processed / "wikivoyage" / "documents" / "alpha-latest.parquet",
    )  # type: ignore[no-untyped-call]
    pq.write_table(
        pa.table({"_placeholder": []}), processed / "wikidata" / "facts" / "alpha-latest.parquet"
    )  # type: ignore[no-untyped-call]

    plan = lm.plan_link_migration(processed)
    blocked = [s for s in plan.stems if s.classification == lm.StemClassification.BLOCKED]
    assert blocked, f"Conflicting duplicate legacy rows must BLOCK the migration; plan={plan}"
    assert (
        "conflict" in blocked[0].reason.lower()
        or "duplicate" in blocked[0].reason.lower()
        or "ambiguous" in blocked[0].reason.lower()
    ), f"Block reason must mention conflict/duplicate/ambiguous; got {blocked[0].reason}"


# ---------------------------------------------------------------------------
# 3. Byte-identical duplicate legacy rows may collapse
# ---------------------------------------------------------------------------


def test_byte_identical_duplicate_legacy_rows_collapse(tmp_path: Path) -> None:
    """Two byte-identical legacy rows for the same
    (polygon_id, article_id) MUST collapse to a single canonical row.
    """
    lm, _, polygon_article_schema = _import_modules()

    processed = tmp_path / "processed"
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
        "augmentation/manifests",
    ):
        (processed / sub).mkdir(parents=True, exist_ok=True)
    _seed_polygons(processed / "polygons" / "alpha-latest.parquet", "p1", ["Q1"])
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "polygon_id": "p1",
                    "article_id": "Q1:en:1:1",
                    "wikidata": "Q1",
                    "language": "en",
                    "source_pbf": "test.osm.pbf",
                    "region": "r",
                    "osm_type": "way",
                    "osm_id": 1,
                    "page_id": 1,
                    "revision_id": 1,
                    "is_best_language": True,
                },
                {
                    "polygon_id": "p1",
                    "article_id": "Q1:en:1:1",
                    "wikidata": "Q1",
                    "language": "en",
                    "source_pbf": "test.osm.pbf",
                    "region": "r",
                    "osm_type": "way",
                    "osm_id": 1,
                    "page_id": 1,
                    "revision_id": 1,
                    "is_best_language": True,
                },
            ],
            schema=polygon_article_schema(),
        ),
        processed / "polygon_articles" / "alpha-latest.parquet",
    )
    _seed_wiki_doc(
        processed / "wikipedia" / "documents" / "alpha-latest.parquet",
        "Q1:wikipedia:en:1:1",
        "Q1",
    )
    (processed / "manifests" / "processed_pbfs.json").write_text(
        '{"test.osm.pbf": {"source_pbf": "test.osm.pbf"}}'
    )
    from osm_polygon_wikidata_only.augmentation.schema import section_schema

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikipedia" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(
        pa.table({"_placeholder": []}),
        processed / "wikivoyage" / "documents" / "alpha-latest.parquet",
    )  # type: ignore[no-untyped-call]
    pq.write_table(
        pa.table({"_placeholder": []}), processed / "wikidata" / "facts" / "alpha-latest.parquet"
    )  # type: ignore[no-untyped-call]

    plan = lm.plan_link_migration(processed)
    assert plan.is_safe_to_apply, (
        f"Byte-identical duplicates must collapse to a single canonical row; plan={plan}"
    )
