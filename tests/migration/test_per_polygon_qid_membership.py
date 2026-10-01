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

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema
from osm_polygon_wikidata_only.pipeline import link_migration as lm
from tests.migration._builders import (
    write_legacy_polygon_article,
    write_polygon_qid_membership,
    write_wikipedia_document,
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
    write_legacy_polygon_article(
        processed / "polygon_articles" / "alpha-latest.parquet", "p1", "Q2"
    )
    links_path = processed / "polygon_articles" / "alpha-latest.parquet"
    legacy_rows = pq.read_table(links_path).to_pylist()
    second_legacy_row = {
        **legacy_rows[0],
        "article_id": "Q2:en:2:1",
        "page_id": 2,
    }
    pq.write_table(
        pa.Table.from_pylist(
            [second_legacy_row, legacy_rows[0]],
            schema=polygon_article_schema(),
        ),
        links_path,
    )
    write_wikipedia_document(processed / "wikipedia" / "documents" / "alpha-latest.parquet", "Q2")

    documents_path = processed / "wikipedia" / "documents" / "alpha-latest.parquet"
    document_rows = pq.read_table(documents_path).to_pylist()
    second_document = {
        **document_rows[0],
        "document_id": "Q2:wikipedia:en:2:1",
        "article_id": "Q2:en:2:1",
        "page_id": 2,
    }
    pq.write_table(
        pa.Table.from_pylist(
            [*document_rows, second_document],
            schema=wikipedia_document_schema(),
        ),
        documents_path,
    )
    (processed / "manifests" / "processed_pbfs.json").write_text(
        '{"test.osm.pbf": {"source_pbf": "test.osm.pbf"}}'
    )
    # Empty sidecars.
    from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikipedia" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / "alpha-latest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([], schema=document_schema()),
        processed / "wikivoyage" / "documents" / "alpha-latest.parquet",
    )  # type: ignore[no-untyped-call]
    pq.write_table(
        pa.table({"_placeholder": []}), processed / "wikidata" / "facts" / "alpha-latest.parquet"
    )  # type: ignore[no-untyped-call]

    plan = lm.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    rejections = lm.plan_link_migration_normalization_rejections(plan)
    assert [rejection["identifier"] for rejection in rejections] == [
        "Q2:en:1:1",
        "Q2:en:2:1",
    ]

    lm.apply_link_migration(processed, plan=plan)
    ledger = json.loads((processed / "integrity" / "rejection_ledger.json").read_text())
    assert [record["identifier"] for record in ledger["records"]] == [
        "Q2:en:1:1",
        "Q2:en:2:1",
    ]
    assert {record["wikidata"] for record in ledger["records"]} == {"Q2"}
    assert {record["reason"] for record in ledger["records"]} == {"wikidata_not_in_polygon_qids"}


# ---------------------------------------------------------------------------
# 2. Conflicting duplicate legacy rows block migration
# ---------------------------------------------------------------------------


def test_conflicting_duplicate_legacy_rows_block_migration_contract(tmp_path: Path) -> None:
    """Two legacy rows for the same (polygon_id, article_id) but with
    different wikidata values must BLOCK the migration -- they cannot
    silently collapse.
    """

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
    write_polygon_qid_membership(processed / "polygons" / "alpha-latest.parquet", "p1", ["Q1"])
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
    write_wikipedia_document(processed / "wikipedia" / "documents" / "alpha-latest.parquet", "Q1")
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
    assert "polygon_id='p1'" in blocked[0].reason
    assert "article_id='Q1:en:1:1'" in blocked[0].reason


# ---------------------------------------------------------------------------
# 3. Byte-identical duplicate legacy rows may collapse
# ---------------------------------------------------------------------------


def test_byte_identical_duplicate_legacy_rows_collapse(tmp_path: Path) -> None:
    """Two byte-identical legacy rows for the same
    (polygon_id, article_id) MUST collapse to a single canonical row.
    """

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
    write_polygon_qid_membership(processed / "polygons" / "alpha-latest.parquet", "p1", ["Q1"])
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
    write_wikipedia_document(processed / "wikipedia" / "documents" / "alpha-latest.parquet", "Q1")
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
    assert lm.plan_link_migration_normalization_rejections(plan) == []
