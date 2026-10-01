"""Phase 2.5 / Defect 7: Both invalid-defect classes must be
audited/normalized before link migration.

The current code only normalizes Wikivoyage documents via
``plan_integrity_normalization``. The 237 invalid legacy Wikipedia
``polygon_articles`` relationships (rows that reference a wikidata
QID that is NOT in any of the polygon's resolved QIDs) are NEVER
audited -- the migration silently carries them through or drops them.

Both defect classes must be normalized:

* Invalid Wikipedia polygon↔article relationships: a legacy
  ``polygon_articles`` row whose ``wikidata`` is not a member of the
  polygon's resolved QID set.
* Invalid Wikivoyage document relationships and cascaded sections.

Every rejected relationship must be recorded in the cumulative ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation import rejection_ledger as rl
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.pipeline import link_migration as lm
from tests.migration._builders import (
    write_legacy_polygon_article,
    write_polygon_qid_membership,
    write_wikipedia_document,
)


def _seed_minimal_sidecars(processed: Path, stem: str) -> None:
    from osm_polygon_wikidata_only.augmentation.schema import section_schema

    # wikipedia + wikivoyage sections need the real schema (read by orchestrator)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikipedia" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / f"{stem}.parquet",
    )
    # For the rest, write an empty table with a single dummy string col.
    for sub in ("wikidata/facts",):
        empty = pa.table({"_placeholder": []})
        pq.write_table(empty, processed / Path(sub) / f"{stem}.parquet")  # type: ignore[no-untyped-call]


def test_invalid_wikipedia_relationships_are_rejected(tmp_path: Path) -> None:
    """An invalid legacy Wikipedia polygon↔article relationship (where
    the wikidata QID is NOT in the polygon's resolved QID set) must be
    rejected and recorded in the cumulative ledger.
    """

    # Polygon p1 has Q1 AND Q2 in its resolved QID set. The legacy
    # polygon_articles file references Q99 for p1 (not in {Q1,Q2}).
    # The wikipedia documents file DOES contain a Q99 row, so the
    # article_id reference is valid -- the only defect is the QID
    # membership.
    stem = "alpha-latest"
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
    write_polygon_qid_membership(processed / "polygons" / f"{stem}.parquet", "p1", ["Q1", "Q2"])
    write_legacy_polygon_article(
        processed / "polygon_articles" / f"{stem}.parquet", "p1", "Q99", page_id=1
    )
    write_wikipedia_document(
        processed / "wikipedia" / "documents" / f"{stem}.parquet", "Q99", page_id=1
    )
    _seed_minimal_sidecars(processed, stem)
    (processed / "manifests" / "processed_pbfs.json").write_text(
        json.dumps({f"{stem}.osm.pbf": {"source_pbf": f"{stem}.osm.pbf"}})
    )

    plan = lm.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    rejections = lm.plan_link_migration_normalization_rejections(plan)
    assert len(rejections) == 1
    assert rejections[0]["wikidata"] == "Q99"


def test_valid_wikipedia_relationships_are_not_rejected(tmp_path: Path) -> None:
    """A valid legacy Wikipedia relationship (where the wikidata QID
    IS in the polygon's resolved QID set) must NOT be rejected.
    """

    stem = "alpha-latest"
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
    write_polygon_qid_membership(processed / "polygons" / f"{stem}.parquet", "p1", ["Q1"])
    write_legacy_polygon_article(processed / "polygon_articles" / f"{stem}.parquet", "p1", "Q1")
    write_wikipedia_document(
        processed / "wikipedia" / "documents" / f"{stem}.parquet", "Q1", page_id=100
    )
    _seed_minimal_sidecars(processed, stem)
    (processed / "manifests" / "processed_pbfs.json").write_text(
        json.dumps({f"{stem}.osm.pbf": {"source_pbf": f"{stem}.osm.pbf"}})
    )

    plan = lm.plan_link_migration(processed)
    rejections = lm.plan_link_migration_normalization_rejections(plan)
    assert rejections == [], f"Valid Wikipedia relationships must NOT be rejected; got {rejections}"


def test_invalid_wikivoyage_relationships_are_rejected(tmp_path: Path) -> None:
    """An invalid Wikivoyage document relationship (wikidata QID
    absent from polygons) must be rejected by
    ``plan_integrity_normalization``.
    """

    stem = "alpha-latest"
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
    write_polygon_qid_membership(processed / "polygons" / f"{stem}.parquet", "p1", ["Q1"])
    # No wikipedia document needed for the wikivoyage defect class.
    write_legacy_polygon_article(processed / "polygon_articles" / f"{stem}.parquet", "p1", "Q1")
    write_wikipedia_document(
        processed / "wikipedia" / "documents" / f"{stem}.parquet", "Q1", page_id=1
    )
    _seed_minimal_sidecars(processed, stem)

    # Add an invalid wikivoyage document (Q99 not in polygons).
    from osm_polygon_wikidata_only.augmentation.schema import document_schema

    invalid_document = {
        "document_id": "Q99:wikivoyage:en:2:1",
        "article_id": "Q99:en:2:1",
        "wikidata": "Q99",
        "project": "wikivoyage",
        "language": "en",
        "site": "enwikivoyage",
        "title": "T",
        "url": "https://en.wikivoyage.org/wiki/T",
        "page_id": 2,
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
    valid_document = {
        **invalid_document,
        "document_id": "Q1:wikivoyage:en:1:1",
        "article_id": "Q1:en:1:1",
        "wikidata": "Q1",
        "page_id": 1,
        "url": "https://en.wikivoyage.org/wiki/Valid",
    }
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [valid_document, invalid_document],
            schema=document_schema(),
        ),
        processed / "wikivoyage" / "documents" / f"{stem}.parquet",
    )
    (processed / "manifests" / "processed_pbfs.json").write_text(
        json.dumps({f"{stem}.osm.pbf": {"source_pbf": f"{stem}.osm.pbf"}})
    )

    dr = DataRoot(tmp_path)
    # Use plan_integrity_normalization directly for wikivoyage.
    plan = rl.plan_integrity_normalization(dr, stem)
    wikivoyage_rejections = [r for r in plan.rejections if r.source_table == "wikivoyage_documents"]
    assert wikivoyage_rejections, (
        f"Invalid Wikivoyage relationships must be rejected; got {plan.rejections}"
    )

    lm.apply_link_migration(processed)

    normalized_documents = pq.read_table(
        processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    ).to_pylist()
    assert [row["document_id"] for row in normalized_documents] == ["Q1:wikivoyage:en:1:1"]
    canonical_links = pq.read_table(processed / "polygon_articles" / f"{stem}.parquet").to_pylist()
    assert ("p1", "Q1:wikivoyage:en:1:1") in {
        (row["polygon_id"], row["document_id"]) for row in canonical_links
    }
    assert all(row["document_id"] != "Q99:wikivoyage:en:2:1" for row in canonical_links)
    assert not (processed / ".link_migration_staging" / stem).exists()
    ledger = json.loads((processed / "integrity" / "rejection_ledger.json").read_text())
    voyage_records = [
        record for record in ledger["records"] if record["source_table"] == "wikivoyage_documents"
    ]
    assert len(voyage_records) == 1
    assert voyage_records[0]["identifier"] == "Q99:wikivoyage:en:2:1"
    assert voyage_records[0]["wikidata"] == "Q99"
