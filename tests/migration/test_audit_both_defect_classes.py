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


def _import_modules():
    from osm_polygon_wikidata_only.augmentation import rejection_ledger as rl
    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.pipeline import link_migration as lm

    return lm, rl, DataRoot


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


def _seed_legacy_links(path: Path, polygon_id: str, wikidata: str, page_id: int = 1) -> None:
    from osm_polygon_wikidata_only.domain.schema import polygon_article_schema

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "polygon_id": polygon_id,
                    "article_id": f"{wikidata}:en:{page_id}:1",
                    "wikidata": wikidata,
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


def _seed_wiki_docs(processed: Path, stem: str, qid: str, page_id: int) -> None:
    from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
        wikipedia_document_schema,
    )

    # Match the canonical document_id format: {qid}:{project}:{language}:{page_id}:{revision_id}
    document_id = f"{qid}:wikipedia:en:{page_id}:1"
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
        processed / "wikipedia" / "documents" / f"{stem}.parquet",
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
    lm, _rl, _DataRoot = _import_modules()

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
    _seed_polygons(processed / "polygons" / f"{stem}.parquet", "p1", ["Q1", "Q2"])
    _seed_legacy_links(processed / "polygon_articles" / f"{stem}.parquet", "p1", "Q99", page_id=1)
    _seed_wiki_docs(processed, stem, "Q99", 1)
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
    lm, _rl, _DataRoot = _import_modules()

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
    _seed_polygons(processed / "polygons" / f"{stem}.parquet", "p1", ["Q1"])
    _seed_legacy_links(processed / "polygon_articles" / f"{stem}.parquet", "p1", "Q1")
    _seed_wiki_docs(processed, stem, "Q1", 100)
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
    _lm, rl, DataRoot = _import_modules()

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
    _seed_polygons(processed / "polygons" / f"{stem}.parquet", "p1", ["Q1"])
    # No wikipedia document needed for the wikivoyage defect class.
    # Skip writing wikipedia documents.
    _seed_legacy_links(processed / "polygon_articles" / f"{stem}.parquet", "p1", "Q1")
    _seed_minimal_sidecars(processed, stem)

    # Add an invalid wikivoyage document (Q99 not in polygons).
    from osm_polygon_wikidata_only.augmentation.schema import document_schema

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "document_id": "Q99:wikivoyage:en:1:1",
                    "article_id": "Q99:en:1:1",
                    "wikidata": "Q99",
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
            ],
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
