"""Phase 2 / Amendment 3: Exact canonical Parquet schema.

The canonical classification must use
``schema.equals(expected, check_metadata=True)`` -- not column names
alone. Migrated tables must be constructed with
``polygon_document_link_schema()`` explicitly. Order, types,
nullability, and field metadata are part of the contract. Mixed,
reordered, mistyped, extra-metadata, and metadata-less lookalike
schemas are rejected.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema


def _import_module():
    try:
        from osm_polygon_wikidata_only.pipeline import link_migration as mod
    except ImportError:
        pytest.fail("link_migration module must exist")
    return mod


def _write_minimal_stem(processed_dir: Path, stem: str) -> None:
    polygons_dir = processed_dir / "polygons"
    polygons_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = processed_dir / "wikipedia" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    links_dir = processed_dir / "polygon_articles"
    links_dir.mkdir(parents=True, exist_ok=True)
    polygons = pa.table(
        {
            "polygon_id": ["p1"],
            "wikidata": ["Q1"],
            "source_pbf": [f"{stem}.osm.pbf"],
            "region": ["r"],
            "osm_type": ["way"],
            "osm_id": [1],
            "name": [""],
            "tags": [""],
            "tag_keys": [""],
            "tag_count": [0],
            "osm_primary_tag": [""],
            "centroid": [""],
            "lat": [0.0],
            "lon": [0.0],
            "bbox": [""],
            "geometry": [""],
            "area_m2": [0.0],
            "area_km2": [0.0],
            "area_bucket": [""],
            "has_name": [False],
            "has_wikidata": [True],
            "has_wikipedia": [False],
            "wikipedia_language_count": [0],
            "wikipedia_languages": [""],
            "wikipedia_article_count": [0],
            "has_english_wikipedia": [False],
            "has_french_wikipedia": [False],
            "text_available": [False],
            "best_language": ["en"],
            "extraction_version": ["test"],
            "extracted_at": ["2026-07-24T00:00:00Z"],
        }
    )
    pq.write_table(polygons, polygons_dir / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    doc = wikipedia_document_schema().empty_table()
    pq.write_table(doc, docs_dir / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    links = polygon_article_schema().empty_table()
    pq.write_table(links, links_dir / f"{stem}.parquet")  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Strict schema classification
# ---------------------------------------------------------------------------


def test_classify_rejects_lookalike_with_extra_metadata(tmp_path: Path) -> None:
    """A table with the canonical column names but extra field metadata is NOT canonical."""
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _write_minimal_stem(processed, stem)

    canonical = polygon_document_link_schema()
    # Add an extra metadata key to one field to create a lookalike schema.
    lookalike = pa.schema(
        [
            pa.field(
                f.name,
                f.type,
                metadata={**f.metadata, b"extra": b"x"} if f.metadata else f.metadata,
            )
            for f in canonical
        ]
    )
    rows_path = processed / "polygon_articles" / f"{stem}.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "polygon_id": "p1",
                "document_id": "Q1:wikipedia:en:100:1",
                "project": "wikipedia",
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": "alpha-latest.osm.pbf",
                "region": "r",
                "osm_type": "way",
                "osm_id": 1,
                "page_id": 100,
                "revision_id": 1,
            }
        ],
        schema=lookalike,
    )
    pq.write_table(table, rows_path)  # type: ignore[no-untyped-call]

    plan = mod.plan_link_migration(processed)
    sp = next(s for s in plan.stems if s.stem == stem)
    assert sp.classification == mod.StemClassification.BLOCKED, (
        f"Schema with extra metadata must not be classified canonical; got {sp.classification}"
    )


def test_classify_rejects_reordered_canonical(tmp_path: Path) -> None:
    """Reordering canonical columns produces a lookalike schema that must be BLOCKED."""
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _write_minimal_stem(processed, stem)

    canonical = polygon_document_link_schema()
    fields = list(canonical)
    # Swap the first two fields.
    reordered = pa.schema([fields[1], fields[0], *fields[2:]])
    rows_path = processed / "polygon_articles" / f"{stem}.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "document_id": "Q1:wikipedia:en:100:1",
                "polygon_id": "p1",
                "project": "wikipedia",
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": "alpha-latest.osm.pbf",
                "region": "r",
                "osm_type": "way",
                "osm_id": 1,
                "page_id": 100,
                "revision_id": 1,
            }
        ],
        schema=reordered,
    )
    pq.write_table(table, rows_path)  # type: ignore[no-untyped-call]

    plan = mod.plan_link_migration(processed)
    sp = next(s for s in plan.stems if s.stem == stem)
    assert sp.classification == mod.StemClassification.BLOCKED, (
        f"Reordered schema must not be classified canonical; got {sp.classification}"
    )


def test_classify_rejects_mistyped_field(tmp_path: Path) -> None:
    """A column with the wrong type is not canonical."""
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _write_minimal_stem(processed, stem)

    canonical = polygon_document_link_schema()
    bad_fields = [
        pa.field(f.name, pa.string() if f.name == "osm_id" else f.type, metadata=f.metadata)
        for f in canonical
    ]
    bad_schema = pa.schema(bad_fields)
    rows_path = processed / "polygon_articles" / f"{stem}.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "polygon_id": "p1",
                "document_id": "Q1:wikipedia:en:100:1",
                "project": "wikipedia",
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": "alpha-latest.osm.pbf",
                "region": "r",
                "osm_type": "way",
                "osm_id": "1",  # should be int64
                "page_id": 100,
                "revision_id": 1,
            }
        ],
        schema=bad_schema,
    )
    pq.write_table(table, rows_path)  # type: ignore[no-untyped-call]

    plan = mod.plan_link_migration(processed)
    sp = next(s for s in plan.stems if s.stem == stem)
    assert sp.classification == mod.StemClassification.BLOCKED


# ---------------------------------------------------------------------------
# Migrated table uses polygon_document_link_schema() explicitly
# ---------------------------------------------------------------------------


def test_apply_writes_canonical_schema_exactly(tmp_path: Path) -> None:
    """The migrated table must have schema == polygon_document_link_schema() (with metadata)."""
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _write_minimal_stem(processed, stem)

    # Write a legacy row + a matching document.
    links = pa.Table.from_pylist(
        [
            {
                "polygon_id": "p1",
                "article_id": "a1",
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": "alpha-latest.osm.pbf",
                "region": "r",
                "osm_type": "way",
                "osm_id": 1,
                "page_id": 100,
                "revision_id": 1,
                "is_best_language": True,
            }
        ],
        schema=polygon_article_schema(),
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        links,
        processed / "polygon_articles" / f"{stem}.parquet",
    )

    doc = pa.Table.from_pylist(
        [
            {
                "document_id": "Q1:wikipedia:en:100:1",
                "article_id": "a1",
                "wikidata": "Q1",
                "language": "en",
                "site": "enwiki",
                "title": "T",
                "url": "https://en.wikipedia.org/wiki/T",
                "page_id": 100,
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
    )
    pq.write_table(doc, processed / "wikipedia" / "documents" / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    mod.apply_link_migration(processed)
    written = pq.read_table(processed / "polygon_articles" / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    assert written.schema.equals(polygon_document_link_schema(), check_metadata=True), (
        f"Migrated table must have canonical schema (with metadata); got {written.schema}"
    )
