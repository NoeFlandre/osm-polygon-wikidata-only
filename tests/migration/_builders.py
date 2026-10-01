"""Shared Parquet fixture builders for link-migration transactions."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema


def write_polygon(processed_dir: Path, stem: str) -> None:
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
    pq.write_table(  # type: ignore[no-untyped-call]
        polygons, processed_dir / "polygons" / f"{stem}.parquet"
    )


def write_polygon_qid_membership(path: Path, polygon_id: str, qids: list[str]) -> None:
    """Write the compact polygon row used by QID-membership migration tests."""
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


def write_document(processed_dir: Path, stem: str) -> None:
    table = pa.Table.from_pylist(
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
    pq.write_table(  # type: ignore[no-untyped-call]
        table, processed_dir / "wikipedia" / "documents" / f"{stem}.parquet"
    )


def write_legacy_link(processed_dir: Path, stem: str) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "polygon_id": "p1",
                "article_id": "a1",
                "wikidata": "Q1",
                "language": "en",
                "source_pbf": f"{stem}.osm.pbf",
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
        table, processed_dir / "polygon_articles" / f"{stem}.parquet"
    )


def seed_processed_migration_stem(processed_dir: Path, stem: str, *, region: str) -> None:
    """Create the complete processed-shard fixture used by migration tests."""
    directories = (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
        "augmentation/manifests",
    )
    for relative_path in directories:
        (processed_dir / relative_path).mkdir(parents=True, exist_ok=True)

    write_polygon(processed_dir, stem)
    write_document(processed_dir, stem)
    write_legacy_link(processed_dir, stem)

    empty_sections = pa.Table.from_pylist([], schema=section_schema())
    pq.write_table(  # type: ignore[no-untyped-call]
        empty_sections,
        processed_dir / "wikipedia" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=document_schema()),
        processed_dir / "wikivoyage" / "documents" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed_dir / "wikivoyage" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"_placeholder": []}),
        processed_dir / "wikidata" / "facts" / f"{stem}.parquet",
    )

    source_pbf = f"{stem}.osm.pbf"
    manifest = {
        source_pbf: {
            "source_pbf": source_pbf,
            "region": region,
            "polygons_path": f"polygons/{stem}.parquet",
            "articles_path": f"wikipedia/documents/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            "extraction_version": "test",
            "processed_at": "2026-07-24T00:00:00Z",
        }
    }
    (processed_dir / "manifests" / "processed_pbfs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
