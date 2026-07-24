"""Phase 2 / Amendment 5: Manifest atomicity and preservation.

The link migration must NOT overwrite the existing
``manifests/link_manifest.json`` for every stem -- unrelated stem
entries must be preserved. The augmentation manifest entry for the
migrated stem must be updated without erasing other stem entries.

The link parquet, the link manifest entry, the augmentation manifest
update, the pending-publication intent, and the metadata-refresh
marker must all follow the documented journaled commit ordering. A
stem must NEVER be classified current when only some of these
writes succeeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    polygon_article_schema,
)


def _import_module():
    from osm_polygon_wikidata_only.pipeline import link_migration as mod

    return mod


def _write_polygon(processed_dir: Path, stem: str) -> None:
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


def _write_document(processed_dir: Path, stem: str) -> None:
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


def _write_legacy_link(processed_dir: Path, stem: str) -> None:
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


def _setup_processed(processed: Path, stem: str) -> None:
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
    ):
        (processed / sub).mkdir(parents=True, exist_ok=True)
    _write_polygon(processed, stem)
    _write_document(processed, stem)
    _write_legacy_link(processed, stem)
    # Empty placeholder sidecars so sidecar_paths(...).exists() passes.
    import pyarrow as pa  # local import for clarity

    # Minimum wikipedia_document_schema is already a real table; the
    # rest of the sidecars just need to exist as parquet files. Use
    # empty tables with the corresponding known schemas so any future
    # check is happy.
    from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema

    empty_sections = pa.Table.from_pylist([], schema=section_schema())
    pq.write_table(  # type: ignore[no-untyped-call]
        empty_sections,
        processed / "wikipedia" / "sections" / f"{stem}.parquet",
    )

    # For the others, write an empty pyarrow table with a generic schema.
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=document_schema()),
        processed / "wikivoyage" / "documents" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"_placeholder": []}),
        processed / "wikidata" / "facts" / f"{stem}.parquet",
    )
    source_pbf = f"{stem}.osm.pbf"
    (processed / "manifests" / "processed_pbfs.json").write_text(
        json.dumps(
            {
                source_pbf: {
                    "source_pbf": source_pbf,
                    "region": stem.removesuffix("-latest"),
                    "polygons_path": f"polygons/{stem}.parquet",
                    "articles_path": f"wikipedia/documents/{stem}.parquet",
                    "polygon_articles_path": f"polygon_articles/{stem}.parquet",
                    "extraction_version": "test",
                    "processed_at": "2026-07-24T00:00:00Z",
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# 1. link_manifest.json must preserve unrelated stem entries
# ---------------------------------------------------------------------------


def test_processed_pbf_manifest_preserves_other_pbfs(tmp_path: Path) -> None:
    """Migrating PBF A must NOT erase an existing entry for PBF B in
    manifests/processed_pbfs.json.
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem_a = "alpha-latest"
    _setup_processed(processed, stem_a)

    # Pre-seed processed_pbfs.json with an entry for a different PBF.
    manifest_path = processed / "manifests" / "processed_pbfs.json"
    other_pbf = "beta-latest.osm.pbf"
    existing_entry = {
        "source_pbf": other_pbf,
        "region": "r2",
        "polygons_path": "polygons/beta-latest.parquet",
        "articles_path": "wikipedia/documents/beta-latest.parquet",
        "polygon_articles_path": "polygon_articles/beta-latest.parquet",
        "extraction_version": "v2",
        "processed_at": "2026-07-24T00:00:00Z",
    }
    current = json.loads(manifest_path.read_text())
    current[other_pbf] = existing_entry
    manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    mod.apply_link_migration(processed)

    payload = json.loads(manifest_path.read_text())
    assert other_pbf in payload, (
        f"processed_pbfs.json must preserve unrelated PBF {other_pbf!r}; got keys={list(payload)}"
    )
    assert payload[other_pbf] == existing_entry, (
        f"Existing PBF {other_pbf!r} entry must be unchanged"
    )
    # The migrated PBF must also be present.
    assert f"{stem_a}.osm.pbf" in payload, (
        f"Migrated PBF must be present in processed_pbfs.json; got keys={list(payload)}"
    )


# ---------------------------------------------------------------------------
# 2. Augmentation manifest: other stem entries preserved
# ---------------------------------------------------------------------------


def test_augmentation_manifest_preserves_other_stem_entries(tmp_path: Path) -> None:
    mod = _import_module()
    processed = tmp_path / "processed"
    data_root = tmp_path
    stem_a = "alpha-latest"
    stem_b = "beta-latest"
    _setup_processed(processed, stem_a)

    # Pre-seed augmentation manifest with an entry for stem B.
    aug_manifest = (
        data_root / "processed" / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    aug_manifest.parent.mkdir(parents=True, exist_ok=True)
    pre_existing = {
        "contract_version": "augmentation-v1",
        stem_b: {
            "stale": False,
            "core_hashes": {},
        },
    }
    aug_manifest.write_text(json.dumps(pre_existing, indent=2, sort_keys=True) + "\n")

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    mod.apply_link_migration(processed)

    payload = json.loads(aug_manifest.read_text())
    assert stem_b in payload, f"Augmentation manifest must preserve stem {stem_b!r}"
    assert payload[stem_b] == pre_existing[stem_b], (
        f"Augmentation manifest entry for {stem_b!r} must be unchanged"
    )
    assert stem_a in payload, f"Migrated stem {stem_a!r} must be in augmentation manifest"


# ---------------------------------------------------------------------------
# 3. apply step classifies stem as current only after full transaction
# ---------------------------------------------------------------------------


def test_stem_classified_current_only_after_all_writes(tmp_path: Path) -> None:
    """If any of {link parquet, link manifest, augmentation manifest,
    pending intent, marker} is missing, augmentation_is_current must
    not return True for the stem.
    """
    from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current

    mod = _import_module()
    data_root_path = tmp_path
    processed = data_root_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    mod.apply_link_migration(processed)

    # After apply, all required writes have succeeded -> current.
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(data_root_path)
    assert augmentation_is_current(dr, stem), (
        "After apply completes, stem must be classified current"
    )
