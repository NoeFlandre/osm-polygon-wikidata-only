"""Phase 2.5 / Defect 4: Manifest amendment 5 (link_schema_version +
link_artifact_sha256) is not implemented.

The approved design requires:

* ``link_schema_version`` and ``link_count`` added to the processed
  manifest entry ``<data-root>/processed/manifests/processed_pbfs.json``
  for the migrated PBF.
* ``link_schema_version`` and ``link_artifact_sha256`` (64-hex SHA-256
  of the canonical link file) added to the augmentation manifest entry
  keyed by stem.
* NO secondary ``manifests/link_manifest.json``. Any such artifact must
  not be created by migration.

Additionally:

* Pending-publication intent must be written for the migrated stem.
* Metadata-refresh marker must be written for the migrated stem,
  carrying the per-region ``link_artifact_sha256`` from the augmentation
  manifest.

Malformed existing manifest JSON must block (never silently replace
corruption with an empty manifest). The manifest update must merge
with existing entries (preserving unrelated fields/stems).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
    pq.write_table(polygons, processed_dir / "polygons" / f"{stem}.parquet")  # type: ignore[no-untyped-call]


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
    pq.write_table(table, processed_dir / "wikipedia" / "documents" / f"{stem}.parquet")  # type: ignore[no-untyped-call]


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
    pq.write_table(table, processed_dir / "polygon_articles" / f"{stem}.parquet")  # type: ignore[no-untyped-call]


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
    from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema

    empty_sections = pa.Table.from_pylist([], schema=section_schema())
    pq.write_table(  # type: ignore[no-untyped-call]
        empty_sections,
        processed / "wikipedia" / "sections" / f"{stem}.parquet",
    )
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
# 1. NO secondary manifests/link_manifest.json is created
# ---------------------------------------------------------------------------


def test_apply_does_not_create_link_manifest_json(tmp_path: Path) -> None:
    """Migration must NOT create a secondary
    ``manifests/link_manifest.json`` -- the approved design updates the
    EXISTING ``processed_pbfs.json`` instead.
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    # Pre-seed the processed manifest with an entry for the PBF.
    processed_manifest = processed / "manifests" / "processed_pbfs.json"
    processed_manifest.write_text(
        json.dumps(
            {
                f"{stem}.osm.pbf": {
                    "source_pbf": f"{stem}.osm.pbf",
                    "region": "r",
                    "polygons_path": f"polygons/{stem}.parquet",
                    "articles_path": f"wikipedia/documents/{stem}.parquet",
                    "polygon_articles_path": f"polygon_articles/{stem}.parquet",
                    "extraction_version": "test",
                    "processed_at": "2026-07-24T00:00:00Z",
                }
            }
        )
    )

    plan = mod.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    mod.apply_link_migration(processed)

    link_manifest = processed / "manifests" / "link_manifest.json"
    assert not link_manifest.exists(), (
        f"Migration must NOT create {link_manifest}; the design updates processed_pbfs.json"
    )


# ---------------------------------------------------------------------------
# 2. processed_pbfs.json: link_schema_version + link_count
# ---------------------------------------------------------------------------


def test_apply_writes_link_schema_version_and_count_to_processed_manifest(tmp_path: Path) -> None:
    """The processed manifest entry for the migrated PBF must include
    ``link_schema_version`` and ``link_count`` (with the actual local
    table row count), and other fields must be preserved.
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    processed_manifest = processed / "manifests" / "processed_pbfs.json"
    pre_existing = {
        "source_pbf": f"{stem}.osm.pbf",
        "region": "r",
        "polygons_path": f"polygons/{stem}.parquet",
        "articles_path": f"wikipedia/documents/{stem}.parquet",
        "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        "extraction_version": "test",
        "processed_at": "2026-07-24T00:00:00Z",
        "unrelated_field": {"preserve": "me"},
    }
    processed_manifest.write_text(
        json.dumps({f"{stem}.osm.pbf": pre_existing}, indent=2, sort_keys=True) + "\n"
    )

    mod.apply_link_migration(processed)

    payload = json.loads(processed_manifest.read_text())
    entry = payload[f"{stem}.osm.pbf"]

    # Required fields.
    assert entry["link_schema_version"] == "polygon-document-links-v1", (
        f"Processed manifest must include link_schema_version; got entry={entry}"
    )
    assert entry["link_count"] == 1, f"link_count must equal local table rows; got {entry}"

    # Every pre-existing field must be preserved.
    for key, value in pre_existing.items():
        if key == "processed_at":
            # processed_at is updated by the migration.
            continue
        assert entry.get(key) == value, (
            f"Field {key!r} must be preserved; got {entry.get(key)} != {value}"
        )

    # No secondary manifest is created.
    link_manifest = processed / "manifests" / "link_manifest.json"
    assert not link_manifest.exists(), f"Migration must NOT create {link_manifest}"


# ---------------------------------------------------------------------------
# 3. augmentation manifest: link_schema_version + link_artifact_sha256
# ---------------------------------------------------------------------------


def test_apply_writes_link_schema_version_and_sha256_to_augmentation_manifest(
    tmp_path: Path,
) -> None:
    """The augmentation manifest entry for the migrated stem must
    include ``link_schema_version`` and ``link_artifact_sha256``
    (64-hex SHA-256 of the canonical link file).
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    mod.apply_link_migration(processed)

    aug_manifest = processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    payload = json.loads(aug_manifest.read_text())
    entry = payload[stem]
    # All required augmentation-manifest fields.
    required_fields = {
        "contract_version",
        "core_hashes",
        "paths",
        "counts",
        "completed_at",
        "link_schema_version",
        "link_artifact_sha256",
    }
    missing = required_fields - set(entry.keys())
    assert not missing, (
        f"Augmentation manifest entry is missing required fields: {missing}; got {entry}"
    )

    assert entry.get("link_schema_version") == "polygon-document-links-v1", (
        f"Augmentation manifest must include link_schema_version; got {entry}"
    )
    sha = entry.get("link_artifact_sha256")
    assert isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
        f"link_artifact_sha256 must be 64 lowercase hex; got {sha!r}"
    )

    # The SHA must equal the hash of the canonical link file.
    link_path = processed / "polygon_articles" / f"{stem}.parquet"
    expected = hashlib.sha256(link_path.read_bytes()).hexdigest()
    assert sha == expected, (
        f"link_artifact_sha256 must equal canonical link hash; got {sha} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 4. Augmentation manifest must merge (preserve other stems)
# ---------------------------------------------------------------------------


def test_augmentation_manifest_merges_preserves_other_stems(tmp_path: Path) -> None:
    """Updating the augmentation manifest for stem A must NOT erase
    prior entries for stem B.
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem_a = "alpha-latest"
    _setup_processed(processed, stem_a)

    aug_manifest = processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    aug_manifest.parent.mkdir(parents=True, exist_ok=True)
    pre_existing = {
        "beta-latest": {
            "stale": False,
            "core_hashes": {},
        },
    }
    aug_manifest.write_text(json.dumps(pre_existing, indent=2, sort_keys=True) + "\n")

    mod.apply_link_migration(processed)

    payload = json.loads(aug_manifest.read_text())
    assert "beta-latest" in payload, (
        f"Augmentation manifest must preserve stem beta-latest; got {payload}"
    )
    assert payload["beta-latest"] == pre_existing["beta-latest"]


# ---------------------------------------------------------------------------
# 5. Pending-publication intent is written
# ---------------------------------------------------------------------------


def test_apply_writes_pending_publication_intent(tmp_path: Path) -> None:
    """After migration, the pending-publication manifest must include
    the migrated stem.
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    # No prior pending manifest.
    mod.apply_link_migration(processed)

    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.pipeline import pending_publications as pp

    dr = DataRoot(tmp_path)
    stems = pp.load_pending_publications(dr)
    assert stem in stems, f"Pending publications must include the migrated stem {stem}; got {stems}"


# ---------------------------------------------------------------------------
# 6. Metadata-refresh marker is written
# ---------------------------------------------------------------------------


def test_apply_writes_metadata_refresh_marker(tmp_path: Path) -> None:
    """After migration, the metadata-refresh marker must include the
    migrated stem with the correct ``fingerprint_hashes`` entry
    (the augmentation manifest's link_artifact_sha256).
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    mod.apply_link_migration(processed)

    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.pipeline import pending_publications as pp

    dr = DataRoot(tmp_path)
    marker = pp.load_metadata_refresh_marker(dr)
    assert marker is not None, "Metadata-refresh marker must be written after migration"

    # Every required marker field must be present and well-formed.
    assert set(marker.keys()) == {"stems", "fingerprint_hashes"}, (
        f"Marker keys must be exactly 'stems' and 'fingerprint_hashes'; got {set(marker.keys())}"
    )
    assert marker["stems"] == [stem], (
        f"Marker stems must be the sorted list of migrated stems; got {marker['stems']}"
    )
    assert set(marker["fingerprint_hashes"].keys()) == {stem}, (
        f"Marker fingerprint_hashes keys must equal the stems list; got {marker['fingerprint_hashes'].keys()}"
    )
    sha = marker["fingerprint_hashes"][stem]
    link_path = processed / "polygon_articles" / f"{stem}.parquet"
    expected = hashlib.sha256(link_path.read_bytes()).hexdigest()
    assert sha == expected, (
        f"Marker fingerprint_hashes[{stem}] must equal canonical link hash; got {sha} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 7. Malformed existing processed manifest JSON blocks migration
# ---------------------------------------------------------------------------


def test_malformed_processed_manifest_blocks_migration(tmp_path: Path) -> None:
    """A malformed existing ``processed_pbfs.json`` must block the
    migration -- never silently replace corruption with an empty manifest.
    """
    mod = _import_module()
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    processed_manifest = processed / "manifests" / "processed_pbfs.json"
    processed_manifest.write_text("{ this is not valid JSON")

    # Plan should still succeed (planning is read-only and a separate
    # file from the migration). The apply stage must refuse to
    # silently overwrite a malformed manifest.
    with pytest.raises((ValueError, json.JSONDecodeError)):
        mod.apply_link_migration(processed)
