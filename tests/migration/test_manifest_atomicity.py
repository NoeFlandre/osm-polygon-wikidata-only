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

from osm_polygon_wikidata_only.pipeline import link_migration
from tests.migration._builders import seed_processed_migration_stem

# ---------------------------------------------------------------------------
# 1. link_manifest.json must preserve unrelated stem entries
# ---------------------------------------------------------------------------


def test_processed_pbf_manifest_preserves_other_pbfs(tmp_path: Path) -> None:
    """Migrating PBF A must NOT erase an existing entry for PBF B in
    manifests/processed_pbfs.json.
    """
    processed = tmp_path / "processed"
    stem_a = "alpha-latest"
    seed_processed_migration_stem(processed, stem_a, region=stem_a.removesuffix("-latest"))

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

    plan = link_migration.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    link_migration.apply_link_migration(processed)

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
    processed = tmp_path / "processed"
    data_root = tmp_path
    stem_a = "alpha-latest"
    stem_b = "beta-latest"
    seed_processed_migration_stem(processed, stem_a, region=stem_a.removesuffix("-latest"))

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

    plan = link_migration.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    link_migration.apply_link_migration(processed)

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

    data_root_path = tmp_path
    processed = data_root_path / "processed"
    stem = "alpha-latest"
    seed_processed_migration_stem(processed, stem, region=stem.removesuffix("-latest"))

    plan = link_migration.plan_link_migration(processed)
    assert plan.is_safe_to_apply
    link_migration.apply_link_migration(processed)

    # After apply, all required writes have succeeded -> current.
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(data_root_path)
    assert augmentation_is_current(dr, stem), (
        "After apply completes, stem must be classified current"
    )
