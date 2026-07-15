"""Tests verifying that ``augmentation_is_current`` validates against the
exact source paths named by the augmentation manifest.

The manifest's ``core_hashes`` dictionary names the file whose hash was
recorded at write time. ``augmentation_is_current`` must read the
``expected`` paths directly -- not select a different policy-based path --
so a region whose manifest has been repointed to the canonical document
while the legacy staging file still exists is correctly recognised as
current instead of being mis-classified as stale.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current
from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.steps import (
    sha256_file,
    update_augmentation_manifest,
    wikipedia_source_paths,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    apply_migration,
    plan_migration,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_retirement import (
    prepare_local_retirement,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.sync_planner import (
    RegionSyncState,
    SyncAction,
    plan_sync_states,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "processed"
STEM = "monaco-latest"


def _write_empty_table(path: Path, schema: pa.Schema) -> None:
    """Write an empty Parquet file with the given schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([], schema=schema), path)


def _ensure_migrated_canonical(data_root: DataRoot) -> None:
    """Run the migration so the canonical document is in canonical schema.

    The fixture document is in the legacy schema; ``apply_migration``
    overwrites it with the lossless canonical version.
    """
    apply_migration(plan_migration(data_root.processed, stems={STEM}))


def _ensure_all_sidecars(data_root: DataRoot) -> None:
    """Create empty sidecar parquets so ``augmentation_is_current`` can pass the sidecar check."""
    _write_empty_table(
        data_root.processed / "wikivoyage" / "documents" / f"{STEM}.parquet",
        document_schema(),
    )
    _write_empty_table(
        data_root.processed / "wikivoyage" / "sections" / f"{STEM}.parquet",
        section_schema(),
    )
    _write_empty_table(
        data_root.processed / "wikidata" / "facts" / f"{STEM}.parquet",
        fact_schema(),
    )


def _seed_post_repoint(tmp_path: Path) -> DataRoot:
    """Seed a DataRoot where both files exist and the manifest is repointed to canonical.

    State:
      * ``processed/articles/{stem}.parquet`` exists (legacy staging).
      * ``processed/wikipedia/documents/{stem}.parquet`` exists (canonical).
      * The augmentation manifest's ``core_hashes`` references the canonical
        document and polygon (post-:func:`prepare_local_retirement`).
    """
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for relative in (
        f"articles/{STEM}.parquet",
        f"polygon_articles/{STEM}.parquet",
        f"polygons/{STEM}.parquet",
        f"wikipedia/documents/{STEM}.parquet",
        f"wikipedia/sections/{STEM}.parquet",
    ):
        src = FIXTURES / relative
        dest = data_root.processed / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    _ensure_migrated_canonical(data_root)
    _ensure_all_sidecars(data_root)

    (data_root.processed_manifests / "processed_pbfs.json").write_text(
        json.dumps(
            {f"{STEM}.osm.pbf": {"wikipedia_documents_path": f"wikipedia/documents/{STEM}.parquet"}}
        ),
        encoding="utf-8",
    )

    sources = wikipedia_source_paths(data_root, STEM)
    polygon_path = data_root.processed_polygons / f"{STEM}.parquet"
    core_hashes = {
        str(sources.canonical): sha256_file(sources.canonical),
        str(polygon_path): sha256_file(polygon_path),
    }
    update_augmentation_manifest(
        data_root,
        stem=STEM,
        paths=(
            sources.canonical,
            data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet",
            data_root.processed / "wikivoyage" / "documents" / f"{STEM}.parquet",
            data_root.processed / "wikivoyage" / "sections" / f"{STEM}.parquet",
            data_root.processed / "wikidata" / "facts" / f"{STEM}.parquet",
        ),
        core_hashes=core_hashes,
        counts={
            "wikipedia_documents": 1,
            "wikipedia_sections": 0,
            "wikivoyage_documents": 0,
            "wikivoyage_sections": 0,
            "wikidata_facts": 0,
        },
        completed_at="2026-07-15T00:00:00Z",
    )
    return data_root


# ---------------------------------------------------------------------------
# Core bug: repointed manifest with legacy still present must be current
# ---------------------------------------------------------------------------


def test_both_files_with_canonical_hashes_is_current(tmp_path: Path) -> None:
    """When both files exist and the manifest references canonical, current is True."""
    data_root = _seed_post_repoint(tmp_path)

    assert (data_root.processed_articles / f"{STEM}.parquet").exists()
    assert (data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet").exists()

    assert augmentation_is_current(data_root, STEM)


def test_pending_current_region_plans_publish_not_augment(tmp_path: Path) -> None:
    data_root = _seed_post_repoint(tmp_path)

    entries = load_manifest(data_root.processed_manifests / "processed_pbfs.json")
    core_stems = {name.removesuffix(".osm.pbf") for name in entries}
    current = {stem for stem in core_stems if augmentation_is_current(data_root, stem)}
    states = plan_sync_states(
        pbfs=[data_root.raw / f"{STEM}.osm.pbf"],
        core_stems=core_stems,
        augmentation_stems=current,
        pending_stems={STEM},
    )
    assert len(states) == 1
    state = states[0]
    assert isinstance(state, RegionSyncState)
    assert state.action is SyncAction.PUBLISH


def test_publication_only_does_not_invoke_augmentation_or_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PUBLISH state must route through load_existing_augmentation only."""
    from osm_polygon_wikidata_only.pipeline import sync_runner

    calls: list[str] = []

    def _extract(_pbf_path: Path) -> Any:
        calls.append("extract_pbf")
        return None

    def _process(_extracted: Any) -> Any:
        calls.append("process_extracted_pbf")
        return None

    def _augment(_state: RegionSyncState) -> Any:
        calls.append("augment_region")
        return None

    def _load(_state: RegionSyncState) -> Any:
        calls.append("load_existing_augmentation")
        return None

    state = RegionSyncState(
        stem=STEM,
        pbf_path=Path("/tmp/non-existent.pbf"),
        action=SyncAction.PUBLISH,
    )
    sync_runner.run_sync(
        [state],
        extract_pbf=_extract,
        process_extracted_pbf=_process,
        augment_region=_augment,
        load_existing_augmentation=_load,
        submit_upload=None,
        build_upload_files=None,
        close_uploads=None,
    )

    assert calls == ["load_existing_augmentation"]


# ---------------------------------------------------------------------------
# Pre-repoint legacy-hashed manifest still validates against legacy
# ---------------------------------------------------------------------------


def test_legacy_hashed_manifest_validates_against_legacy_file(tmp_path: Path) -> None:
    """Before repointing, a manifest that references the legacy file must validate."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for relative in (
        f"articles/{STEM}.parquet",
        f"polygons/{STEM}.parquet",
        f"wikipedia/documents/{STEM}.parquet",
        f"wikipedia/sections/{STEM}.parquet",
    ):
        src = FIXTURES / relative
        dest = data_root.processed / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    _ensure_all_sidecars(data_root)

    legacy = data_root.processed_articles / f"{STEM}.parquet"
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    update_augmentation_manifest(
        data_root,
        stem=STEM,
        paths=(
            data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet",
            data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet",
            data_root.processed / "wikivoyage" / "documents" / f"{STEM}.parquet",
            data_root.processed / "wikivoyage" / "sections" / f"{STEM}.parquet",
            data_root.processed / "wikidata" / "facts" / f"{STEM}.parquet",
        ),
        core_hashes={
            str(legacy): sha256_file(legacy),
            str(polygon): sha256_file(polygon),
        },
        counts={
            "wikipedia_documents": 1,
            "wikipedia_sections": 0,
            "wikivoyage_documents": 0,
            "wikivoyage_sections": 0,
            "wikidata_facts": 0,
        },
        completed_at="2026-07-15T00:00:00Z",
    )

    assert augmentation_is_current(data_root, STEM)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_mismatched_canonical_hash_returns_false(tmp_path: Path) -> None:
    """When the recorded canonical hash does not match, return False."""
    data_root = _seed_post_repoint(tmp_path)

    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_key = next(iter(payload[STEM]["core_hashes"]))
    payload[STEM]["core_hashes"][canonical_key] = "deadbeef" * 8
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not augmentation_is_current(data_root, STEM)


def test_missing_source_returns_false(tmp_path: Path) -> None:
    """If a manifest-named source file is missing, return False."""
    data_root = _seed_post_repoint(tmp_path)
    (data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet").unlink()

    assert not augmentation_is_current(data_root, STEM)


# ---------------------------------------------------------------------------
# Post-retirement: canonical-only state remains current
# ---------------------------------------------------------------------------


def test_canonical_only_after_retirement_is_current(tmp_path: Path) -> None:
    """After legacy deletion, canonical-only state must remain current."""
    data_root = _seed_post_repoint(tmp_path)

    prepare_local_retirement(data_root, STEM)
    (data_root.processed_articles / f"{STEM}.parquet").unlink()

    assert augmentation_is_current(data_root, STEM)
