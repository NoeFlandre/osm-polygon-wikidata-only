"""Characterise how the V2 runner chooses each region's durable action.

These tests use real canonical artifacts on disk and public collaborators, so
they pin the extract / reconcile / publish / skip decisions without patching
private seams.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2 import runner as v2_runner
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION

REGION = "region-latest"


def _artifact_paths(stem: str) -> tuple[str, ...]:
    return (
        f"polygons/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"polygon_document_links/{stem}.parquet",
    )


ARTIFACT_PATHS = _artifact_paths(REGION)


class _RecordingHashCache:
    """Injected digest source that records every artifact it was asked to hash."""

    def __init__(self, digest_of: Callable[[Path], str]) -> None:
        self.digest_of = digest_of
        self.requested: list[Path] = []

    def digest(self, path: Path) -> str:
        self.requested.append(path)
        return self.digest_of(path)


def _root(tmp_path: Path) -> DataRoot:
    root = DataRoot(tmp_path)
    root.ensure()
    return root


def _write_current_region(root: DataRoot, stem: str = REGION) -> dict[str, Any]:
    """Write the four canonical artifacts and return a manifest entry matching them."""
    file_hashes: dict[str, str] = {}
    for relative in _artifact_paths(stem):
        path = root.processed_v2 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact {relative}\n".encode())
        file_hashes[relative] = sha256_file(path)
    return {
        "contract_version": V2_CONTRACT_VERSION,
        "polygons_path": f"polygons/{stem}.parquet",
        "documents_path": f"wikipedia/documents/{stem}.parquet",
        "sections_path": f"wikipedia/sections/{stem}.parquet",
        "links_path": f"polygon_document_links/{stem}.parquet",
        "file_hashes": file_hashes,
    }


def _plan(
    root: DataRoot,
    stems: tuple[str, ...] = (REGION,),
    *,
    manifest: dict[str, dict[str, Any]],
    settings: Settings | None = None,
    push: bool = False,
    remote_inventory: RemoteInventory | None = None,
    hash_cache: Any = None,
) -> tuple[v2_runner._RegionPlan, ...]:
    pbfs = [root.raw / f"{stem}.osm.pbf" for stem in stems]
    return v2_runner._plan_regions(
        pbfs,
        data_root=root,
        settings=Settings(skip_existing=True) if settings is None else settings,
        manifest=manifest,
        push=push,
        remote_inventory=remote_inventory,
        hash_cache=hash_cache,
    )


def _actions(plans: tuple[v2_runner._RegionPlan, ...]) -> list[tuple[str, str]]:
    return [(plan.stem, plan.action) for plan in plans]


def test_region_without_manifest_entry_is_extracted(tmp_path: Path) -> None:
    root = _root(tmp_path)

    plans = _plan(root, manifest={})

    assert _actions(plans) == [(REGION, "extract")]


def test_skip_existing_disabled_reextracts_a_current_region(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True

    plans = _plan(root, manifest={REGION: entry}, settings=Settings(skip_existing=False))

    assert _actions(plans) == [(REGION, "extract")]


def test_force_reextracts_a_current_region(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True

    plans = _plan(root, manifest={REGION: entry}, settings=Settings(skip_existing=True, force=True))

    assert _actions(plans) == [(REGION, "extract")]


def test_tampered_artifact_requires_extraction(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True
    (root.processed_v2 / ARTIFACT_PATHS[2]).write_bytes(b"edited after manifest\n")

    plans = _plan(root, manifest={REGION: entry})

    assert _actions(plans) == [(REGION, "extract")]


def test_current_reconciled_region_is_skipped_without_push(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True

    plans = _plan(root, manifest={REGION: entry}, push=False)

    assert _actions(plans) == [(REGION, "skip")]


def test_unreconciled_index_selects_reconcile_before_any_remote_check(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = False

    plans = _plan(
        root,
        manifest={REGION: entry},
        push=True,
        remote_inventory=RemoteInventory(set()),
    )

    assert _actions(plans) == [(REGION, "reconcile")]


def test_missing_reconciliation_flag_counts_as_reconciled(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)

    plans = _plan(root, manifest={REGION: entry}, push=False)

    assert _actions(plans) == [(REGION, "skip")]


def test_push_without_remote_inventory_publishes_a_current_region(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True

    plans = _plan(root, manifest={REGION: entry}, push=True, remote_inventory=None)

    assert _actions(plans) == [(REGION, "publish")]


def test_push_with_partial_remote_inventory_publishes_a_current_region(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True

    plans = _plan(
        root,
        manifest={REGION: entry},
        push=True,
        remote_inventory=RemoteInventory({ARTIFACT_PATHS[0]}),
    )

    assert _actions(plans) == [(REGION, "publish")]


def test_push_with_complete_remote_inventory_skips_a_current_region(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True

    plans = _plan(
        root,
        manifest={REGION: entry},
        push=True,
        remote_inventory=RemoteInventory(set(ARTIFACT_PATHS)),
    )

    assert _actions(plans) == [(REGION, "skip")]


def test_remote_inventory_is_ignored_when_not_pushing(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True

    plans = _plan(
        root,
        manifest={REGION: entry},
        push=False,
        remote_inventory=RemoteInventory(set()),
    )

    assert _actions(plans) == [(REGION, "skip")]


def test_injected_hash_cache_is_used_for_every_canonical_artifact(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True
    cache = _RecordingHashCache(sha256_file)

    plans = _plan(root, manifest={REGION: entry}, hash_cache=cache)

    assert _actions(plans) == [(REGION, "skip")]
    assert sorted(path.relative_to(root.processed_v2).as_posix() for path in cache.requested) == (
        sorted(ARTIFACT_PATHS)
    )


def test_injected_hash_cache_mismatch_requires_extraction(tmp_path: Path) -> None:
    root = _root(tmp_path)
    entry = _write_current_region(root)
    entry["v1_index_reconciled"] = True
    cache = _RecordingHashCache(lambda _path: "0" * 64)

    plans = _plan(root, manifest={REGION: entry}, hash_cache=cache)

    assert _actions(plans) == [(REGION, "extract")]


def test_plans_keep_input_order_and_carry_the_source_path(tmp_path: Path) -> None:
    root = _root(tmp_path)
    current = _write_current_region(root, stem="region-b")
    current["v1_index_reconciled"] = True
    unreconciled = _write_current_region(root, stem="region-c")
    unreconciled["v1_index_reconciled"] = False
    manifest = {"region-b": current, "region-c": unreconciled}

    plans = _plan(root, ("region-a", "region-b", "region-c"), manifest=manifest)

    assert _actions(plans) == [
        ("region-a", "extract"),
        ("region-b", "skip"),
        ("region-c", "reconcile"),
    ]
    assert [plan.pbf for plan in plans] == [
        root.raw / "region-a.osm.pbf",
        root.raw / "region-b.osm.pbf",
        root.raw / "region-c.osm.pbf",
    ]


def test_no_pbfs_plans_nothing(tmp_path: Path) -> None:
    root = _root(tmp_path)

    assert _plan(root, (), manifest={}) == ()
