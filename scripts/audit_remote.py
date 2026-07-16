#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import resolve_data_root
from osm_polygon_wikidata_only.hf.reconciliation import ReconciliationPlanner
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit remote versus local canonical dataset files."
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Local dataset root")
    parser.add_argument(
        "--repo-id", default="NoeFlandre/osm-polygon-wikidata-only", help="Hugging Face repo id"
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face auth token")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    try:
        data_root = resolve_data_root(args.data_root, repo_root=repo_root)
    except Exception as e:
        print(f"Error resolving data root: {e}")
        return 1

    print(f"Fetching remote inventory for {args.repo_id}...")
    try:
        inventory = RemoteInventory.fetch(repo_id=args.repo_id, token=args.hf_token)
    except Exception as e:
        print(f"Failed to fetch remote inventory: {e}")
        return 1

    # Audit script checks the entire local dataset (all local stems)
    local_stems = {p.stem for p in data_root.processed_polygons.glob("*.parquet")}
    print(f"Local finalized stems found: {len(local_stems)}")

    print(f"Checking augmentation status for {len(local_stems)} stems...")
    from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current

    augmentation_current = {stem: augmentation_is_current(data_root, stem) for stem in local_stems}

    planner = ReconciliationPlanner(
        data_root,
        inventory,
        stems=local_stems,
        augmentation_current=augmentation_current,
    )
    try:
        plan = planner.plan()
    except Exception as e:
        print(f"Failed to compute reconciliation plan: {e}")
        return 1

    print("\n=== AUDIT RESULTS ===")

    missing_polygons = [stem for stem, corp in plan.missing if corp == "polygons"]
    print(f"\nMissing remote polygons ({len(missing_polygons)}):")
    for stem in sorted(missing_polygons):
        print(f"  - {stem}.parquet")

    missing_links = [stem for stem, corp in plan.missing if corp == "polygon_articles"]
    print(f"\nMissing remote polygon links ({len(missing_links)}):")
    for stem in sorted(missing_links):
        print(f"  - {stem}.parquet")

    # Audit the five augmentation corpora
    augmentation_corpora = [
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    ]
    print("\nMissing remote files in the five augmentation corpora:")
    for corpus in augmentation_corpora:
        missing_corp_stems = [stem for stem, corp in plan.missing if corp == corpus]
        print(f"  {corpus}: {len(missing_corp_stems)} missing")
        for stem in sorted(missing_corp_stems):
            print(f"    - {stem}.parquet")

    if plan.unexpected:
        print(f"\nUnexpected remote canonical files ({len(plan.unexpected)}):")
        for f in sorted(plan.unexpected):
            print(f"  - {f}")

    if plan.repository_refresh:
        print(f"\nMissing repository-level metadata assets ({len(plan.repository_refresh)}):")
        for f in sorted(plan.repository_refresh):
            print(f"  - {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
