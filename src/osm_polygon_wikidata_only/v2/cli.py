"""CLI boundary for the explicitly selected V2 sync workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID
from osm_polygon_wikidata_only.v2.runner import run_v2_sync


def execute_v2(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run V2 with the normal shared Wikimedia runtime and uploader."""
    runtime = build_wikimedia_runtime(settings, data_root=data_root)
    hub = StubHfHub() if args.dry_run else None

    def upload(ops: list[PublicationOp], message: str) -> None:
        upload_files(
            V2_REPO_ID,
            ops=ops,
            hub=hub,
            token=settings.hf_token,
            commit_message=args.commit_message or message,
            num_threads=args.upload_threads,
        )

    return run_v2_sync(
        Path(args.input),
        data_root=data_root,
        settings=settings,
        wikipedia_client=runtime.wikipedia,
        push=bool(args.push),
        upload=upload if args.push else None,
    )


__all__ = ["execute_v2"]
