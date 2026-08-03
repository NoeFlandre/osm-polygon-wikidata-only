"""CLI boundary for the explicitly selected V2 sync workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.uploader import UploadError, upload_files
from osm_polygon_wikidata_only.v2.runner import run_v2_sync

LOGGER = logging.getLogger(__name__)


def execute_v2(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run V2 with the normal shared Wikimedia runtime and uploader."""
    runtime = build_wikimedia_runtime(settings, data_root=data_root)
    # ``commands.main`` supplies the V2 default, while an explicit
    # ``--repo-id`` remains an intentional operator override.
    repo_id = settings.repo_id
    hub = StubHfHub() if args.dry_run else None
    remote_inventory = None
    if args.push:
        try:
            remote_inventory = RemoteInventory.fetch(
                repo_id,
                hub=hub,
                token=settings.hf_token,
            )
        except UploadError as error:
            LOGGER.info("V2 Hub repository is new or not listable yet: %s", error)

    def upload(ops: list[PublicationOp], message: str) -> None:
        upload_files(
            repo_id,
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
        remote_inventory=remote_inventory,
    )


__all__ = ["execute_v2"]
