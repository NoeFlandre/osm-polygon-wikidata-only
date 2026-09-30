"""CLI composition root for ``sync-dir``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from osm_polygon_wikidata_only.cli import sync_publication, sync_runtime
from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.pipeline import sync_planning

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


def execute(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
    build_upload_files: Callable[..., list[PublicationOp]] | None = None,
    _remote_inventory: RemoteInventory | None = None,
    _hub: HfHub | None = None,
) -> int:
    """Run the ``sync-dir`` CLI command.

    Prepares the unified sync plan (local states plus remote
    reconciliation), builds the Wikimedia runtime, augmentation client
    and optional upload queue, enqueues containment retirement, logs the
    plan counts, and then delegates execution to
    :class:`cli.sync_application.SyncApplication`, which decides whether
    publication runs and calls :func:`pipeline.sync_runner.run_sync`.

    ``build_upload_files`` optionally replaces the default region
    publication builder when ``--push`` is enabled; production passes
    ``None``.
    """
    push_enabled = bool(getattr(args, "push", False))
    dry_run = bool(getattr(args, "dry_run", False))
    prepared = sync_planning.prepare_sync_plan(
        args,
        data_root=data_root,
        settings=settings,
        push_enabled=push_enabled,
        dry_run=dry_run,
        remote_inventory=_remote_inventory,
        hub=_hub,
    )
    runtime = build_wikimedia_runtime(settings, data_root=data_root)
    augmentation_client = sync_runtime.build_augmentation_client(data_root, runtime)
    upload_queue = sync_publication.build_upload_queue(
        push=push_enabled,
        dry_run=dry_run,
        settings=settings,
        data_root=data_root,
        num_threads=getattr(args, "upload_threads", 2),
        _hub=_hub,
    )
    containment_enqueued = sync_publication.enqueue_containment_retirement(
        data_root,
        settings,
        prepared.remote_state.containment_publications,
        upload_queue,
        push_enabled=push_enabled,
    )
    sync_planning.log_sync_plan(prepared.states)
    return sync_runtime.run_sync_application(
        args,
        data_root=data_root,
        settings=settings,
        runtime=runtime,
        augmentation_client=augmentation_client,
        prepared=prepared,
        push_enabled=push_enabled,
        dry_run=dry_run,
        upload_queue=upload_queue,
        containment_enqueued=containment_enqueued,
        publish_builder=build_upload_files,
    )


__all__ = ["execute"]
