"""CLI boundary for the explicitly selected V2 sync workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.cli.dependencies import WikimediaRuntime, build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.uploader import UploadError, upload_files
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.v2.card import V2CardStats, write_v2_card
from osm_polygon_wikidata_only.v2.config import (
    V2_CACHE_CONTRACT_VERSION,
    V2_TRACKIO_RUN_NAME,
)
from osm_polygon_wikidata_only.v2.runner import run_v2_sync
from osm_polygon_wikidata_only.v2.sat import DEFAULT_SAT_MODEL_REVISION, SaT3lSegmenter
from osm_polygon_wikidata_only.v2.sentence_runner import run_v2_sentence_split

LOGGER = logging.getLogger(__name__)


def execute_v2(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run V2 with the normal shared Wikimedia runtime and uploader."""
    runtime = build_wikimedia_runtime(settings, data_root=data_root)
    section_client = _build_section_client(settings, data_root, runtime)
    # ``commands.main`` supplies the V2 default, while an explicit
    # ``--repo-id`` remains an intentional operator override.
    repo_id = settings.repo_id
    hub = StubHfHub() if args.dry_run else None
    remote_inventory = _fetch_inventory(args, repo_id, settings, hub)
    upload = _build_uploader(args, repo_id, settings, hub)
    trackio_publish = _build_trackio_publisher(args, data_root)

    return run_v2_sync(
        Path(args.input),
        data_root=data_root,
        settings=settings,
        wikipedia_client=runtime.wikipedia,
        section_client=section_client,
        section_workers=settings.enrichment_site_workers,
        push=bool(args.push),
        upload=upload if args.push else None,
        remote_inventory=remote_inventory,
        trackio_publish=trackio_publish,
    )


def execute_v2_sentence_split(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Materialize V2 sentence sidecars and optionally publish them."""
    segmenter = SaT3lSegmenter(
        cache_dir=data_root.hf_cache / "models" / "sat-3l-sm",
        revision=DEFAULT_SAT_MODEL_REVISION,
        inference_batch_size=getattr(args, "inference_batch_size", 16),
    )
    result = run_v2_sentence_split(
        data_root,
        segmenter=segmenter,
        batch_size=args.batch_size,
    )
    write_v2_card(data_root.processed_v2)
    if args.push:
        from osm_polygon_wikidata_only.v2.publication import sentence_publication_ops

        stems = tuple(sorted({summary.stem for summary in result.regions}))
        operations = sentence_publication_ops(data_root.processed_v2, stems)
        upload_files(
            settings.repo_id,
            ops=operations,
            hub=StubHfHub() if args.dry_run else None,
            token=settings.hf_token,
            commit_message=args.commit_message or "Add V2 sentence sidecars",
            num_threads=args.upload_threads,
        )
    LOGGER.info("Sentence splitting complete for %d region/project table(s)", len(result.regions))
    return 0


def _build_section_client(
    settings: Settings, data_root: DataRoot, runtime: WikimediaRuntime
) -> AugmentationWikimediaClient:
    return AugmentationWikimediaClient(
        settings,
        JsonFileCache(data_root.v2_cache / "sections", contract_version=V2_CACHE_CONTRACT_VERSION),
        scheduler=runtime.scheduler,
        session=runtime.session,
    )


def _fetch_inventory(
    args: argparse.Namespace,
    repo_id: str,
    settings: Settings,
    hub: StubHfHub | None,
) -> RemoteInventory | None:
    if not args.push:
        return None
    try:
        return RemoteInventory.fetch(repo_id, hub=hub, token=settings.hf_token)
    except UploadError as error:
        LOGGER.info("V2 Hub repository is new or not listable yet: %s", error)
        return None


def _build_uploader(
    args: argparse.Namespace,
    repo_id: str,
    settings: Settings,
    hub: StubHfHub | None,
):
    if not args.push:
        return None

    def upload(ops: list[PublicationOp], message: str) -> None:
        upload_files(
            repo_id,
            ops=ops,
            hub=hub,
            token=settings.hf_token,
            commit_message=args.commit_message or message,
            num_threads=args.upload_threads,
        )

    return upload


def _build_trackio_publisher(args: argparse.Namespace, data_root: DataRoot):
    if not args.push or args.dry_run:
        return None

    def publish_snapshot(stats: V2CardStats) -> None:
        from osm_polygon_wikidata_only.hf.v2_trackio_snapshot import publish_v2_trackio_snapshot

        publish_v2_trackio_snapshot(
            output_dir=data_root.cache / "trackio" / V2_TRACKIO_RUN_NAME,
            stats=stats,
        )

    return publish_snapshot


__all__ = ["execute_v2", "execute_v2_sentence_split"]
