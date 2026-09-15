"""CLI entry point.

Two commands:

- ``process-pbf <path>``: extract + enrich one PBF file.
- ``process-dir <path>``: process every ``*.pbf`` under a directory.

Shared options: ``--push``, ``--repo-id``, ``--data-root``,
``--skip-existing``, ``--force``, ``--languages``, ``--all-languages``,
``--no-full-text``, ``--max-articles-per-qid``, ``--limit``,
``--commit-message``, ``--log-level``.

This module owns argparse, runtime construction, and HF
authentication. Publication assembly lives in
:mod:`osm_polygon_wikidata_only.hf.publication`; CLI code here
submits the file lists it returns through the upload queue or the
direct ``upload_files`` helper.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    AugmentationResult,
    augment_region,
    augmentation_is_current,
    completed_region_stems,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID, Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import (
    StubHfHub,
    UploadError,
    resolve_hf_token,
    upload_files,
    verify_hf_token,
    verify_repo_authorization,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.io.run_lock import RunLockError, exclusive_run_lock
from osm_polygon_wikidata_only.pipeline.orchestrator import orchestrate
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    clear_metadata_refresh_marker,
    load_metadata_refresh_marker,
    set_metadata_refresh_marker,
)
from osm_polygon_wikidata_only.pipeline.processor import (
    ProcessResult,
)
from osm_polygon_wikidata_only.utils.logging import configure_logging
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID

from .dependencies import build_clients as _build_clients
from .dependencies import resolve_cli_data_root as _resolve_data_root
from .parser import build_parser
from .parser import build_settings as _build_settings

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


def _enqueue_core_upload(
    upload_queue: BackgroundUploadQueue,
    *,
    data_root: DataRoot,
    repo_id: str,
    commit_message: str,
    result: ProcessResult,
    defer_metadata_assets: bool = False,
) -> None:
    """Submit one core publication via :mod:`hf.publication`.

    Thin CLI adapter: builds the ordered file list through
    :func:`hf.publication.assemble_core_upload` and submits it
    once through the upload queue. No assembly logic lives here.
    The legacy ``Could not fetch world land data; map will omit
    continents`` WARNING is preserved on the CLI logger.
    """
    from osm_polygon_wikidata_only.hf.publication import assemble_core_upload

    ops = assemble_core_upload(
        data_root=data_root,
        repo_id=repo_id,
        core=result,
        world_land_warning=LOGGER.warning,
        defer_metadata_assets=defer_metadata_assets,
    )
    upload_queue.submit(ops, commit_message)


def _drain_uploads(
    upload_queue: BackgroundUploadQueue,
    *,
    data_root: DataRoot,
    repo_id: str,
    deferring: bool,
    published_regions: int,
) -> list[str]:
    """Drain the regional queue, then refresh deferred metadata.

    The queue is closed before anything else, so a region whose upload
    exhausted its retries never gets a manifest, card, map, or statistics
    report describing it, and nothing after the drain -- an assembly
    failure, or a malformed refresh marker -- can leave the queue's
    non-daemon worker waiting for a sentinel that never arrives. Reading
    the marker therefore happens here rather than in the caller's
    argument list, while its validation error still propagates.

    The refresh runs synchronously on the drained queue. Its marker is
    retired only once it succeeds, so a failure leaves the intent on
    disk for the next run to repair.
    """
    failures = upload_queue.close_and_wait()
    if failures or not deferring:
        return failures
    if not _metadata_refresh_requested(data_root, published_regions=published_regions):
        return failures
    return _refresh_repository_metadata(upload_queue, data_root=data_root, repo_id=repo_id)


def _refresh_repository_metadata(
    upload_queue: BackgroundUploadQueue,
    *,
    data_root: DataRoot,
    repo_id: str,
) -> list[str]:
    """Publish the deferred repository-wide assets and retire their marker."""
    try:
        _upload_metadata_refresh(upload_queue, data_root=data_root, repo_id=repo_id)
    # ``except Exception`` retained: assembling the refresh runs the
    # fail-closed statistics scan and the map renderers, which raise a
    # broad, unstable set of types. The documented behavior is to report
    # the refresh as a failed upload and keep its marker for the next run,
    # never to escape the caller's ``finally`` block.
    except Exception as error:
        LOGGER.error("Repository metadata refresh failed: %s", error)
        return [f"Refresh repository metadata and maps: {error}"]
    clear_metadata_refresh_marker(data_root)
    return []


def _refresh_allowed(deferring: bool, processing_completed: bool) -> bool:
    """Only a run that finished processing may publish deferred assets."""
    return deferring and processing_completed


def _metadata_refresh_requested(data_root: DataRoot, *, published_regions: int) -> bool:
    """A deferred run refreshes after publishing, or when a marker survives.

    A resumed run can publish nothing -- every PBF is already processed
    and skipped -- while an earlier run's marker records that the
    repository-wide assets never made it. Reading the marker is what
    makes that run repair them instead of leaving them stale.
    """
    if published_regions:
        return True
    return load_metadata_refresh_marker(data_root) is not None


def _record_deferred_metadata(
    data_root: DataRoot,
    stems: dict[str, str],
    result: ProcessResult,
) -> None:
    """Persist the intent to refresh the repository-wide assets.

    A directory run defers those assets, so they stay stale until the
    final refresh succeeds. The marker survives a crash or a failed
    refresh and is what a later sync run repairs from; without it every
    remote path would look present and nothing would be scheduled.
    """
    stem = Path(str(result.manifest_entry["source_pbf"])).stem.removesuffix(".osm")
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    if not polygons_path.is_file():
        return
    stems[stem] = sha256_file(polygons_path)
    set_metadata_refresh_marker(data_root, sorted(stems), dict(stems))


def _upload_metadata_refresh(
    upload_queue: BackgroundUploadQueue,
    *,
    data_root: DataRoot,
    repo_id: str,
) -> None:
    """Publish the repository-wide assets once a directory run has drained.

    A directory run defers the maps, the statistics report, the hero, and
    the README so the full-dataset scan behind them happens once, not
    once per processed PBF.
    """
    from osm_polygon_wikidata_only.hf.publication import assemble_metadata_only_upload

    LOGGER.info("Refreshing repository metadata after the processed directory")
    upload_queue.upload_synchronously(
        assemble_metadata_only_upload(
            data_root=data_root,
            repo_id=repo_id,
            world_land_warning=LOGGER.warning,
        ),
        "Refresh repository metadata and maps",
    )


def _processing_inputs(command: str, input_path: Path) -> list[Path]:
    """Return the processing input list for a parsed process command."""
    del command
    return [input_path]


def _augmentation_stems(
    command: str,
    stem: str | None,
    data_root: DataRoot,
) -> list[str]:
    """Select the region stems addressed by an augmentation command."""
    if command == "augment-region":
        assert stem is not None
        return [stem]
    return completed_region_stems(data_root)


def _prepare_runtime(
    args: argparse.Namespace,
) -> tuple[DataRoot, Settings]:
    """Configure logging and construct the immutable runtime inputs."""
    configure_logging(level=getattr(logging, args.log_level))
    data_root = _resolve_data_root(args)
    data_root.ensure()
    settings = _build_settings(args)
    if getattr(args, "dataset_version", "v1") == "v2" and settings.repo_id == DEFAULT_REPO_ID:
        settings = replace(settings, repo_id=V2_REPO_ID)
    return data_root, settings


def _require_push_token(parser: argparse.ArgumentParser, settings: Settings) -> None:
    """Fail with an actionable parser error when a supplied token is invalid."""
    resolved = resolve_hf_token(settings.hf_token)
    if resolved:
        return
    env_token = os.environ.get("HF_TOKEN")
    explicit = bool(settings.hf_token)
    if env_token or explicit:
        source = "--hf-token" if explicit else "HF_TOKEN"
        parser.error(
            f"--push: {source} is set but Hugging Face rejected it as invalid. "
            "Generate a fresh write token at https://huggingface.co/settings/tokens "
            "and replace the current value."
        )
    parser.error(
        "--push requires a Hugging Face write token: pass --hf-token, "
        "set HF_TOKEN, or run `huggingface-cli login`."
    )


def _verify_push_access(parser: argparse.ArgumentParser, settings: Settings) -> None:
    """Authenticate and authorize a publishing run."""
    LOGGER.info("Connecting to Hugging Face using bounded IPv4 transport (connect timeout: 10s)")
    try:
        username = verify_hf_token(settings.hf_token)
    except UploadError as error:
        parser.error(str(error))
    try:
        verify_repo_authorization(settings.hf_token, settings.repo_id)
    except UploadError as error:
        parser.error(str(error))
    LOGGER.info("Authenticated to Hugging Face as %s (target: %s)", username, settings.repo_id)


def _authenticate_for_push(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    settings: Settings,
) -> None:
    """Validate Hugging Face credentials when a real push was requested."""
    if not args.push or args.dry_run:
        return
    _require_push_token(parser, settings)
    _verify_push_access(parser, settings)


def _run_v2_sync(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the v2 sync while holding the shared lock."""
    from osm_polygon_wikidata_only.v2.cli import execute_v2

    try:
        with exclusive_run_lock(data_root.cache / "sync.lock"):
            return execute_v2(args, data_root=data_root, settings=settings)
    except RunLockError as error:
        parser.error(str(error))


def _run_v2_sentence_split(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run V2 sentence sidecars while holding their dedicated lock."""
    from osm_polygon_wikidata_only.v2.cli import execute_v2_sentence_split

    try:
        with exclusive_run_lock(data_root.cache / "sentence-splitting.lock"):
            return execute_v2_sentence_split(args, data_root=data_root, settings=settings)
    except RunLockError as error:
        parser.error(str(error))


def _run_v1_sync(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the v1 sync while holding the shared lock."""
    from .run_sync import execute as cli_run_sync

    try:
        with exclusive_run_lock(data_root.cache / "sync.lock"):
            return cli_run_sync(
                args,
                data_root=data_root,
                settings=settings,
                # No build_upload_files override: the CLI shell owns the
                # production region-publication builder via
                # hf.publication.assemble_region_upload.
                build_upload_files=None,
            )
    except RunLockError as error:
        parser.error(str(error))


def _run_sync_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Dispatch the selected sync dataset version."""
    if getattr(args, "dataset_version", "v1") == "v2":
        return _run_v2_sync(parser, args, data_root=data_root, settings=settings)
    return _run_v1_sync(parser, args, data_root=data_root, settings=settings)


def _load_augmentation_result(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    stem: str,
    augmentation_client: AugmentationWikimediaClient,
) -> AugmentationResult | None:
    """Load a current result or perform one region augmentation."""
    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        load_existing_augmentation_result,
    )
    from osm_polygon_wikidata_only.pipeline.link_migration import (
        apply_link_migration,
        plan_link_migration,
    )

    if args.skip_existing and augmentation_is_current(data_root, stem):
        migration = plan_link_migration(data_root.processed, stems={stem})
        if not migration.stems or migration.stems[0].classification.value == "canonical":
            LOGGER.info("Skipping augmentation for %s (already current)", stem)
            return None
        apply_link_migration(data_root.processed, stems={stem})
        LOGGER.info(
            "Migrated %s to unified polygon-document links without Wikimedia requests",
            stem,
        )
    else:
        augment_region(data_root, stem, augmentation_client)
        apply_link_migration(data_root.processed, stems={stem})
    return load_existing_augmentation_result(data_root, stem)


def _publish_augmentation(
    args: argparse.Namespace,
    settings: Settings,
    *,
    data_root: DataRoot,
    stem: str,
    result: AugmentationResult,
) -> None:
    """Publish one augmentation result when requested."""
    if not args.push:
        return
    from osm_polygon_wikidata_only.hf.publication import assemble_augmentation_upload

    hub = StubHfHub() if args.dry_run else None
    ops = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=settings.repo_id,
        augmentation=result,
    )
    upload_files(
        settings.repo_id,
        ops=ops,
        hub=hub,
        token=settings.hf_token,
        commit_message=args.commit_message or f"Add text augmentation for {stem}",
        num_threads=args.upload_threads,
    )


def _run_augmentation_command(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run augmentation for one region or all completed regions."""
    augmentation_client = AugmentationWikimediaClient(
        settings,
        JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
    )
    stems = _augmentation_stems(args.command, getattr(args, "stem", None), data_root)
    augmentation_results: list[AugmentationResult] = []
    for stem in stems:
        result = _load_augmentation_result(
            args,
            data_root=data_root,
            stem=stem,
            augmentation_client=augmentation_client,
        )
        if result is None:
            continue
        augmentation_results.append(result)
        LOGGER.info("Augmented %s: %s", stem, result.counts)
        _publish_augmentation(args, settings, data_root=data_root, stem=stem, result=result)
    LOGGER.info("Done. %d region augmentation(s).", len(augmentation_results))
    return 0


def _build_upload_queue(
    args: argparse.Namespace,
    settings: Settings,
    *,
    data_root: DataRoot,
) -> BackgroundUploadQueue | None:
    """Build and resume the bounded background upload queue when pushing."""
    if not args.push:
        return None
    hub = StubHfHub() if args.dry_run else None

    def upload_job(ops: list[PublicationOp], message: str) -> None:
        upload_files(
            settings.repo_id,
            ops=ops,
            hub=hub,
            token=settings.hf_token,
            commit_message=message,
            num_threads=args.upload_threads,
        )

    upload_queue = BackgroundUploadQueue(
        upload=upload_job,
        max_pending=2,
        state_dir=data_root.cache / "upload_jobs",
    )
    resumed = upload_queue.resume_pending()
    if resumed:
        LOGGER.info("Resumed %d pending background upload(s)", resumed)
    return upload_queue


def _log_process_results(results: Sequence[ProcessResult]) -> None:
    """Log the stable summary emitted after core processing completes."""
    LOGGER.info(
        "Done. %d PBF(s), %d polygons processed.",
        len(results),
        sum(r.polygon_count for r in results),
    )
    for result in results:
        LOGGER.info(
            "Stage timings for %s: %s",
            result.manifest_entry["source_pbf"],
            ", ".join(f"{name}={seconds:.3f}s" for name, seconds in result.stage_timings_s.items()),
        )


def _run_processing_command(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the core PBF processor and drain any requested uploads."""
    wd, wiki, cache = _build_clients(settings, data_root=data_root)
    inputs = _processing_inputs(args.command, args.input)
    upload_queue = _build_upload_queue(args, settings, data_root=data_root)
    # A directory run publishes one region per PBF; the repository-wide
    # assets are produced once after the queue drains instead of after
    # each region.
    defer_metadata_assets = args.command == "process-dir"
    published_regions = 0
    deferred_stems: dict[str, str] = {}

    def enqueue_upload(result: ProcessResult) -> None:
        nonlocal published_regions
        if upload_queue is None:
            return
        # Recorded before the job is submitted: a kill between the two
        # would otherwise leave an uploadable region with no record that
        # the repository-wide assets still owe it a refresh.
        if defer_metadata_assets:
            _record_deferred_metadata(data_root, deferred_stems, result)
        _enqueue_core_upload(
            upload_queue,
            data_root=data_root,
            repo_id=settings.repo_id,
            commit_message=args.commit_message
            or f"Update PBF {result.manifest_entry['source_pbf']}",
            result=result,
            defer_metadata_assets=defer_metadata_assets,
        )
        published_regions += 1

    upload_failures: list[str] = []
    processing_completed = False
    try:
        results = orchestrate(
            inputs,
            data_root=data_root,
            settings=settings,
            wikidata_client=wd,
            wikipedia_client=wiki,
            cache=cache,
            on_complete=enqueue_upload,
        )
        processing_completed = True
    finally:
        if upload_queue is not None:
            upload_failures = _drain_uploads(
                upload_queue,
                data_root=data_root,
                repo_id=settings.repo_id,
                # An aborted run still drains the queue, but never
                # publishes repository-wide assets: a region whose
                # assembly or submission raised has local artifacts the
                # remote does not have. The marker survives for the next
                # run to repair.
                deferring=_refresh_allowed(defer_metadata_assets, processing_completed),
                published_regions=published_regions,
            )
    _log_process_results(results)
    if upload_failures:
        LOGGER.error("%d background upload(s) failed", len(upload_failures))
        return 1
    return 0


def _dispatch_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the handler for the parsed command."""
    if args.command == "split-v2-sentences":
        return _run_v2_sentence_split(parser, args, data_root=data_root, settings=settings)
    if args.command == "sync-dir":
        return _run_sync_command(parser, args, data_root=data_root, settings=settings)
    if args.command in {"augment-region", "augment-dir"}:
        return _run_augmentation_command(args, data_root=data_root, settings=settings)
    return _run_processing_command(args, data_root=data_root, settings=settings)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_root, settings = _prepare_runtime(args)
    _authenticate_for_push(parser, args, settings)
    return _dispatch_command(parser, args, data_root=data_root, settings=settings)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
