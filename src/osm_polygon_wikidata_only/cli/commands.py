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
import json
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
    # Keep command-only publication imports lazy so parser/help startup stays light.
    from osm_polygon_wikidata_only.hf.publication import assemble_core_upload  # noqa: PLC0415

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
    published_stems: set[str],
    dry_run: bool,
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
    if not _metadata_refresh_requested(data_root, published_stems=published_stems):
        _warn_metadata_refresh_pending(data_root)
        return failures
    return _refresh_repository_metadata(
        upload_queue, data_root=data_root, repo_id=repo_id, dry_run=dry_run
    )


def _warn_metadata_refresh_pending(data_root: DataRoot) -> None:
    """Say which regions still owe a refresh this run could not publish.

    This command cannot verify that a region it did not publish itself --
    one an earlier run stranded, or one the upload queue resumed from its
    durable state -- actually reached the remote, so it leaves the
    repository-wide assets alone. The operator needs to know that, and
    which command repairs it.
    """
    marker = load_metadata_refresh_marker(data_root)
    if marker is None:
        return
    LOGGER.warning(
        "Repository metadata still owes %s; run sync-dir to reconcile those regions "
        "and refresh the statistics, maps, and README",
        ", ".join(marker["stems"]),
    )


def _refresh_repository_metadata(
    upload_queue: BackgroundUploadQueue,
    *,
    data_root: DataRoot,
    repo_id: str,
    dry_run: bool,
) -> list[str]:
    """Publish the deferred repository-wide assets and retire their marker.

    A dry run simulates the upload against the stub hub, so nothing
    reaches the remote and the marker is left alone: retiring it would
    tell the next run that the assets are current when they were never
    published.
    """
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
    if not dry_run:
        clear_metadata_refresh_marker(data_root)
    return []


def _refresh_allowed(deferring: bool, processing_completed: bool) -> bool:
    """Only a run that finished processing may publish deferred assets."""
    return deferring and processing_completed


def _metadata_refresh_requested(data_root: DataRoot, *, published_stems: set[str]) -> bool:
    """Refresh only when every marked region was published by this run.

    The marker names the regions whose repository-wide assets are still
    owed. A run that published none of them -- a resumed run that skipped
    every locally processed PBF, or one whose marked region died before
    its upload was submitted -- must not publish a manifest, statistics
    report, or map describing a region the remote never received. The
    marker survives such a run, and the sync command, which reconciles
    against the remote and republishes missing regional artifacts before
    refreshing, is what repairs it.
    """
    if not published_stems:
        return False
    marker = load_metadata_refresh_marker(data_root)
    if marker is None:
        return True
    return set(marker["stems"]) <= published_stems


def _result_stem(result: ProcessResult) -> str:
    """Return the region stem of one processed PBF."""
    return Path(str(result.manifest_entry["source_pbf"])).stem.removesuffix(".osm")


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

    A surviving marker is merged rather than replaced: an earlier run may
    have marked a region whose upload never happened, and dropping it
    here would let this run's own region license a repository-wide
    refresh describing the stranded one.
    """
    stem = _result_stem(result)
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    if not polygons_path.is_file():
        return
    stems[stem] = sha256_file(polygons_path)
    merged = {**_recorded_marker_hashes(data_root), **stems}
    set_metadata_refresh_marker(data_root, sorted(merged), merged)


def _recorded_marker_hashes(data_root: DataRoot) -> dict[str, str]:
    """Return the stems an existing refresh marker already names."""
    marker = load_metadata_refresh_marker(data_root)
    if marker is None:
        return {}
    return {str(stem): str(digest) for stem, digest in marker["fingerprint_hashes"].items()}


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
    # Load the publication stack only for the metadata-refresh command path.
    from osm_polygon_wikidata_only.hf.publication import (  # noqa: PLC0415
        assemble_metadata_only_upload,
    )

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
    release_apply = _release_apply_requested(args)
    if not _push_authentication_required(args, release_apply) or args.dry_run:
        return
    if release_apply:
        _authenticate_release_targets(parser, args, settings)
        return
    _require_push_token(parser, settings)
    _verify_push_access(parser, settings)


def _release_apply_requested(args: argparse.Namespace) -> bool:
    return getattr(args, "command", None) in {
        "release-stats",
        "publish-language-splits",
    } and getattr(args, "apply", False)


def _push_authentication_required(args: argparse.Namespace, release_apply: bool) -> bool:
    return bool(args.push or release_apply)


def _authenticate_release_targets(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    settings: Settings,
) -> None:
    for target in _selected_release_targets(args.dataset_version):
        target_settings = replace(settings, repo_id=_RELEASE_TARGETS[target])
        _require_push_token(parser, target_settings)
        _verify_push_access(parser, target_settings)


def _run_v2_sync(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the v2 sync while holding the shared lock."""
    # Keep the selected workflow lazy so the CLI can expose help without V2 dependencies.
    from osm_polygon_wikidata_only.v2.cli import execute_v2  # noqa: PLC0415

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
    # Sentence splitting has optional model dependencies and is loaded only when selected.
    from osm_polygon_wikidata_only.v2.cli import execute_v2_sentence_split  # noqa: PLC0415

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
    # The V1 runner is loaded only after argument dispatch and lock selection.
    from .run_sync import execute as cli_run_sync  # noqa: PLC0415

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
    # These imports are command collaborators; keeping them local avoids importing
    # the full augmentation and migration stacks for unrelated CLI commands.
    from osm_polygon_wikidata_only.augmentation.orchestrator import (  # noqa: PLC0415
        load_existing_augmentation_result,
    )
    from osm_polygon_wikidata_only.pipeline.link_migration import (  # noqa: PLC0415
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
    # Do not load Hugging Face publication code for local-only augmentation runs.
    from osm_polygon_wikidata_only.hf.publication import (  # noqa: PLC0415
        assemble_augmentation_upload,
    )

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
    # A simulated push must not touch durable publication state.
    dry_run = bool(getattr(args, "dry_run", False))
    published_stems: set[str] = set()
    deferred_stems: dict[str, str] = {}

    def enqueue_upload(result: ProcessResult) -> None:
        if upload_queue is None:
            return
        # Recorded before the job is submitted: a kill between the two
        # would otherwise leave an uploadable region with no record that
        # the repository-wide assets still owe it a refresh.
        if defer_metadata_assets and not dry_run:
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
        published_stems.add(_result_stem(result))

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
                published_stems=published_stems,
                dry_run=dry_run,
            )
    _log_process_results(results)
    if upload_failures:
        LOGGER.error("%d background upload(s) failed", len(upload_failures))
        return 1
    return 0


_RELEASE_TARGETS: dict[str, str] = {"v1": DEFAULT_REPO_ID, "v2": V2_REPO_ID}


def _selected_release_targets(dataset_version: str) -> tuple[str, ...]:
    """Return the ordered contracts a release run publishes."""
    if dataset_version == "both":
        return ("v1", "v2")
    return (dataset_version,)


def _release_confirmations(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    targets: tuple[str, ...],
) -> dict[str, str]:
    """Map each released contract to its required exact repository id."""
    supplied = list(getattr(args, "confirm_repo", None) or [])
    expected = [_RELEASE_TARGETS[target] for target in targets]
    if sorted(supplied) != sorted(expected):
        parser.error(
            "release-stats requires one --confirm-repo per released dataset: " + ", ".join(expected)
        )
    return dict(zip(targets, expected, strict=True))


def _run_release_stats(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
) -> int:
    """Recompute and publish only the card and statistics report."""
    # Release-only dependencies are intentionally isolated from normal sync startup.
    from osm_polygon_wikidata_only.hf.stats_release import (  # noqa: PLC0415
        StatsReleaseError,
        release_v1_polygon_stats,
        release_v2_polygon_stats,
    )

    targets = _selected_release_targets(args.dataset_version)
    confirmations = _release_confirmations(parser, args, targets)
    releases = {"v1": release_v1_polygon_stats, "v2": release_v2_polygon_stats}
    hub = StubHfHub() if args.dry_run else None
    reports = []
    for target in targets:
        try:
            report = releases[target](
                data_root,
                confirm_repo=confirmations[target],
                apply=args.apply,
                hub=hub,
                token=getattr(args, "hf_token", None),
                source_revision=getattr(args, "source_revision", None),
                data_revision=getattr(args, "data_revision", None),
                generated_on=getattr(args, "generated_on", None),
            )
        except StatsReleaseError as error:
            parser.error(str(error))
        reports.append(report)
        print(json.dumps(report.to_payload(), sort_keys=True))
    return 0


def _run_language_splits(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
) -> int:
    """Plan or generate both language-split contracts without publication."""
    # Language generation is an explicit command and may load large optional readers.
    from osm_polygon_wikidata_only.hf.language_split_release import (  # noqa: PLC0415
        LanguageSplitReleaseError,
        run_language_split_release,
    )

    try:
        result = run_language_split_release(
            data_root,
            dataset_version=args.dataset_version,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    except LanguageSplitReleaseError as error:
        parser.error(str(error))
    print(result.to_json())
    return 0


def _run_publish_language_splits(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
) -> int:
    """Generate and publish exact-target language partitions."""
    # Keep publication and its network client out of non-publication CLI paths.
    from osm_polygon_wikidata_only.hf.language_split_publication import (  # noqa: PLC0415
        LanguagePublicationError,
        run_language_split_publication,
    )

    try:
        result = run_language_split_publication(
            data_root,
            dataset_version=args.dataset_version,
            batch_size=args.batch_size,
            confirm_repos=tuple(args.confirm_repo or ()),
            apply=args.apply,
            dry_run=args.dry_run,
            token=args.hf_token,
        )
    except LanguagePublicationError as error:
        parser.error(str(error))
    print(result.to_json())
    return 0


def _dispatch_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the language-split facade or delegate to an existing command."""
    if args.command == "language-splits":
        return _run_language_splits(parser, args, data_root=data_root)
    if args.command == "publish-language-splits":
        return _run_publish_language_splits(parser, args, data_root=data_root)
    return _dispatch_existing_command(parser, args, data_root=data_root, settings=settings)


def _dispatch_existing_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the handler for the established processing and release commands."""
    if args.command == "split-v2-sentences":
        return _run_v2_sentence_split(parser, args, data_root=data_root, settings=settings)
    if args.command == "release-stats":
        return _run_release_stats(parser, args, data_root=data_root)
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
