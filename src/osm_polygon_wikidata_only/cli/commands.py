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
from osm_polygon_wikidata_only.io.run_lock import RunLockError, exclusive_run_lock
from osm_polygon_wikidata_only.pipeline.orchestrator import orchestrate
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
    )
    upload_queue.submit(ops, commit_message)


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

    def enqueue_upload(result: ProcessResult) -> None:
        if upload_queue is None:
            return
        _enqueue_core_upload(
            upload_queue,
            data_root=data_root,
            repo_id=settings.repo_id,
            commit_message=args.commit_message
            or f"Update PBF {result.manifest_entry['source_pbf']}",
            result=result,
        )

    upload_failures: list[str] = []
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
    finally:
        if upload_queue is not None:
            upload_failures = upload_queue.close_and_wait()
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
