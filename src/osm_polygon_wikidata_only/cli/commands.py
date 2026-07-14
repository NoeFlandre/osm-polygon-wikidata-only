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

import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    AugmentationResult,
    augment_region,
    augmentation_is_current,
    completed_region_stems,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level))
    if not args.data_root or not args.data_root.exists():
        # defer to resolver
        pass
    data_root = _resolve_data_root(args)
    data_root.ensure()
    settings = _build_settings(args)

    if args.push and not args.dry_run:
        resolved = resolve_hf_token(settings.hf_token)
        if not resolved:
            env_token = os.environ.get("HF_TOKEN")
            explicit = bool(settings.hf_token)
            if env_token or explicit:
                # Token was supplied but huggingface_hub rejected it (placeholder,
                # expired, revoked, malformed). Give a more actionable hint than
                # the generic "no token" message.
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
        try:
            username = verify_hf_token(settings.hf_token)
        except UploadError as error:
            parser.error(str(error))
        try:
            verify_repo_authorization(settings.hf_token, settings.repo_id)
        except UploadError as error:
            parser.error(str(error))
        LOGGER.info("Authenticated to Hugging Face as %s (target: %s)", username, settings.repo_id)

    if args.command == "sync-dir":
        from .run_sync import execute as cli_run_sync

        try:
            with exclusive_run_lock(data_root.cache / "sync.lock"):
                return cli_run_sync(
                    args,
                    data_root=data_root,
                    settings=settings,
                    # No build_upload_files override: the CLI shell owns
                    # the production region-publication builder via
                    # hf.publication.assemble_region_upload.
                    build_upload_files=None,
                )
        except RunLockError as error:
            parser.error(str(error))

    if args.command in {"augment-region", "augment-dir"}:
        from osm_polygon_wikidata_only.hf.publication import assemble_augmentation_upload

        augmentation_client = AugmentationWikimediaClient(
            settings,
            JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
        )
        stems = (
            [args.stem] if args.command == "augment-region" else completed_region_stems(data_root)
        )
        augmentation_results: list[AugmentationResult] = []
        for stem in stems:
            if settings.skip_existing and augmentation_is_current(data_root, stem):
                LOGGER.info("Skipping augmentation for %s (already current)", stem)
                continue
            augmentation_result = augment_region(data_root, stem, augmentation_client)
            augmentation_results.append(augmentation_result)
            LOGGER.info("Augmented %s: %s", stem, augmentation_result.counts)
            if args.push:
                hub = StubHfHub() if args.dry_run else None

                def _submit(
                    ops: list[PublicationOp],
                    message: str,
                    _hub: StubHfHub | None = hub,
                ) -> None:
                    upload_files(
                        settings.repo_id,
                        ops=ops,
                        hub=_hub,
                        token=settings.hf_token,
                        commit_message=message,
                        num_threads=args.upload_threads,
                    )

                ops = assemble_augmentation_upload(
                    data_root=data_root,
                    repo_id=settings.repo_id,
                    augmentation=augmentation_result,
                )
                _submit(
                    ops,
                    args.commit_message or f"Add text augmentation for {stem}",
                )
        LOGGER.info("Done. %d region augmentation(s).", len(augmentation_results))
        return 0

    wd, wiki, cache = _build_clients(settings, data_root=data_root)

    inputs: list[Path]
    if args.command == "process-pbf" or args.command == "process-dir":
        inputs = [args.input]
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")
        return 2

    upload_queue: BackgroundUploadQueue | None = None
    hub = StubHfHub() if args.push and args.dry_run else None
    if args.push:

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
    if upload_failures:
        LOGGER.error("%d background upload(s) failed", len(upload_failures))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
