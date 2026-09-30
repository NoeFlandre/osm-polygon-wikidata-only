"""Run core processing with its optional, durable HF publication lifecycle.

The CLI supplies the processing callback and command options. This service owns
the queue, regional uploads, deferred repository metadata, and refresh marker so
publication policy stays out of command dispatch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import StubHfHub, upload_files
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    clear_metadata_refresh_marker,
    load_metadata_refresh_marker,
    set_metadata_refresh_marker,
)
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


class _ProcessingOptions(Protocol):
    """CLI values needed to publish completed processing results."""

    push: bool
    dry_run: bool
    upload_threads: int
    commit_message: str | None


@dataclass(frozen=True, slots=True)
class _CorePublicationOutcome:
    """Processing results and any background publication failures."""

    results: list[ProcessResult]
    upload_failures: list[str]


Process = Callable[[Callable[[ProcessResult], None]], list[ProcessResult]]


def run_core_publication(
    process: Process,
    options: _ProcessingOptions,
    settings: Settings,
    *,
    data_root: DataRoot,
    defer_metadata_assets: bool,
) -> _CorePublicationOutcome:
    """Run processing and reliably drain its optional core publication queue."""
    upload_queue = _build_upload_queue(options, settings, data_root=data_root)
    dry_run = bool(getattr(options, "dry_run", False))
    published_stems: set[str] = set()
    enqueue_upload = _processing_upload_enqueuer(
        options,
        settings,
        data_root=data_root,
        upload_queue=upload_queue,
        defer_metadata_assets=defer_metadata_assets,
        dry_run=dry_run,
        published_stems=published_stems,
    )
    upload_failures: list[str] = []
    processing_completed = False
    results: list[ProcessResult] = []
    try:
        results = process(enqueue_upload)
        processing_completed = True
    finally:
        if upload_queue is not None:
            upload_failures = _drain_uploads(
                upload_queue,
                data_root=data_root,
                repo_id=settings.repo_id,
                # An aborted run still drains the queue, but never publishes
                # repository-wide assets for regions absent from the remote.
                deferring=_refresh_allowed(defer_metadata_assets, processing_completed),
                published_stems=published_stems,
                dry_run=dry_run,
            )
    return _CorePublicationOutcome(results=results, upload_failures=upload_failures)


def _build_upload_queue(
    options: _ProcessingOptions,
    settings: Settings,
    *,
    data_root: DataRoot,
) -> BackgroundUploadQueue | None:
    """Build and resume the bounded background upload queue when pushing."""
    if not options.push:
        return None
    hub = StubHfHub() if getattr(options, "dry_run", False) else None

    def upload_job(ops: list[PublicationOp], message: str) -> None:
        upload_files(
            settings.repo_id,
            ops=ops,
            hub=hub,
            token=settings.hf_token,
            commit_message=message,
            num_threads=options.upload_threads,
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


def _processing_upload_enqueuer(
    options: _ProcessingOptions,
    settings: Settings,
    *,
    data_root: DataRoot,
    upload_queue: BackgroundUploadQueue | None,
    defer_metadata_assets: bool,
    dry_run: bool,
    published_stems: set[str],
) -> Callable[[ProcessResult], None]:
    """Return the per-region callback that records and queues core uploads."""
    deferred_stems: dict[str, str] = {}

    def enqueue_upload(result: ProcessResult) -> None:
        if upload_queue is None:
            return
        if defer_metadata_assets and not dry_run:
            _record_deferred_metadata(data_root, deferred_stems, result)
        _enqueue_core_upload(
            upload_queue,
            data_root=data_root,
            repo_id=settings.repo_id,
            commit_message=options.commit_message
            or f"Update PBF {result.manifest_entry['source_pbf']}",
            result=result,
            defer_metadata_assets=defer_metadata_assets,
        )
        published_stems.add(_result_stem(result))

    return enqueue_upload


def _enqueue_core_upload(
    upload_queue: BackgroundUploadQueue,
    *,
    data_root: DataRoot,
    repo_id: str,
    commit_message: str,
    result: ProcessResult,
    defer_metadata_assets: bool = False,
) -> None:
    """Assemble and submit one core publication to the background queue."""
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
    """Drain regional jobs and refresh deferred metadata only after success."""
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
    """Explain which regions still need the deferred metadata refresh."""
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
    """Publish repository-wide assets and retire their marker after success."""
    try:
        _upload_metadata_refresh(upload_queue, data_root=data_root, repo_id=repo_id)
    except Exception as error:  # noqa: BLE001 -- fail-soft refresh retains its marker
        LOGGER.error("Repository metadata refresh failed: %s", error)
        return [f"Refresh repository metadata and maps: {error}"]
    if not dry_run:
        clear_metadata_refresh_marker(data_root)
    return []


def _upload_metadata_refresh(
    upload_queue: BackgroundUploadQueue,
    *,
    data_root: DataRoot,
    repo_id: str,
) -> None:
    """Publish repository-wide metadata after a directory run has drained."""
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


def _refresh_allowed(deferring: bool, processing_completed: bool) -> bool:
    """Only a run that finished processing may publish deferred assets."""
    return deferring and processing_completed


def _metadata_refresh_requested(data_root: DataRoot, *, published_stems: set[str]) -> bool:
    """Refresh only when every marked region was published by this run."""
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
    """Persist and merge the intent to refresh repository-wide assets."""
    stem = _result_stem(result)
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    if not polygons_path.is_file():
        return
    stems[stem] = sha256_file(polygons_path)
    merged = {**_recorded_marker_hashes(data_root), **stems}
    set_metadata_refresh_marker(data_root, sorted(merged), merged)


def _recorded_marker_hashes(data_root: DataRoot) -> dict[str, str]:
    """Return hashes a previous refresh marker already names."""
    marker = load_metadata_refresh_marker(data_root)
    if marker is None:
        return {}
    return {str(stem): str(digest) for stem, digest in marker["fingerprint_hashes"].items()}


__all__ = ["run_core_publication"]
