"""Upload queue lifecycle and post-publication local retirement for ``sync-dir``."""

from __future__ import annotations

import logging

from osm_polygon_wikidata_only.augmentation.wikipedia_retirement import finalize_local_retirement
from osm_polygon_wikidata_only.cli._sync.retirement import paired_retirement_stems
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.pipeline.pending_publications import remove_pending_publications

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


def _assemble_containment_retirement_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    parent_children: dict[str, tuple[str, ...]],
) -> list[PublicationOp]:
    # The retirement plan is only assembled when containment migration found work.
    from osm_polygon_wikidata_only.hf.publication import (  # noqa: PLC0415
        assemble_containment_retirement_upload,
    )

    return assemble_containment_retirement_upload(
        data_root=data_root,
        repo_id=repo_id,
        parent_children=parent_children,
        world_land_warning=None,
    )


def enqueue_containment_retirement(
    data_root: DataRoot,
    settings: Settings,
    parent_children: dict[str, tuple[str, ...]],
    upload_queue: BackgroundUploadQueue | None,
    *,
    push_enabled: bool,
) -> bool:
    if not push_enabled or not parent_children or upload_queue is None:
        return False
    operations = _assemble_containment_retirement_upload(
        data_root=data_root,
        repo_id=settings.repo_id,
        parent_children=parent_children,
    )
    upload_queue.submit(operations, "Retire losslessly contained regional dataset shards")
    LOGGER.info(
        "Enqueued containment retirement for %d child region(s)",
        sum(len(children) for children in parent_children.values()),
    )
    return True


def post_upload_publication_cleanup(
    data_root: DataRoot,
    ops: list[PublicationOp],
    *,
    dry_run: bool,
) -> None:
    """Retire local legacy articles and clear pending intent after a confirmed upload.

    Runs *after* the Hub upload succeeds. A stem is retired only when
    the operation list contains BOTH the canonical
    ``add wikipedia/documents/<stem>.parquet`` and the matching legacy
    ``delete articles/<stem>.parquet`` for the same stem. An add
    without its matching delete, a delete for another stem, a delete
    without an add, nested or traversal paths, lookalike prefixes, or
    conflicting duplicate adds do NOT authorize local retirement or
    pending-intent cleanup. Pending intent is cleared only after
    every selected stem's local retirement succeeds.

    The commit message is never inspected; stems are derived strictly
    from ``PublicationOp`` entries that match the canonical layout.
    When ``dry_run`` is true the local filesystem is left untouched so
    repeated dry-runs remain safe.
    """
    if dry_run:
        return

    paired_stems = paired_retirement_stems(data_root, ops)
    if not paired_stems:
        return

    retired: list[str] = []
    for stem in sorted(paired_stems):
        finalize_local_retirement(data_root, stem)
        retired.append(stem)

    remove_pending_publications(data_root, set(retired))


def execute_upload_job(
    *,
    data_root: DataRoot,
    settings: Settings,
    ops: list[PublicationOp],
    message: str,
    num_threads: int,
    hub: HfHub | None,
    dry_run: bool,
) -> None:
    """Production upload-job callback: upload, then clean up local state.

    The upload is the network boundary. When it raises, the cleanup
    helper is never invoked, so the local legacy staging file and the
    durable pending-publications manifest survive intact for the next
    invocation to retry.

    When the upload queue has snapshotted an op (``op.snapshot_path``
    is set), the upload reads from the snapshot rather than the
    canonical local_path so the upload is durable across canonical
    mutation. The canonical local_path is preserved on the op for
    the post-upload retirement check.
    """
    upload_ops = [
        PublicationOp(
            action=op.action,
            path_in_repo=op.path_in_repo,
            local_path=op.snapshot_path or op.local_path,
        )
        for op in ops
    ]
    upload_files(
        settings.repo_id,
        ops=upload_ops,
        hub=hub,
        token=settings.hf_token,
        commit_message=message,
        num_threads=num_threads,
    )
    post_upload_publication_cleanup(data_root, ops, dry_run=dry_run)


def build_upload_queue(
    *,
    push: bool,
    dry_run: bool,
    settings: Settings,
    data_root: DataRoot,
    num_threads: int,
    _hub: HfHub | None = None,
) -> BackgroundUploadQueue | None:
    """Open the documented ``BackgroundUploadQueue`` and resume pending jobs."""
    if not push:
        return None

    hub = _hub if _hub is not None else (StubHfHub() if dry_run else None)

    def upload_job(ops: list[PublicationOp], message: str) -> None:
        execute_upload_job(
            data_root=data_root,
            settings=settings,
            ops=ops,
            message=message,
            num_threads=num_threads,
            hub=hub,
            dry_run=dry_run,
        )

    queue = BackgroundUploadQueue(
        upload=upload_job,
        max_pending=2,
        state_dir=data_root.cache / "sync_upload_jobs",
    )
    resumed = queue.resume_pending()
    if resumed:
        LOGGER.info("Resumed %d pending background upload(s)", resumed)
    return queue
