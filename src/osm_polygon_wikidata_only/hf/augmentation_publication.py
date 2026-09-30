"""Publish completed text augmentation artifacts to Hugging Face."""

from __future__ import annotations

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.uploader import StubHfHub, upload_files


def publish_augmentation(
    *,
    data_root: DataRoot,
    repo_id: str,
    token: str | None,
    dry_run: bool,
    commit_message: str,
    upload_threads: int,
    result: AugmentationResult,
) -> None:
    """Assemble and upload the operations for one augmentation result."""
    from osm_polygon_wikidata_only.hf.publication import (  # noqa: PLC0415
        assemble_augmentation_upload,
    )

    ops = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=repo_id,
        augmentation=result,
    )
    upload_files(
        repo_id,
        ops=ops,
        hub=StubHfHub() if dry_run else None,
        token=token,
        commit_message=commit_message,
        num_threads=upload_threads,
    )


__all__ = ["publish_augmentation"]
