"""Publish the public ``final-dataset-snapshot`` Trackio run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from osm_polygon_wikidata_only.cli.parser import add_trackio_snapshot_arguments
from osm_polygon_wikidata_only.config.paths import repository_root, resolve_data_root

from ._trackio.models import (
    DATASET_PRESENTATION_URL,
    FINAL_DATASET_SNAPSHOT,
    TRACKIO_DATASET_ID,
    TRACKIO_PROJECT,
    TRACKIO_RUN_NAME,
    TRACKIO_SPACE_ID,
    TRACKIO_SPACE_URL,
    FinalDatasetSnapshot,
)
from ._trackio.publisher import TrackioSnapshotArtifacts, publish_trackio_snapshot
from ._trackio.rendering import render_snapshot_charts, render_snapshot_markdown

STANDALONE_PROG = "osm-polygon-wikidata-only-trackio"
STANDALONE_DESCRIPTION = "Publish the frozen final dataset snapshot to Trackio."


def publish(data_root: Path | None = None, space_id: str = TRACKIO_SPACE_ID) -> None:
    """Publish one static run and exactly three plots."""
    resolved = resolve_data_root(data_root, repo_root=repository_root())
    artifacts = publish_trackio_snapshot(
        output_dir=resolved.cache / "trackio" / TRACKIO_RUN_NAME,
        space_id=space_id,
    )
    print(f"Trackio run published: https://huggingface.co/spaces/{space_id}")
    print(f"Artifacts: {artifacts.output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    """Installed console-script entry point for ``osm-polygon-wikidata-only-trackio``."""
    parser = argparse.ArgumentParser(prog=STANDALONE_PROG, description=STANDALONE_DESCRIPTION)
    add_trackio_snapshot_arguments(parser)
    args = parser.parse_args(argv)
    publish(data_root=args.data_root, space_id=args.space_id or TRACKIO_SPACE_ID)
    return 0


__all__ = [
    "DATASET_PRESENTATION_URL",
    "FINAL_DATASET_SNAPSHOT",
    "TRACKIO_DATASET_ID",
    "TRACKIO_PROJECT",
    "TRACKIO_RUN_NAME",
    "TRACKIO_SPACE_ID",
    "TRACKIO_SPACE_URL",
    "FinalDatasetSnapshot",
    "TrackioSnapshotArtifacts",
    "main",
    "publish",
    "publish_trackio_snapshot",
    "render_snapshot_charts",
    "render_snapshot_markdown",
]
