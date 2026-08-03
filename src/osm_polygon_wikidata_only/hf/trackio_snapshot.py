"""Publish the public ``final-dataset-snapshot`` Trackio run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from osm_polygon_wikidata_only.config.paths import resolve_data_root

from ._trackio.models import (
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

app = typer.Typer(
    add_completion=False, help="Publish the frozen final dataset snapshot to Trackio."
)


@app.command()
def publish(
    data_root: Annotated[
        Path | None,
        typer.Option(help="Local data root used for Trackio artifact storage."),
    ] = None,
    space_id: Annotated[
        str,
        typer.Option(help="Public Hugging Face Space receiving the Trackio run."),
    ] = TRACKIO_SPACE_ID,
) -> None:
    """Publish one static run and exactly three plots."""
    resolved = resolve_data_root(data_root, repo_root=Path(__file__).resolve().parents[3])
    artifacts = publish_trackio_snapshot(
        output_dir=resolved.cache / "trackio" / TRACKIO_RUN_NAME,
        space_id=space_id,
    )
    typer.echo(f"Trackio run published: https://huggingface.co/spaces/{space_id}")
    typer.echo(f"Artifacts: {artifacts.output_dir}")


def run() -> None:
    """Installed console-script entry point."""
    app()


__all__ = [
    "FINAL_DATASET_SNAPSHOT",
    "TRACKIO_DATASET_ID",
    "TRACKIO_PROJECT",
    "TRACKIO_RUN_NAME",
    "TRACKIO_SPACE_ID",
    "TRACKIO_SPACE_URL",
    "FinalDatasetSnapshot",
    "TrackioSnapshotArtifacts",
    "app",
    "publish",
    "publish_trackio_snapshot",
    "render_snapshot_charts",
    "render_snapshot_markdown",
    "run",
]
