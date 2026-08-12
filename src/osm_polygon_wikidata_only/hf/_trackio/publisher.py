"""Trackio publisher for the single final dataset snapshot."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_write_text

from .models import (
    DATASET_PRESENTATION_URL,
    FINAL_DATASET_SNAPSHOT,
    TRACKIO_DATASET_ID,
    TRACKIO_PROJECT,
    TRACKIO_RUN_NAME,
    TRACKIO_SPACE_ID,
    FinalDatasetSnapshot,
)
from .rendering import render_snapshot_charts, render_snapshot_markdown


@dataclass(frozen=True, slots=True)
class TrackioSnapshotArtifacts:
    """Paths written locally for one Trackio publication."""

    output_dir: Path
    chart_paths: tuple[Path, ...]
    manifest_path: Path


def publish_trackio_snapshot(
    *,
    output_dir: Path,
    space_id: str | None = TRACKIO_SPACE_ID,
    snapshot: FinalDatasetSnapshot = FINAL_DATASET_SNAPSHOT,
    trackio_module: Any | None = None,
    project: str = TRACKIO_PROJECT,
    run_name: str = TRACKIO_RUN_NAME,
    dataset_id: str = TRACKIO_DATASET_ID,
    dataset_repo_id: str = "NoeFlandre/osm-polygon-wikidata-only",
    presentation_url: str | None = DATASET_PRESENTATION_URL,
) -> TrackioSnapshotArtifacts:
    """Write the three plots and log one immutable Trackio run.

    Trackio's automatic CPU/GPU logging is explicitly disabled. The existing
    local run with this frozen name is replaced before logging, so rerunning
    the command keeps one run with one step instead of a construction timeline.
    """
    output_dir = Path(output_dir)
    # Keep Trackio's local SQLite database and media on the configured data
    # root. The production command passes the Seagate-backed cache directory.
    os.environ.setdefault("TRACKIO_DIR", str(output_dir / ".trackio"))
    chart_paths_by_name = render_snapshot_charts(output_dir, snapshot)
    manifest_path = output_dir / "snapshot.json"
    atomic_write_text(
        manifest_path,
        json.dumps(
            {
                "run_name": run_name,
                "project": project,
                "metrics": snapshot.metrics(),
                "plots": list(chart_paths_by_name),
                "table": [list(row) for row in snapshot.table_rows()],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    if trackio_module is None:
        import trackio as trackio_module

    resolved_space_id = space_id or os.environ.get("TRACKIO_SPACE_ID")
    _reset_frozen_run(trackio_module, project=project, run_name=run_name)
    run = trackio_module.init(
        project=project,
        name=run_name,
        # Log locally first. ``sync(..., sdk='static')`` below is the public,
        # free dashboard path and avoids requiring a running Gradio Space.
        space_id=None,
        config={
            "snapshot": run_name,
            "dataset": dataset_repo_id,
            "static": True,
        },
        resume="never",
        embed=False,
        auto_log_cpu=False,
        auto_log_gpu=False,
        private=False,
    )
    del run
    try:
        metrics: dict[str, Any] = dict(snapshot.metrics())
        metrics.update(
            {
                "report/summary": trackio_module.Markdown(
                    render_snapshot_markdown(
                        snapshot,
                        run_name=run_name,
                        space_url=(
                            f"https://huggingface.co/spaces/{resolved_space_id}"
                            if resolved_space_id
                            else ""
                        ),
                        presentation_url=presentation_url,
                    )
                ),
                "report/table": trackio_module.Table(
                    columns=["Metric", "Value"],
                    data=[list(row) for row in snapshot.table_rows()],
                ),
                "plot/text_coverage_funnel": trackio_module.Image(
                    chart_paths_by_name["text_coverage_funnel"],
                    caption="Text coverage funnel",
                ),
                "plot/top_10_wikipedia_languages": trackio_module.Image(
                    chart_paths_by_name["top_10_wikipedia_languages"],
                    caption="Top 10 Wikipedia languages plus Other languages",
                ),
                "plot/dataset_composition": trackio_module.Image(
                    chart_paths_by_name["dataset_composition"],
                    caption="Dataset composition on a logarithmic scale",
                ),
            }
        )
        trackio_module.log(metrics, step=0)
    finally:
        trackio_module.finish()

    sync = getattr(trackio_module, "sync", None)
    if resolved_space_id is not None and callable(sync):
        sync(
            project=project,
            space_id=resolved_space_id,
            dataset_id=dataset_id,
            sdk="static",
            force=True,
        )

    return TrackioSnapshotArtifacts(
        output_dir=output_dir,
        chart_paths=tuple(chart_paths_by_name.values()),
        manifest_path=manifest_path,
    )


def _reset_frozen_run(trackio_module: Any, *, project: str, run_name: str) -> None:
    """Remove only the previous local run with the fixed snapshot name."""
    if getattr(trackio_module, "__name__", "") != "trackio":
        return
    from trackio.sqlite_storage import SQLiteStorage

    SQLiteStorage.delete_run(project, run_name)


__all__ = ["TrackioSnapshotArtifacts", "publish_trackio_snapshot"]
