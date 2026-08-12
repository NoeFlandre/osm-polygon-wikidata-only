"""Publish the immutable Trackio snapshot for the V2 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from osm_polygon_wikidata_only.config.paths import resolve_data_root
from osm_polygon_wikidata_only.hf._trackio.models import FinalDatasetSnapshot
from osm_polygon_wikidata_only.hf._trackio.publisher import (
    TrackioSnapshotArtifacts,
    publish_trackio_snapshot,
)
from osm_polygon_wikidata_only.v2.card import V2CardStats, compute_v2_card_stats
from osm_polygon_wikidata_only.v2.config import (
    V2_REPO_ID,
    V2_TRACKIO_DATASET_ID,
    V2_TRACKIO_PROJECT,
    V2_TRACKIO_RUN_NAME,
    V2_TRACKIO_SPACE_ID,
    V2_TRACKIO_SPACE_URL,
)

app = typer.Typer(add_completion=False, help="Publish the frozen V2 dataset snapshot to Trackio.")


def snapshot_from_v2_stats(stats: V2CardStats) -> FinalDatasetSnapshot:
    """Adapt factual V2 card statistics to the shared static Trackio schema."""
    return FinalDatasetSnapshot(
        polygons=stats.polygons,
        unique_wikidata_entities=stats.unique_wikidata_entities,
        documents=stats.documents,
        document_words=stats.document_words,
        languages=stats.languages,
        geographic_regions=stats.regions,
        wikipedia_documents=stats.wikipedia_documents,
        wikipedia_sections=stats.wikipedia_sections,
        wikivoyage_documents=stats.wikivoyage_documents,
        wikivoyage_sections=stats.wikivoyage_sections,
        wikidata_facts=stats.wikidata_facts,
        wikipedia_polygon_document_links=stats.polygon_document_links,
        polygon_link_storage_gb=stats.polygon_link_storage_bytes / 1_000_000_000,
        total_parquet_storage_gb=stats.total_parquet_storage_bytes / 1_000_000_000,
        text_coverage_funnel=stats.text_coverage_funnel,
        top_wikipedia_languages=stats.top_wikipedia_languages,
    )


def publish_v2_trackio_snapshot(
    *,
    output_dir: Path,
    stats: V2CardStats,
    space_id: str = V2_TRACKIO_SPACE_ID,
    trackio_module: object | None = None,
) -> TrackioSnapshotArtifacts:
    """Publish exactly one V2 run with the current data-derived metrics."""
    return publish_trackio_snapshot(
        output_dir=output_dir,
        space_id=space_id,
        snapshot=snapshot_from_v2_stats(stats),
        trackio_module=trackio_module,
        project=V2_TRACKIO_PROJECT,
        run_name=V2_TRACKIO_RUN_NAME,
        dataset_id=V2_TRACKIO_DATASET_ID,
        dataset_repo_id=V2_REPO_ID,
        presentation_url=f"https://huggingface.co/datasets/{V2_REPO_ID}",
    )


@app.command()
def publish(
    data_root: Annotated[
        Path | None,
        typer.Option(help="Local data root used for V2 Trackio artifact storage."),
    ] = None,
    space_id: Annotated[
        str,
        typer.Option(help="Public Hugging Face Space receiving the V2 run."),
    ] = V2_TRACKIO_SPACE_ID,
) -> None:
    """Publish the V2 card metrics and three static plots."""
    resolved = resolve_data_root(data_root, repo_root=Path(__file__).resolve().parents[3])
    stats = compute_v2_card_stats(resolved.processed_v2, v1_processed=resolved.processed)
    artifacts = publish_v2_trackio_snapshot(
        output_dir=resolved.cache / "trackio" / V2_TRACKIO_RUN_NAME,
        stats=stats,
        space_id=space_id,
    )
    typer.echo(f"Trackio run published: {V2_TRACKIO_SPACE_URL}")
    typer.echo(f"Artifacts: {artifacts.output_dir}")


def run() -> None:
    """Installed console-script entry point."""
    app()


__all__ = [
    "V2_TRACKIO_RUN_NAME",
    "app",
    "publish",
    "publish_v2_trackio_snapshot",
    "run",
    "snapshot_from_v2_stats",
]
