"""Markdown and chart rendering for the frozen Trackio snapshot."""

from __future__ import annotations

from pathlib import Path

from .models import FINAL_DATASET_SNAPSHOT, TRACKIO_SPACE_URL, FinalDatasetSnapshot


def render_snapshot_markdown(snapshot: FinalDatasetSnapshot = FINAL_DATASET_SNAPSHOT) -> str:
    """Render the public description shared by the README and dataset card."""
    return (
        "## Trackio snapshot\n\n"
        f"The finished dataset is recorded in the single public Trackio run "
        f"[`final-dataset-snapshot`]({TRACKIO_SPACE_URL}). It contains exactly "
        "three plots: a text-coverage funnel, the top ten Wikipedia languages "
        "plus `Other languages`, and dataset composition on a logarithmic scale. "
        "The run is a static snapshot with no pipeline timeline.\n\n"
        "| Snapshot metric | Value |\n"
        "| --- | ---: |\n"
        f"| Polygons | {snapshot.polygons:,} |\n"
        f"| Wikipedia + Wikivoyage documents | {snapshot.documents:,} |\n"
        f"| Document words | {snapshot.document_words:,} |\n"
        f"| Languages | {snapshot.languages:,} |\n"
        f"| Geographic regions | {snapshot.geographic_regions:,} |\n\n"
        "| Small snapshot table | Value |\n"
        "| --- | ---: |\n"
        f"| Wikipedia polygon-document links | {snapshot.wikipedia_polygon_document_links:,} |\n"
        f"| Polygon/link-table storage | {snapshot.polygon_link_storage_gb:.1f} GB |\n"
        f"| Total Parquet storage | {snapshot.total_parquet_storage_gb:.1f} GB |\n\n"
        "The funnel's language thresholds use the canonical Wikipedia polygon "
        "fields. Wikivoyage is included in the combined document, word, and "
        "dataset-composition totals.\n"
    )


def render_snapshot_charts(
    output_dir: Path,
    snapshot: FinalDatasetSnapshot = FINAL_DATASET_SNAPSHOT,
) -> dict[str, Path]:
    """Create exactly the three Trackio PNG plots and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "text_coverage_funnel": output_dir / "text_coverage_funnel.png",
        "top_10_wikipedia_languages": output_dir / "top_10_wikipedia_languages.png",
        "dataset_composition": output_dir / "dataset_composition.png",
    }
    _render_funnel(paths["text_coverage_funnel"], snapshot)
    _render_languages(paths["top_10_wikipedia_languages"], snapshot)
    _render_composition(paths["dataset_composition"], snapshot)
    return paths


def _new_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(8.4, 4.8), dpi=150)


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Date": None})
    figure.clear()


def _render_funnel(path: Path, snapshot: FinalDatasetSnapshot) -> None:
    figure, axis = _new_figure()
    labels = [label for label, _ in snapshot.text_coverage_funnel]
    values = [value for _, value in snapshot.text_coverage_funnel]
    bars = axis.barh(labels[::-1], values[::-1], color="#2563eb")
    axis.set_title("Text coverage funnel")
    axis.set_xlabel("Polygons")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values[::-1], strict=True):
        axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,}", va="center")
    _save_figure(figure, path)


def _render_languages(path: Path, snapshot: FinalDatasetSnapshot) -> None:
    figure, axis = _new_figure()
    labels = [label for label, _ in snapshot.top_wikipedia_languages] + ["Other languages"]
    values = [value for _, value in snapshot.top_wikipedia_languages] + [
        snapshot.other_wikipedia_languages
    ]
    colors = ["#2563eb"] * 10 + ["#94a3b8"]
    axis.bar(labels, values, color=colors)
    axis.set_title("Top 10 Wikipedia languages")
    axis.set_ylabel("Unique Wikipedia documents")
    axis.tick_params(axis="x", labelrotation=45)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    _save_figure(figure, path)


def _render_composition(path: Path, snapshot: FinalDatasetSnapshot) -> None:
    figure, axis = _new_figure()
    labels = [
        "Wikipedia\ndocuments",
        "Wikipedia\nsections",
        "Wikivoyage\ndocuments",
        "Wikivoyage\nsections",
        "Wikidata\nfacts",
    ]
    values = [
        snapshot.wikipedia_documents,
        snapshot.wikipedia_sections,
        snapshot.wikivoyage_documents,
        snapshot.wikivoyage_sections,
        snapshot.wikidata_facts,
    ]
    axis.bar(labels, values, color="#7c3aed")
    axis.set_yscale("log")
    axis.set_title("Dataset composition")
    axis.set_ylabel("Rows, logarithmic scale")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    _save_figure(figure, path)


__all__ = ["render_snapshot_charts", "render_snapshot_markdown"]
