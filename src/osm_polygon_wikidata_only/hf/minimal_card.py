"""Shared compact renderer for the public Hugging Face dataset cards.

The release cards deliberately contain a small, stable summary.  Detailed
statistics remain machine-readable in ``stats.json``; this module only knows
how to render the common public presentation and never reads the dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MAX_CARD_BODY_BYTES = 8 * 1024


@dataclass(frozen=True, slots=True)
class ContinentCoverage:
    """One row of the compact continent coverage table."""

    name: str
    polygons: int
    wikipedia_documents: int
    wikivoyage_documents: int
    wikipedia_text_polygons: int
    text_polygons: int


@dataclass(frozen=True, slots=True)
class MinimalCardSnapshot:
    """All values needed to render a compact card, already computed."""

    front_matter: str
    repo_id: str
    title: str
    description: str
    polygon_rows: int
    unique_polygon_identities: int
    polygons_with_text: int
    documents: int
    sections: int
    languages: int
    regions: int
    total_parquet_bytes: int
    continent_rows: Sequence[ContinentCoverage] = ()
    generated_on: str | None = None
    source_url: str = "https://github.com/NoeFlandre/osm-polygon-wikidata-only"
    stats_path: str = "stats.json"
    coverage_map_path: str = "assets/coverage_map.png"
    text_presence_map_path: str = "assets/geographic_text_presence.png"
    text_density_map_path: str = "assets/geographic_text_density.png"
    viewer_url: str | None = None


def render_minimal_card(snapshot: MinimalCardSnapshot) -> str:
    """Render one compact card and fail closed if its body grows too large."""
    body = "\n".join(_body_lines(snapshot)).rstrip() + "\n"
    if len(body.encode("utf-8")) >= MAX_CARD_BODY_BYTES:
        raise ValueError("minimal dataset card body must be smaller than 8 KiB")
    front_matter = snapshot.front_matter.rstrip() + "\n"
    return front_matter + body


def _body_lines(snapshot: MinimalCardSnapshot) -> list[str]:
    generated = [f"Generated on {snapshot.generated_on}.", ""] if snapshot.generated_on else []
    viewer = (
        f"[Dataset Viewer schema]({snapshot.viewer_url})"
        if snapshot.viewer_url
        else "The YAML front matter declares the Dataset Viewer configurations."
    )
    return [
        f"![{snapshot.repo_id} dataset overview](assets/dataset_hero.png)",
        "",
        f"# {snapshot.title}",
        "",
        snapshot.description,
        "",
        *generated,
        f"Source code: [GitHub repository]({snapshot.source_url}).",
        "",
        "## Dataset snapshot",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Polygon rows across regional extracts | {_integer(snapshot.polygon_rows)} |",
        f"| Unique polygon identities (osm_type, osm_id) | {_integer(snapshot.unique_polygon_identities)} |",
        f"| Polygons with successful non-empty text (unique OSM identities) | {_integer(snapshot.polygons_with_text)} |",
        f"| Wikipedia + Wikivoyage documents | {_integer(snapshot.documents)} |",
        f"| Wikipedia + Wikivoyage sections | {_integer(snapshot.sections)} |",
        f"| Wikipedia + Wikivoyage languages | {_integer(snapshot.languages)} |",
        f"| Geographic regions | {_integer(snapshot.regions)} |",
        f"| Total Parquet size | {_size(snapshot.total_parquet_bytes)} |",
        "",
        "Polygon rows preserve regional records; identity and text metrics count each "
        "`(osm_type, osm_id)` once. Text requires `fetch_status=ok` and non-empty `full_text`.",
        "",
        "## Coverage maps",
        "",
        "### All polygon identities",
        "",
        f"![All dataset polygons]({snapshot.coverage_map_path})",
        "",
        "One point per unique retained polygon identity, including identities without text.",
        "",
        "### Polygon identities with text",
        "",
        f"![Polygons with text]({snapshot.text_presence_map_path})",
        "",
        f"One point per unique identity with successful non-empty text ({_integer(snapshot.polygons_with_text)}).",
        "",
        "### H3 text density",
        "",
        f"![Geographic text density]({snapshot.text_density_map_path})",
        "",
        "Each H3 cell shows the absolute count of unique text-covered polygon identities; it is not a proportion.",
        "",
        "## Geographic distribution by continent",
        "",
        "Counts use one deterministic representative per polygon identity and the same text definition as the maps.",
        "",
        "| Continent | Polygons | Wikipedia documents | Wikivoyage documents | Polygons with Wikipedia text | Polygons with Wikipedia or Wikivoyage text | Text coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(_continent_line(row) for row in sorted(snapshot.continent_rows, key=lambda row: row.name)),
        "",
        "## Polygon area and geometry",
        "",
        f"Complete area, histogram, geometry, extent, and per-source statistics: [`{snapshot.stats_path}`]({snapshot.stats_path}).",
        "",
        "## Schema",
        "",
        f"{viewer}. The published tables are documented by their Dataset Viewer configurations.",
        "",
        "## Data sources & licenses",
        "",
        "OpenStreetMap polygons are ODbL; Wikidata is CC0; Wikipedia and Wikivoyage text are CC BY-SA 4.0.",
        "",
        "## How to load",
        "",
        "```python",
        "from datasets import load_dataset",
        f'ds = load_dataset("parquet", data_files={{"polygons": "hf://datasets/{snapshot.repo_id}/polygons/*.parquet"}})',
        "```",
        "",
        "## Citation",
        "",
        "Cite the dataset using [`CITATION.cff`](CITATION.cff).",
        "",
    ]


def _continent_line(row: ContinentCoverage) -> str:
    rate = row.text_polygons / row.polygons if row.polygons else 0.0
    return (
        f"| {row.name} | {_integer(row.polygons)} | {_integer(row.wikipedia_documents)} | "
        f"{_integer(row.wikivoyage_documents)} | {_integer(row.wikipedia_text_polygons)} | "
        f"{_integer(row.text_polygons)} | {rate:.1%} |"
    )


def _integer(value: int) -> str:
    return f"{value:,}"


def _size(value: int) -> str:
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.1f} MB"
    return f"{value / 1_000_000_000:.1f} GB"


__all__ = [
    "MAX_CARD_BODY_BYTES",
    "ContinentCoverage",
    "MinimalCardSnapshot",
    "render_minimal_card",
]
