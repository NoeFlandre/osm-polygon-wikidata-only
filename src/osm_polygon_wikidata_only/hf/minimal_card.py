"""Shared compact renderer for the public Hugging Face dataset cards.

The release cards deliberately contain a small, stable summary.  Detailed
statistics remain machine-readable in ``stats.json``; this module only knows
how to render the common public presentation and never reads the dataset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
class SentenceCoverage:
    """Input-unit coverage for the sentence splitter."""

    eligible_units: int
    supported_units: int
    unsupported_units: int
    top_unsupported_languages: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if min(self.eligible_units, self.supported_units, self.unsupported_units) < 0:
            raise ValueError("sentence coverage counts must be non-negative")
        if self.supported_units + self.unsupported_units != self.eligible_units:
            raise ValueError("sentence coverage counts must add up to eligible units")

    @property
    def supported_percentage(self) -> float:
        return _percentage(self.supported_units, self.eligible_units)

    @property
    def unsupported_percentage(self) -> float:
        return _percentage(self.unsupported_units, self.eligible_units)


def continent_coverage_rows(
    rows: Iterable[tuple[str, int, int, int, int, int]],
) -> tuple[ContinentCoverage, ...]:
    """Adapt raw continent tuples into typed coverage rows.

    Both release cards read the same ``compute_continent_stats`` tuples, so the
    positional-to-named adaptation lives here rather than being repeated.
    """
    return tuple(
        ContinentCoverage(
            name=name,
            polygons=polygons,
            wikipedia_documents=wikipedia_documents,
            wikivoyage_documents=wikivoyage_documents,
            wikipedia_text_polygons=wikipedia_text_polygons,
            text_polygons=text_polygons,
        )
        for (
            name,
            polygons,
            wikipedia_documents,
            wikivoyage_documents,
            wikipedia_text_polygons,
            text_polygons,
        ) in rows
    )


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
    document_words: int | None = None
    sentence_rows: int | None = None
    sentence_coverage: SentenceCoverage | None = None
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
        f"| Document words | {_optional_integer(snapshot.document_words)} |",
        f"| Wikipedia + Wikivoyage sections | {_integer(snapshot.sections)} |",
        f"| Sentence rows | {_sentence_rows(snapshot.sentence_rows)} |",
        f"| Wikipedia + Wikivoyage languages | {_integer(snapshot.languages)} |",
        f"| Geographic regions | {_integer(snapshot.regions)} |",
        f"| Total Parquet size | {_size(snapshot.total_parquet_bytes)} |",
        "",
        "Polygon rows preserve regional records; identity and text metrics count each "
        "`(osm_type, osm_id)` once. Text requires `fetch_status=ok` and non-empty `full_text`.",
        "",
        "Document words count full Wikipedia and Wikivoyage document text; section rows "
        "are excluded. Sentence rows include split and explicitly unsplit unsupported-language "
        "rows when sentence sidecars exist.",
        "",
        *_sentence_coverage_lines(snapshot.sentence_coverage),
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
        *(
            _continent_line(row)
            for row in sorted(snapshot.continent_rows, key=lambda row: row.name)
        ),
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


def _optional_integer(value: int | None) -> str:
    return "Not available" if value is None else _integer(value)


def _sentence_rows(value: int | None) -> str:
    return "Not generated for this dataset version" if value is None else _integer(value)


def _sentence_coverage_lines(coverage: SentenceCoverage | None) -> list[str]:
    if coverage is None:
        return []
    top_languages = (
        "; ".join(
            f"`{_language_label(language)}` ({_integer(count)})"
            for language, count in coverage.top_unsupported_languages[:10]
        )
        or "None"
    )
    return [
        "## Sentence-splitting coverage",
        "",
        "Measured over input text units (sections), not output sentence rows.",
        f"- Eligible text units: {_integer(coverage.eligible_units)}",
        f"- Units split because language is supported: {_integer(coverage.supported_units)}",
        f"- Units left unsplit because language is unsupported: {_integer(coverage.unsupported_units)}",
        f"- Supported-language coverage: {coverage.supported_percentage:.1%}",
        f"- Unsupported-language share: {coverage.unsupported_percentage:.1%}",
        f"- Top unsupported languages by count: {top_languages}",
        "",
    ]


def _language_label(language: str) -> str:
    return language or "missing"


def _percentage(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _size(value: int) -> str:
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.1f} MB"
    return f"{value / 1_000_000_000:.1f} GB"


__all__ = [
    "MAX_CARD_BODY_BYTES",
    "ContinentCoverage",
    "MinimalCardSnapshot",
    "SentenceCoverage",
    "continent_coverage_rows",
    "render_minimal_card",
]
