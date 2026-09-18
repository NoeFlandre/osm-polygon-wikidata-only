"""Deterministic public dataset card and statistics for V2."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.hf._dataset_stats.rendering import demote_headings
from osm_polygon_wikidata_only.hf._geographic.polygon_identities import (
    PolygonIdentity,
    PolygonIndex,
    load_unique_polygon_records,
)
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import render_polygon_stats_section
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.comparison import (
    select_v2_added_wikipedia_tag_document_polygon_ids_from_files,
)
from osm_polygon_wikidata_only.v2.config import (
    V1_DATASET_URL,
    V2_ADDED_WIKIPEDIA_TAG_MAP_PATH,
    V2_CONTRACT_VERSION,
    V2_DATASET_CARD_VERSION,
    V2_GITHUB_URL,
    V2_REPO_ID,
    V2_TRACKIO_RUN_NAME,
    V2_TRACKIO_SPACE_ID,
)
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

_METADATA_READ_WORKERS = 4
_SUCCESSFUL_FETCH_STATUS = "ok"
_TEXT_DOCUMENT_PROJECTS = frozenset({"wikipedia", "wikivoyage"})


@dataclass(frozen=True, slots=True)
class V2CardStats:
    """Factual aggregate used by the V2 card and Trackio snapshot."""

    regions: int
    polygons: int
    unique_wikidata_entities: int
    wikipedia_documents: int
    wikipedia_sections: int
    wikivoyage_documents: int
    wikivoyage_sections: int
    wikidata_facts: int
    polygon_document_links: int
    wikipedia_tag_only_polygons: int
    document_words: int
    languages: int
    new_polygons_vs_v1: int | None
    new_wikipedia_documents_vs_v1: int | None
    text_coverage_funnel: tuple[tuple[str, int], ...]
    top_wikipedia_languages: tuple[tuple[str, int], ...]
    polygon_link_storage_bytes: int
    total_parquet_storage_bytes: int
    additional_document_words_vs_v1: int | None = None
    additional_sections_vs_v1: int | None = None
    new_polygons_wikipedia_tag_vs_v1: int | None = None
    new_polygons_wikidata_only_vs_v1: int | None = None
    new_wikipedia_tag_polygons_without_document: int | None = None
    new_wikipedia_document_identity_words_vs_v1: int | None = None
    new_wikipedia_documents_sharing_v1_content: int | None = None
    additional_unique_sections_vs_v1: int | None = None
    new_wikipedia_tag_document_polygons_vs_v1: int | None = None
    non_empty_text_polygons: int | None = None
    unique_polygon_identities: int | None = None
    sentence_stats: _SentenceCardStats | None = None

    @property
    def documents(self) -> int:
        return self.wikipedia_documents + self.wikivoyage_documents

    @property
    def other_wikipedia_languages(self) -> int:
        return max(
            0,
            self.wikipedia_documents - sum(value for _, value in self.top_wikipedia_languages),
        )


@dataclass(frozen=True, slots=True)
class _CardFiles:
    stems: tuple[str, ...]
    polygon_files: list[Path]
    wikipedia_document_files: list[Path]
    wikipedia_section_files: list[Path]
    wikivoyage_document_files: list[Path]
    wikivoyage_section_files: list[Path]
    wikidata_fact_files: list[Path]
    link_files: list[Path]
    parquet_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _CardMetrics:
    polygon_ids: set[str]
    document_ids: set[str]
    qids: set[str]
    languages: set[str]
    wikipedia_tag_only: int
    document_words: int
    wikipedia_section_count: int
    wikivoyage_section_count: int
    text_funnel: tuple[tuple[str, int], ...]
    top_languages: tuple[tuple[str, int], ...]
    polygon_row_count: int
    non_empty_text_polygon_count: int
    wikipedia_document_row_count: int
    wikivoyage_document_row_count: int
    wikidata_fact_row_count: int
    link_row_count: int
    unique_polygon_count: int


@dataclass(frozen=True, slots=True)
class _SentenceCardStats:
    """Data-derived counts for the optional sentence sidecars."""

    total_rows: int
    split_rows: int
    unsupported_rows: int
    polygon_count: int
    supported_language_count: int
    wikipedia_sidecars: int
    wikivoyage_sidecars: int


@dataclass(slots=True)
class _DocumentMetrics:
    """Metrics collected in one columnar pass over document files."""

    document_ids: set[str]
    languages: set[str]
    text_document_languages: dict[str, str]
    wikipedia_language_counts: Counter[str]
    non_empty_text_document_keys: set[tuple[str, str]] = field(default_factory=set)
    successful_text_document_languages: dict[tuple[str, str], str] = field(default_factory=dict)
    wikipedia_document_row_count: int = 0
    wikivoyage_document_row_count: int = 0
    document_words: int = 0


@dataclass(slots=True)
class _PolygonMetrics:
    """Metrics collected in one columnar pass over polygon files."""

    polygon_ids: set[str]
    qids: set[str]
    polygon_row_count: int = 0
    wikipedia_tag_only: int = 0


@dataclass(frozen=True, slots=True)
class _V1Baseline:
    polygon_ids: set[str]
    document_ids: set[str]
    wikipedia_document_files: list[Path]
    document_files: list[Path]
    document_words: int
    section_count: int


@dataclass(frozen=True, slots=True)
class _V1Comparison:
    new_polygons: int | None = None
    new_documents: int | None = None
    document_words: int | None = None
    sections: int | None = None
    wikipedia_tag_polygons: int | None = None
    wikidata_only_polygons: int | None = None
    tag_polygons_without_document: int | None = None
    document_identity_words: int | None = None
    documents_sharing_content: int | None = None
    unique_sections: int | None = None
    wikipedia_tag_document_polygons: int | None = None


def compute_v2_card_stats(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
) -> V2CardStats:
    """Compute V2 counts from live Parquet files and optional V1 files."""
    files = _collect_card_files(processed_v2)
    metrics = _compute_card_metrics(files)
    comparison = _compute_v1_comparison(v1_processed, files, metrics)
    sentence_stats = _compute_sentence_stats(processed_v2)
    return _build_card_stats(files, metrics, comparison, sentence_stats=sentence_stats)


def _collect_card_files(processed_v2: Path) -> _CardFiles:
    stems = tuple(sorted(load_v2_manifest(processed_v2)))
    polygon_files = _manifest_files(processed_v2 / "polygons", stems)
    wikipedia_document_files = _manifest_files(processed_v2 / "wikipedia/documents", stems)
    wikipedia_section_files = _manifest_files(processed_v2 / "wikipedia/sections", stems)
    wikivoyage_document_files = _manifest_files(processed_v2 / "wikivoyage/documents", stems)
    wikivoyage_section_files = _manifest_files(processed_v2 / "wikivoyage/sections", stems)
    wikidata_fact_files = _manifest_files(processed_v2 / "wikidata/facts", stems)
    link_files = _manifest_files(processed_v2 / "polygon_document_links", stems)
    parquet_files = (
        tuple(polygon_files)
        + tuple(wikipedia_document_files)
        + tuple(wikipedia_section_files)
        + tuple(wikivoyage_document_files)
        + tuple(wikivoyage_section_files)
        + tuple(wikidata_fact_files)
        + tuple(link_files)
    )
    return _CardFiles(
        stems=stems,
        polygon_files=polygon_files,
        wikipedia_document_files=wikipedia_document_files,
        wikipedia_section_files=wikipedia_section_files,
        wikivoyage_document_files=wikivoyage_document_files,
        wikivoyage_section_files=wikivoyage_section_files,
        wikidata_fact_files=wikidata_fact_files,
        link_files=link_files,
        parquet_files=parquet_files,
    )


def _compute_card_metrics(files: _CardFiles) -> _CardMetrics:
    document_files = files.wikipedia_document_files + files.wikivoyage_document_files
    document_metrics = _scan_document_metrics(
        document_files,
        files.wikipedia_document_files,
    )
    polygon_index = _load_polygon_index(files.polygon_files)
    text_funnel, top_languages = _text_metrics_from_scanned(
        document_metrics.successful_text_document_languages,
        document_metrics.wikipedia_language_counts,
        files.link_files,
        polygon_index,
    )
    polygon_metrics = _scan_polygon_metrics(files.polygon_files)
    non_empty_text_polygon_count = _count_linked_non_empty_text_polygons(
        files.link_files,
        document_metrics.non_empty_text_document_keys,
        polygon_index,
    )
    wikipedia_section_count = _sum_metadata(files.wikipedia_section_files)
    wikivoyage_section_count = _sum_metadata(files.wikivoyage_section_files)
    wikidata_fact_count = _sum_metadata(files.wikidata_fact_files)
    link_count = _sum_metadata(files.link_files)
    return _CardMetrics(
        polygon_ids=polygon_metrics.polygon_ids,
        document_ids=document_metrics.document_ids,
        qids=polygon_metrics.qids,
        languages=document_metrics.languages,
        wikipedia_tag_only=polygon_metrics.wikipedia_tag_only,
        document_words=document_metrics.document_words,
        wikipedia_section_count=wikipedia_section_count,
        wikivoyage_section_count=wikivoyage_section_count,
        text_funnel=(("All polygons", len(polygon_index.records)), *text_funnel[1:]),
        top_languages=top_languages,
        polygon_row_count=polygon_metrics.polygon_row_count,
        non_empty_text_polygon_count=non_empty_text_polygon_count,
        wikipedia_document_row_count=document_metrics.wikipedia_document_row_count,
        wikivoyage_document_row_count=document_metrics.wikivoyage_document_row_count,
        wikidata_fact_row_count=wikidata_fact_count,
        link_row_count=link_count,
        unique_polygon_count=len(polygon_index.records),
    )


def _load_polygon_index(paths: Iterable[Path]) -> PolygonIndex:
    """Load canonical polygon identities while retaining incomplete-test compatibility."""
    materialized = tuple(paths)
    if not materialized:
        return PolygonIndex(records={}, by_polygon_id={})
    if any("polygon_id" not in pq.read_schema(path).names for path in materialized):
        return PolygonIndex(records={}, by_polygon_id={})
    return load_unique_polygon_records(materialized)


def _load_v1_baseline(processed: Path) -> _V1Baseline:
    polygon_ids = set(
        _unique_values(sorted(processed.joinpath("polygons").glob("*.parquet")), "polygon_id")
    )
    wikipedia_document_files = _v1_wikipedia_document_files(processed)
    document_files = _v1_document_files(processed)
    document_ids = set(_unique_values(wikipedia_document_files, "document_id"))
    if not document_ids:
        document_ids = set(_unique_values(wikipedia_document_files, "article_id"))
    return _V1Baseline(
        polygon_ids=polygon_ids,
        document_ids=document_ids,
        wikipedia_document_files=wikipedia_document_files,
        document_files=document_files,
        document_words=_sum_first_available(
            document_files,
            ("article_length_words", "text_length_words"),
        ),
        section_count=_sum_metadata(_v1_section_files(processed)),
    )


def _compute_v1_comparison(
    v1_processed: Path | None,
    files: _CardFiles,
    metrics: _CardMetrics,
) -> _V1Comparison:
    if v1_processed is None:
        return _V1Comparison()
    baseline = _load_v1_baseline(v1_processed)
    polygon_comparison = _compare_polygon_sources(files, metrics, baseline)
    document_comparison = _compare_document_content(files, metrics, baseline)
    unique_sections = _compare_unique_sections(files, v1_processed)
    new_polygon_ids = metrics.polygon_ids - baseline.polygon_ids
    new_document_ids = metrics.document_ids - baseline.document_ids
    wikipedia_tag_document_polygons = len(
        select_v2_added_wikipedia_tag_document_polygon_ids_from_files(
            files.polygon_files,
            files.link_files,
            new_polygon_ids=new_polygon_ids,
            new_document_ids=new_document_ids,
        )
    )
    return _V1Comparison(
        new_polygons=len(metrics.polygon_ids - baseline.polygon_ids),
        new_documents=len(metrics.document_ids - baseline.document_ids),
        document_words=metrics.document_words - baseline.document_words,
        sections=(
            metrics.wikipedia_section_count
            + metrics.wikivoyage_section_count
            - baseline.section_count
        ),
        wikipedia_tag_polygons=polygon_comparison[0],
        wikidata_only_polygons=polygon_comparison[1],
        tag_polygons_without_document=polygon_comparison[2],
        document_identity_words=document_comparison[0],
        documents_sharing_content=document_comparison[1],
        unique_sections=unique_sections,
        wikipedia_tag_document_polygons=wikipedia_tag_document_polygons,
    )


def _compare_polygon_sources(
    files: _CardFiles,
    metrics: _CardMetrics,
    baseline: _V1Baseline,
) -> tuple[int, int, int]:
    source_sets = _polygon_source_sets(
        files.polygon_files,
        metrics.polygon_ids - baseline.polygon_ids,
    )
    wikipedia_tag = sum("wikipedia_tag" in sources for sources in source_sets.values())
    wikidata_only = sum(sources == {"wikidata"} for sources in source_sets.values())
    linked_polygon_ids = _polygon_ids_with_link_source(files.link_files, "osm_wikipedia_tag")
    without_document = sum(
        "wikipedia_tag" in sources and polygon_id not in linked_polygon_ids
        for polygon_id, sources in source_sets.items()
    )
    return wikipedia_tag, wikidata_only, without_document


def _compare_document_content(
    files: _CardFiles,
    metrics: _CardMetrics,
    baseline: _V1Baseline,
) -> tuple[int, int]:
    v2_words = _unique_numeric_values(
        files.wikipedia_document_files,
        "document_id",
        ("article_length_words", "text_length_words"),
    )
    v1_words = _v1_document_words_by_id(baseline.wikipedia_document_files)
    new_identity_words = _new_identity_words(v2_words, v1_words)
    v1_content_hashes = _unique_values(baseline.wikipedia_document_files, "content_hash")
    new_ids = set(v2_words) - set(v1_words)
    v2_new_hashes = _field_values_for_ids(
        files.wikipedia_document_files,
        "document_id",
        "content_hash",
        new_ids,
    )
    sharing_content = _shared_content_count(v2_new_hashes, v1_content_hashes)
    return new_identity_words, sharing_content


def _v1_document_words_by_id(paths: list[Path]) -> dict[str, int]:
    words = _unique_numeric_values(
        paths,
        "document_id",
        ("article_length_words", "text_length_words"),
    )
    if words:
        return words
    return _unique_numeric_values(
        paths,
        "article_id",
        ("article_length_words", "text_length_words"),
    )


def _new_identity_words(v2_words: dict[str, int], v1_words: dict[str, int]) -> int:
    return sum(value for identity, value in v2_words.items() if identity not in v1_words)


def _shared_content_count(
    new_hashes: dict[str, str],
    v1_hashes: set[str],
) -> int:
    return sum(content_hash in v1_hashes for content_hash in new_hashes.values() if content_hash)


def _compare_unique_sections(files: _CardFiles, v1_processed: Path) -> int:
    v2_section_ids = _unique_values(
        files.wikipedia_section_files + files.wikivoyage_section_files,
        "section_id",
    )
    v1_section_ids = _unique_values(_v1_section_files(v1_processed), "section_id")
    return len(v2_section_ids - v1_section_ids)


def _build_card_stats(
    files: _CardFiles,
    metrics: _CardMetrics,
    comparison: _V1Comparison,
    *,
    sentence_stats: _SentenceCardStats | None = None,
) -> V2CardStats:
    return V2CardStats(
        regions=len(files.stems),
        polygons=metrics.polygon_row_count,
        unique_wikidata_entities=len(metrics.qids),
        wikipedia_documents=metrics.wikipedia_document_row_count,
        wikipedia_sections=metrics.wikipedia_section_count,
        wikivoyage_documents=metrics.wikivoyage_document_row_count,
        wikivoyage_sections=metrics.wikivoyage_section_count,
        wikidata_facts=metrics.wikidata_fact_row_count,
        polygon_document_links=metrics.link_row_count,
        wikipedia_tag_only_polygons=metrics.wikipedia_tag_only,
        document_words=metrics.document_words,
        languages=len(metrics.languages),
        new_polygons_vs_v1=comparison.new_polygons,
        new_wikipedia_documents_vs_v1=comparison.new_documents,
        text_coverage_funnel=metrics.text_funnel,
        top_wikipedia_languages=metrics.top_languages,
        polygon_link_storage_bytes=sum(path.stat().st_size for path in files.link_files),
        total_parquet_storage_bytes=sum(path.stat().st_size for path in files.parquet_files),
        additional_document_words_vs_v1=comparison.document_words,
        additional_sections_vs_v1=comparison.sections,
        new_polygons_wikipedia_tag_vs_v1=comparison.wikipedia_tag_polygons,
        new_polygons_wikidata_only_vs_v1=comparison.wikidata_only_polygons,
        new_wikipedia_tag_polygons_without_document=comparison.tag_polygons_without_document,
        new_wikipedia_document_identity_words_vs_v1=comparison.document_identity_words,
        new_wikipedia_documents_sharing_v1_content=comparison.documents_sharing_content,
        additional_unique_sections_vs_v1=comparison.unique_sections,
        new_wikipedia_tag_document_polygons_vs_v1=comparison.wikipedia_tag_document_polygons,
        non_empty_text_polygons=metrics.non_empty_text_polygon_count,
        unique_polygon_identities=metrics.unique_polygon_count,
        sentence_stats=sentence_stats,
    )


def render_v2_card(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
    stats: V2CardStats | None = None,
    generated_on: str | None = None,
) -> str:
    """Render a concise, viewer-compatible card from V2 files on disk."""
    snapshot = stats or compute_v2_card_stats(processed_v2, v1_processed=v1_processed)
    front_matter = _render_front_matter(snapshot, processed_v2=processed_v2)
    comparison = _render_comparison(snapshot)
    generated_block = [f"Generated on {generated_on}.", ""] if generated_on else []
    return (
        front_matter
        + "\n"
        + "\n".join(
            [
                "![NoeFlandre/osm-polygon-wikidata-and-wikipedia dataset overview](assets/dataset_hero.png)",
                "",
                "# OSM Polygon Wikidata + Wikipedia, V2",
                "",
                f"V2 builds on the [V1 Wikidata-only dataset]({V1_DATASET_URL}), which retained "
                f"OSM polygons carrying `wikidata=*` and enriched them with multilingual "
                f"Wikipedia and Wikivoyage text; V2 adds valid multilingual `wikipedia=*` "
                f"references, including polygons without a Wikidata QID. "
                f"The code is maintained in the [GitHub repository]({V2_GITHUB_URL}).",
                "",
                *generated_block,
                f"The public V2 Trackio snapshot is [`{V2_TRACKIO_RUN_NAME}`](https://huggingface.co/spaces/{V2_TRACKIO_SPACE_ID}).",
                "",
                "## Dataset snapshot",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Polygon rows across regional extracts | {snapshot.polygons:,} |",
                f"| Unique polygon identities (osm_type, osm_id) | {_unique_polygon_count(snapshot):,} |",
                f"| Polygons with successful non-empty text (unique OSM identities) | {_non_empty_text_polygon_count(snapshot):,} |",
                f"| Wikipedia documents | {snapshot.wikipedia_documents:,} |",
                f"| Wikipedia sections | {snapshot.wikipedia_sections:,} |",
                f"| Wikipedia + Wikivoyage languages | {snapshot.languages:,} |",
                f"| Geographic regions | {snapshot.regions:,} |",
                f"| Total Parquet size | {snapshot.total_parquet_storage_bytes / 1_000_000_000:.1f} GB |",
                "",
                "Polygon rows retain regional copies. Identity metrics use one deterministic representative per `(osm_type, osm_id)`; successful text additionally requires `fetch_status=ok` and trimmed non-empty `full_text`.",
                "",
                "<details>",
                "<summary>Full snapshot</summary>",
                "",
                f"- **Hugging Face dataset:** [{V2_REPO_ID}](https://huggingface.co/datasets/{V2_REPO_ID})",
                f"- **Unique Wikidata entities:** {snapshot.unique_wikidata_entities:,}",
                f"- **Wikivoyage documents:** {snapshot.wikivoyage_documents:,}",
                f"- **Wikipedia + Wikivoyage documents:** {snapshot.documents:,}",
                f"- **Wikivoyage sections:** {snapshot.wikivoyage_sections:,}",
                f"- **Wikidata facts:** {snapshot.wikidata_facts:,}",
                f"- **Polygon-document links:** {snapshot.polygon_document_links:,}",
                f"- **Wikipedia-tag-only polygons:** {snapshot.wikipedia_tag_only_polygons:,}",
                f"- **Document words:** {snapshot.document_words:,}",
                f"- **Polygon/link-table storage:** {snapshot.polygon_link_storage_bytes / 1_000_000_000:.1f} GB",
                "",
                "</details>",
                "",
                comparison,
                "<details>",
                "<summary>Polygon surface and geometry</summary>",
                "",
                demote_headings(render_polygon_stats_section(processed_v2).strip()),
                "",
                "</details>",
                "",
                "## Coverage maps",
                "",
                "### All V2 dataset polygons",
                "",
                "![All V2 dataset polygons](assets/coverage_map.png)",
                "",
                "Every point is one globally unique retained V2 `(osm_type, osm_id)` identity, including identities without text. Regional polygon rows remain separate source/provenance records.",
                "",
                "### Polygons with Wikipedia or Wikivoyage text",
                "",
                "![V2 polygons with text](assets/geographic_text_presence.png)",
                "",
                "Each point is one globally unique `(osm_type, osm_id)` identity linked to at least one successfully extracted (`fetch_status=ok`) Wikipedia or Wikivoyage document with trimmed non-empty `full_text`. Overlapping regional rows and multiple qualifying documents count once.",
                "",
                "### H3 density of polygons with text",
                "",
                "![V2 geographic text density](assets/geographic_text_density.png)",
                "",
                "Each H3 cell shows the raw count of unique V2 `(osm_type, osm_id)` identities with successfully extracted (`fetch_status=ok`) non-empty Wikipedia or Wikivoyage text. Colour uses a logarithmic scale; it is not a proportion.",
                "",
                "## Deduplication and provenance",
                "",
                "V2 deduplicates documents by `document_id` and polygon-document links by `(polygon_id, project, document_id)` within each region. Byte-identical repeats collapse deterministically; conflicting rows fail closed. `discovery_sources` explains how a polygon was included: `wikidata` means the polygon came from an OSM `wikidata=*` tag, while `wikipedia_tag` means it came from an OSM `wikipedia=*` tag. `link_sources` explains each polygon-document relationship: `wikidata_sitelink` means the relationship came from a Wikidata sitelink, while `osm_wikipedia_tag` means it came directly from an OSM `wikipedia=*` tag. A relationship can list both when both routes agree.",
                "",
                "Regional extracts can overlap, so the same OSM object or document may appear in more than one regional file. We keep those copies to preserve regional membership and provenance. Row-based snapshot, document, link, and storage metrics retain those regional copies; map points, text funnels, and text-covered card metrics use one deterministic representative per global `(osm_type, osm_id)` identity.",
                "",
                "## Sentence-level text",
                "",
                *_sentence_section_lines(snapshot.sentence_stats),
                "",
                "## Repository layout",
                "",
                "- `polygons/<stem>.parquet` — one row per retained OSM polygon.",
                "- `wikipedia/documents/<stem>.parquet` and `wikipedia/sections/<stem>.parquet` — multilingual Wikipedia documents and their sections.",
                "  Wikipedia sections retain the exact V1 22-column section schema for lossless reuse.",
                "- `wikivoyage/documents/<stem>.parquet` and `wikivoyage/sections/<stem>.parquet` — Wikivoyage documents and sections reused from V1 where available.",
                "- `wikipedia/sentences/<stem>.parquet` and `wikivoyage/sentences/<stem>.parquet` — optional sentence rows with explicit split/unsplit provenance.",
                "- `polygon_document_links/<stem>.parquet` — unified Wikipedia and Wikivoyage polygon links.",
                "- `wikidata/facts/<stem>.parquet` — structured Wikidata facts.",
                "- `stats.json` — the complete machine-readable polygon surface and geometry report.",
                "",
                "## Reproducibility",
                "",
                "Run V2 explicitly with `sync-dir --dataset-version v2`. V1 remains a separate contract and is not modified by V2 processing.",
                "",
                "## Citation",
                "",
                "If you use this dataset, please cite it. Download the dataset citation metadata from [`CITATION.cff`](CITATION.cff).",
                "",
            ]
        )
    )


def write_v2_card(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
    stats: V2CardStats | None = None,
    generated_on: str | None = None,
) -> Path:
    """Write the deterministic V2 card atomically and return its path."""
    path = processed_v2 / "README.md"
    atomic_write_text(
        path,
        render_v2_card(
            processed_v2,
            v1_processed=v1_processed,
            stats=stats,
            generated_on=generated_on,
        ),
    )
    return path


def _render_front_matter(snapshot: V2CardStats, *, processed_v2: Path) -> str:
    configs = [
        ("polygons", "polygons", "polygons/*.parquet"),
        ("polygon_document_links", "polygon_document_links", "polygon_document_links/*.parquet"),
        ("wikipedia_documents", "wikipedia_documents", "wikipedia/documents/*.parquet"),
        ("wikipedia_sections", "wikipedia_sections", "wikipedia/sections/*.parquet"),
        ("wikivoyage_documents", "wikivoyage_documents", "wikivoyage/documents/*.parquet"),
        ("wikivoyage_sections", "wikivoyage_sections", "wikivoyage/sections/*.parquet"),
        ("wikidata_facts", "wikidata_facts", "wikidata/facts/*.parquet"),
    ]
    if _has_parquet(processed_v2 / "wikipedia/sentences"):
        configs.append(
            ("wikipedia_sentences", "wikipedia_sentences", "wikipedia/sentences/*.parquet")
        )
    if _has_parquet(processed_v2 / "wikivoyage/sentences"):
        configs.append(
            ("wikivoyage_sentences", "wikivoyage_sentences", "wikivoyage/sentences/*.parquet")
        )
    lines = [
        "---",
        "license: odbl",
        "language:",
        "  - en",
        "tags:",
        "  - openstreetmap",
        "  - wikidata",
        "  - wikipedia",
        "  - wikivoyage",
        "  - geospatial",
        "configs:",
    ]
    for config_name, split, path in configs:
        lines.extend(
            [
                f"  - config_name: {config_name}",
                "    data_files:",
                f"      - split: {split}",
                f"        path: {path}",
            ]
        )
    lines.extend(
        [
            "dataset_info:",
            f"  version: {V2_DATASET_CARD_VERSION}",
            f"  regions: {snapshot.regions}",
            f"  polygons: {snapshot.polygons}",
            f"  documents: {snapshot.documents}",
            f"dataset_contract: {V2_CONTRACT_VERSION}",
            "---",
        ]
    )
    return "\n".join(lines) + "\n"


def _has_parquet(directory: Path) -> bool:
    return any(directory.glob("*.parquet"))


def _compute_sentence_stats(processed_v2: Path) -> _SentenceCardStats | None:
    sentence_paths = {
        project: sorted((processed_v2 / project / "sentences").glob("*.parquet"))
        for project in ("wikipedia", "wikivoyage")
    }
    manifest_path = processed_v2 / "manifests" / "sentence_splitting.json"
    if not manifest_path.is_file() or not any(sentence_paths.values()):
        return None

    regions, supported_languages = _load_sentence_manifest(manifest_path)
    total_rows, unsupported_rows = _sentence_manifest_totals(regions, manifest_path)
    document_ids = {
        project: _sentence_document_ids(paths) for project, paths in sentence_paths.items()
    }
    return _SentenceCardStats(
        total_rows=total_rows,
        split_rows=total_rows - unsupported_rows,
        unsupported_rows=unsupported_rows,
        polygon_count=_sentence_polygon_count(processed_v2, document_ids),
        supported_language_count=len(supported_languages),
        wikipedia_sidecars=len(sentence_paths["wikipedia"]),
        wikivoyage_sidecars=len(sentence_paths["wikivoyage"]),
    )


def _load_sentence_manifest(manifest_path: Path) -> tuple[list[object], list[object]]:
    raw_manifest = json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, Mapping):
        raise ValueError(f"Invalid sentence manifest: {manifest_path}")
    regions = raw_manifest.get("regions")
    supported_languages = raw_manifest.get("supported_languages")
    if not isinstance(regions, list) or not isinstance(supported_languages, list):
        raise ValueError(f"Invalid sentence manifest: {manifest_path}")
    return cast(list[object], regions), cast(list[object], supported_languages)


def _sentence_manifest_totals(regions: Iterable[object], manifest_path: Path) -> tuple[int, int]:
    total_rows = 0
    unsupported_rows = 0
    for region in regions:
        region_total_rows, region_unsupported_rows = _sentence_region_totals(region, manifest_path)
        total_rows += region_total_rows
        unsupported_rows += region_unsupported_rows
    if unsupported_rows > total_rows:
        raise ValueError(f"Invalid sentence row totals: {manifest_path}")
    return total_rows, unsupported_rows


def _sentence_region_totals(region: object, manifest_path: Path) -> tuple[int, int]:
    if not isinstance(region, Mapping):
        raise ValueError(f"Invalid sentence manifest region: {manifest_path}")
    values = cast(Mapping[str, object], region)
    # Keep the historical permissive int coercion for legacy JSON values.
    return int(cast(Any, values.get("sentence_rows", 0))), int(
        cast(Any, values.get("unsplit_sections", 0))
    )


def _sentence_document_ids(paths: Iterable[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        _update_sentence_document_ids(path, values)
    return values


def _update_sentence_document_ids(path: Path, values: set[str]) -> None:
    with pq.ParquetFile(path) as parquet_file:
        if "document_id" not in parquet_file.schema_arrow.names:
            return
        for batch in parquet_file.iter_batches(columns=["document_id"], batch_size=65_536):
            values.update(_sentence_document_ids_batch(batch))


def _sentence_document_ids_batch(batch: pa.RecordBatch) -> set[str]:
    return {str(value) for value in _compute_array("unique", batch.column(0)).to_pylist() if value}


def _sentence_polygon_count(
    processed_v2: Path,
    document_ids: Mapping[str, set[str]],
) -> int:
    value_sets = {
        project: pa.array(sorted(values), type=pa.string())
        for project, values in document_ids.items()
        if values
    }
    if not value_sets:
        return 0

    polygon_index = _load_polygon_index(sorted((processed_v2 / "polygons").glob("*.parquet")))
    polygon_ids: set[PolygonIdentity] = set()
    for path in sorted((processed_v2 / "polygon_document_links").glob("*.parquet")):
        _collect_sentence_polygon_ids(path, value_sets, polygon_index, polygon_ids)
    return len(polygon_ids)


def _collect_sentence_polygon_ids(
    path: Path,
    value_sets: Mapping[str, pa.Array],
    polygon_index: PolygonIndex,
    polygon_ids: set[PolygonIdentity],
) -> None:
    with pq.ParquetFile(path) as parquet_file:
        columns = {"polygon_id", "document_id", "project"}
        if not columns.issubset(parquet_file.schema_arrow.names):
            return
        for batch in parquet_file.iter_batches(
            columns=["polygon_id", "document_id", "project"],
            batch_size=65_536,
        ):
            polygon_ids.update(_sentence_polygon_ids_from_batch(batch, value_sets, polygon_index))


def _sentence_polygon_ids_from_batch(
    batch: pa.RecordBatch,
    value_sets: Mapping[str, pa.Array],
    polygon_index: PolygonIndex,
) -> set[PolygonIdentity]:
    polygon_column, document_column, project_column = batch.columns
    polygon_ids: set[PolygonIdentity] = set()
    for project, value_set in value_sets.items():
        matches = _compute_array(
            "and",
            _compute_array("equal", project_column, project),
            _compute_array(
                "is_in",
                document_column,
                options=pc.SetLookupOptions(value_set),
            ),
        )
        polygon_ids.update(
            polygon_identity
            for value in _compute_array("filter", polygon_column, matches).to_pylist()
            if value
            and (polygon_identity := polygon_index.by_polygon_id.get(str(value))) is not None
        )
    return polygon_ids


def _compute_array(function: str, *arguments: Any, options: Any = None) -> Any:
    return pc.call_function(function, list(arguments), options=options)


def _unique_polygon_count(snapshot: V2CardStats) -> int:
    if snapshot.unique_polygon_identities is not None:
        return snapshot.unique_polygon_identities
    return snapshot.polygons


def _non_empty_text_polygon_count(snapshot: V2CardStats) -> int:
    if snapshot.non_empty_text_polygons is not None:
        return snapshot.non_empty_text_polygons
    return dict(snapshot.text_coverage_funnel).get("With non-empty text", 0)


def _sentence_section_lines(stats: _SentenceCardStats | None) -> tuple[str, ...]:
    lines = (
        "Sentence sidecars are opt-in and use `sat-3l-sm` only for the exact ISO codes listed in `docs/sentence-splitting.md` in the [source repository](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/docs/sentence-splitting.md). Any other language code remains one unsplit row with `segmentation_status=unsupported_language`; it is never passed to SaT.",
    )
    if stats is not None:
        lines += (
            f"Data-derived totals: {stats.split_rows:,} split sentence rows plus {stats.unsupported_rows:,} unsupported-language rows retained unsplit = {stats.total_rows:,} total rows; {stats.supported_language_count:,} supported language codes; {stats.polygon_count:,} polygons linked to sentence-sidecar documents across {stats.wikipedia_sidecars:,} Wikipedia and {stats.wikivoyage_sidecars:,} Wikivoyage sidecars.",
        )
    return (
        *lines,
        "When generated, sentence rows are stored in `wikipedia/sentences/<stem>.parquet` and `wikivoyage/sentences/<stem>.parquet`; `manifests/sentence_splitting.json` records the model, revision, and observed routing.",
    )


def _render_comparison(snapshot: V2CardStats) -> str:
    lines = ["## V2 compared with V1", ""]
    if any(
        value is None
        for value in (
            snapshot.new_polygons_vs_v1,
            snapshot.new_wikipedia_documents_vs_v1,
            snapshot.additional_document_words_vs_v1,
            snapshot.additional_sections_vs_v1,
            snapshot.new_polygons_wikipedia_tag_vs_v1,
            snapshot.new_polygons_wikidata_only_vs_v1,
            snapshot.new_wikipedia_tag_polygons_without_document,
            snapshot.new_wikipedia_document_identity_words_vs_v1,
            snapshot.new_wikipedia_documents_sharing_v1_content,
            snapshot.additional_unique_sections_vs_v1,
            snapshot.new_wikipedia_tag_document_polygons_vs_v1,
        )
    ):
        lines.append(
            "The local V1 artifact root was not supplied for this card render, so the delta is not estimated."
        )
    else:
        lines.extend(
            [
                "The following deltas are computed from the local V1 and V2 Parquet snapshots:",
                "",
                f"- **Additional polygon identities:** {snapshot.new_polygons_vs_v1:,}",
                f"- **Of those, polygons with a Wikipedia tag:** {snapshot.new_polygons_wikipedia_tag_vs_v1:,}",
                f"- **Of those, Wikidata-only polygons:** {snapshot.new_polygons_wikidata_only_vs_v1:,}",
                f"- **Additional Wikipedia document identities:** {snapshot.new_wikipedia_documents_vs_v1:,}",
                f"- **Additional document-row words in V2 (Wikipedia + Wikivoyage):** {snapshot.additional_document_words_vs_v1:,}",
                f"- **Words in newly added Wikipedia document identities:** {snapshot.new_wikipedia_document_identity_words_vs_v1:,}",
                f"- **New Wikipedia document identities sharing content with V1:** {snapshot.new_wikipedia_documents_sharing_v1_content:,}",
                f"- **Additional section rows in V2 (Wikipedia + Wikivoyage):** {snapshot.additional_sections_vs_v1:,}",
                f"- **Additional unique section identities:** {snapshot.additional_unique_sections_vs_v1:,}",
                f"- **New Wikipedia-tag polygons without a matching page at the snapshot:** {snapshot.new_wikipedia_tag_polygons_without_document:,}",
                f"- **V2-added polygons with a new Wikipedia-tag document and no Wikidata discovery:** {snapshot.new_wikipedia_tag_document_polygons_vs_v1:,}",
                "",
                "### V2-added polygons with Wikipedia-tag documents",
                "",
                f"![V2-added polygons with Wikipedia-tag documents]({V2_ADDED_WIKIPEDIA_TAG_MAP_PATH})",
                "",
                "This map shows only polygon identities absent from V1 whose discovery provenance is exactly `wikipedia_tag` and which link to at least one Wikipedia document identity new in V2 through `osm_wikipedia_tag`.",
                "",
                "The polygon and document figures are set differences of stable identities, while row-word and row-section figures include regional copies. V2 keeps regional copies to preserve source membership and provenance; a direct Wikipedia reference without a matching page remains represented in the polygon table and is not counted as a document.",
                "",
            ]
        )
    return "\n".join(lines)


def _v1_wikipedia_document_files(processed: Path) -> list[Path]:
    wikipedia = sorted((processed / "wikipedia/documents").glob("*.parquet"))
    if not wikipedia:
        wikipedia = sorted((processed / "articles").glob("*.parquet"))
    return wikipedia


def _v1_document_files(processed: Path) -> list[Path]:
    return _v1_wikipedia_document_files(processed) + sorted(
        (processed / "wikivoyage/documents").glob("*.parquet")
    )


def _v1_section_files(processed: Path) -> list[Path]:
    return sorted((processed / "wikipedia/sections").glob("*.parquet")) + sorted(
        (processed / "wikivoyage/sections").glob("*.parquet")
    )


def _manifest_files(directory: Path, stems: Iterable[str]) -> list[Path]:
    return [path for stem in stems if (path := directory / f"{stem}.parquet").is_file()]


def _sum_metadata(paths: Iterable[Path]) -> int:
    materialized = tuple(paths)
    if not materialized:
        return 0
    workers = min(_METADATA_READ_WORKERS, len(materialized))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="v2-card-metadata",
    ) as executor:
        return sum(executor.map(_metadata_row_count, materialized))


def _metadata_row_count(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def _unique_values(paths: Iterable[Path], column: str) -> set[str]:
    values: set[str] = set()
    for path in paths:
        values.update(_unique_values_file(path, column))
    return values


def _unique_values_file(path: Path, column: str) -> set[str]:
    values: set[str] = set()
    with pq.ParquetFile(path) as parquet_file:
        if column not in parquet_file.schema_arrow.names:
            return values
        for batch in parquet_file.iter_batches(columns=[column], batch_size=65_536):
            values.update(_non_empty_strings(batch.column(0).to_pylist()))
    return values


def _sum_first_available(paths: Iterable[Path], columns: tuple[str, ...]) -> int:
    return sum(_sum_first_available_file(path, columns) for path in paths)


def _sum_first_available_file(path: Path, columns: tuple[str, ...]) -> int:
    with pq.ParquetFile(path) as parquet_file:
        column = _first_present_column(set(parquet_file.schema_arrow.names), columns)
        if column is None:
            return 0
        return sum(
            sum(int(value or 0) for value in batch.column(0).to_pylist())
            for batch in parquet_file.iter_batches(columns=[column], batch_size=65_536)
        )


def _first_present_column(names: set[str], columns: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in columns if candidate in names), None)


def _unique_numeric_values(
    paths: Iterable[Path], key_column: str, value_columns: tuple[str, ...]
) -> dict[str, int]:
    """Return one stable numeric value for each document identity."""
    values: dict[str, int] = {}
    for path in paths:
        _merge_numeric_file(values, path, key_column, value_columns)
    return values


def _merge_numeric_file(
    values: dict[str, int],
    path: Path,
    key_column: str,
    value_columns: tuple[str, ...],
) -> None:
    with pq.ParquetFile(path) as parquet_file:
        names = set(parquet_file.schema_arrow.names)
        value_column = _first_present_column(names, value_columns)
        if key_column not in names or value_column is None:
            return
        _merge_numeric_batches(values, parquet_file, key_column, value_column)


def _merge_numeric_batches(
    values: dict[str, int],
    parquet_file: Any,
    key_column: str,
    value_column: str,
) -> None:
    for batch in parquet_file.iter_batches(columns=[key_column, value_column], batch_size=65_536):
        _merge_numeric_batch(values, batch, value_column)


def _merge_numeric_batch(values: dict[str, int], batch: Any, value_column: str) -> None:
    for identity, value in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        _record_numeric_value(values, identity, value, value_column)


def _record_numeric_value(
    values: dict[str, int],
    identity: Any,
    value: Any,
    value_column: str,
) -> None:
    if identity in (None, ""):
        return
    key = str(identity)
    numeric = int(value or 0)
    previous = values.get(key)
    if previous is not None and previous != numeric:
        raise ValueError(f"Inconsistent {value_column} for document {key!r}")
    values[key] = numeric


def _field_values_for_ids(
    paths: Iterable[Path],
    key_column: str,
    value_column: str,
    identities: set[str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    if not identities:
        return values
    for path in paths:
        _merge_field_values_file(values, path, key_column, value_column, identities)
    return values


def _merge_field_values_file(
    values: dict[str, str],
    path: Path,
    key_column: str,
    value_column: str,
    identities: set[str],
) -> None:
    with pq.ParquetFile(path) as parquet_file:
        if not {key_column, value_column}.issubset(parquet_file.schema_arrow.names):
            return
        for batch in parquet_file.iter_batches(
            columns=[key_column, value_column], batch_size=65_536
        ):
            _merge_field_values_batch(values, batch, identities)


def _merge_field_values_batch(
    values: dict[str, str],
    batch: Any,
    identities: set[str],
) -> None:
    for identity, value in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        if identity in identities and value not in (None, ""):
            values[str(identity)] = str(value)


def _polygon_source_sets(paths: Iterable[Path], identities: set[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for path in paths:
        values.update(_polygon_source_file(path, identities))
    return values


def _polygon_source_file(path: Path, identities: set[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    with pq.ParquetFile(path) as parquet_file:
        if not {"polygon_id", "discovery_sources"}.issubset(parquet_file.schema_arrow.names):
            return values
        for batch in parquet_file.iter_batches(
            columns=["polygon_id", "discovery_sources"], batch_size=65_536
        ):
            _merge_polygon_sources(values, batch, identities, path)
    return values


def _merge_polygon_sources(
    values: dict[str, set[str]],
    batch: Any,
    identities: set[str],
    path: Path,
) -> None:
    for identity, raw_sources in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        if identity not in identities:
            continue
        parsed = _parse_source_list(raw_sources, identity, path, "discovery_sources")
        values[str(identity)] = set(parsed)


def _parse_source_list(
    raw_sources: Any,
    identity: Any,
    path: Path,
    field: str,
) -> list[str]:
    try:
        parsed = json_loads(str(raw_sources or "[]"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for polygon {identity!r} in {path}") from exc
    return _validated_source_list(parsed, identity, field)


def _validated_source_list(parsed: Any, identity: Any, field: str) -> list[str]:
    if not isinstance(parsed, list):
        raise ValueError(f"Invalid {field} for polygon {identity!r}")
    if not all(isinstance(source, str) for source in parsed):
        raise ValueError(f"Invalid {field} for polygon {identity!r}")
    return parsed


def _polygon_ids_with_link_source(paths: Iterable[Path], source: str) -> set[str]:
    values: set[str] = set()
    for path in paths:
        values.update(_link_source_file(path, source))
    return values


def _link_source_file(path: Path, source: str) -> set[str]:
    values: set[str] = set()
    with pq.ParquetFile(path) as parquet_file:
        if not {"polygon_id", "link_sources"}.issubset(parquet_file.schema_arrow.names):
            return values
        for batch in parquet_file.iter_batches(
            columns=["polygon_id", "link_sources"], batch_size=65_536
        ):
            _merge_link_sources(values, batch, source, path)
    return values


def _merge_link_sources(values: set[str], batch: Any, source: str, path: Path) -> None:
    for identity, raw_sources in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        parsed = _parse_source_list(raw_sources, identity, path, "link_sources")
        if identity and source in parsed:
            values.add(str(identity))


def _text_metrics_from_scanned(
    document_languages: dict[tuple[str, str], str],
    wikipedia_language_counts: Counter[str],
    link_paths: Iterable[Path],
    polygon_index: PolygonIndex,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Build text metrics from document columns already scanned once."""
    languages_by_polygon = _polygon_languages(link_paths, document_languages, polygon_index)
    return (
        _text_funnel(languages_by_polygon),
        tuple(wikipedia_language_counts.most_common(10)),
    )


def _scan_document_metrics(
    all_document_paths: Iterable[Path],
    wikipedia_document_paths: Iterable[Path],
) -> _DocumentMetrics:
    """Collect document identities, languages, words, and text routing once."""
    metrics = _DocumentMetrics(set(), set(), {}, Counter())
    wikipedia_paths = set(wikipedia_document_paths)
    for path in all_document_paths:
        _scan_document_file(path, is_wikipedia=path in wikipedia_paths, metrics=metrics)
    return metrics


def _scan_document_file(path: Path, *, is_wikipedia: bool, metrics: _DocumentMetrics) -> None:
    with pq.ParquetFile(path) as parquet_file:
        metadata = parquet_file.metadata
        row_count = 0 if metadata is None else int(metadata.num_rows)
        if is_wikipedia:
            metrics.wikipedia_document_row_count += row_count
        else:
            metrics.wikivoyage_document_row_count += row_count
        names = set(parquet_file.schema_arrow.names)
        word_column = _word_column(names)
        has_document_id = "document_id" in names
        has_language = "language" in names
        columns = _document_columns(
            names,
            word_column=word_column,
            has_document_id=has_document_id,
        )
        if not columns:
            return
        _scan_document_batches(
            parquet_file,
            columns=columns,
            is_wikipedia=is_wikipedia,
            has_document_id=has_document_id,
            has_language=has_language,
            word_column=word_column,
            metrics=metrics,
        )


def _document_columns(
    names: set[str],
    *,
    word_column: str | None,
    has_document_id: bool,
) -> list[str]:
    columns = _document_metric_columns(
        names, word_column=word_column, has_document_id=has_document_id
    )
    if has_document_id:
        columns.extend(column for column in ("fetch_status", "full_text") if column in names)
    return columns


def _document_metric_columns(
    names: set[str],
    *,
    word_column: str | None,
    has_document_id: bool,
) -> list[str]:
    if has_document_id and word_column is not None:
        # Keep the historical failure for a text-bearing document file
        # without a language column: the old text scan requested it too.
        return ["document_id", "language", word_column]
    return [column for column in _document_column_names(word_column) if column in names]


def _document_column_names(word_column: str | None) -> tuple[str, ...]:
    return (
        ("document_id", "language")
        if word_column is None
        else (
            "document_id",
            "language",
            word_column,
        )
    )


def _scan_document_batches(
    parquet_file: pq.ParquetFile,
    *,
    columns: list[str],
    is_wikipedia: bool,
    has_document_id: bool,
    has_language: bool,
    word_column: str | None,
    metrics: _DocumentMetrics,
) -> None:
    positions = {column: index for index, column in enumerate(columns)}
    for batch in parquet_file.iter_batches(columns=columns, batch_size=65_536):
        _scan_document_batch(
            batch,
            positions=positions,
            is_wikipedia=is_wikipedia,
            has_document_id=has_document_id,
            has_language=has_language,
            word_column=word_column,
            metrics=metrics,
        )


def _document_batch_columns(
    batch: Any,
    positions: dict[str, int],
    *,
    has_document_id: bool,
    has_language: bool,
    word_column: str | None,
) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
    return (
        _batch_column(batch, positions, "document_id" if has_document_id else None),
        _batch_column(batch, positions, "language" if has_language else None),
        _batch_column(batch, positions, word_column),
    )


def _batch_column(
    batch: Any,
    positions: dict[str, int],
    column: str | None,
) -> list[Any] | None:
    if column is None:
        return None
    return batch.column(positions[column]).to_pylist()


def _batch_value(values: list[Any] | None, index: int) -> Any:
    return None if values is None else values[index]


def _scan_document_batch(
    batch: Any,
    *,
    positions: dict[str, int],
    is_wikipedia: bool,
    has_document_id: bool,
    has_language: bool,
    word_column: str | None,
    metrics: _DocumentMetrics,
) -> None:
    identities, languages, words = _document_batch_columns(
        batch,
        positions,
        has_document_id=has_document_id,
        has_language=has_language,
        word_column=word_column,
    )
    fetch_status = _batch_column(
        batch,
        positions,
        "fetch_status" if "fetch_status" in positions else None,
    )
    full_text = _batch_column(
        batch,
        positions,
        "full_text" if "full_text" in positions else None,
    )
    has_text_columns = has_document_id and word_column is not None
    for index in range(batch.num_rows):
        _record_document_row(
            metrics,
            identity=_batch_value(identities, index),
            language=_batch_value(languages, index),
            word_count=_batch_value(words, index),
            is_wikipedia=is_wikipedia,
            has_document_id=has_document_id,
            has_text_columns=has_text_columns,
            has_word_column=words is not None,
            fetch_status=_batch_value(fetch_status, index),
            full_text=_batch_value(full_text, index),
        )


def _record_document_row(
    metrics: _DocumentMetrics,
    *,
    identity: Any,
    language: Any,
    word_count: Any,
    is_wikipedia: bool,
    has_document_id: bool,
    has_text_columns: bool,
    has_word_column: bool,
    fetch_status: Any,
    full_text: Any,
) -> None:
    _record_document_language(
        metrics,
        language,
        is_wikipedia=is_wikipedia,
        has_document_id=has_document_id,
    )
    _record_wikipedia_document_identity(metrics, identity, is_wikipedia=is_wikipedia)
    _record_document_text(
        metrics,
        identity,
        language,
        word_count,
        has_text_columns=has_text_columns,
    )
    _record_document_words(metrics, word_count, has_word_column=has_word_column)
    _record_non_empty_text_document(
        metrics,
        identity,
        language=language,
        fetch_status=fetch_status,
        full_text=full_text,
        is_wikipedia=is_wikipedia,
    )


def _record_non_empty_text_document(
    metrics: _DocumentMetrics,
    identity: Any,
    *,
    language: Any = None,
    fetch_status: Any,
    full_text: Any,
    is_wikipedia: bool,
) -> None:
    if not _is_successful_non_empty_text(identity, fetch_status, full_text):
        return
    project = "wikipedia" if is_wikipedia else "wikivoyage"
    if project in _TEXT_DOCUMENT_PROJECTS:
        key = (project, str(identity))
        metrics.non_empty_text_document_keys.add(key)
        metrics.successful_text_document_languages[key] = str(language or "")


def _is_successful_non_empty_text(identity: Any, fetch_status: Any, full_text: Any) -> bool:
    return bool(
        identity
        and fetch_status == _SUCCESSFUL_FETCH_STATUS
        and isinstance(full_text, str)
        and full_text.strip()
    )


def _record_wikipedia_document_identity(
    metrics: _DocumentMetrics,
    identity: Any,
    *,
    is_wikipedia: bool,
) -> None:
    if is_wikipedia and identity:
        metrics.document_ids.add(str(identity))


def _record_document_text(
    metrics: _DocumentMetrics,
    identity: Any,
    language: Any,
    word_count: Any,
    *,
    has_text_columns: bool,
) -> None:
    if has_text_columns and _has_non_empty_words(identity, word_count):
        metrics.text_document_languages[str(identity)] = str(language or "")


def _record_document_words(
    metrics: _DocumentMetrics,
    word_count: Any,
    *,
    has_word_column: bool,
) -> None:
    if has_word_column:
        metrics.document_words += int(word_count or 0)


def _record_document_language(
    metrics: _DocumentMetrics,
    language: Any,
    *,
    is_wikipedia: bool,
    has_document_id: bool,
) -> None:
    if not language:
        return
    language_value = str(language)
    metrics.languages.add(language_value)
    if is_wikipedia and has_document_id:
        metrics.wikipedia_language_counts[language_value] += 1


def _count_linked_non_empty_text_polygons(
    paths: Iterable[Path],
    eligible_document_keys: set[tuple[str, str]],
    polygon_index: PolygonIndex,
) -> int:
    """Count unique OSM identities linked to successful non-empty documents."""
    polygon_identities: set[PolygonIdentity] = set()
    if not eligible_document_keys:
        return 0
    for path in paths:
        _collect_linked_non_empty_text_polygons(
            path,
            eligible_document_keys,
            polygon_index,
            polygon_identities,
        )
    return len(polygon_identities)


def _collect_linked_non_empty_text_polygons(
    path: Path,
    eligible_document_keys: set[tuple[str, str]],
    polygon_index: PolygonIndex,
    polygon_identities: set[PolygonIdentity],
) -> None:
    with pq.ParquetFile(path) as parquet_file:
        columns = {"polygon_id", "document_id", "project"}
        if not columns.issubset(parquet_file.schema_arrow.names):
            return
        for batch in parquet_file.iter_batches(
            columns=["polygon_id", "document_id", "project"],
            batch_size=65_536,
        ):
            _merge_linked_non_empty_text_polygons(
                batch,
                eligible_document_keys,
                polygon_index,
                polygon_identities,
            )


def _merge_linked_non_empty_text_polygons(
    batch: Any,
    eligible_document_keys: set[tuple[str, str]],
    polygon_index: PolygonIndex,
    polygon_identities: set[PolygonIdentity],
) -> None:
    for polygon_id, document_id, project in zip(
        batch.column(0).to_pylist(),
        batch.column(1).to_pylist(),
        batch.column(2).to_pylist(),
        strict=True,
    ):
        document_key = (str(project), str(document_id))
        if document_key not in eligible_document_keys:
            continue
        identity = polygon_index.by_polygon_id.get(str(polygon_id))
        if identity is not None:
            polygon_identities.add(identity)


def _osm_polygon_identity(osm_type: Any, osm_id: Any) -> tuple[str, int] | None:
    if osm_type in (None, "") or osm_id in (None, ""):
        return None
    try:
        return str(osm_type), int(osm_id)
    except (TypeError, ValueError):
        return None


def _scan_polygon_metrics(paths: Iterable[Path]) -> _PolygonMetrics:
    """Collect polygon identities, QIDs, and source counts in one pass."""
    metrics = _PolygonMetrics(set(), set())
    for path in paths:
        _scan_polygon_file(path, metrics)
    return metrics


def _scan_polygon_file(path: Path, metrics: _PolygonMetrics) -> None:
    with pq.ParquetFile(path) as parquet_file:
        metadata = parquet_file.metadata
        metrics.polygon_row_count += 0 if metadata is None else int(metadata.num_rows)
        names = set(parquet_file.schema_arrow.names)
        columns = _polygon_columns(names)
        if not columns:
            return
        _scan_polygon_batches(parquet_file, columns=columns, metrics=metrics)


def _polygon_columns(names: set[str]) -> list[str]:
    return [column for column in ("polygon_id", "wikidata", "has_wikidata") if column in names]


def _scan_polygon_batches(
    parquet_file: pq.ParquetFile,
    *,
    columns: list[str],
    metrics: _PolygonMetrics,
) -> None:
    positions = {column: index for index, column in enumerate(columns)}
    for batch in parquet_file.iter_batches(columns=columns, batch_size=65_536):
        _scan_polygon_batch(batch, positions=positions, metrics=metrics)


def _scan_polygon_batch(
    batch: Any,
    *,
    positions: dict[str, int],
    metrics: _PolygonMetrics,
) -> None:
    polygon_ids = _batch_column(
        batch,
        positions,
        "polygon_id" if "polygon_id" in positions else None,
    )
    wikidata = _batch_column(batch, positions, "wikidata" if "wikidata" in positions else None)
    has_wikidata = _batch_column(
        batch,
        positions,
        "has_wikidata" if "has_wikidata" in positions else None,
    )
    for index in range(batch.num_rows):
        _record_polygon_row(
            metrics,
            polygon_id=_batch_value(polygon_ids, index),
            wikidata=_batch_value(wikidata, index),
            has_wikidata=_batch_value(has_wikidata, index),
        )


def _record_polygon_row(
    metrics: _PolygonMetrics,
    *,
    polygon_id: Any,
    wikidata: Any,
    has_wikidata: Any,
) -> None:
    if polygon_id:
        polygon_identity = str(polygon_id)
        metrics.polygon_ids.add(polygon_identity)
    if wikidata:
        metrics.qids.update(qids_from_osm_tag(str(wikidata)))
    if has_wikidata is False:
        metrics.wikipedia_tag_only += 1


def _word_column(names: set[str]) -> str | None:
    if "article_length_words" in names:
        return "article_length_words"
    if "text_length_words" in names:
        return "text_length_words"
    return None


def _has_non_empty_words(identity: Any, word_count: Any) -> bool:
    return bool(identity and int(word_count or 0) > 0)


def _non_empty_strings(values: list[Any]) -> list[str]:
    return [str(value) for value in values if value]


def _polygon_languages(
    paths: Iterable[Path],
    document_languages: dict[tuple[str, str], str],
    polygon_index: PolygonIndex,
) -> defaultdict[PolygonIdentity, set[str]]:
    languages_by_polygon: defaultdict[PolygonIdentity, set[str]] = defaultdict(set)
    for path in paths:
        languages_by_path = _polygon_languages_file(path, document_languages, polygon_index)
        for identity, languages in languages_by_path.items():
            languages_by_polygon[identity].update(languages)
    return languages_by_polygon


def _polygon_languages_file(
    path: Path,
    document_languages: dict[tuple[str, str], str],
    polygon_index: PolygonIndex,
) -> dict[PolygonIdentity, set[str]]:
    values: defaultdict[PolygonIdentity, set[str]] = defaultdict(set)
    with pq.ParquetFile(path) as parquet_file:
        if not {"polygon_id", "document_id"}.issubset(parquet_file.schema_arrow.names):
            return values
        columns = ["polygon_id", "document_id"]
        has_project = "project" in parquet_file.schema_arrow.names
        if has_project:
            columns.append("project")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=65_536):
            _merge_polygon_languages(
                values,
                batch,
                document_languages,
                polygon_index,
                has_project=has_project,
            )
    return values


def _merge_polygon_languages(
    values: defaultdict[PolygonIdentity, set[str]],
    batch: Any,
    document_languages: dict[tuple[str, str], str],
    polygon_index: PolygonIndex,
    *,
    has_project: bool,
) -> None:
    projects = batch.column(2).to_pylist() if has_project else ["wikipedia"] * batch.num_rows
    for polygon_id, document_id, project in zip(
        batch.column(0).to_pylist(),
        batch.column(1).to_pylist(),
        projects,
        strict=True,
    ):
        identity = polygon_index.by_polygon_id.get(str(polygon_id))
        language = document_languages.get((str(project), str(document_id)))
        if identity is not None and language is not None:
            values[identity].add(language)


def _text_funnel(
    languages_by_polygon: dict[PolygonIdentity, set[str]],
) -> tuple[tuple[str, int], ...]:
    all_text = len(languages_by_polygon)
    english = sum("en" in languages for languages in languages_by_polygon.values())
    return (
        ("All polygons", 0),
        ("With non-empty text", all_text),
        ("English coverage", english),
        ("Non-English-only coverage", all_text - english),
        ("2+ languages", sum(len(languages) >= 2 for languages in languages_by_polygon.values())),
        ("5+ languages", sum(len(languages) >= 5 for languages in languages_by_polygon.values())),
        ("10+ languages", sum(len(languages) >= 10 for languages in languages_by_polygon.values())),
    )


__all__ = ["V2CardStats", "compute_v2_card_stats", "render_v2_card", "write_v2_card"]
