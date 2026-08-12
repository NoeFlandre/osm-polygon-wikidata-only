"""Deterministic public dataset card and statistics for V2."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.v2.config import (
    V2_CONTRACT_VERSION,
    V2_GITHUB_URL,
    V2_REPO_ID,
    V2_TRACKIO_RUN_NAME,
    V2_TRACKIO_SPACE_ID,
)
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest


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

    @property
    def documents(self) -> int:
        return self.wikipedia_documents + self.wikivoyage_documents

    @property
    def other_wikipedia_languages(self) -> int:
        return max(
            0,
            self.wikipedia_documents - sum(value for _, value in self.top_wikipedia_languages),
        )


def compute_v2_card_stats(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
) -> V2CardStats:
    """Compute V2 counts from live Parquet files and optional V1 files."""
    manifest = load_v2_manifest(processed_v2)
    stems = tuple(sorted(manifest))
    polygon_files = _manifest_files(processed_v2 / "polygons", stems)
    wikipedia_document_files = _manifest_files(processed_v2 / "wikipedia/documents", stems)
    wikipedia_section_files = _manifest_files(processed_v2 / "wikipedia/sections", stems)
    wikivoyage_document_files = _manifest_files(processed_v2 / "wikivoyage/documents", stems)
    wikivoyage_section_files = _manifest_files(processed_v2 / "wikivoyage/sections", stems)
    wikidata_fact_files = _manifest_files(processed_v2 / "wikidata/facts", stems)
    link_files = _manifest_files(processed_v2 / "polygon_document_links", stems)
    parquet_files = (
        polygon_files
        + wikipedia_document_files
        + wikipedia_section_files
        + wikivoyage_document_files
        + wikivoyage_section_files
        + wikidata_fact_files
        + link_files
    )

    polygon_ids = _unique_values(polygon_files, "polygon_id")
    document_ids = _unique_values(wikipedia_document_files, "document_id")
    qids = _unique_qids(polygon_files)
    languages = _unique_values(
        wikipedia_document_files + wikivoyage_document_files,
        "language",
    )
    wikipedia_tag_only = _count_boolean_false(polygon_files, "has_wikidata")
    document_words = _sum_first_available(
        wikipedia_document_files + wikivoyage_document_files,
        ("article_length_words", "text_length_words"),
    )
    text_funnel, top_languages = _text_metrics(
        wikipedia_document_files + wikivoyage_document_files,
        wikipedia_document_files,
        link_files,
    )
    text_funnel = (("All polygons", len(polygon_ids)), *text_funnel[1:])

    v1_polygon_ids: set[str] | None = None
    v1_document_ids: set[str] | None = None
    if v1_processed is not None:
        v1_polygon_ids = set(
            _unique_values(
                sorted(v1_processed.joinpath("polygons").glob("*.parquet")), "polygon_id"
            )
        )
        v1_document_files = sorted((v1_processed / "wikipedia/documents").glob("*.parquet"))
        if not v1_document_files:
            v1_document_files = sorted((v1_processed / "articles").glob("*.parquet"))
        v1_document_ids = set(_unique_values(v1_document_files, "document_id"))
        if not v1_document_ids:
            v1_document_ids = set(_unique_values(v1_document_files, "article_id"))

    return V2CardStats(
        regions=len(stems),
        polygons=_sum_metadata(polygon_files),
        unique_wikidata_entities=len(qids),
        wikipedia_documents=_sum_metadata(wikipedia_document_files),
        wikipedia_sections=_sum_metadata(wikipedia_section_files),
        wikivoyage_documents=_sum_metadata(wikivoyage_document_files),
        wikivoyage_sections=_sum_metadata(wikivoyage_section_files),
        wikidata_facts=_sum_metadata(wikidata_fact_files),
        polygon_document_links=_sum_metadata(link_files),
        wikipedia_tag_only_polygons=wikipedia_tag_only,
        document_words=document_words,
        languages=len(languages),
        new_polygons_vs_v1=(
            len(polygon_ids - v1_polygon_ids) if v1_polygon_ids is not None else None
        ),
        new_wikipedia_documents_vs_v1=(
            len(document_ids - v1_document_ids) if v1_document_ids is not None else None
        ),
        text_coverage_funnel=text_funnel,
        top_wikipedia_languages=top_languages,
        polygon_link_storage_bytes=sum(path.stat().st_size for path in link_files),
        total_parquet_storage_bytes=sum(path.stat().st_size for path in parquet_files),
    )


def render_v2_card(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
    stats: V2CardStats | None = None,
) -> str:
    """Render a concise, viewer-compatible card from V2 files on disk."""
    snapshot = stats or compute_v2_card_stats(processed_v2, v1_processed=v1_processed)
    front_matter = _render_front_matter(snapshot)
    comparison = _render_comparison(snapshot)
    return (
        front_matter
        + "\n"
        + "\n".join(
            [
                "# OSM Polygon Wikidata + Wikipedia, V2",
                "",
                "![V2 dataset overview](assets/dataset_hero.png)",
                "",
                f"V2 keeps the V1 Wikidata-derived corpus and adds valid multilingual "
                f"OSM `wikipedia=*` references, including polygons without a Wikidata QID. "
                f"The code is maintained in the [GitHub repository]({V2_GITHUB_URL}).",
                "",
                f"The public V2 Trackio snapshot is [`{V2_TRACKIO_RUN_NAME}`](https://huggingface.co/spaces/{V2_TRACKIO_SPACE_ID}).",
                "",
                "## Snapshot",
                "",
                f"- **Hugging Face dataset:** [{V2_REPO_ID}](https://huggingface.co/datasets/{V2_REPO_ID})",
                f"- **Regions:** {snapshot.regions:,}",
                f"- **Polygons:** {snapshot.polygons:,}",
                f"- **Unique Wikidata entities:** {snapshot.unique_wikidata_entities:,}",
                f"- **Wikipedia documents:** {snapshot.wikipedia_documents:,}",
                f"- **Wikivoyage documents:** {snapshot.wikivoyage_documents:,}",
                f"- **Wikipedia + Wikivoyage documents:** {snapshot.documents:,}",
                f"- **Wikipedia sections:** {snapshot.wikipedia_sections:,}",
                f"- **Wikivoyage sections:** {snapshot.wikivoyage_sections:,}",
                f"- **Wikidata facts:** {snapshot.wikidata_facts:,}",
                f"- **Polygon-document links:** {snapshot.polygon_document_links:,}",
                f"- **Wikipedia-tag-only polygons:** {snapshot.wikipedia_tag_only_polygons:,}",
                f"- **Document words:** {snapshot.document_words:,}",
                f"- **Languages represented:** {snapshot.languages:,}",
                f"- **Polygon/link-table storage:** {snapshot.polygon_link_storage_bytes / 1_000_000_000:.1f} GB",
                f"- **Total Parquet storage:** {snapshot.total_parquet_storage_bytes / 1_000_000_000:.1f} GB",
                "",
                comparison,
                "## Coverage maps",
                "",
                "### All V2 dataset polygons",
                "",
                "![All V2 dataset polygons](assets/coverage_map.png)",
                "",
                "Every point is one retained V2 polygon, including polygons without text.",
                "",
                "### Polygons with Wikipedia or Wikivoyage text",
                "",
                "![V2 polygons with text](assets/geographic_text_presence.png)",
                "",
                "Each point is one polygon linked to at least one non-empty Wikipedia or Wikivoyage document. A polygon is counted once even when several documents qualify.",
                "",
                "### H3 density of polygons with text",
                "",
                "![V2 geographic text density](assets/geographic_text_density.png)",
                "",
                "Each H3 cell shows the raw count of unique V2 polygons with non-empty Wikipedia or Wikivoyage text. Colour uses a logarithmic scale; it is not a proportion.",
                "",
                "## Deduplication and provenance",
                "",
                "V2 deduplicates documents by `document_id` and polygon-document links by `(polygon_id, project, document_id)` within each region. Byte-identical repeats collapse deterministically; conflicting rows fail closed. The `discovery_sources` and `link_sources` fields distinguish Wikidata-derived and direct Wikipedia-tag relationships.",
                "",
                "## Repository layout",
                "",
                "- `polygons/<stem>.parquet` — one row per retained OSM polygon.",
                "- `wikipedia/documents/<stem>.parquet` and `wikipedia/sections/<stem>.parquet` — multilingual Wikipedia documents and their sections.",
                "  Wikipedia sections retain the exact V1 22-column section schema for lossless reuse.",
                "- `wikivoyage/documents/<stem>.parquet` and `wikivoyage/sections/<stem>.parquet` — Wikivoyage documents and sections reused from V1 where available.",
                "- `polygon_document_links/<stem>.parquet` — unified Wikipedia and Wikivoyage polygon links.",
                "- `wikidata/facts/<stem>.parquet` — structured Wikidata facts.",
                "",
                "## Reproducibility",
                "",
                "Run V2 explicitly with `sync-dir --dataset-version v2`. V1 remains a separate contract and is not modified by V2 processing.",
                "",
            ]
        )
    )


def write_v2_card(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
    stats: V2CardStats | None = None,
) -> Path:
    """Write the deterministic V2 card atomically and return its path."""
    path = processed_v2 / "README.md"
    atomic_write_text(
        path,
        render_v2_card(processed_v2, v1_processed=v1_processed, stats=stats),
    )
    return path


def _render_front_matter(snapshot: V2CardStats) -> str:
    configs = (
        ("polygons", "polygons", "polygons/*.parquet"),
        ("polygon_document_links", "polygon_document_links", "polygon_document_links/*.parquet"),
        ("wikipedia_documents", "wikipedia_documents", "wikipedia/documents/*.parquet"),
        ("wikipedia_sections", "wikipedia_sections", "wikipedia/sections/*.parquet"),
        ("wikivoyage_documents", "wikivoyage_documents", "wikivoyage/documents/*.parquet"),
        ("wikivoyage_sections", "wikivoyage_sections", "wikivoyage/sections/*.parquet"),
        ("wikidata_facts", "wikidata_facts", "wikidata/facts/*.parquet"),
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
            f"  version: {V2_CONTRACT_VERSION}",
            f"  regions: {snapshot.regions}",
            f"  polygons: {snapshot.polygons}",
            f"  documents: {snapshot.documents}",
            "---",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_comparison(snapshot: V2CardStats) -> str:
    lines = ["## V2 compared with V1", ""]
    if snapshot.new_polygons_vs_v1 is None or snapshot.new_wikipedia_documents_vs_v1 is None:
        lines.append(
            "The local V1 artifact root was not supplied for this card render, so the delta is not estimated."
        )
    else:
        lines.extend(
            [
                "The following deltas are computed from local V1 and V2 artifact identities, not entered manually:",
                "",
                f"- **Additional polygons discovered through V2 Wikipedia tags:** {snapshot.new_polygons_vs_v1:,}",
                f"- **Additional Wikipedia document identities:** {snapshot.new_wikipedia_documents_vs_v1:,}",
                "",
            ]
        )
    return "\n".join(lines)


def _manifest_files(directory: Path, stems: Iterable[str]) -> list[Path]:
    return [path for stem in stems if (path := directory / f"{stem}.parquet").is_file()]


def _sum_metadata(paths: Iterable[Path]) -> int:
    return sum(int(pq.read_metadata(path).num_rows) for path in paths)  # type: ignore[no-untyped-call]


def _unique_values(paths: Iterable[Path], column: str) -> set[str]:
    values: set[str] = set()
    for path in paths:
        names = pq.read_schema(path).names  # type: ignore[no-untyped-call]
        if column not in names:
            continue
        with pq.ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(columns=[column], batch_size=65_536):
                values.update(
                    str(value) for value in batch.column(0).to_pylist() if value not in (None, "")
                )
    return values


def _unique_qids(paths: Iterable[Path]) -> set[str]:
    values: set[str] = set()
    for raw in _unique_values(paths, "wikidata"):
        values.update(qids_from_osm_tag(raw))
    return values


def _count_boolean_false(paths: Iterable[Path], column: str) -> int:
    total = 0
    for path in paths:
        names = pq.read_schema(path).names  # type: ignore[no-untyped-call]
        if column not in names:
            continue
        with pq.ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(columns=[column], batch_size=65_536):
                total += sum(value is False for value in batch.column(0).to_pylist())
    return total


def _sum_first_available(paths: Iterable[Path], columns: tuple[str, ...]) -> int:
    total = 0
    for path in paths:
        names = set(pq.read_schema(path).names)  # type: ignore[no-untyped-call]
        column = next((candidate for candidate in columns if candidate in names), None)
        if column is None:
            continue
        with pq.ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(columns=[column], batch_size=65_536):
                total += sum(int(value or 0) for value in batch.column(0).to_pylist())
    return total


def _text_metrics(
    all_document_paths: Iterable[Path],
    wikipedia_document_paths: Iterable[Path],
    link_paths: Iterable[Path],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Return a combined text funnel and top Wikipedia document languages."""
    text_documents: set[str] = set()
    document_languages: dict[str, str] = {}
    wikipedia_language_counts: Counter[str] = Counter()
    for path in all_document_paths:
        names = set(pq.read_schema(path).names)  # type: ignore[no-untyped-call]
        words_column = (
            "article_length_words" if "article_length_words" in names else "text_length_words"
        )
        if "document_id" not in names or words_column not in names:
            continue
        with pq.ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(
                columns=["document_id", "language", words_column], batch_size=65_536
            ):
                ids = batch.column(0).to_pylist()
                langs = batch.column(1).to_pylist()
                words = batch.column(2).to_pylist()
                for identity, language, word_count in zip(ids, langs, words, strict=True):
                    if identity and int(word_count or 0) > 0:
                        document_id = str(identity)
                        text_documents.add(document_id)
                        document_languages[document_id] = str(language or "")
    for path in wikipedia_document_paths:
        names = set(pq.read_schema(path).names)  # type: ignore[no-untyped-call]
        if "language" not in names or "document_id" not in names:
            continue
        with pq.ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(columns=["language"], batch_size=65_536):
                wikipedia_language_counts.update(
                    str(value) for value in batch.column(0).to_pylist() if value
                )

    languages_by_polygon: defaultdict[str, set[str]] = defaultdict(set)
    for path in link_paths:
        names = set(pq.read_schema(path).names)  # type: ignore[no-untyped-call]
        if not {"polygon_id", "document_id"}.issubset(names):
            continue
        with pq.ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(
                columns=["polygon_id", "document_id"], batch_size=65_536
            ):
                for polygon_id, document_id in zip(
                    batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
                ):
                    language = document_languages.get(str(document_id))
                    if polygon_id and language is not None:
                        languages_by_polygon[str(polygon_id)].add(language)

    all_text = len(languages_by_polygon)
    english = sum("en" in languages for languages in languages_by_polygon.values())
    return (
        (
            ("All polygons", 0),
            ("With non-empty text", all_text),
            ("English coverage", english),
            ("Non-English-only coverage", all_text - english),
            (
                "2+ languages",
                sum(len(languages) >= 2 for languages in languages_by_polygon.values()),
            ),
            (
                "5+ languages",
                sum(len(languages) >= 5 for languages in languages_by_polygon.values()),
            ),
            (
                "10+ languages",
                sum(len(languages) >= 10 for languages in languages_by_polygon.values()),
            ),
        ),
        tuple(wikipedia_language_counts.most_common(10)),
    )


__all__ = ["V2CardStats", "compute_v2_card_stats", "render_v2_card", "write_v2_card"]
