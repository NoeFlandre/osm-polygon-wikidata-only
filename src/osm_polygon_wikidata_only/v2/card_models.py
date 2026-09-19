"""Data models for deterministic V2 card computation and publication."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


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


# Focused card modules consume these as explicit collaborators. Keep the
# historical private class names for compatibility with the facade and tests.
CardFiles = _CardFiles
CardMetrics = _CardMetrics
SentenceCardStats = _SentenceCardStats
DocumentMetrics = _DocumentMetrics
PolygonMetrics = _PolygonMetrics
V1Baseline = _V1Baseline
V1Comparison = _V1Comparison
