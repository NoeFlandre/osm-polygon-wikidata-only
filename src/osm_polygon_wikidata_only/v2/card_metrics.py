"""Aggregate V2 card metrics from focused scanning primitives."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.v2.card_models import (
    CardFiles as _CardFiles,
)
from osm_polygon_wikidata_only.v2.card_models import (
    CardMetrics as _CardMetrics,
)
from osm_polygon_wikidata_only.v2.card_models import (
    SentenceCardStats as _SentenceCardStats,
)
from osm_polygon_wikidata_only.v2.card_models import (
    V1Baseline as _V1Baseline,
)
from osm_polygon_wikidata_only.v2.card_models import (
    V1Comparison as _V1Comparison,
)
from osm_polygon_wikidata_only.v2.card_models import (
    V2CardStats,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    collect_card_files as _collect_card_files,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    field_values_for_ids as _field_values_for_ids,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    sum_first_available as _sum_first_available,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    sum_metadata as _sum_metadata,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    unique_numeric_values as _unique_numeric_values,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    unique_values as _unique_values,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    v1_document_files as _v1_document_files,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    v1_section_files as _v1_section_files,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    v1_wikipedia_document_files as _v1_wikipedia_document_files,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    scan_document_metrics as _scan_document_metrics,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    count_linked_non_empty_text_polygons as _count_linked_non_empty_text_polygons,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    text_metrics_from_scanned as _text_metrics_from_scanned,
)
from osm_polygon_wikidata_only.v2.card_scanning_polygons import (
    load_polygon_index as _load_polygon_index,
)
from osm_polygon_wikidata_only.v2.card_scanning_polygons import (
    polygon_ids_with_link_source as _polygon_ids_with_link_source,
)
from osm_polygon_wikidata_only.v2.card_scanning_polygons import (
    polygon_source_sets as _polygon_source_sets,
)
from osm_polygon_wikidata_only.v2.card_scanning_polygons import (
    scan_polygon_metrics as _scan_polygon_metrics,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    compute_sentence_stats as _compute_sentence_stats,
)
from osm_polygon_wikidata_only.v2.comparison import (
    select_v2_added_wikipedia_tag_document_polygon_ids_from_files,
)


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
    document_comparison = _compare_document_content(files, baseline)
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


# Public collaborator spellings used by the compatibility facade.
compute_card_metrics = _compute_card_metrics
load_v1_baseline = _load_v1_baseline
compute_v1_comparison = _compute_v1_comparison
compare_polygon_sources = _compare_polygon_sources
compare_document_content = _compare_document_content
v1_document_words_by_id = _v1_document_words_by_id
new_identity_words = _new_identity_words
shared_content_count = _shared_content_count
compare_unique_sections = _compare_unique_sections
build_card_stats = _build_card_stats
