"""Compatibility facade for V2 card metrics, rendering, and publication.

The private imports below intentionally preserve the historical test and
extension seams while implementation code lives in focused modules.
"""

# ruff: noqa: F401

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.v2.card_metrics import (
    build_card_stats as _build_card_stats,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    compare_document_content as _compare_document_content,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    compare_polygon_sources as _compare_polygon_sources,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    compare_unique_sections as _compare_unique_sections,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    compute_card_metrics as _compute_card_metrics,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    compute_v1_comparison as _compute_v1_comparison,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    compute_v2_card_stats,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    load_v1_baseline as _load_v1_baseline,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    new_identity_words as _new_identity_words,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    shared_content_count as _shared_content_count,
)
from osm_polygon_wikidata_only.v2.card_metrics import (
    v1_document_words_by_id as _v1_document_words_by_id,
)
from osm_polygon_wikidata_only.v2.card_models import (
    CardFiles as _CardFiles,
)
from osm_polygon_wikidata_only.v2.card_models import (
    CardMetrics as _CardMetrics,
)
from osm_polygon_wikidata_only.v2.card_models import (
    DocumentMetrics as _DocumentMetrics,
)
from osm_polygon_wikidata_only.v2.card_models import (
    PolygonMetrics as _PolygonMetrics,
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
from osm_polygon_wikidata_only.v2.card_release import (
    MinimalV2ReleaseSnapshot,
    build_minimal_v2_release_snapshot,
)
from osm_polygon_wikidata_only.v2.card_release import (
    jsonable as _jsonable,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    has_parquet as _has_parquet,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    non_empty_text_polygon_count as _non_empty_text_polygon_count,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    render_comparison as _render_comparison,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    render_front_matter as _render_front_matter,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    render_v2_card,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    sentence_section_lines as _sentence_section_lines,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    unique_polygon_count as _unique_polygon_count,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    batch_column as _batch_column,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    batch_value as _batch_value,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    collect_card_files as _collect_card_files,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    field_values_for_ids as _field_values_for_ids,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    first_present_column as _first_present_column,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    has_non_empty_words as _has_non_empty_words,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    link_source_file as _link_source_file,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    load_polygon_index as _load_polygon_index,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    manifest_files as _manifest_files,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    merge_field_values_batch as _merge_field_values_batch,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    merge_field_values_file as _merge_field_values_file,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    merge_link_sources as _merge_link_sources,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    merge_numeric_batch as _merge_numeric_batch,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    merge_numeric_batches as _merge_numeric_batches,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    merge_numeric_file as _merge_numeric_file,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    merge_polygon_sources as _merge_polygon_sources,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    metadata_row_count as _metadata_row_count,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    non_empty_strings as _non_empty_strings,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    osm_polygon_identity as _osm_polygon_identity,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    parse_source_list as _parse_source_list,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    polygon_columns as _polygon_columns,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    polygon_ids_with_link_source as _polygon_ids_with_link_source,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    polygon_source_file as _polygon_source_file,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    polygon_source_sets as _polygon_source_sets,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    record_numeric_value as _record_numeric_value,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    record_polygon_row as _record_polygon_row,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    scan_polygon_batch as _scan_polygon_batch,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    scan_polygon_batches as _scan_polygon_batches,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    scan_polygon_file as _scan_polygon_file,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    scan_polygon_metrics as _scan_polygon_metrics,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    sum_first_available as _sum_first_available,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    sum_first_available_file as _sum_first_available_file,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    sum_metadata as _scan_sum_metadata,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    unique_numeric_values as _unique_numeric_values,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    unique_values as _unique_values,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    unique_values_file as _unique_values_file,
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
from osm_polygon_wikidata_only.v2.card_scanning import (
    validated_source_list as _validated_source_list,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    word_column as _word_column,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    document_batch_columns as _document_batch_columns,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    document_column_names as _document_column_names,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    document_columns as _document_columns,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    document_metric_columns as _document_metric_columns,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    record_document_language as _record_document_language,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    record_document_row as _record_document_row,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    record_document_text as _record_document_text,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    record_document_words as _record_document_words,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    record_non_empty_text_document as _record_non_empty_text_document,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    scan_document_batch as _scan_document_batch,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    scan_document_batches as _scan_document_batches,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    scan_document_file as _scan_document_file,
)
from osm_polygon_wikidata_only.v2.card_scanning_documents import (
    scan_document_metrics as _scan_document_metrics,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    collect_linked_non_empty_text_polygons as _collect_linked_non_empty_text_polygons,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    count_linked_non_empty_text_polygons as _count_linked_non_empty_text_polygons,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    merge_linked_non_empty_text_polygons as _merge_linked_non_empty_text_polygons,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    merge_polygon_languages as _merge_polygon_languages,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    polygon_languages as _polygon_languages,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    polygon_languages_file as _polygon_languages_file,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    text_funnel as _text_funnel,
)
from osm_polygon_wikidata_only.v2.card_scanning_links import (
    text_metrics_from_scanned as _text_metrics_from_scanned,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    collect_sentence_polygon_ids as _collect_sentence_polygon_ids,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    compute_array as _compute_array,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    compute_sentence_stats as _compute_sentence_stats,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    load_sentence_manifest as _load_sentence_manifest,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    sentence_document_ids as _sentence_document_ids,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    sentence_document_ids_batch as _sentence_document_ids_batch,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    sentence_manifest_totals as _sentence_manifest_totals,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    sentence_polygon_count as _sentence_polygon_count,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    sentence_polygon_ids_from_batch as _sentence_polygon_ids_from_batch,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    sentence_region_totals as _sentence_region_totals,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    update_sentence_document_ids as _update_sentence_document_ids,
)


def _sum_metadata(paths: Iterable[Path]) -> int:
    """Preserve the historical patch seam for bounded metadata reads."""
    return _scan_sum_metadata(paths, executor_factory=ThreadPoolExecutor)


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


__all__ = [
    "MinimalV2ReleaseSnapshot",
    "V2CardStats",
    "build_minimal_v2_release_snapshot",
    "compute_v2_card_stats",
    "render_v2_card",
    "write_v2_card",
]
