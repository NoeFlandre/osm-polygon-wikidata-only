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
    _build_card_stats,
    _compare_document_content,
    _compare_polygon_sources,
    _compare_unique_sections,
    _compute_card_metrics,
    _compute_v1_comparison,
    _load_v1_baseline,
    _new_identity_words,
    _shared_content_count,
    _v1_document_words_by_id,
    compute_v2_card_stats,
)
from osm_polygon_wikidata_only.v2.card_models import (
    V2CardStats,
    _CardFiles,
    _CardMetrics,
    _DocumentMetrics,
    _PolygonMetrics,
    _SentenceCardStats,
    _V1Baseline,
    _V1Comparison,
)
from osm_polygon_wikidata_only.v2.card_release import (
    MinimalV2ReleaseSnapshot,
    _jsonable,
    build_minimal_v2_release_snapshot,
)
from osm_polygon_wikidata_only.v2.card_rendering import (
    _has_parquet,
    _non_empty_text_polygon_count,
    _render_comparison,
    _render_front_matter,
    _sentence_section_lines,
    _unique_polygon_count,
    render_v2_card,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    _batch_column,
    _batch_value,
    _collect_card_files,
    _collect_linked_non_empty_text_polygons,
    _count_linked_non_empty_text_polygons,
    _document_batch_columns,
    _document_column_names,
    _document_columns,
    _document_metric_columns,
    _field_values_for_ids,
    _first_present_column,
    _has_non_empty_words,
    _link_source_file,
    _load_polygon_index,
    _manifest_files,
    _merge_field_values_batch,
    _merge_field_values_file,
    _merge_link_sources,
    _merge_linked_non_empty_text_polygons,
    _merge_numeric_batch,
    _merge_numeric_batches,
    _merge_numeric_file,
    _merge_polygon_languages,
    _merge_polygon_sources,
    _metadata_row_count,
    _non_empty_strings,
    _osm_polygon_identity,
    _parse_source_list,
    _polygon_columns,
    _polygon_ids_with_link_source,
    _polygon_languages,
    _polygon_languages_file,
    _polygon_source_file,
    _polygon_source_sets,
    _record_document_language,
    _record_document_row,
    _record_document_text,
    _record_document_words,
    _record_non_empty_text_document,
    _record_numeric_value,
    _record_polygon_row,
    _scan_document_batch,
    _scan_document_batches,
    _scan_document_file,
    _scan_document_metrics,
    _scan_polygon_batch,
    _scan_polygon_batches,
    _scan_polygon_file,
    _scan_polygon_metrics,
    _sum_first_available,
    _sum_first_available_file,
    _text_funnel,
    _text_metrics_from_scanned,
    _unique_numeric_values,
    _unique_values,
    _unique_values_file,
    _v1_document_files,
    _v1_section_files,
    _v1_wikipedia_document_files,
    _validated_source_list,
    _word_column,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    _sum_metadata as _scan_sum_metadata,
)
from osm_polygon_wikidata_only.v2.card_sentences import (
    _collect_sentence_polygon_ids,
    _compute_array,
    _compute_sentence_stats,
    _load_sentence_manifest,
    _sentence_document_ids,
    _sentence_document_ids_batch,
    _sentence_manifest_totals,
    _sentence_polygon_count,
    _sentence_polygon_ids_from_batch,
    _sentence_region_totals,
    _update_sentence_document_ids,
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
