"""Parquet discovery and columnar scanning primitives for V2 cards."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.v2.card_models import (
    CardFiles as _CardFiles,
)
from osm_polygon_wikidata_only.v2.card_scanning_polygons import (
    link_source_file,
    load_polygon_index,
    merge_link_sources,
    merge_polygon_sources,
    osm_polygon_identity,
    parse_source_list,
    polygon_columns,
    polygon_ids_with_link_source,
    polygon_source_file,
    polygon_source_sets,
    record_polygon_row,
    scan_polygon_batch,
    scan_polygon_batches,
    scan_polygon_file,
    scan_polygon_metrics,
    validated_source_list,
)
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

_METADATA_READ_WORKERS = 4
_SUCCESSFUL_FETCH_STATUS = "ok"
_TEXT_DOCUMENT_PROJECTS = frozenset({"wikipedia", "wikivoyage"})


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


def _sum_metadata(paths: Iterable[Path], *, executor_factory=ThreadPoolExecutor) -> int:
    materialized = tuple(paths)
    if not materialized:
        return 0
    workers = min(_METADATA_READ_WORKERS, len(materialized))
    with executor_factory(
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
    with open_parquet(path) as parquet_file:
        if column not in parquet_file.schema_arrow.names:
            return values
        for batch in iter_record_batches(parquet_file, columns=[column], batch_size=65_536):
            values.update(_non_empty_strings(batch.column(0).to_pylist()))
    return values


def _sum_first_available(paths: Iterable[Path], columns: tuple[str, ...]) -> int:
    return sum(_sum_first_available_file(path, columns) for path in paths)


def _sum_first_available_file(path: Path, columns: tuple[str, ...]) -> int:
    with open_parquet(path) as parquet_file:
        column = _first_present_column(set(parquet_file.schema_arrow.names), columns)
        if column is None:
            return 0
        return sum(
            sum(int(value or 0) for value in batch.column(0).to_pylist())
            for batch in iter_record_batches(parquet_file, columns=[column], batch_size=65_536)
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
    with open_parquet(path) as parquet_file:
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
    for batch in iter_record_batches(
        parquet_file, columns=[key_column, value_column], batch_size=65_536
    ):
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
    with open_parquet(path) as parquet_file:
        if not {key_column, value_column}.issubset(parquet_file.schema_arrow.names):
            return
        for batch in iter_record_batches(
            parquet_file, columns=[key_column, value_column], batch_size=65_536
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


# Compatibility aliases for the polygon scanner moved to its focused module.
_load_polygon_index = load_polygon_index
_polygon_source_sets = polygon_source_sets
_polygon_source_file = polygon_source_file
_merge_polygon_sources = merge_polygon_sources
_parse_source_list = parse_source_list
_validated_source_list = validated_source_list
_polygon_ids_with_link_source = polygon_ids_with_link_source
_link_source_file = link_source_file
_merge_link_sources = merge_link_sources
_osm_polygon_identity = osm_polygon_identity
_scan_polygon_metrics = scan_polygon_metrics
_scan_polygon_file = scan_polygon_file
_polygon_columns = polygon_columns
_scan_polygon_batches = scan_polygon_batches
_scan_polygon_batch = scan_polygon_batch
_record_polygon_row = record_polygon_row


# Public collaborator spellings used by the focused metric and document
# scanners. The private names remain the compatibility surface of this module.
collect_card_files = _collect_card_files
load_polygon_index = _load_polygon_index
v1_wikipedia_document_files = _v1_wikipedia_document_files
v1_document_files = _v1_document_files
v1_section_files = _v1_section_files
manifest_files = _manifest_files
sum_metadata = _sum_metadata
metadata_row_count = _metadata_row_count
unique_values = _unique_values
unique_values_file = _unique_values_file
sum_first_available = _sum_first_available
sum_first_available_file = _sum_first_available_file
first_present_column = _first_present_column
unique_numeric_values = _unique_numeric_values
merge_numeric_file = _merge_numeric_file
merge_numeric_batches = _merge_numeric_batches
merge_numeric_batch = _merge_numeric_batch
record_numeric_value = _record_numeric_value
field_values_for_ids = _field_values_for_ids
merge_field_values_file = _merge_field_values_file
merge_field_values_batch = _merge_field_values_batch
polygon_source_sets = _polygon_source_sets
polygon_source_file = _polygon_source_file
merge_polygon_sources = _merge_polygon_sources
parse_source_list = _parse_source_list
validated_source_list = _validated_source_list
polygon_ids_with_link_source = _polygon_ids_with_link_source
link_source_file = _link_source_file
batch_column = _batch_column
batch_value = _batch_value
merge_link_sources = _merge_link_sources
osm_polygon_identity = _osm_polygon_identity
scan_polygon_metrics = _scan_polygon_metrics
scan_polygon_file = _scan_polygon_file
polygon_columns = _polygon_columns
scan_polygon_batches = _scan_polygon_batches
scan_polygon_batch = _scan_polygon_batch
record_polygon_row = _record_polygon_row
word_column = _word_column
has_non_empty_words = _has_non_empty_words
non_empty_strings = _non_empty_strings


_DOCUMENT_EXPORTS = frozenset(
    {
        "_document_batch_columns",
        "_document_column_names",
        "_document_columns",
        "_document_metric_columns",
        "_record_document_language",
        "_record_document_row",
        "_record_document_text",
        "_record_document_words",
        "_record_non_empty_text_document",
        "_scan_document_batch",
        "_scan_document_batches",
        "_scan_document_file",
        "_scan_document_metrics",
    }
)
_LINK_EXPORTS = frozenset(
    {
        "_collect_linked_non_empty_text_polygons",
        "_count_linked_non_empty_text_polygons",
        "_merge_linked_non_empty_text_polygons",
        "_merge_polygon_languages",
        "_polygon_languages",
        "_polygon_languages_file",
        "_text_funnel",
        "_text_metrics_from_scanned",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily expose moved scanners for old private import seams."""
    if name in _DOCUMENT_EXPORTS:
        # Compatibility lookup avoids importing both scanner modules at startup.
        card_scanning_documents = import_module(f"{__package__}.card_scanning_documents")
        return getattr(card_scanning_documents, name)
    if name in _LINK_EXPORTS:
        # Compatibility lookup avoids importing both scanner modules at startup.
        card_scanning_links = import_module(f"{__package__}.card_scanning_links")
        return getattr(card_scanning_links, name)
    raise AttributeError(name)
