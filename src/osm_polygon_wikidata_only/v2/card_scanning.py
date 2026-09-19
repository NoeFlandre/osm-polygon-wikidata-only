"""Parquet discovery and columnar scanning primitives for V2 cards."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.hf._geographic.polygon_identities import (
    PolygonIndex,
    load_unique_polygon_records,
)
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.card_models import (
    CardFiles as _CardFiles,
)
from osm_polygon_wikidata_only.v2.card_models import (
    PolygonMetrics as _PolygonMetrics,
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

def _load_polygon_index(paths: Iterable[Path]) -> PolygonIndex:
    """Load canonical polygon identities while retaining incomplete-test compatibility."""
    materialized = tuple(paths)
    if not materialized:
        return PolygonIndex(records={}, by_polygon_id={})
    if any("polygon_id" not in pq.read_schema(path).names for path in materialized):
        return PolygonIndex(records={}, by_polygon_id={})
    return load_unique_polygon_records(materialized)

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

def _polygon_source_sets(paths: Iterable[Path], identities: set[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for path in paths:
        values.update(_polygon_source_file(path, identities))
    return values

def _polygon_source_file(path: Path, identities: set[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    with open_parquet(path) as parquet_file:
        if not {"polygon_id", "discovery_sources"}.issubset(parquet_file.schema_arrow.names):
            return values
        for batch in iter_record_batches(
            parquet_file, columns=["polygon_id", "discovery_sources"], batch_size=65_536
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
    with open_parquet(path) as parquet_file:
        if not {"polygon_id", "link_sources"}.issubset(parquet_file.schema_arrow.names):
            return values
        for batch in iter_record_batches(
            parquet_file, columns=["polygon_id", "link_sources"], batch_size=65_536
        ):
            _merge_link_sources(values, batch, source, path)
    return values


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

def _merge_link_sources(values: set[str], batch: Any, source: str, path: Path) -> None:
    for identity, raw_sources in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        parsed = _parse_source_list(raw_sources, identity, path, "link_sources")
        if identity and source in parsed:
            values.add(str(identity))

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
    with open_parquet(path) as parquet_file:
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
    for batch in iter_record_batches(parquet_file, columns=columns, batch_size=65_536):
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
