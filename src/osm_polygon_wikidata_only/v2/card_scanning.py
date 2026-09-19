"""Parquet discovery and columnar scanning primitives for V2 cards."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.hf._geographic.polygon_identities import (
    PolygonIdentity,
    PolygonIndex,
    load_unique_polygon_records,
)
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.card_models import (
    _CardFiles,
    _DocumentMetrics,
    _PolygonMetrics,
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
    with open_parquet(path) as parquet_file:
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
    for batch in iter_record_batches(parquet_file, columns=columns, batch_size=65_536):
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
    with open_parquet(path) as parquet_file:
        columns = {"polygon_id", "document_id", "project"}
        if not columns.issubset(parquet_file.schema_arrow.names):
            return
        for batch in iter_record_batches(
            parquet_file,
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
    with open_parquet(path) as parquet_file:
        if not {"polygon_id", "document_id"}.issubset(parquet_file.schema_arrow.names):
            return values
        columns = ["polygon_id", "document_id"]
        has_project = "project" in parquet_file.schema_arrow.names
        if has_project:
            columns.append("project")
        for batch in iter_record_batches(parquet_file, columns=columns, batch_size=65_536):
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


