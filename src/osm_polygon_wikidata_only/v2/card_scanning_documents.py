"""Single-pass document scanning for V2 card metrics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.v2.card_models import DocumentMetrics as _DocumentMetrics
from osm_polygon_wikidata_only.v2.card_scanning import (
    batch_column as _batch_column,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    batch_value as _batch_value,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    has_non_empty_words as _has_non_empty_words,
)
from osm_polygon_wikidata_only.v2.card_scanning import (
    word_column as _word_column,
)

_SUCCESSFUL_FETCH_STATUS = "ok"
_TEXT_DOCUMENT_PROJECTS = frozenset({"wikipedia", "wikivoyage"})


def _scan_document_metrics(
    all_document_paths: list[Path] | tuple[Path, ...],
    wikipedia_document_paths: list[Path] | tuple[Path, ...],
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


# Public collaborator spellings keep the split scanner modules independent of
# the legacy facade's private compatibility names.
scan_document_metrics = _scan_document_metrics
scan_document_file = _scan_document_file
document_batch_columns = _document_batch_columns
document_column_names = _document_column_names
document_columns = _document_columns
document_metric_columns = _document_metric_columns
record_document_language = _record_document_language
record_document_row = _record_document_row
record_document_text = _record_document_text
record_document_words = _record_document_words
record_non_empty_text_document = _record_non_empty_text_document
scan_document_batch = _scan_document_batch
scan_document_batches = _scan_document_batches
