"""Parquet discovery and columnar scanning primitives for V2 cards."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.v2.card_models import (
    CardFiles as _CardFiles,
)
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

_METADATA_READ_WORKERS = 4
_SUCCESSFUL_FETCH_STATUS = "ok"
_TEXT_DOCUMENT_PROJECTS = frozenset({"wikipedia", "wikivoyage"})


def collect_card_files(processed_v2: Path) -> _CardFiles:
    stems = tuple(sorted(load_v2_manifest(processed_v2)))
    polygon_files = manifest_files(processed_v2 / "polygons", stems)
    wikipedia_document_files = manifest_files(processed_v2 / "wikipedia/documents", stems)
    wikipedia_section_files = manifest_files(processed_v2 / "wikipedia/sections", stems)
    wikivoyage_document_files = manifest_files(processed_v2 / "wikivoyage/documents", stems)
    wikivoyage_section_files = manifest_files(processed_v2 / "wikivoyage/sections", stems)
    wikidata_fact_files = manifest_files(processed_v2 / "wikidata/facts", stems)
    link_files = manifest_files(processed_v2 / "polygon_document_links", stems)
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


def v1_wikipedia_document_files(processed: Path) -> list[Path]:
    wikipedia = sorted((processed / "wikipedia/documents").glob("*.parquet"))
    if not wikipedia:
        wikipedia = sorted((processed / "articles").glob("*.parquet"))
    return wikipedia


def v1_document_files(processed: Path) -> list[Path]:
    return v1_wikipedia_document_files(processed) + sorted(
        (processed / "wikivoyage/documents").glob("*.parquet")
    )


def v1_section_files(processed: Path) -> list[Path]:
    return sorted((processed / "wikipedia/sections").glob("*.parquet")) + sorted(
        (processed / "wikivoyage/sections").glob("*.parquet")
    )


def manifest_files(directory: Path, stems: Iterable[str]) -> list[Path]:
    return [path for stem in stems if (path := directory / f"{stem}.parquet").is_file()]


def sum_metadata(paths: Iterable[Path], *, executor_factory=ThreadPoolExecutor) -> int:
    materialized = tuple(paths)
    if not materialized:
        return 0
    workers = min(_METADATA_READ_WORKERS, len(materialized))
    with executor_factory(
        max_workers=workers,
        thread_name_prefix="v2-card-metadata",
    ) as executor:
        return sum(executor.map(metadata_row_count, materialized))


def metadata_row_count(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def unique_values(paths: Iterable[Path], column: str) -> set[str]:
    values: set[str] = set()
    for path in paths:
        values.update(unique_values_file(path, column))
    return values


def unique_values_file(path: Path, column: str) -> set[str]:
    values: set[str] = set()
    with open_parquet(path) as parquet_file:
        if column not in parquet_file.schema_arrow.names:
            return values
        for batch in iter_record_batches(parquet_file, columns=[column], batch_size=65_536):
            values.update(non_empty_strings(batch.column(0).to_pylist()))
    return values


def sum_first_available(paths: Iterable[Path], columns: tuple[str, ...]) -> int:
    return sum(sum_first_available_file(path, columns) for path in paths)


def sum_first_available_file(path: Path, columns: tuple[str, ...]) -> int:
    with open_parquet(path) as parquet_file:
        column = first_present_column(set(parquet_file.schema_arrow.names), columns)
        if column is None:
            return 0
        return sum(
            sum(int(value or 0) for value in batch.column(0).to_pylist())
            for batch in iter_record_batches(parquet_file, columns=[column], batch_size=65_536)
        )


def first_present_column(names: set[str], columns: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in columns if candidate in names), None)


def unique_numeric_values(
    paths: Iterable[Path], key_column: str, value_columns: tuple[str, ...]
) -> dict[str, int]:
    """Return one stable numeric value for each document identity."""
    values: dict[str, int] = {}
    for path in paths:
        merge_numeric_file(values, path, key_column, value_columns)
    return values


def merge_numeric_file(
    values: dict[str, int],
    path: Path,
    key_column: str,
    value_columns: tuple[str, ...],
) -> None:
    with open_parquet(path) as parquet_file:
        names = set(parquet_file.schema_arrow.names)
        value_column = first_present_column(names, value_columns)
        if key_column not in names or value_column is None:
            return
        merge_numeric_batches(values, parquet_file, key_column, value_column)


def merge_numeric_batches(
    values: dict[str, int],
    parquet_file: Any,
    key_column: str,
    value_column: str,
) -> None:
    for batch in iter_record_batches(
        parquet_file, columns=[key_column, value_column], batch_size=65_536
    ):
        merge_numeric_batch(values, batch, value_column)


def merge_numeric_batch(values: dict[str, int], batch: Any, value_column: str) -> None:
    for identity, value in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        record_numeric_value(values, identity, value, value_column)


def record_numeric_value(
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


def field_values_for_ids(
    paths: Iterable[Path],
    key_column: str,
    value_column: str,
    identities: set[str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    if not identities:
        return values
    for path in paths:
        merge_field_values_file(values, path, key_column, value_column, identities)
    return values


def merge_field_values_file(
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
            merge_field_values_batch(values, batch, identities)


def merge_field_values_batch(
    values: dict[str, str],
    batch: Any,
    identities: set[str],
) -> None:
    for identity, value in zip(
        batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
    ):
        if identity in identities and value not in (None, ""):
            values[str(identity)] = str(value)


def word_column(names: set[str]) -> str | None:
    if "article_length_words" in names:
        return "article_length_words"
    if "text_length_words" in names:
        return "text_length_words"
    return None


def has_non_empty_words(identity: Any, word_count: Any) -> bool:
    return bool(identity and int(word_count or 0) > 0)


def non_empty_strings(values: list[Any]) -> list[str]:
    return [str(value) for value in values if value]
