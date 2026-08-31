"""Counter aggregation and final DatasetStats construction.

Owns the pure computation that turns the processed parquet files into
a :class:`DatasetStats` instance. The logic, the column set, and the
malformed-language JSON handling are unchanged from the documented
behavior. Dataset-size accounting is applied to every file we
attempted to read, including files we then skipped due to a PyArrow
read error -- the bytes-on-disk figure must be honest even when the
parquet content is unreadable.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from .models import DatasetStats
from .scanning import safe_metadata_row_count, safe_table, sorted_parquets

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.dataset_stats")
_METADATA_READ_WORKERS = 4
_LANGUAGE_CACHE_MAX_ENTRIES = 100_000


@dataclass(slots=True)
class _StatsAccumulator:
    polygon_count: int = 0
    unique_wikidata: set[str] = field(default_factory=set)
    polygons_with_wikipedia: int = 0
    polygons_with_text: int = 0
    polygons_with_english: int = 0
    polygons_with_no_english_other_lang: int = 0
    polygons_with_2plus_langs: int = 0
    polygons_with_5plus_langs: int = 0
    polygons_with_10plus_langs: int = 0
    distinct_regions: set[str] = field(default_factory=set)
    polygons_per_language: Counter[str] = field(default_factory=Counter)
    article_count: int = 0
    total_words: int = 0
    total_tokens_estimate: int = 0
    articles_per_language: Counter[str] = field(default_factory=Counter)
    link_count: int = 0
    dataset_size_bytes: int = 0
    language_lists: dict[str | bytes, tuple[str, ...]] = field(
        default_factory=dict,
        repr=False,
    )

    def to_stats(self) -> DatasetStats:
        return DatasetStats(
            polygon_count=self.polygon_count,
            unique_wikidata_count=len(self.unique_wikidata),
            article_count=self.article_count,
            link_count=self.link_count,
            language_count=len(self.articles_per_language),
            region_count=len(self.distinct_regions),
            total_words=self.total_words,
            total_tokens_estimate=self.total_tokens_estimate,
            dataset_size_bytes=self.dataset_size_bytes,
            polygons_with_wikipedia=self.polygons_with_wikipedia,
            polygons_with_text=self.polygons_with_text,
            polygons_with_english=self.polygons_with_english,
            polygons_with_no_english_other_lang=self.polygons_with_no_english_other_lang,
            polygons_with_2plus_langs=self.polygons_with_2plus_langs,
            polygons_with_5plus_langs=self.polygons_with_5plus_langs,
            polygons_with_10plus_langs=self.polygons_with_10plus_langs,
            articles_per_language=dict(self.articles_per_language.most_common()),
            polygons_per_language=dict(self.polygons_per_language.most_common()),
        )


def compute_dataset_stats(processed_dir: Path) -> DatasetStats:
    """Read all processed parquet files and compute factual stats.

    Reads only the columns needed for each metric (columnar pruning).
    Returns a :class:`DatasetStats` with every value derived from the
    data, never hardcoded.
    """
    canonical_documents_dir = processed_dir / "wikipedia" / "documents"
    articles_dir = (
        canonical_documents_dir if canonical_documents_dir.exists() else processed_dir / "articles"
    )
    stats = _StatsAccumulator()
    _accumulate_polygon_files(stats, processed_dir / "polygons")
    _accumulate_article_files(stats, articles_dir)
    _accumulate_link_files(stats, processed_dir / "polygon_articles")
    return stats.to_stats()


def _accumulate_polygon_files(stats: _StatsAccumulator, polygons_dir: Path) -> None:
    if not polygons_dir.exists():
        return
    columns = [
        "wikidata",
        "region",
        "has_wikipedia",
        "text_available",
        "has_english_wikipedia",
        "wikipedia_language_count",
        "wikipedia_languages",
    ]
    for parquet_path in sorted_parquets(polygons_dir):
        stats.dataset_size_bytes += parquet_path.stat().st_size
        table = safe_table(parquet_path, columns)
        if table is not None:
            _accumulate_polygon_table(stats, table)


def _accumulate_polygon_table(stats: _StatsAccumulator, table: pa.Table) -> None:
    stats.polygon_count += table.num_rows
    _update_unique_values(
        _unique_column_values(table.column("wikidata")),
        stats.unique_wikidata,
    )
    _update_unique_values(
        _unique_column_values(table.column("region")),
        stats.distinct_regions,
    )
    has_wikipedia = table.column("has_wikipedia")
    has_english = table.column("has_english_wikipedia")
    stats.polygons_with_wikipedia += _count_truthy_column(has_wikipedia)
    stats.polygons_with_text += _count_truthy_column(table.column("text_available"))
    stats.polygons_with_english += _count_truthy_column(has_english)
    stats.polygons_with_no_english_other_lang += _count_non_english_columns(
        has_wikipedia,
        has_english,
    )
    buckets = _count_language_bucket_column(table.column("wikipedia_language_count"))
    stats.polygons_with_2plus_langs += buckets[0]
    stats.polygons_with_5plus_langs += buckets[1]
    stats.polygons_with_10plus_langs += buckets[2]
    stats.polygons_per_language.update(
        _count_polygon_language_column(
            table.column("wikipedia_languages"),
            stats.language_lists,
        )
    )


def _update_unique_values(values: list[object], target: set[str]) -> None:
    for value in values:
        if value:
            target.add(str(value))


def _unique_column_values(values: pa.ChunkedArray) -> list[object]:
    if not _is_serialized_string(values.type):
        return values.to_pylist()  # type: ignore[no-untyped-call]
    return _compute_array("unique", values).to_pylist()  # type: ignore[no-untyped-call]


def _count_truthy(values: list[object]) -> int:
    return sum(1 for value in values if value)


def _count_truthy_column(values: pa.ChunkedArray) -> int:
    if not pa.types.is_boolean(values.type):
        return _count_truthy(values.to_pylist())  # type: ignore[no-untyped-call]
    return _scalar_int(_compute_scalar("sum", values))


def _count_non_english(wikipedia: list[object], english: list[object]) -> int:
    return sum(
        1
        for has_wiki, has_english in zip(wikipedia, english, strict=True)
        if has_wiki and not has_english
    )


def _count_non_english_columns(
    wikipedia: pa.ChunkedArray,
    english: pa.ChunkedArray,
) -> int:
    if not pa.types.is_boolean(wikipedia.type) or not pa.types.is_boolean(english.type):
        return _count_non_english(
            wikipedia.to_pylist(),  # type: ignore[no-untyped-call]
            english.to_pylist(),  # type: ignore[no-untyped-call]
        )
    has_no_english = _compute_array("invert", pc.fill_null(english, False))
    active = _compute_array("and", wikipedia, has_no_english)
    return _scalar_int(_compute_scalar("sum", active))


def _count_language_buckets(values: list[Any]) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for value in values:
        if value is None:
            continue
        counts[0] += int(value >= 2)
        counts[1] += int(value >= 5)
        counts[2] += int(value >= 10)
    return counts[0], counts[1], counts[2]


def _count_language_bucket_column(values: pa.ChunkedArray) -> tuple[int, int, int]:
    if not pa.types.is_integer(values.type):
        return _count_language_buckets(values.to_pylist())  # type: ignore[no-untyped-call]
    return (
        _count_at_least(values, 2),
        _count_at_least(values, 5),
        _count_at_least(values, 10),
    )


def _count_at_least(values: pa.ChunkedArray, minimum: int) -> int:
    matches = _compute_array("greater_equal", values, pa.scalar(minimum))
    return _scalar_int(_compute_scalar("sum", matches))


def _count_polygon_languages(values: list[object]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in values:
        languages = _parse_language_list(value)
        for language in languages:
            counts[str(language)] += 1
    return counts


def _count_polygon_language_column(
    values: pa.ChunkedArray,
    cache: dict[str | bytes, tuple[str, ...]],
) -> Counter[str]:
    if not _is_serialized_string(values.type):
        return _count_polygon_languages(values.to_pylist())  # type: ignore[no-untyped-call]
    counts: Counter[str] = Counter()
    for value, frequency in _arrow_value_counts(values):
        for language in _cached_languages(value, cache):
            counts[language] += frequency
    return counts


def _cached_languages(
    value: object,
    cache: dict[str | bytes, tuple[str, ...]],
) -> tuple[str, ...]:
    serialized = _serialized_language_value(value)
    if serialized is None:
        return ()
    cached = cache.get(serialized)
    if cached is not None:
        return cached
    return _decode_languages(serialized, cache)


def _serialized_language_value(value: object) -> str | bytes | None:
    if not isinstance(value, (str, bytes)):
        return None
    if not value or value == "[]":
        return None
    return value


def _decode_languages(
    value: str | bytes,
    cache: dict[str | bytes, tuple[str, ...]],
) -> tuple[str, ...]:
    languages = tuple(str(language) for language in _decoded_language_list(value))
    if len(cache) < _LANGUAGE_CACHE_MAX_ENTRIES:
        cache[value] = languages
    return languages


def _is_serialized_string(data_type: pa.DataType) -> bool:
    return bool(
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
    )


def _parse_language_list(value: object) -> list[object]:
    if not value or value == "[]":
        return []
    if not isinstance(value, (str, bytes, bytearray)):
        return []
    return _decoded_language_list(value)


def _decoded_language_list(value: str | bytes | bytearray) -> list[object]:
    try:
        languages = json.loads(value)
    except (ValueError, TypeError):
        return []
    return languages if isinstance(languages, list) else []


def _accumulate_article_files(stats: _StatsAccumulator, articles_dir: Path) -> None:
    if not articles_dir.exists():
        return
    columns = ["language", "article_length_words", "article_length_tokens_estimate"]
    for parquet_path in sorted_parquets(articles_dir):
        stats.dataset_size_bytes += parquet_path.stat().st_size
        table = safe_table(parquet_path, columns)
        if table is not None:
            _accumulate_article_table(stats, table)


def _accumulate_article_table(stats: _StatsAccumulator, table: pa.Table) -> None:
    stats.article_count += table.num_rows
    stats.articles_per_language.update(_count_value_strings(table.column("language")))
    stats.total_words += _sum_numeric_column(table.column("article_length_words"))
    stats.total_tokens_estimate += _sum_numeric_column(
        table.column("article_length_tokens_estimate")
    )


def _count_value_strings(values: pa.ChunkedArray) -> Counter[str]:
    if not _is_serialized_string(values.type):
        return _count_python_value_strings(values.to_pylist())  # type: ignore[no-untyped-call]
    return _count_serialized_value_strings(values)


def _count_python_value_strings(values: list[object]) -> Counter[str]:
    return Counter(str(value) for value in values if value)


def _count_serialized_value_strings(values: pa.ChunkedArray) -> Counter[str]:
    return Counter(
        {str(value): frequency for value, frequency in _arrow_value_counts(values) if value}
    )


def _arrow_value_counts(values: pa.ChunkedArray) -> list[tuple[object, int]]:
    frequencies = _compute_array("value_counts", values)
    distinct_values = frequencies.field("values").to_pylist()  # type: ignore[no-untyped-call]
    counts = frequencies.field("counts").to_pylist()  # type: ignore[no-untyped-call]
    return [(value, int(count)) for value, count in zip(distinct_values, counts, strict=True)]


def _sum_numeric(values: list[Any]) -> int:
    return sum(int(value) for value in values if value is not None)


def _sum_numeric_column(values: pa.ChunkedArray) -> int:
    if not pa.types.is_integer(values.type):
        return _sum_numeric(values.to_pylist())  # type: ignore[no-untyped-call]
    return _scalar_int(_compute_scalar("sum", values))


def _compute_array(
    function: str,
    *arguments: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Array | pa.ChunkedArray:
    result: Any = pc.call_function(function, list(arguments))
    return result


def _compute_scalar(
    function: str,
    *arguments: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Scalar:
    result: Any = pc.call_function(function, list(arguments))
    return result


def _scalar_int(value: pa.Scalar) -> int:
    raw = value.as_py()
    return 0 if raw is None else int(raw)


def _accumulate_link_files(stats: _StatsAccumulator, links_dir: Path) -> None:
    if not links_dir.exists():
        return
    parquet_paths = sorted_parquets(links_dir)
    stats.dataset_size_bytes += sum(path.stat().st_size for path in parquet_paths)
    stats.link_count += _count_link_rows(parquet_paths)


def _count_link_rows(parquet_paths: list[Path]) -> int:
    if not parquet_paths:
        return 0
    workers = min(_METADATA_READ_WORKERS, len(parquet_paths))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="dataset-stats-metadata",
    ) as executor:
        row_counts = executor.map(safe_metadata_row_count, parquet_paths)
        return sum(n_rows for n_rows in row_counts if n_rows is not None)


__all__ = ["compute_dataset_stats"]
