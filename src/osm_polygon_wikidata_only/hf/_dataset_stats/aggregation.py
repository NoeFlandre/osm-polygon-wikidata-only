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
from typing import Any, cast

import pyarrow as pa

from .models import DatasetStats
from .scanning import safe_metadata_row_count, safe_table, sorted_parquets

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.dataset_stats")
_METADATA_READ_WORKERS = 4


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
    _update_unique_values(table.column("wikidata").to_pylist(), stats.unique_wikidata)  # type: ignore[no-untyped-call]
    _update_unique_values(table.column("region").to_pylist(), stats.distinct_regions)  # type: ignore[no-untyped-call]
    stats.polygons_with_wikipedia += _count_truthy(table.column("has_wikipedia").to_pylist())  # type: ignore[no-untyped-call]
    stats.polygons_with_text += _count_truthy(table.column("text_available").to_pylist())  # type: ignore[no-untyped-call]
    stats.polygons_with_english += _count_truthy(table.column("has_english_wikipedia").to_pylist())  # type: ignore[no-untyped-call]
    stats.polygons_with_no_english_other_lang += _count_non_english(  # type: ignore[no-untyped-call]
        table.column("has_wikipedia").to_pylist(), table.column("has_english_wikipedia").to_pylist()
    )
    buckets = _count_language_buckets(table.column("wikipedia_language_count").to_pylist())  # type: ignore[no-untyped-call]
    stats.polygons_with_2plus_langs += buckets[0]
    stats.polygons_with_5plus_langs += buckets[1]
    stats.polygons_with_10plus_langs += buckets[2]
    stats.polygons_per_language.update(  # type: ignore[no-untyped-call]
        _count_polygon_languages(table.column("wikipedia_languages").to_pylist())
    )


def _update_unique_values(values: list[object], target: set[str]) -> None:
    for value in values:
        if value:
            target.add(str(value))


def _count_truthy(values: list[object]) -> int:
    return sum(1 for value in values if value)


def _count_non_english(wikipedia: list[object], english: list[object]) -> int:
    return sum(
        1
        for has_wiki, has_english in zip(wikipedia, english, strict=True)
        if has_wiki and not has_english
    )


def _count_language_buckets(values: list[object]) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for value in values:
        if value is None:
            continue
        number = cast(Any, value)
        counts[0] += int(number >= 2)
        counts[1] += int(number >= 5)
        counts[2] += int(number >= 10)
    return counts[0], counts[1], counts[2]


def _count_polygon_languages(values: list[object]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in values:
        languages = _parse_language_list(value)
        for language in languages:
            counts[str(language)] += 1
    return counts


def _parse_language_list(value: object) -> list[object]:
    if not value or value == "[]":
        return []
    if not isinstance(value, (str, bytes, bytearray)):
        return []
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
    stats.articles_per_language.update(  # type: ignore[no-untyped-call]
        str(language)
        for language in table.column("language").to_pylist()  # type: ignore[no-untyped-call]
        if language
    )
    stats.total_words += _sum_numeric(table.column("article_length_words").to_pylist())  # type: ignore[no-untyped-call]
    stats.total_tokens_estimate += _sum_numeric(  # type: ignore[union-attr]
        table.column("article_length_tokens_estimate").to_pylist()
    )


def _sum_numeric(values: list[object]) -> int:
    return sum(int(cast(Any, value)) for value in values if value is not None)


def _accumulate_link_files(stats: _StatsAccumulator, links_dir: Path) -> None:
    if not links_dir.exists():
        return
    parquet_paths = sorted_parquets(links_dir)
    for parquet_path in parquet_paths:
        stats.dataset_size_bytes += parquet_path.stat().st_size
    if not parquet_paths:
        return
    workers = min(_METADATA_READ_WORKERS, len(parquet_paths))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="dataset-stats-metadata",
    ) as executor:
        row_counts = executor.map(safe_metadata_row_count, parquet_paths)
        for n_rows in row_counts:
            if n_rows is not None:
                stats.link_count += n_rows


__all__ = ["compute_dataset_stats"]
