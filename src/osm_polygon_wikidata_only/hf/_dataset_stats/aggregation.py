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
from pathlib import Path

from .models import DatasetStats
from .scanning import safe_metadata_row_count, safe_table, sorted_parquets

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.dataset_stats")


def compute_dataset_stats(processed_dir: Path) -> DatasetStats:
    """Read all processed parquet files and compute factual stats.

    Reads only the columns needed for each metric (columnar pruning).
    Returns a :class:`DatasetStats` with every value derived from the
    data, never hardcoded.
    """
    polygons_dir = processed_dir / "polygons"
    canonical_documents_dir = processed_dir / "wikipedia" / "documents"
    legacy_articles_dir = processed_dir / "articles"
    articles_dir = (
        canonical_documents_dir if canonical_documents_dir.exists() else legacy_articles_dir
    )
    links_dir = processed_dir / "polygon_articles"

    polygon_count = 0
    unique_wikidata: set[str] = set()
    polygons_with_wikipedia = 0
    polygons_with_text = 0
    polygons_with_english = 0
    polygons_with_no_english_other_lang = 0
    polygons_with_2plus_langs = 0
    polygons_with_5plus_langs = 0
    polygons_with_10plus_langs = 0
    distinct_regions: set[str] = set()
    polygons_per_language: Counter[str] = Counter()
    dataset_size_bytes = 0

    if polygons_dir.exists():
        for parquet_path in sorted_parquets(polygons_dir):
            dataset_size_bytes += parquet_path.stat().st_size
            table = safe_table(
                parquet_path,
                [
                    "wikidata",
                    "region",
                    "has_wikipedia",
                    "text_available",
                    "has_english_wikipedia",
                    "wikipedia_language_count",
                    "wikipedia_languages",
                ],
            )
            if table is None:
                continue
            n = table.num_rows
            polygon_count += n

            for w in table.column("wikidata").to_pylist():
                if w:
                    unique_wikidata.add(str(w))

            for r in table.column("region").to_pylist():
                if r:
                    distinct_regions.add(str(r))

            for hw in table.column("has_wikipedia").to_pylist():
                if hw:
                    polygons_with_wikipedia += 1

            for tx in table.column("text_available").to_pylist():
                if tx:
                    polygons_with_text += 1

            for en in table.column("has_english_wikipedia").to_pylist():
                if en:
                    polygons_with_english += 1

            wiki = table.column("has_wikipedia").to_pylist()
            en_list = table.column("has_english_wikipedia").to_pylist()
            for hw, en in zip(wiki, en_list, strict=True):
                if hw and not en:
                    polygons_with_no_english_other_lang += 1

            for c in table.column("wikipedia_language_count").to_pylist():
                if c is not None:
                    if c >= 2:
                        polygons_with_2plus_langs += 1
                    if c >= 5:
                        polygons_with_5plus_langs += 1
                    if c >= 10:
                        polygons_with_10plus_langs += 1

            for langs_json in table.column("wikipedia_languages").to_pylist():
                if not langs_json:
                    continue
                try:
                    langs = json.loads(langs_json)
                except (ValueError, TypeError):
                    continue
                if isinstance(langs, list):
                    for lang in langs:
                        if lang:
                            polygons_per_language[str(lang)] += 1

    article_count = 0
    total_words = 0
    total_tokens_estimate = 0
    articles_per_language: Counter[str] = Counter()

    if articles_dir.exists():
        for parquet_path in sorted_parquets(articles_dir):
            dataset_size_bytes += parquet_path.stat().st_size
            table = safe_table(
                parquet_path,
                [
                    "language",
                    "article_length_words",
                    "article_length_tokens_estimate",
                ],
            )
            if table is None:
                continue
            article_count += table.num_rows
            for lang in table.column("language").to_pylist():
                if lang:
                    articles_per_language[str(lang)] += 1
            for w in table.column("article_length_words").to_pylist():
                if w is not None:
                    total_words += int(w)
            for t in table.column("article_length_tokens_estimate").to_pylist():
                if t is not None:
                    total_tokens_estimate += int(t)

    link_count = 0
    if links_dir.exists():
        for parquet_path in sorted_parquets(links_dir):
            dataset_size_bytes += parquet_path.stat().st_size
            n_rows = safe_metadata_row_count(parquet_path)
            if n_rows is None:
                continue
            link_count += n_rows

    return DatasetStats(
        polygon_count=polygon_count,
        unique_wikidata_count=len(unique_wikidata),
        article_count=article_count,
        link_count=link_count,
        language_count=len(articles_per_language),
        region_count=len(distinct_regions),
        total_words=total_words,
        total_tokens_estimate=total_tokens_estimate,
        dataset_size_bytes=dataset_size_bytes,
        polygons_with_wikipedia=polygons_with_wikipedia,
        polygons_with_text=polygons_with_text,
        polygons_with_english=polygons_with_english,
        polygons_with_no_english_other_lang=polygons_with_no_english_other_lang,
        polygons_with_2plus_langs=polygons_with_2plus_langs,
        polygons_with_5plus_langs=polygons_with_5plus_langs,
        polygons_with_10plus_langs=polygons_with_10plus_langs,
        articles_per_language=dict(articles_per_language.most_common()),
        polygons_per_language=dict(polygons_per_language.most_common()),
    )


__all__ = ["compute_dataset_stats"]
