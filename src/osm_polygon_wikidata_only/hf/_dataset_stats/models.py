"""DatasetStats frozen dataclass.

Re-exported by the :mod:`osm_polygon_wikidata_only.hf.dataset_stats`
facade. Field order, frozen flag, and ``slots=True`` are part of the
documented contract; do not reorder.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Factual snapshot of the processed dataset.

    All counts and aggregates are computed from the processed parquet
    files at the time :func:`compute_dataset_stats` is called.
    """

    polygon_count: int
    unique_wikidata_count: int
    article_count: int
    link_count: int
    language_count: int
    region_count: int
    total_words: int
    total_tokens_estimate: int
    dataset_size_bytes: int

    # Wikipedia coverage funnel
    polygons_with_wikipedia: int
    polygons_with_text: int
    polygons_with_english: int
    polygons_with_no_english_other_lang: int
    polygons_with_2plus_langs: int
    polygons_with_5plus_langs: int
    polygons_with_10plus_langs: int

    # Language distribution (sorted by count descending)
    articles_per_language: dict[str, int]
    polygons_per_language: dict[str, int]
