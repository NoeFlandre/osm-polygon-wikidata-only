"""Dataset statistics: compute factual stats from processed data and render them as markdown.

This module reads the processed parquet files directly (columnar pruning
keeps it fast) and produces a :class:`DatasetStats` snapshot. The
:func:`render_stats_section` function turns that snapshot into the
factual README sections: dataset snapshot table, Wikipedia coverage
funnel, and language distribution.

All values are computed from the data, never hardcoded. The tests in
``tests/test_dataset_stats.py`` cross-check the computed values
against a manual count over known fixture data.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)


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


def compute_dataset_stats(processed_dir: Path) -> DatasetStats:
    """Read all processed parquet files and compute factual stats.

    Reads only the columns needed for each metric (columnar pruning).
    Returns a :class:`DatasetStats` with every value derived from the
    data, never hardcoded.
    """
    polygons_dir = processed_dir / "polygons"
    articles_dir = processed_dir / "articles"
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
        for parquet_path in sorted(polygons_dir.glob("*.parquet")):
            dataset_size_bytes += parquet_path.stat().st_size
            try:
                table = pq.read_table(  # type: ignore[no-untyped-call]
                    parquet_path,
                    columns=[
                        "wikidata",
                        "region",
                        "has_wikipedia",
                        "text_available",
                        "has_english_wikipedia",
                        "wikipedia_language_count",
                        "wikipedia_languages",
                    ],
                )
            except (OSError, KeyError) as e:
                LOGGER.warning("Skipping %s: %s", parquet_path, e)
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
        for parquet_path in sorted(articles_dir.glob("*.parquet")):
            dataset_size_bytes += parquet_path.stat().st_size
            try:
                table = pq.read_table(  # type: ignore[no-untyped-call]
                    parquet_path,
                    columns=[
                        "language",
                        "article_length_words",
                        "article_length_tokens_estimate",
                    ],
                )
            except (OSError, KeyError) as e:
                LOGGER.warning("Skipping %s: %s", parquet_path, e)
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
        for parquet_path in sorted(links_dir.glob("*.parquet")):
            dataset_size_bytes += parquet_path.stat().st_size
            try:
                link_count += pq.read_metadata(parquet_path).num_rows  # type: ignore[no-untyped-call]
            except (OSError, KeyError) as e:
                LOGGER.warning("Skipping %s: %s", parquet_path, e)

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


def render_stats_section(stats: DatasetStats) -> str:
    """Render the factual README stats sections as markdown."""
    parts: list[str] = []
    parts.append("## Dataset snapshot\n")
    parts.append(_render_headline_table(stats))
    parts.append("\n## Wikipedia coverage funnel\n")
    parts.append(_render_funnel_table(stats))
    parts.append("\n## Language distribution\n")
    parts.append(_render_language_section(stats))
    return "\n".join(parts) + "\n"


def _render_headline_table(stats: DatasetStats) -> str:
    rows = [
        ("Polygons", _fmt_int(stats.polygon_count)),
        ("Unique Wikidata entities", _fmt_int(stats.unique_wikidata_count)),
        ("Wikipedia articles", _fmt_int(stats.article_count)),
        ("Polygon-article links", _fmt_int(stats.link_count)),
        ("Languages", _fmt_int(stats.language_count)),
        ("Geographic regions", _fmt_int(stats.region_count)),
        ("Total words", _fmt_int(stats.total_words)),
        ("Dataset size on disk", _fmt_size(stats.dataset_size_bytes)),
    ]
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _render_funnel_table(stats: DatasetStats) -> str:
    total = max(stats.polygon_count, 1)
    stages: list[tuple[str, int]] = [
        ("All polygons", stats.polygon_count),
        ("With >=1 article", stats.polygons_with_wikipedia),
        ("With non-empty text", stats.polygons_with_text),
        ("With English coverage", stats.polygons_with_english),
        ("No English, another language", stats.polygons_with_no_english_other_lang),
        ("2+ languages", stats.polygons_with_2plus_langs),
        ("5+ languages", stats.polygons_with_5plus_langs),
        ("10+ languages", stats.polygons_with_10plus_langs),
    ]
    lines = ["| Stage | Count | % of all polygons |", "| --- | ---: | ---: |"]
    for label, count in stages:
        pct = (count / total) * 100.0
        lines.append(f"| {label} | {_fmt_int(count)} | {_fmt_pct(pct)} |")
    return "\n".join(lines)


def _render_language_section(stats: DatasetStats) -> str:
    if not stats.articles_per_language:
        return "No language data yet.\n"

    top_articles = list(stats.articles_per_language.items())[:20]
    top_polygons = dict(stats.polygons_per_language)

    lines = ["Top 20 languages by article count:", ""]
    lines.append("| Language | Articles | % of total | Polygons |")
    lines.append("| --- | ---: | ---: | ---: |")
    total_articles = max(stats.article_count, 1)
    for lang, count in top_articles:
        pct = (count / total_articles) * 100.0
        poly_count = top_polygons.get(lang, 0)
        lines.append(f"| {lang} | {_fmt_int(count)} | {_fmt_pct(pct)} | {_fmt_int(poly_count)} |")

    lines.append("")
    lines.append("**Concentration:**")
    for n in (1, 5, 10, 20):
        top_n_sum = sum(c for _, c in list(stats.articles_per_language.items())[:n])
        pct = (top_n_sum / total_articles) * 100.0
        lines.append(f"- Top {n} language{'s' if n > 1 else ''}: {_fmt_pct(pct)} of all articles")

    lines.append("")
    lines.append("**Long-tail:**")
    lines.append(f"- {stats.language_count} language(s) total")
    tail_counts = _count_long_tail(stats.articles_per_language, stats.polygons_per_language)
    for threshold_key, threshold_label in (
        ("articles_lt1", "1"),
        ("articles_lt5", "5"),
        ("articles_lt10", "10"),
    ):
        lines.append(
            f"- {tail_counts[threshold_key]} language(s) appear in fewer than "
            f"{threshold_label} article(s)"
        )
    lines.append(f"- {tail_counts['polygons_lt5']} language(s) appear in fewer than 5 polygons")
    return "\n".join(lines)


def _count_long_tail(
    articles_per_language: dict[str, int],
    polygons_per_language: dict[str, int],
) -> dict[str, int]:
    out: dict[str, int] = {
        "articles_lt1": 0,
        "articles_lt5": 0,
        "articles_lt10": 0,
        "polygons_lt5": 0,
    }
    for count in articles_per_language.values():
        if count < 1:
            out["articles_lt1"] += 1
        if count < 5:
            out["articles_lt5"] += 1
        if count < 10:
            out["articles_lt10"] += 1
    for count in polygons_per_language.values():
        if count < 5:
            out["polygons_lt5"] += 1
    return out


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


__all__ = [
    "DatasetStats",
    "compute_dataset_stats",
    "render_stats_section",
]
