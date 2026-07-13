"""Markdown rendering for the dataset stats.

Owns the exact Markdown sections, table layouts, whitespace, and
number/percentage/size formatters that turn a :class:`DatasetStats`
into the README stats block. Output is byte-stable across runs for a
given stats instance.
"""

from __future__ import annotations

from .models import DatasetStats

__all__ = ["render_stats_section"]


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
