"""Markdown rendering for the dataset stats.

Owns the exact Markdown sections, table layouts, whitespace, and
number/percentage/size formatters that turn a :class:`DatasetStats`
into the README stats block. Output is byte-stable across runs for a
given stats instance. When an :class:`AugmentationStats` snapshot is
provided, additional factual sections are appended in a documented
order. The legacy three-section output is preserved byte-for-byte
when no augmentation snapshot is provided.

This module is intentionally side-effect free. It does not import
the scanner and does not compute anything itself; the publication
layer is responsible for producing both snapshots, and the renderer
just renders whatever it is given.
"""

from __future__ import annotations

from .models import AugmentationStats, DatasetStats, ProjectTextStats

__all__ = ["render_stats_section"]


def render_stats_section(
    stats: DatasetStats,
    *,
    augmentation_stats: AugmentationStats | None = None,
) -> str:
    """Render the factual README stats sections as markdown.

    Existing callers that pass only ``stats`` receive the legacy
    three sections unchanged. When ``augmentation_stats`` is supplied,
    the headline table is extended with concise augmentation totals
    and additional sections are appended after the legacy three.
    """
    parts: list[str] = []
    parts.append("## Dataset snapshot\n")
    parts.append(_render_headline_table(stats, augmentation_stats))
    parts.append("\n## Wikipedia coverage funnel\n")
    parts.append(_render_funnel_table(stats))
    parts.append("\n## Language distribution\n")
    parts.append(_render_language_section(stats))
    if augmentation_stats is not None:
        parts.append("\n## Augmentation coverage\n")
        parts.append(_render_augmentation_coverage_table(augmentation_stats))
        parts.append("\n## Storage accounting\n")
        parts.append(_render_storage_size_rows(augmentation_stats))
        parts.append("\n## Wikipedia text corpus\n")
        parts.append(
            _render_project_section(
                "Documents",
                augmentation_stats.wikipedia_documents,
                kind="documents",
            )
            + "\n"
            + _render_project_section(
                "Sections",
                augmentation_stats.wikipedia_sections,
                kind="sections",
            )
            + "\n"
            + "### Languages\n\n"
            + _render_top_languages(augmentation_stats.wikipedia_documents)
            + "\n"
        )
        parts.append("\n## Wikivoyage text corpus\n")
        parts.append(
            _render_project_section(
                "Documents",
                augmentation_stats.wikivoyage_documents,
                kind="documents",
            )
            + "\n"
            + _render_project_section(
                "Sections",
                augmentation_stats.wikivoyage_sections,
                kind="sections",
            )
            + "\n"
            + "### Languages\n\n"
            + _render_top_languages(augmentation_stats.wikivoyage_documents)
            + "\n"
        )
        parts.append("\n## Wikidata facts\n")
        parts.append(_render_wikidata_facts_section(augmentation_stats) + "\n")
        if augmentation_stats.unreadable_file_count > 0:
            parts.append(
                "\n> Statistics exclude "
                f"{augmentation_stats.unreadable_file_count} unreadable sidecar "
                "file(s); see generation logs.\n"
            )
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Legacy tables (unchanged)
# ---------------------------------------------------------------------------


def _render_headline_table(
    stats: DatasetStats,
    augmentation_stats: AugmentationStats | None = None,
) -> str:
    """Render the headline table.

    Without augmentation, the rows are exactly the documented eight;
    the legacy ``Dataset size on disk`` label MUST stay preserved
    byte-for-byte. With augmentation, the last row is renamed to
    ``Core tables size`` and additional augmentation totals follow.
    """
    rows = [
        ("Polygons", _fmt_int(stats.polygon_count)),
        ("Unique Wikidata entities", _fmt_int(stats.unique_wikidata_count)),
        ("Wikipedia articles", _fmt_int(stats.article_count)),
        ("Polygon-article links", _fmt_int(stats.link_count)),
        ("Languages", _fmt_int(stats.language_count)),
        ("Geographic regions", _fmt_int(stats.region_count)),
        ("Total words", _fmt_int(stats.total_words)),
        (
            "Core tables size" if augmentation_stats is not None else "Dataset size on disk",
            _fmt_size(stats.dataset_size_bytes),
        ),
    ]
    if augmentation_stats is not None:
        aug = augmentation_stats
        rows.extend(
            [
                (
                    "Wikipedia documents",
                    _fmt_int(aug.wikipedia_documents.rows),
                ),
                (
                    "Wikipedia sections",
                    _fmt_int(aug.wikipedia_sections.rows),
                ),
                (
                    "Wikivoyage documents",
                    _fmt_int(aug.wikivoyage_documents.rows),
                ),
                (
                    "Wikivoyage sections",
                    _fmt_int(aug.wikivoyage_sections.rows),
                ),
                (
                    "Wikidata facts",
                    _fmt_int(aug.wikidata_facts.rows),
                ),
                (
                    "Fully augmented regions",
                    _fmt_int(aug.fully_augmented_count),
                ),
                (
                    "Document corpus words",
                    _fmt_int(_document_corpus_words(aug)),
                ),
                (
                    "Augmentation tables size",
                    _fmt_size(aug.augmentation_parquet_bytes),
                ),
                (
                    "Total Parquet size",
                    _fmt_size(aug.total_parquet_bytes),
                ),
            ]
        )
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _document_corpus_words(aug: AugmentationStats) -> int:
    """Sum the document-only word totals across Wikipedia + Wikivoyage.

    We deliberately EXCLUDE the section totals so the headline is not
    inflated by double-counting the same underlying text (each
    Wikipedia document's body is also split into sections). Section
    word totals stay in their individual sections.
    """
    return aug.wikipedia_documents.total_words + aug.wikivoyage_documents.total_words


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


# ---------------------------------------------------------------------------
# Augmentation tables
# ---------------------------------------------------------------------------


def _render_augmentation_coverage_table(stats: AugmentationStats) -> str:
    total = max(stats.core_region_count, 1)
    lines = ["| Metric | Count | Percentage |", "| --- | ---: | ---: |"]
    lines.append(f"| Core regions | {_fmt_int(stats.core_region_count)} | 100.0% |")
    fully_pct = (stats.fully_augmented_count / total) * 100.0
    partial_pct = (stats.partial_augmented_count / total) * 100.0
    not_pct = (stats.not_augmented_count / total) * 100.0
    lines.append(
        f"| Fully augmented | {_fmt_int(stats.fully_augmented_count)} | {_fmt_pct(fully_pct)} |"
    )
    lines.append(
        f"| Partially augmented | {_fmt_int(stats.partial_augmented_count)} | {_fmt_pct(partial_pct)} |"
    )
    lines.append(f"| Not augmented | {_fmt_int(stats.not_augmented_count)} | {_fmt_pct(not_pct)} |")
    orphan_text = _fmt_int(len(stats.orphan_sidecar_stems)) if stats.orphan_sidecar_stems else "0"
    lines.append(f"| Orphan sidecar stems | {orphan_text} | - |")
    lines.append("")
    lines.append(
        "Augmentation is additive and a zero-row sidecar may still "
        "represent a completed region. Orphan sidecars (a sidecar with "
        "no matching core polygon) do not count toward core regions."
    )
    if stats.orphan_sidecar_stems:
        orphan_list = ", ".join(stats.orphan_sidecar_stems)
        lines.append("")
        lines.append(f"Orphan stems: {orphan_list}")
    return "\n".join(lines)


def _render_project_section(title: str, project: ProjectTextStats, *, kind: str) -> str:
    """Render a Wikipedia or Wikivoyage documents / sections subsection.

    Distinguishes three sidecar states:

    * Missing sub-directory (``subdir_present is False``) → "No data
      exists yet."
    * Present sidecar with ``rows == 0`` → "This sidecar is present
      but empty."
    * Present sidecar with rows → the metric table.
    """
    lines = [f"### {title}", ""]
    if not project.subdir_present:
        lines.append("No data exists yet.")
        return "\n".join(lines)
    if project.rows == 0:
        lines.append("This sidecar is present but empty.")
        return "\n".join(lines)
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    if kind == "documents":
        lines.append(f"| Document rows | {_fmt_int(project.rows)} |")
        lines.append(f"| Unique documents | {_fmt_int(project.unique_documents)} |")
    else:
        lines.append(f"| Section rows | {_fmt_int(project.rows)} |")
        lines.append(f"| Unique sections | {_fmt_int(project.unique_section_ids)} |")
        lines.append(f"| Documents represented | {_fmt_int(project.unique_documents)} |")
        lines.append(
            f"| Avg sections per represented document | {_avg_sections(project.avg_sections_per_doc)} |"
        )
        lines.append(f"| Non-empty section rate | {_fmt_pct(project.non_empty_rate * 100.0)} |")
    lines.append(f"| Unique Wikidata entities | {_fmt_int(project.unique_qids)} |")
    lines.append(f"| Languages | {_fmt_int(project.language_count)} |")
    lines.append(f"| Non-empty text | {_fmt_int(project.non_empty)} |")
    lines.append(f"| Empty / null text | {_fmt_int(project.empty_or_null)} |")
    lines.append(f"| Characters | {_fmt_int(project.total_chars)} |")
    lines.append(f"| Words | {_fmt_int(project.total_words)} |")
    lines.append(f"| Estimated tokens | {_fmt_int(project.total_tokens_estimate)} |")
    lines.append(f"| Regions / files represented | {_fmt_int(project.region_count)} |")
    return "\n".join(lines) + "\n"


def _render_top_languages(project: ProjectTextStats) -> str:
    if not project.top_languages:
        return "No language data yet."
    lines = []
    lines.append("| Language | Documents | % of total |")
    lines.append("| --- | ---: | ---: |")
    total_rows = max(project.rows, 1)
    for lang, count in project.top_languages:
        pct = (count / total_rows) * 100.0
        lines.append(f"| {lang} | {_fmt_int(count)} | {_fmt_pct(pct)} |")
    return "\n".join(lines)


def _render_wikidata_facts_section(stats: AugmentationStats) -> str:
    facts = stats.wikidata_facts
    lines: list[str] = []
    if not facts.subdir_present:
        lines.append("No data exists yet.")
        return "\n".join(lines)
    if facts.rows == 0:
        lines.append("This sidecar is present but empty.")
        return "\n".join(lines)
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Fact rows | {_fmt_int(facts.rows)} |")
    lines.append(f"| Unique facts | {_fmt_int(facts.unique_facts)} |")
    lines.append(f"| Unique subjects | {_fmt_int(facts.unique_subjects)} |")
    lines.append(f"| Distinct properties | {_fmt_int(facts.distinct_property_ids)} |")
    lines.append(f"| Non-empty English property label | {_fmt_int(facts.with_property_en_label)} |")
    lines.append(f"| Non-empty English value label | {_fmt_int(facts.with_value_en_label)} |")
    lines.append(f"| With qualifiers | {_fmt_int(facts.with_qualifiers)} |")
    lines.append(f"| With references | {_fmt_int(facts.with_references)} |")
    lines.append(f"| Unreadable qualifier JSON | {_fmt_int(facts.unavailable_qualifiers)} |")
    lines.append(f"| Unreadable references JSON | {_fmt_int(facts.unavailable_references)} |")
    lines.append(f"| Regions / files represented | {_fmt_int(facts.region_count)} |")
    if facts.value_type_distribution:
        lines.append("")
        lines.append("**Value-type distribution:**")
        lines.append("")
        lines.append("| Value type | Facts |")
        lines.append("| --- | ---: |")
        for value_type, count in facts.value_type_distribution:
            lines.append(f"| {value_type} | {_fmt_int(count)} |")
    lines.append("")
    lines.append("**Top properties:**")
    lines.append("")
    lines.append("| Property ID | English label | Facts |")
    lines.append("| --- | --- | ---: |")
    for property_id, label, count in facts.top_properties:
        display_label = label if label else "(no English label)"
        lines.append(f"| {property_id} | {display_label} | {_fmt_int(count)} |")
    lines.append("")
    lines.append(
        "English labels are requested where available; multilingual labels are "
        "preserved verbatim in the Parquet column `property_labels`."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Storage accounting
# ---------------------------------------------------------------------------


def _render_storage_size_rows(stats: AugmentationStats) -> str:
    """Storage-size table rendered at the end of the snapshot block."""
    lines = [
        "Core tables size and total storage size are additive:",
        "",
        "| Metric | Bytes | Human-readable |",
        "| --- | ---: | --- |",
        f"| Core tables size | {_fmt_int(stats.core_parquet_bytes)} | "
        f"{_fmt_size(stats.core_parquet_bytes)} |",
        f"| Augmentation tables size | {_fmt_int(stats.augmentation_parquet_bytes)} | "
        f"{_fmt_size(stats.augmentation_parquet_bytes)} |",
        f"| Total Parquet size | {_fmt_int(stats.total_parquet_bytes)} | "
        f"{_fmt_size(stats.total_parquet_bytes)} |",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _avg_sections(value: float) -> str:
    """Render the average sections per represented document float.

    Returns only the two-decimal float string. The caller (the section
    metrics renderer) is responsible for closing the markdown row.
    """
    return f"{value:.2f}"
