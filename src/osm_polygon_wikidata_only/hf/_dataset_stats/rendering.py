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

from collections.abc import Sequence

from .models import (
    AugmentationStats,
    CombinedLanguageStats,
    DatasetStats,
    ProjectTextStats,
    WikidataFactStats,
)

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
    parts = ["## Dataset snapshot\n", _render_headline_table(stats, augmentation_stats)]
    if augmentation_stats is None:
        parts.extend(["\n## Wikipedia coverage funnel\n", _render_funnel_table(stats)])
    parts.extend(
        ["\n## Language distribution\n", _render_language_distribution(stats, augmentation_stats)]
    )
    if augmentation_stats is not None:
        parts.extend(_render_augmentation_sections(augmentation_stats))
    return "\n".join(parts) + "\n"


def _render_language_distribution(
    stats: DatasetStats,
    augmentation_stats: AugmentationStats | None,
) -> str:
    if (
        augmentation_stats is not None
        and augmentation_stats.combined_languages.documents_per_language
    ):
        return _render_combined_language_section(augmentation_stats.combined_languages)
    return _render_language_section(stats)


def _render_augmentation_sections(stats: AugmentationStats) -> list[str]:
    parts = [
        "\n## Storage accounting\n",
        _render_storage_size_rows(stats),
        "\n## Wikipedia text corpus\n",
        _render_project_corpus(stats.wikipedia_documents, stats.wikipedia_sections),
        "\n## Wikivoyage text corpus\n",
        _render_project_corpus(stats.wikivoyage_documents, stats.wikivoyage_sections),
        "\n## Wikidata facts\n",
        _render_wikidata_facts_section(stats) + "\n",
    ]
    note = _render_unreadable_note(stats)
    if note:
        parts.append(note)
    return parts


def _render_project_corpus(documents: ProjectTextStats, sections: ProjectTextStats) -> str:
    return (
        _render_project_section("Documents", documents, kind="documents")
        + "\n"
        + _render_project_section("Sections", sections, kind="sections")
        + "\n### Languages\n\n"
        + _render_top_languages(documents)
        + "\n"
    )


def _render_unreadable_note(stats: AugmentationStats) -> str:
    if stats.unreadable_file_count <= 0:
        return ""
    return (
        "\n> Statistics exclude "
        f"{stats.unreadable_file_count} unreadable sidecar file(s); see generation logs.\n"
    )


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

    When augmentation is present the legacy ``Wikipedia articles``
    and ``Total words`` rows are dropped: the augmentation-aware
    headline already carries the canonical Wikipedia document count
    and a precise, project-broken-down word total, so the legacy rows
    would be redundant or ambiguous.
    """
    rows = _headline_rows(stats, augmentation_stats)
    lines = ["| Metric | Value |", "| --- | ---: |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    if augmentation_stats is not None:
        lines.extend(
            [
                "",
                "Wikipedia + Wikivoyage document words sums the full Wikipedia "
                "and Wikivoyage documents and excludes section rows because "
                "sections duplicate document text.",
            ]
        )
    return "\n".join(lines)


def _headline_rows(
    stats: DatasetStats,
    augmentation_stats: AugmentationStats | None,
) -> list[tuple[str, str]]:
    rows = [
        ("Polygons", _fmt_int(stats.polygon_count)),
        ("Unique Wikidata entities", _fmt_int(stats.unique_wikidata_count)),
    ]
    if augmentation_stats is None:
        # Legacy-only render: all eight documented rows, unchanged.
        rows.extend(
            [
                ("Wikipedia articles", _fmt_int(stats.article_count)),
                ("Polygon-article links", _fmt_int(stats.link_count)),
                ("Languages", _fmt_int(stats.language_count)),
                ("Geographic regions", _fmt_int(stats.region_count)),
                ("Total words", _fmt_int(stats.total_words)),
                ("Dataset size on disk", _fmt_size(stats.dataset_size_bytes)),
            ]
        )
    else:
        # Augmentation-aware render: drop the redundant/ambiguous
        # legacy rows, keep the augmentation-broken-down totals.
        rows.extend(
            [
                ("Wikipedia polygon-document links", _fmt_int(stats.link_count)),
                (
                    "Wikipedia + Wikivoyage languages",
                    _fmt_int(
                        augmentation_stats.combined_languages.language_count or stats.language_count
                    ),
                ),
                ("Geographic regions", _fmt_int(stats.region_count)),
                ("Polygon and link tables size", _fmt_size(stats.dataset_size_bytes)),
            ]
        )
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
                    "Wikipedia + Wikivoyage document words",
                    _fmt_int(_document_corpus_words(aug)),
                ),
                (
                    "Total Parquet size",
                    _fmt_size(aug.total_parquet_bytes),
                ),
            ]
        )
    return rows


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

    total_articles = max(stats.article_count, 1)
    lines = _language_table_lines(top_articles, top_polygons, total_articles)
    lines.extend(_language_concentration_lines(stats.articles_per_language, total_articles))
    lines.extend(_language_tail_lines(stats))
    return "\n".join(lines)


def _language_table_lines(
    top_articles: list[tuple[str, int]], top_polygons: dict[str, int], total: int
) -> list[str]:
    lines = [
        "Top 20 languages by Wikipedia document count:",
        "",
        "| Language | Wikipedia documents | % of total | Polygons |",
        "| --- | ---: | ---: | ---: |",
    ]
    for lang, count in top_articles:
        lines.append(
            f"| {lang} | {_fmt_int(count)} | {_fmt_pct(count / total * 100.0)} | "
            f"{_fmt_int(top_polygons.get(lang, 0))} |"
        )
    return lines


def _language_concentration_lines(articles_per_language: dict[str, int], total: int) -> list[str]:
    lines = ["", "**Concentration:**"]
    for n in (1, 5, 10, 20):
        count = sum(value for _, value in list(articles_per_language.items())[:n])
        lines.append(
            f"- Top {n} language{'s' if n > 1 else ''}: "
            f"{_fmt_pct(count / total * 100.0)} of all Wikipedia documents"
        )
    return lines


def _language_tail_lines(stats: DatasetStats) -> list[str]:
    tail_counts = _count_long_tail(stats.articles_per_language, stats.polygons_per_language)
    lines = ["", "**Long-tail:**", f"- {stats.language_count} language(s) total"]
    for key, label in (("articles_lt1", "1"), ("articles_lt5", "5"), ("articles_lt10", "10")):
        lines.append(
            f"- {tail_counts[key]} language(s) appear in fewer than {label} Wikipedia document(s)"
        )
    lines.append(f"- {tail_counts['polygons_lt5']} language(s) appear in fewer than 5 polygons")
    return lines


def _render_combined_language_section(stats: CombinedLanguageStats) -> str:
    """Render public language metrics across both text projects."""
    if not stats.documents_per_language:
        return "No language data yet.\n"
    polygons = dict(stats.polygons_per_language)
    denominator = max(stats.document_count, 1)
    lines = [
        "Top 20 languages across Wikipedia and Wikivoyage documents:",
        "",
        "Document counts include canonical rows from both projects. Polygon counts "
        "deduplicate places per language and require non-empty text: Wikipedia uses "
        "the `polygon_articles` link table, while Wikivoyage joins through the shared "
        "Wikidata QID.",
        "",
        "| Language | Documents | % of total | Polygons with non-empty text |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        _combined_language_table_lines(stats.documents_per_language, polygons, denominator)
    )
    lines.extend(_combined_concentration_lines(stats.documents_per_language, denominator))
    lines.extend(_combined_tail_lines(stats))
    return "\n".join(lines)


def _combined_language_table_lines(
    documents: Sequence[tuple[str, int]], polygons: dict[str, int], denominator: int
) -> list[str]:
    lines: list[str] = []
    for language, count in documents[:20]:
        lines.append(
            f"| {language} | {_fmt_int(count)} | {_fmt_pct(count / denominator * 100.0)} | "
            f"{_fmt_int(polygons.get(language, 0))} |"
        )
    return lines


def _combined_concentration_lines(
    documents: Sequence[tuple[str, int]], denominator: int
) -> list[str]:
    lines = ["", "**Concentration:**"]
    for n in (1, 5, 10, 20):
        count = sum(value for _, value in documents[:n])
        lines.append(
            f"- Top {n} language{'s' if n > 1 else ''}: "
            f"{_fmt_pct(count / denominator * 100.0)} of all Wikipedia + Wikivoyage documents"
        )
    return lines


def _combined_tail_lines(stats: CombinedLanguageStats) -> list[str]:
    documents = stats.documents_per_language
    return [
        "",
        "**Long-tail:**",
        f"- {stats.language_count} language(s) total",
        f"- {sum(count < 5 for _, count in documents)} language(s) appear in fewer than 5 "
        "Wikipedia + Wikivoyage documents",
        f"- {sum(count < 10 for _, count in documents)} language(s) appear in fewer than 10 "
        "Wikipedia + Wikivoyage documents",
        f"- {sum(count < 5 for _, count in stats.polygons_per_language)} language(s) appear in "
        "fewer than 5 polygons with non-empty text",
    ]


def _count_long_tail(
    articles_per_language: dict[str, int],
    polygons_per_language: dict[str, int],
) -> dict[str, int]:
    return {
        **_article_tail_counts(articles_per_language),
        "polygons_lt5": sum(count < 5 for count in polygons_per_language.values()),
    }


def _article_tail_counts(articles_per_language: dict[str, int]) -> dict[str, int]:
    return {
        "articles_lt1": sum(count < 1 for count in articles_per_language.values()),
        "articles_lt5": sum(count < 5 for count in articles_per_language.values()),
        "articles_lt10": sum(count < 10 for count in articles_per_language.values()),
    }


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
    lines.append("")
    lines.append(
        "Estimated tokens use a transparent approximation: `characters // 4` "
        "per non-empty text row, with a minimum of one token per non-empty row."
    )
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
    if not facts.subdir_present:
        return "No data exists yet."
    if facts.rows == 0:
        return "This sidecar is present but empty."
    lines = _fact_metric_lines(facts)
    lines.extend(_fact_distribution_lines(facts))
    lines.extend(_fact_property_lines(facts))
    lines.extend(
        [
            "",
            "English labels are requested where available; multilingual labels are "
            "preserved verbatim in the Parquet column `property_labels`.",
        ]
    )
    return "\n".join(lines)


def _fact_metric_lines(facts: WikidataFactStats) -> list[str]:
    return [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Fact rows | {_fmt_int(facts.rows)} |",
        f"| Unique facts | {_fmt_int(facts.unique_facts)} |",
        f"| Unique subjects | {_fmt_int(facts.unique_subjects)} |",
        f"| Distinct properties | {_fmt_int(facts.distinct_property_ids)} |",
        f"| Non-empty English property label | {_fmt_int(facts.with_property_en_label)} |",
        f"| Non-empty English value label | {_fmt_int(facts.with_value_en_label)} |",
        f"| With qualifiers | {_fmt_int(facts.with_qualifiers)} |",
        f"| With references | {_fmt_int(facts.with_references)} |",
        f"| Unreadable qualifier JSON | {_fmt_int(facts.unavailable_qualifiers)} |",
        f"| Unreadable references JSON | {_fmt_int(facts.unavailable_references)} |",
        f"| Regions / files represented | {_fmt_int(facts.region_count)} |",
    ]


def _fact_distribution_lines(facts: WikidataFactStats) -> list[str]:
    lines: list[str] = []
    if facts.value_type_distribution:
        lines.append("")
        lines.append("**Value-type distribution:**")
        lines.append("")
        lines.append("| Value type | Facts |")
        lines.append("| --- | ---: |")
        for value_type, count in facts.value_type_distribution:
            lines.append(f"| {value_type} | {_fmt_int(count)} |")
    return lines


def _fact_property_lines(facts: WikidataFactStats) -> list[str]:
    lines = [
        "",
        "**Top properties:**",
        "",
        "| Property ID | English label | Facts |",
        "| --- | --- | ---: |",
    ]
    for property_id, label, count in facts.top_properties:
        display_label = label if label else "(no English label)"
        lines.append(f"| {property_id} | {display_label} | {_fmt_int(count)} |")
    return lines


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
        f"| Polygon and link tables size | {_fmt_int(stats.core_parquet_bytes)} | "
        f"{_fmt_size(stats.core_parquet_bytes)} |",
        f"| Wikipedia, Wikivoyage, and Wikidata tables size | "
        f"{_fmt_int(stats.augmentation_parquet_bytes)} | "
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
