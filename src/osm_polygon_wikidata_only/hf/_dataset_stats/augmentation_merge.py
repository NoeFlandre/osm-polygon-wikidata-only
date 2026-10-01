"""Lossless merge of per-file augmentation summaries into project statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .augmentation_scan import KIND_SECTION
from .models import (
    PerFileSummary,
    ProjectTextStats,
    WikidataFactStats,
)

# Top-N cut-offs used by the merge step.
TOP_LANGUAGES_LIMIT = 10
TOP_PROPERTIES_LIMIT = 10


# ---------------------------------------------------------------------------
# Merge: per-file summaries -> per-project aggregates
# ---------------------------------------------------------------------------


def merge_project_text(
    summaries: list[PerFileSummary], *, subdir_present: bool
) -> ProjectTextStats:
    """Merge a list of per-file project summaries into one
    :class:`ProjectTextStats`. ``summaries`` may be empty (a missing
    sidecar sub-directory). Skip summaries with ``scan_failed`` so
    their bytes still count in storage accounting but not in the row
    metrics.

    ``subdir_present`` distinguishes a missing sub-directory
    (``False``) from a present-but-empty one (``True`` with no
    summaries).
    """
    metrics = merge_project_text_metrics(summaries)
    unique_section_ids, unique_documents, avg = _section_metrics(summaries, metrics)
    non_empty_rate = metrics["non_empty"] / metrics["rows"] if metrics["rows"] > 0 else 0.0
    top_languages = _top_counts(metrics["languages"])
    return ProjectTextStats(
        subdir_present=subdir_present,
        rows=metrics["rows"],
        unique_documents=unique_documents,
        unique_section_ids=unique_section_ids,
        unique_qids=len(metrics["qids"]),
        language_count=len(metrics["languages"]),
        region_count=len(summaries),
        non_empty=metrics["non_empty"],
        empty_or_null=metrics["empty_or_null"],
        non_empty_rate=non_empty_rate,
        total_chars=metrics["total_chars"],
        total_words=metrics["total_words"],
        total_tokens_estimate=metrics["total_tokens"],
        avg_sections_per_doc=avg,
        top_languages=top_languages,
    )


def merge_project_text_metrics(summaries: list[PerFileSummary]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "rows": 0,
        "non_empty": 0,
        "empty_or_null": 0,
        "total_chars": 0,
        "total_words": 0,
        "total_tokens": 0,
        "document_ids": set(),
        "section_ids": set(),
        "qids": set(),
        "languages": Counter(),
    }
    for summary in summaries:
        if summary.scan_failed:
            continue
        _add_project_summary(metrics, summary)
    return metrics


def _add_project_summary(metrics: dict[str, Any], summary: PerFileSummary) -> None:
    metrics["rows"] += summary.rows
    metrics["non_empty"] += summary.non_empty
    metrics["empty_or_null"] += summary.empty_or_null
    metrics["total_chars"] += summary.total_chars
    metrics["total_words"] += summary.total_words
    metrics["total_tokens"] += summary.total_tokens_estimate
    metrics["document_ids"].update(summary.document_ids)
    metrics["section_ids"].update(summary.section_ids)
    metrics["qids"].update(summary.qids)
    for language, count in summary.languages.items():
        metrics["languages"][language] += count


def _section_metrics(
    summaries: list[PerFileSummary], metrics: dict[str, Any]
) -> tuple[int, int, float]:
    unique_documents = len(metrics["document_ids"])
    if summaries and summaries[0].kind == KIND_SECTION:
        unique_sections = len(metrics["section_ids"])
        average = metrics["rows"] / unique_documents if unique_documents else 0.0
    else:
        unique_sections = 0
        average = 0.0
    return unique_sections, unique_documents, average


def _top_counts(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ordered[:TOP_LANGUAGES_LIMIT])


def merge_wikidata_facts(
    summaries: list[PerFileSummary], *, subdir_present: bool
) -> WikidataFactStats:
    metrics = merge_fact_metrics(summaries)
    top_properties = _top_properties(metrics["property_counts"], metrics["property_labels"])
    value_type_distribution = _sorted_counts(metrics["value_types"])
    return WikidataFactStats(
        subdir_present=subdir_present,
        rows=metrics["rows"],
        unique_facts=len(metrics["fact_ids"]),
        unique_subjects=len(metrics["subjects"]),
        distinct_property_ids=len(metrics["properties"]),
        with_property_en_label=metrics["with_prop_en"],
        with_value_en_label=metrics["with_value_en"],
        with_qualifiers=metrics["with_qualifiers"],
        with_references=metrics["with_references"],
        unavailable_qualifiers=metrics["unavailable_qualifiers"],
        unavailable_references=metrics["unavailable_references"],
        region_count=len(summaries),
        value_type_distribution=value_type_distribution,
        top_properties=top_properties,
    )


def merge_fact_metrics(summaries: list[PerFileSummary]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "rows": 0,
        "fact_ids": set(),
        "subjects": set(),
        "properties": set(),
        "property_labels": {},
        "property_counts": Counter(),
        "with_prop_en": 0,
        "with_value_en": 0,
        "with_qualifiers": 0,
        "with_references": 0,
        "unavailable_qualifiers": 0,
        "unavailable_references": 0,
        "value_types": Counter(),
    }
    for summary in summaries:
        if summary.scan_failed:
            continue
        _add_fact_summary(metrics, summary)
    return metrics


def _add_fact_summary(metrics: dict[str, Any], summary: PerFileSummary) -> None:
    metrics["rows"] += summary.fact_rows
    metrics["fact_ids"].update(summary.fact_ids)
    metrics["subjects"].update(summary.subject_qids)
    metrics["properties"].update(summary.property_ids)
    for pid, label in summary.property_labels.items():
        metrics["property_labels"].setdefault(pid, label)
    for pid, count in summary.property_counts.items():
        metrics["property_counts"][pid] += count
    for name in (
        "with_prop_en",
        "with_value_en",
        "with_qualifiers",
        "with_references",
        "unavailable_qualifiers",
        "unavailable_references",
    ):
        field = {
            "with_prop_en": "with_property_en_label",
            "with_value_en": "with_value_en_label",
            "with_qualifiers": "with_qualifiers",
            "with_references": "with_references",
            "unavailable_qualifiers": "unavailable_qualifiers",
            "unavailable_references": "unavailable_references",
        }[name]
        metrics[name] += getattr(summary, field)
    for value_type, count in summary.value_type_counts.items():
        metrics["value_types"][value_type] += count


def _top_properties(
    counts: Counter[str], labels: dict[str, str]
) -> tuple[tuple[str, str, int], ...]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_PROPERTIES_LIMIT]
    return tuple((pid, labels.get(pid, ""), count) for pid, count in ordered)


def _sorted_counts(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
