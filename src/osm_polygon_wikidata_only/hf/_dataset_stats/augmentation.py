"""Private aggregation for the augmentation sidecar statistics.

Owns the cache-aware per-file scanner that produces
:class:`PerFileSummary` records (one per sidecar) and the lossless
merge that turns them into a single :class:`AugmentationStats`
instance.

The scanner is purely deterministic. Identical inputs produce
identical outputs. Replacing a sidecar invalidates that file's cache
entry by fingerprint only. Removing a sidecar invalidates it by
absence: the cache index is rebuilt from the live filesystem on
every refresh, so a deleted file disappears from the next
:class:`AugmentationStats`.

Architecture
------------
* :class:`PerFileSummary` lives in :mod:`models.py`. Each summary
  captures all the row-level information needed to merge back into
  per-project aggregates losslessly (counters, sets, scalars).

* The on-disk cache index lives under
  ``<cache_index_dir>/index.json`` (callers pass
  ``data_root.cache``). It is rewritten on every refresh. Each
  entry stores a single ``PerFileSummary`` in JSON form (Counter
  as ``dict`` + ``frozenset`` as sorted lists).

* :func:`compute_augmentation_stats` orchestrates the cache:
  enumerate sidecars → load index → reuse matching summaries →
  scan the rest once → rewrite the index.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .cache import (
    _file_fingerprint,
    _relative_path,
    _scan_paths,
    load_cache_index,
    write_cache_index,
)
from .combined_languages import compute_combined_language_stats
from .models import (
    AugmentationStats,
    PerFileSummary,
    ProjectTextStats,
    WikidataFactStats,
)
from .scanning import safe_table, sorted_parquets
from .summary_codec import summary_from_json as _summary_from_json
from .summary_codec import summary_to_json as _summary_to_json

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.dataset_stats")

# Sidecar directories under <processed>, sorted.
AUGMENTATION_SUBDIRS: tuple[str, ...] = (
    "wikipedia/documents",
    "wikipedia/sections",
    "wikivoyage/documents",
    "wikivoyage/sections",
    "wikidata/facts",
)

# Document columns actually used by the scanner.
DOCUMENT_COLUMNS: tuple[str, ...] = (
    "document_id",
    "wikidata",
    "project",
    "language",
    "full_text",
    "article_length_chars",
    "article_length_words",
    "article_length_tokens_estimate",
)
SECTION_COLUMNS: tuple[str, ...] = (
    "section_id",
    "document_id",
    "wikidata",
    "project",
    "language",
    "text",
    "text_length_chars",
    "text_length_words",
    "text_length_tokens_estimate",
)
FACT_COLUMNS: tuple[str, ...] = (
    "fact_id",
    "wikidata",
    "property_id",
    "property_label_en",
    "property_labels",
    "value_type",
    "value_entity_id",
    "value_label_en",
    "value_labels",
    "value_text",
    "qualifiers",
    "references",
)

# Top-N cut-offs used by the merge step.
TOP_LANGUAGES_LIMIT = 10
TOP_PROPERTIES_LIMIT = 10

KIND_DOCUMENT = "documents"
KIND_SECTION = "sections"
KIND_FACT = "facts"

# Core sub-directories whose parquet sizes count toward core_parquet_bytes.
CORE_SUBDIRS: tuple[str, ...] = ("polygons", "polygon_articles")


# ---------------------------------------------------------------------------
# JSON helper detection
# ---------------------------------------------------------------------------


def _has_json_content(value: object) -> bool:
    """A column cell counts as "present JSON" when it is a valid
    non-empty JSON array or object."""
    if value is None:
        return False
    if isinstance(value, str):
        return _json_text_has_content(value)
    return _json_collection_has_content(value)


def _json_text_has_content(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return False
    return _json_collection_has_content(parsed)


def _json_collection_has_content(value: object) -> bool:
    return isinstance(value, (list, dict)) and len(value) > 0


def _section_row_metrics(
    text: Any, chars: Any, words: Any, tokens: Any
) -> tuple[int, int, int, int, int]:
    """Return non-empty/empty flags and numeric text lengths for one row."""
    text_value = text if isinstance(text, str) else ""
    non_empty = int(bool(text_value and text_value.strip()))
    empty_or_null = 1 - non_empty
    return (
        non_empty,
        empty_or_null,
        _length_or_zero(chars),
        _length_or_zero(words),
        _length_or_zero(tokens),
    )


def _length_or_zero(value: Any) -> int:
    """Convert an optional stored length to an integer."""
    return int(str(value)) if value is not None else 0


def _optional_string(value: Any) -> tuple[str, ...]:
    """Return a one-item string tuple when a cell has a value."""
    return (str(value),) if value else ()


def _section_row_identity(
    arrays: dict[str, list[Any]], index: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract set/counter updates for one section row."""
    return (
        _optional_string(arrays.get("section_id", [None])[index]),
        _optional_string(arrays.get("document_id", [None])[index]),
        _optional_string(arrays.get("wikidata", [None])[index]),
        _optional_string(arrays.get("language", [None])[index]),
    )


def _document_row_identity(
    arrays: dict[str, list[Any]], index: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract document, subject, and language updates for one row."""
    return (
        _optional_string(arrays.get("document_id", [None])[index]),
        _optional_string(arrays.get("wikidata", [None])[index]),
        _optional_string(arrays.get("language", [None])[index]),
    )


def _json_field_counts(value: object) -> tuple[int, int]:
    """Return ``(available, unavailable)`` counts for one JSON cell."""
    if _has_json_content(value):
        return 1, 0
    if isinstance(value, str) and value.strip():
        return 0, 1
    return 0, 0


def _fact_row_identity(
    arrays: dict[str, list[Any]], index: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract fact, subject, and property updates for one row."""
    return (
        _optional_string(arrays.get("fact_id", [None])[index]),
        _optional_string(arrays.get("wikidata", [None])[index]),
        _optional_string(arrays.get("property_id", [None])[index]),
    )


def _nonempty_text(value: object) -> int:
    """Return one when a value is a non-blank string."""
    return int(isinstance(value, str) and bool(value.strip()))


def _present_string(value: object) -> tuple[str, ...]:
    """Return one non-empty string value without coercing other types."""
    return (value,) if isinstance(value, str) and value else ()


def _property_label_update(
    property_id: object, property_label: object
) -> tuple[tuple[str, str], ...]:
    """Return a first-seen property-label candidate for one fact row."""
    if not property_id or not isinstance(property_label, str):
        return ()
    return ((str(property_id), property_label.strip()),)


def _record_property_updates(
    property_ids: tuple[str, ...],
    property_label: object,
    property_counts: Counter[str],
    property_labels: dict[str, str],
) -> None:
    """Merge one row's property count and first-seen label."""
    property_counts.update(property_ids)
    for property_id, label in _property_label_update(
        property_ids[0] if property_ids else None, property_label
    ):
        property_labels.setdefault(property_id, label)


# ---------------------------------------------------------------------------
# Per-file scanning
# ---------------------------------------------------------------------------


def _scan_documents_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary:
    """Aggregate one ``wikipedia/documents`` or ``wikivoyage/documents``."""
    rel = _relative_path(processed_dir, parquet_path)
    fp = _file_fingerprint(parquet_path)
    file_size = parquet_path.stat().st_size
    table = safe_table(parquet_path, list(DOCUMENT_COLUMNS))
    if table is None:
        return PerFileSummary(
            relative_path=rel,
            fingerprint=fp,
            file_size_bytes=file_size,
            kind=KIND_DOCUMENT,
            scan_failed=True,
        )

    arrays: dict[str, list[Any]] = {
        col: table.column(col).to_pylist() for col in DOCUMENT_COLUMNS if col in table.schema.names
    }

    rows = table.num_rows
    document_ids: set[str] = set()
    qids: set[str] = set()
    languages: Counter[str] = Counter()
    non_empty = 0
    empty_or_null = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0

    for i in range(rows):
        row_document_ids, row_qids, row_languages = _document_row_identity(arrays, i)
        full_text = arrays.get("full_text", [None])[i]
        document_ids.update(row_document_ids)
        qids.update(row_qids)
        languages.update(row_languages)
        chars = arrays.get("article_length_chars", [None])[i]
        words = arrays.get("article_length_words", [None])[i]
        tokens = arrays.get("article_length_tokens_estimate", [None])[i]
        row_non_empty, row_empty, row_chars, row_words, row_tokens = _section_row_metrics(
            full_text, chars, words, tokens
        )
        non_empty += row_non_empty
        empty_or_null += row_empty
        total_chars += row_chars
        total_words += row_words
        total_tokens += row_tokens

    return PerFileSummary(
        relative_path=rel,
        fingerprint=fp,
        file_size_bytes=file_size,
        kind=KIND_DOCUMENT,
        rows=rows,
        non_empty=non_empty,
        empty_or_null=empty_or_null,
        total_chars=total_chars,
        total_words=total_words,
        total_tokens_estimate=total_tokens,
        document_ids=frozenset(document_ids),
        qids=frozenset(qids),
        languages=dict(languages),
    )


def _scan_sections_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary:
    """Aggregate one ``wikipedia/sections`` or ``wikivoyage/sections``."""
    rel = _relative_path(processed_dir, parquet_path)
    fp = _file_fingerprint(parquet_path)
    file_size = parquet_path.stat().st_size
    table = safe_table(parquet_path, list(SECTION_COLUMNS))
    if table is None:
        return PerFileSummary(
            relative_path=rel,
            fingerprint=fp,
            file_size_bytes=file_size,
            kind=KIND_SECTION,
            scan_failed=True,
        )

    arrays: dict[str, list[Any]] = {
        col: table.column(col).to_pylist() for col in SECTION_COLUMNS if col in table.schema.names
    }

    rows = table.num_rows
    document_ids: set[str] = set()
    section_ids: set[str] = set()
    qids: set[str] = set()
    languages: Counter[str] = Counter()
    non_empty = 0
    empty_or_null = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0

    for i in range(rows):
        row_section_ids, row_document_ids, row_qids, row_languages = _section_row_identity(
            arrays, i
        )
        text = arrays.get("text", [None])[i]
        section_ids.update(row_section_ids)
        document_ids.update(row_document_ids)
        qids.update(row_qids)
        languages.update(row_languages)
        chars = arrays.get("text_length_chars", [None])[i]
        words = arrays.get("text_length_words", [None])[i]
        tokens = arrays.get("text_length_tokens_estimate", [None])[i]
        row_non_empty, row_empty, row_chars, row_words, row_tokens = _section_row_metrics(
            text, chars, words, tokens
        )
        non_empty += row_non_empty
        empty_or_null += row_empty
        total_chars += row_chars
        total_words += row_words
        total_tokens += row_tokens

    return PerFileSummary(
        relative_path=rel,
        fingerprint=fp,
        file_size_bytes=file_size,
        kind=KIND_SECTION,
        rows=rows,
        non_empty=non_empty,
        empty_or_null=empty_or_null,
        total_chars=total_chars,
        total_words=total_words,
        total_tokens_estimate=total_tokens,
        document_ids=frozenset(document_ids),
        section_ids=frozenset(section_ids),
        qids=frozenset(qids),
        languages=dict(languages),
    )


def _scan_facts_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary:
    """Aggregate one ``wikidata/facts``."""
    rel = _relative_path(processed_dir, parquet_path)
    fp = _file_fingerprint(parquet_path)
    file_size = parquet_path.stat().st_size
    table = safe_table(parquet_path, list(FACT_COLUMNS))
    if table is None:
        return PerFileSummary(
            relative_path=rel,
            fingerprint=fp,
            file_size_bytes=file_size,
            kind=KIND_FACT,
            scan_failed=True,
        )

    arrays: dict[str, list[Any]] = {
        col: table.column(col).to_pylist() for col in FACT_COLUMNS if col in table.schema.names
    }

    rows = table.num_rows
    fact_ids: set[str] = set()
    subjects: set[str] = set()
    properties: set[str] = set()
    property_labels: dict[str, str] = {}
    property_counts: Counter[str] = Counter()
    with_prop_en = 0
    with_value_en = 0
    with_qualifiers = 0
    with_references = 0
    unavailable_qualifiers = 0
    unavailable_references = 0
    value_types: Counter[str] = Counter()

    for i in range(rows):
        row_fact_ids, row_subjects, row_properties = _fact_row_identity(arrays, i)
        property_label_en = arrays.get("property_label_en", [None])[i]
        value_type = arrays.get("value_type", [None])[i]
        value_label_en = arrays.get("value_label_en", [None])[i]
        qualifiers = arrays.get("qualifiers", [None])[i]
        references = arrays.get("references", [None])[i]
        fact_ids.update(row_fact_ids)
        subjects.update(row_subjects)
        properties.update(row_properties)
        with_prop_en += _nonempty_text(property_label_en)
        with_value_en += _nonempty_text(value_label_en)
        qualifier_count, unavailable_qualifier_count = _json_field_counts(qualifiers)
        with_qualifiers += qualifier_count
        unavailable_qualifiers += unavailable_qualifier_count
        reference_count, unavailable_reference_count = _json_field_counts(references)
        with_references += reference_count
        unavailable_references += unavailable_reference_count
        value_types.update(_present_string(value_type))
        _record_property_updates(
            row_properties, property_label_en, property_counts, property_labels
        )

    return PerFileSummary(
        relative_path=rel,
        fingerprint=fp,
        file_size_bytes=file_size,
        kind=KIND_FACT,
        fact_rows=rows,
        fact_ids=frozenset(fact_ids),
        subject_qids=frozenset(subjects),
        property_ids=frozenset(properties),
        property_labels=dict(property_labels),
        property_counts=dict(property_counts),
        with_property_en_label=with_prop_en,
        with_value_en_label=with_value_en,
        with_qualifiers=with_qualifiers,
        with_references=with_references,
        unavailable_qualifiers=unavailable_qualifiers,
        unavailable_references=unavailable_references,
        value_type_counts=dict(value_types),
    )


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _kind_for_rel(rel: str) -> str:
    """Return the augmentation kind for a sidecar path relative to
    ``<processed>/``."""
    for kind, prefixes in (
        (KIND_DOCUMENT, ("wikipedia/documents/", "wikivoyage/documents/")),
        (KIND_SECTION, ("wikipedia/sections/", "wikivoyage/sections/")),
        (KIND_FACT, ("wikidata/facts/",)),
    ):
        if rel.startswith(prefixes):
            return kind
    return ""


def _scan_one_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary | None:
    """Dispatch a single sidecar file to its specialized scanner."""
    try:
        rel = _relative_path(processed_dir, parquet_path)
    except ValueError:
        return None
    kind = _kind_for_rel(rel)
    if kind == KIND_DOCUMENT:
        return _scan_documents_file(processed_dir, parquet_path)
    if kind == KIND_SECTION:
        return _scan_sections_file(processed_dir, parquet_path)
    if kind == KIND_FACT:
        return _scan_facts_file(processed_dir, parquet_path)
    return None


# ---------------------------------------------------------------------------
# Merge: per-file summaries -> per-project aggregates
# ---------------------------------------------------------------------------


def _merge_project_text(
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
    metrics = _merge_project_text_metrics(summaries)
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


def _merge_project_text_metrics(summaries: list[PerFileSummary]) -> dict[str, Any]:
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


def _merge_wikidata_facts(
    summaries: list[PerFileSummary], *, subdir_present: bool
) -> WikidataFactStats:
    metrics = _merge_fact_metrics(summaries)
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


def _merge_fact_metrics(summaries: list[PerFileSummary]) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Core coverage classification
# ---------------------------------------------------------------------------


def _core_stems(processed: Path) -> set[str]:
    if not (processed / "polygons").exists():
        return set()
    return {path.stem for path in sorted_parquets(processed / "polygons")}


def _all_sidecar_stems(processed: Path) -> set[str]:
    stems: set[str] = set()
    for rel in AUGMENTATION_SUBDIRS:
        directory = processed / rel
        if not directory.exists():
            continue
        stems.update(path.stem for path in sorted_parquets(directory))
    return stems


def _fully_or_partial(processed: Path, cores: set[str]) -> tuple[set[str], set[str]]:
    fully: set[str] = set()
    partial: set[str] = set()
    for stem in sorted(cores):
        _classify_augmentation_stem(processed, stem, fully, partial)
    return fully, partial


def _classify_augmentation_stem(
    processed: Path,
    stem: str,
    fully: set[str],
    partial: set[str],
) -> None:
    present = sum(
        1 for rel in AUGMENTATION_SUBDIRS if (processed / rel / f"{stem}.parquet").exists()
    )
    if present == len(AUGMENTATION_SUBDIRS):
        fully.add(stem)
    elif present > 0:
        partial.add(stem)


# ---------------------------------------------------------------------------
# Storage accounting helpers
# ---------------------------------------------------------------------------


def _core_bytes(processed: Path) -> int:
    total = 0
    for rel in CORE_SUBDIRS:
        directory = processed / rel
        if not directory.exists():
            continue
        for path in sorted_parquets(directory):
            total += path.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_augmentation_stats(
    processed_dir: Path,
    *,
    cache_index_dir: Path,
) -> AugmentationStats:
    """Compute :class:`AugmentationStats` from local finalized parquets.

    Cache layer
    -----------
    A per-file summary cache lives under ``<cache_index_dir>``. The
    cache is keyed by ``relative_path + "@" + fingerprint``. The
    cache index is rewritten from the live filesystem on every call:

    * A sidecar with a matching fingerprint is reused without a
      Parquet table read.
    * A sidecar whose fingerprint or path is not in the index is
      scanned once and added.
    * A sidecar that no longer exists on disk disappears from the
      index and from the aggregates.

    Unreadability
    -------------
    A sidecar whose Parquet content cannot be parsed is recorded with
    ``scan_failed=True`` and its bytes still count toward
    :attr:`AugmentationStats.augmentation_parquet_bytes`. The
    :attr:`AugmentationStats.unreadable_file_count` private metric
    surfaces a one-line warning under the documented logger.

    Storage accounting
    ------------------
    Core parquet bytes include every file under ``polygons/`` and
    ``polygon_articles/``. Canonical Wikipedia documents are counted
    once with the text sidecars; retired local ``articles/`` staging
    files are deliberately excluded from published-dataset storage.
    Augmentation parquet bytes include every file under the sidecar
    sub-directories. The invariant
    ``core + augmentation == total`` always holds.
    """
    processed_dir = Path(processed_dir)
    cache_index_dir = Path(cache_index_dir)
    cores = _core_stems(processed_dir)
    fully, partial = _fully_or_partial(processed_dir, cores)
    orphans = sorted(_all_sidecar_stems(processed_dir) - cores)
    new_index, by_subdir, unreadable = _scan_augmentation_files(
        processed_dir,
        load_cache_index(cache_index_dir),
    )
    write_cache_index(cache_index_dir, new_index)
    return _build_augmentation_stats(
        processed_dir,
        cache_index_dir,
        cores,
        fully,
        partial,
        orphans,
        by_subdir,
        unreadable,
    )


def _empty_subdir_summaries() -> dict[str, list[PerFileSummary]]:
    return {prefix: [] for prefix in AUGMENTATION_SUBDIRS}


def _scan_augmentation_files(
    processed_dir: Path,
    existing_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[PerFileSummary]], int]:
    new_index: dict[str, dict[str, Any]] = {}
    by_subdir = _empty_subdir_summaries()
    unreadable = 0
    for parquet_path in _scan_paths(processed_dir, AUGMENTATION_SUBDIRS):
        rel = _relative_path(processed_dir, parquet_path)
        summary = _load_or_scan_summary(processed_dir, parquet_path, existing_index.get(rel))
        if summary is None:
            continue
        new_index[rel] = _summary_to_json(summary)
        unreadable += int(summary.scan_failed)
        _append_summary(by_subdir, rel, summary)
    return new_index, by_subdir, unreadable


def _load_or_scan_summary(
    processed_dir: Path,
    parquet_path: Path,
    cached: dict[str, Any] | None,
) -> PerFileSummary | None:
    fingerprint = _file_fingerprint(parquet_path)
    if (
        cached is not None
        and cached.get("fingerprint") == fingerprint
        and cached.get("scan_failed") is not True
    ):
        summary = _summary_from_json(cached)
        return summary if summary is not None else _scan_one_file(processed_dir, parquet_path)
    return _scan_one_file(processed_dir, parquet_path)


def _append_summary(
    by_subdir: dict[str, list[PerFileSummary]],
    rel: str,
    summary: PerFileSummary,
) -> None:
    for prefix, summaries in by_subdir.items():
        if rel.startswith(prefix + "/"):
            summaries.append(summary)
            return


def _build_augmentation_stats(
    processed_dir: Path,
    cache_index_dir: Path,
    cores: set[str],
    fully: set[str],
    partial: set[str],
    orphans: list[str],
    by_subdir: dict[str, list[PerFileSummary]],
    unreadable: int,
) -> AugmentationStats:
    present = {prefix: _has_readable_summary(summaries) for prefix, summaries in by_subdir.items()}
    projects = {
        "wikipedia_documents": _merge_project_text(
            by_subdir["wikipedia/documents"], subdir_present=present["wikipedia/documents"]
        ),
        "wikipedia_sections": _merge_project_text(
            by_subdir["wikipedia/sections"], subdir_present=present["wikipedia/sections"]
        ),
        "wikivoyage_documents": _merge_project_text(
            by_subdir["wikivoyage/documents"], subdir_present=present["wikivoyage/documents"]
        ),
        "wikivoyage_sections": _merge_project_text(
            by_subdir["wikivoyage/sections"], subdir_present=present["wikivoyage/sections"]
        ),
    }
    facts = _merge_wikidata_facts(
        by_subdir["wikidata/facts"], subdir_present=present["wikidata/facts"]
    )
    core_bytes = _core_bytes(processed_dir)
    augmentation_bytes = _augmentation_bytes(by_subdir)
    return AugmentationStats(
        core_region_count=len(cores),
        fully_augmented_count=len(fully),
        partial_augmented_count=len(partial),
        not_augmented_count=len(cores) - len(fully) - len(partial),
        orphan_sidecar_stems=tuple(orphans),
        wikipedia_documents=projects["wikipedia_documents"],
        wikipedia_sections=projects["wikipedia_sections"],
        wikivoyage_documents=projects["wikivoyage_documents"],
        wikivoyage_sections=projects["wikivoyage_sections"],
        wikidata_facts=facts,
        core_parquet_bytes=core_bytes,
        augmentation_parquet_bytes=augmentation_bytes,
        total_parquet_bytes=core_bytes + augmentation_bytes,
        unreadable_file_count=unreadable,
        combined_languages=compute_combined_language_stats(
            processed_dir, cache_index_dir=cache_index_dir
        ),
    )


def _augmentation_bytes(by_subdir: dict[str, list[PerFileSummary]]) -> int:
    return sum(summary.file_size_bytes for summaries in by_subdir.values() for summary in summaries)


def _has_readable_summary(summaries: list[PerFileSummary]) -> bool:
    return any(not summary.scan_failed for summary in summaries)


__all__ = [
    "AUGMENTATION_SUBDIRS",
    "CORE_SUBDIRS",
    "DOCUMENT_COLUMNS",
    "FACT_COLUMNS",
    "KIND_DOCUMENT",
    "KIND_FACT",
    "KIND_SECTION",
    "SECTION_COLUMNS",
    "compute_augmentation_stats",
]
