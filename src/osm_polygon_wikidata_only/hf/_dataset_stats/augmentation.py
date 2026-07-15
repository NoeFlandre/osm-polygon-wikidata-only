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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cache import (
    _file_fingerprint,
    _relative_path,
    _scan_paths,
    load_cache_index,
    write_cache_index,
)
from .models import (
    AugmentationStats,
    PerFileSummary,
    ProjectTextStats,
    WikidataFactStats,
)
from .scanning import safe_table, sorted_parquets

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
        text = value.strip()
        if not text:
            return False
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return False
        return isinstance(parsed, (list, dict)) and len(parsed) > 0
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return False


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
        did = arrays.get("document_id", [None])[i]
        qid = arrays.get("wikidata", [None])[i]
        lang = arrays.get("language", [None])[i]
        full_text = arrays.get("full_text", [None])[i]
        if did:
            document_ids.add(str(did))
        if qid:
            qids.add(str(qid))
        if lang:
            languages[str(lang)] += 1
        text = full_text if isinstance(full_text, str) else ""
        if text and text.strip():
            non_empty += 1
        else:
            empty_or_null += 1
        chars = arrays.get("article_length_chars", [None])[i]
        words = arrays.get("article_length_words", [None])[i]
        tokens = arrays.get("article_length_tokens_estimate", [None])[i]
        if chars is not None:
            total_chars += int(str(chars))
        if words is not None:
            total_words += int(str(words))
        if tokens is not None:
            total_tokens += int(str(tokens))

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
        sid = arrays.get("section_id", [None])[i]
        did = arrays.get("document_id", [None])[i]
        qid = arrays.get("wikidata", [None])[i]
        lang = arrays.get("language", [None])[i]
        text = arrays.get("text", [None])[i]
        if sid:
            section_ids.add(str(sid))
        if did:
            document_ids.add(str(did))
        if qid:
            qids.add(str(qid))
        if lang:
            languages[str(lang)] += 1
        text_value = text if isinstance(text, str) else ""
        if text_value and text_value.strip():
            non_empty += 1
        else:
            empty_or_null += 1
        chars = arrays.get("text_length_chars", [None])[i]
        words = arrays.get("text_length_words", [None])[i]
        tokens = arrays.get("text_length_tokens_estimate", [None])[i]
        if chars is not None:
            total_chars += int(str(chars))
        if words is not None:
            total_words += int(str(words))
        if tokens is not None:
            total_tokens += int(str(tokens))

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
        fact_id = arrays.get("fact_id", [None])[i]
        wikidata = arrays.get("wikidata", [None])[i]
        property_id = arrays.get("property_id", [None])[i]
        property_label_en = arrays.get("property_label_en", [None])[i]
        value_type = arrays.get("value_type", [None])[i]
        value_label_en = arrays.get("value_label_en", [None])[i]
        qualifiers = arrays.get("qualifiers", [None])[i]
        references = arrays.get("references", [None])[i]
        if fact_id:
            fact_ids.add(str(fact_id))
        if wikidata:
            subjects.add(str(wikidata))
        if property_id:
            properties.add(str(property_id))
        if isinstance(property_label_en, str) and property_label_en.strip():
            with_prop_en += 1
        if isinstance(value_label_en, str) and value_label_en.strip():
            with_value_en += 1
        if _has_json_content(qualifiers):
            with_qualifiers += 1
        elif isinstance(qualifiers, str) and qualifiers.strip():
            unavailable_qualifiers += 1
        if _has_json_content(references):
            with_references += 1
        elif isinstance(references, str) and references.strip():
            unavailable_references += 1
        if isinstance(value_type, str) and value_type:
            value_types[value_type] += 1
        if property_id:
            pid = str(property_id)
            property_counts[pid] += 1
            if pid not in property_labels and isinstance(property_label_en, str):
                property_labels[pid] = property_label_en.strip()

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
# Cache (de)serialization
# ---------------------------------------------------------------------------


def _summary_to_json(summary: PerFileSummary) -> dict[str, Any]:
    return {
        "relative_path": summary.relative_path,
        "fingerprint": summary.fingerprint,
        "file_size_bytes": summary.file_size_bytes,
        "kind": summary.kind,
        "scan_failed": summary.scan_failed,
        "rows": summary.rows,
        "non_empty": summary.non_empty,
        "empty_or_null": summary.empty_or_null,
        "total_chars": summary.total_chars,
        "total_words": summary.total_words,
        "total_tokens_estimate": summary.total_tokens_estimate,
        "document_ids": sorted(summary.document_ids),
        "section_ids": sorted(summary.section_ids),
        "qids": sorted(summary.qids),
        "languages": dict(sorted(summary.languages.items())),
        "fact_rows": summary.fact_rows,
        "fact_ids": sorted(summary.fact_ids),
        "subject_qids": sorted(summary.subject_qids),
        "property_ids": sorted(summary.property_ids),
        "property_labels": dict(sorted(summary.property_labels.items())),
        "property_counts": dict(sorted(summary.property_counts.items())),
        "with_property_en_label": summary.with_property_en_label,
        "with_value_en_label": summary.with_value_en_label,
        "with_qualifiers": summary.with_qualifiers,
        "with_references": summary.with_references,
        "unavailable_qualifiers": summary.unavailable_qualifiers,
        "unavailable_references": summary.unavailable_references,
        "value_type_counts": dict(sorted(summary.value_type_counts.items())),
    }


def _summary_from_json(blob: Mapping[str, object]) -> PerFileSummary | None:
    """Inverse of :func:`_summary_to_json`. Returns ``None`` on a
    structurally-incompatible cache entry."""

    def _get_list(key: str) -> list[str]:
        if key in blob:
            value = blob[key]
            if isinstance(value, list):
                return [str(x) for x in value]
        return []

    def _get_dict_str(key: str) -> dict[str, str]:
        if key in blob and isinstance(blob[key], dict):
            value = blob[key]
            inner: dict[object, object] = value  # type: ignore[assignment]
            return {str(k): str(v) for k, v in inner.items()}
        return {}

    def _get_dict_int(key: str) -> dict[str, int]:
        if key in blob and isinstance(blob[key], dict):
            value = blob[key]
            inner: dict[object, object] = value  # type: ignore[assignment]
            items: list[tuple[str, int]] = []
            for k, v in inner.items():
                coerced_v = v
                coerced_int = int(coerced_v)  # type: ignore[call-overload]
                items.append((str(k), coerced_int))
            return dict(items)
        return {}

    required = ("relative_path", "fingerprint", "file_size_bytes", "kind")
    if not all(key in blob for key in required):
        return None

    def _i(key: str) -> int:
        raw = blob.get(key)
        return int(raw) if isinstance(raw, (int, float, str, bytes)) else 0

    def _b(key: str) -> bool:
        raw = blob.get(key)
        return bool(raw) if raw is not None else False

    def _s(key: str) -> str:
        return str(blob.get(key)) if blob.get(key) is not None else ""

    return PerFileSummary(
        relative_path=_s("relative_path"),
        fingerprint=_s("fingerprint"),
        file_size_bytes=_i("file_size_bytes"),
        kind=_s("kind"),
        scan_failed=_b("scan_failed"),
        rows=_i("rows"),
        non_empty=_i("non_empty"),
        empty_or_null=_i("empty_or_null"),
        total_chars=_i("total_chars"),
        total_words=_i("total_words"),
        total_tokens_estimate=_i("total_tokens_estimate"),
        document_ids=frozenset(_get_list("document_ids")),
        section_ids=frozenset(_get_list("section_ids")),
        qids=frozenset(_get_list("qids")),
        languages=_get_dict_int("languages"),
        fact_rows=_i("fact_rows"),
        fact_ids=frozenset(_get_list("fact_ids")),
        subject_qids=frozenset(_get_list("subject_qids")),
        property_ids=frozenset(_get_list("property_ids")),
        property_labels=_get_dict_str("property_labels"),
        property_counts=_get_dict_int("property_counts"),
        with_property_en_label=_i("with_property_en_label"),
        with_value_en_label=_i("with_value_en_label"),
        with_qualifiers=_i("with_qualifiers"),
        with_references=_i("with_references"),
        unavailable_qualifiers=_i("unavailable_qualifiers"),
        unavailable_references=_i("unavailable_references"),
        value_type_counts=_get_dict_int("value_type_counts"),
    )


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _kind_for_rel(rel: str) -> str:
    """Return the augmentation kind for a sidecar path relative to
    ``<processed>/``."""
    if rel.startswith("wikipedia/documents/") or rel.startswith("wikivoyage/documents/"):
        return KIND_DOCUMENT
    if rel.startswith("wikipedia/sections/") or rel.startswith("wikivoyage/sections/"):
        return KIND_SECTION
    if rel.startswith("wikidata/facts/"):
        return KIND_FACT
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
    rows = 0
    non_empty = 0
    empty_or_null = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0
    document_ids: set[str] = set()
    section_ids: set[str] = set()
    qids: set[str] = set()
    languages: Counter[str] = Counter()
    region_count = 0
    for summary in summaries:
        region_count += 1
        if summary.scan_failed:
            continue
        rows += summary.rows
        non_empty += summary.non_empty
        empty_or_null += summary.empty_or_null
        total_chars += summary.total_chars
        total_words += summary.total_words
        total_tokens += summary.total_tokens_estimate
        document_ids.update(summary.document_ids)
        section_ids.update(summary.section_ids)
        qids.update(summary.qids)
        for lang, count in summary.languages.items():
            languages[lang] += count

    if summaries and summaries[0].kind == KIND_SECTION:
        unique_section_ids = len(section_ids)
        unique_documents = len(document_ids)
        avg = (rows / unique_documents) if unique_documents > 0 else 0.0
    else:
        unique_section_ids = 0
        unique_documents = len(document_ids)
        avg = 0.0

    non_empty_rate = (non_empty / rows) if rows > 0 else 0.0
    by_code = sorted(languages.items(), key=lambda item: (-item[1], item[0]))
    top_languages = tuple(by_code[:TOP_LANGUAGES_LIMIT])
    return ProjectTextStats(
        subdir_present=subdir_present,
        rows=rows,
        unique_documents=unique_documents,
        unique_section_ids=unique_section_ids,
        unique_qids=len(qids),
        language_count=len(languages),
        region_count=region_count,
        non_empty=non_empty,
        empty_or_null=empty_or_null,
        non_empty_rate=non_empty_rate,
        total_chars=total_chars,
        total_words=total_words,
        total_tokens_estimate=total_tokens,
        avg_sections_per_doc=avg,
        top_languages=top_languages,
    )


def _merge_wikidata_facts(
    summaries: list[PerFileSummary], *, subdir_present: bool
) -> WikidataFactStats:
    rows = 0
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
    for summary in summaries:
        if summary.scan_failed:
            continue
        rows += summary.fact_rows
        fact_ids.update(summary.fact_ids)
        subjects.update(summary.subject_qids)
        properties.update(summary.property_ids)
        for pid, label in summary.property_labels.items():
            if pid not in property_labels:
                property_labels[pid] = label
        for pid, count in summary.property_counts.items():
            property_counts[pid] += count
        with_prop_en += summary.with_property_en_label
        with_value_en += summary.with_value_en_label
        with_qualifiers += summary.with_qualifiers
        with_references += summary.with_references
        unavailable_qualifiers += summary.unavailable_qualifiers
        unavailable_references += summary.unavailable_references
        for value_type, count in summary.value_type_counts.items():
            value_types[value_type] += count

    sorted_properties = sorted(property_counts.items(), key=lambda item: (-item[1], item[0]))[
        :TOP_PROPERTIES_LIMIT
    ]
    top_properties = tuple(
        (pid, property_labels.get(pid, ""), count) for pid, count in sorted_properties
    )
    value_type_distribution = tuple(
        sorted(value_types.items(), key=lambda item: (-item[1], item[0]))
    )
    return WikidataFactStats(
        subdir_present=subdir_present,
        rows=rows,
        unique_facts=len(fact_ids),
        unique_subjects=len(subjects),
        distinct_property_ids=len(properties),
        with_property_en_label=with_prop_en,
        with_value_en_label=with_value_en,
        with_qualifiers=with_qualifiers,
        with_references=with_references,
        unavailable_qualifiers=unavailable_qualifiers,
        unavailable_references=unavailable_references,
        region_count=len(summaries),
        value_type_distribution=value_type_distribution,
        top_properties=top_properties,
    )


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
        present = sum(
            1 for rel in AUGMENTATION_SUBDIRS if (processed / rel / f"{stem}.parquet").exists()
        )
        if present == len(AUGMENTATION_SUBDIRS):
            fully.add(stem)
        elif present > 0:
            partial.add(stem)
    return fully, partial


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

    existing_index = load_cache_index(cache_index_dir)

    new_index: dict[str, dict[str, Any]] = {}
    by_subdir: dict[str, list[PerFileSummary]] = {
        "wikipedia/documents": [],
        "wikipedia/sections": [],
        "wikivoyage/documents": [],
        "wikivoyage/sections": [],
        "wikidata/facts": [],
    }
    unreadable = 0
    # ``subdir_present`` is True iff at least one readable, valid
    # Parquet sidecar was located inside the sub-directory. Directory
    # existence alone is not enough: a sidecar sub-directory may
    # exist but contain no Parquets (and is therefore rendered as
    # "No data exists yet.").
    present_subdirs: set[str] = set()

    live_paths = _scan_paths(processed_dir, AUGMENTATION_SUBDIRS)
    for parquet_path in live_paths:
        rel = _relative_path(processed_dir, parquet_path)
        fp = _file_fingerprint(parquet_path)
        cached = existing_index.get(rel)
        summary: PerFileSummary | None
        if (
            cached is not None
            and cached.get("fingerprint") == fp
            and cached.get("scan_failed") is not True
        ):
            summary = _summary_from_json(cached)
            # Defensive: a malformed cache entry is dropped and the
            # file is re-scanned. A successfully recovered scan
            # (``scan_failed=False``) is preserved across the refresh.
            if summary is None:
                summary = _scan_one_file(processed_dir, parquet_path)
        else:
            summary = _scan_one_file(processed_dir, parquet_path)
        if summary is None:
            continue
        new_index[rel] = _summary_to_json(summary)
        if summary.scan_failed:
            unreadable += 1
        else:
            present_subdirs.add(rel.split("/", 1)[0] + "/" + rel.split("/", 2)[1])
        for subdir_prefix in by_subdir:
            if rel.startswith(subdir_prefix + "/"):
                by_subdir[subdir_prefix].append(summary)
                break

    subdir_present_flags = {sub: sub in present_subdirs for sub in by_subdir}

    wikipedia_documents = _merge_project_text(
        by_subdir["wikipedia/documents"],
        subdir_present=subdir_present_flags["wikipedia/documents"],
    )
    wikipedia_sections = _merge_project_text(
        by_subdir["wikipedia/sections"],
        subdir_present=subdir_present_flags["wikipedia/sections"],
    )
    wikivoyage_documents = _merge_project_text(
        by_subdir["wikivoyage/documents"],
        subdir_present=subdir_present_flags["wikivoyage/documents"],
    )
    wikivoyage_sections = _merge_project_text(
        by_subdir["wikivoyage/sections"],
        subdir_present=subdir_present_flags["wikivoyage/sections"],
    )
    wikidata_facts = _merge_wikidata_facts(
        by_subdir["wikidata/facts"],
        subdir_present=subdir_present_flags["wikidata/facts"],
    )

    core_bytes = _core_bytes(processed_dir)
    aug_bytes = sum(
        summary.file_size_bytes for summaries in by_subdir.values() for summary in summaries
    )
    total_bytes = core_bytes + aug_bytes

    write_cache_index(cache_index_dir, new_index)

    return AugmentationStats(
        core_region_count=len(cores),
        fully_augmented_count=len(fully),
        partial_augmented_count=len(partial),
        not_augmented_count=len(cores) - len(fully) - len(partial),
        orphan_sidecar_stems=tuple(orphans),
        wikipedia_documents=wikipedia_documents,
        wikipedia_sections=wikipedia_sections,
        wikivoyage_documents=wikivoyage_documents,
        wikivoyage_sections=wikivoyage_sections,
        wikidata_facts=wikidata_facts,
        core_parquet_bytes=core_bytes,
        augmentation_parquet_bytes=aug_bytes,
        total_parquet_bytes=total_bytes,
        unreadable_file_count=unreadable,
    )


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
