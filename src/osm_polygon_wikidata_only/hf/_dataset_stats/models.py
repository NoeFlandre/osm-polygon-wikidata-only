"""DatasetStats frozen dataclass.

Re-exported by the :mod:`osm_polygon_wikidata_only.hf.dataset_stats`
facade. Field order, frozen flag, and ``slots=True`` are part of the
documented contract; do not reorder.

Augmentation-specific dataclasses (``ProjectTextStats``,
``WikidataFactStats`` and ``AugmentationStats``) are PRIVATE
implementation details: they are NOT exported by the public
``hf.dataset_stats`` facade, and the canonical dataset-card renderer
is the only sanctioned consumer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


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


# ---------------------------------------------------------------------------
# Private augmentation models. NOT part of the public facade contract.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectTextStats:
    """Private aggregation container for a per-project text sidecar set.

    Used for both Wikipedia and Wikivoyage documents and sections. Not
    re-exported by :mod:`osm_polygon_wikidata_only.hf.dataset_stats`.

    Field semantics:

    * ``subdir_present`` — ``True`` when the sidecar sub-directory was
      located during the last scan. Lets the renderer distinguish a
      missing sub-directory ("No data exists yet.") from a
      present-but-empty one ("This sidecar is present but empty.").
    * ``rows`` — total rows scanned across all sidecar files. Always
      reflects the latest scan; an empty, but present, sidecar set
      still gets ``rows == 0``.
    * ``unique_documents`` — count of distinct ``document_id`` values.
    * ``unique_section_ids`` — count of distinct ``section_id`` values
      (only meaningful for section sidecars, ``0`` for document
      sidecars). Distinct from ``unique_documents`` which is the count
      of distinct documents those sections belong to.
    * ``unique_qids`` — count of distinct Wikidata QIDs across all rows.
    * ``language_count`` — count of distinct languages across all rows.
    * ``region_count`` — number of distinct sidecar files
      (one per PBF stem) located during the last scan.
    * ``non_empty`` — rows whose textual content is a non-null, non-blank
      string.
    * ``empty_or_null`` — rows whose textual content is null, empty, or
      whitespace-only.
    * ``non_empty_rate`` — fraction ``0.0..1.0`` of non-empty rows out
      of all rows. Always defined, even when ``rows == 0`` (returns 0.0).
    * ``avg_sections_per_doc`` — for the section sidecar, ``rows /
      unique_documents`` with a guarded denominator of ``1``.
    * ``total_chars`` / ``total_words`` / ``total_tokens_estimate`` —
      summed across rows.
    * ``top_languages`` — top-N languages sorted by (-count, language).
    """

    subdir_present: bool = False
    rows: int = 0
    unique_documents: int = 0
    unique_section_ids: int = 0
    unique_qids: int = 0
    language_count: int = 0
    region_count: int = 0
    non_empty: int = 0
    empty_or_null: int = 0
    non_empty_rate: float = 0.0
    total_chars: int = 0
    total_words: int = 0
    total_tokens_estimate: int = 0
    avg_sections_per_doc: float = 0.0
    top_languages: tuple[tuple[str, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class WikidataFactStats:
    """Private aggregation container for the ``wikidata/facts`` sidecar.

    Not re-exported by :mod:`osm_polygon_wikidata_only.hf.dataset_stats`.
    """

    subdir_present: bool = False
    rows: int = 0
    unique_facts: int = 0
    unique_subjects: int = 0
    distinct_property_ids: int = 0
    with_property_en_label: int = 0
    with_value_en_label: int = 0
    with_qualifiers: int = 0
    with_references: int = 0
    unavailable_qualifiers: int = 0
    unavailable_references: int = 0
    region_count: int = 0
    value_type_distribution: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    top_properties: tuple[tuple[str, str, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CombinedLanguageStats:
    """Combined Wikipedia and Wikivoyage document/language distribution."""

    document_count: int = 0
    language_count: int = 0
    documents_per_language: tuple[tuple[str, int], ...] = ()
    polygons_per_language: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class PerFileSummary:
    """Per-file aggregation cached on disk across README refreshes.

    Each entry is the deterministic, side-effect-free aggregation of a
    single finalized Parquet file. The cache layer keys these
    summaries by ``(relative_path, fingerprint)`` so that replacing a
    file with a new one (e.g. a new revision of the same sidecar)
    invalidates the cache automatically.

    The :class:`AugmentationStats` is purely the merge of these
    summaries across the live filesystem, never the source of truth.
    The cache layer always walks the live filesystem first; missing
    files disappear from aggregates.

    A single sidecar file is either a "documents" sidecar (used for
    Wikipedia and Wikivoyage documents), a "sections" sidecar (used
    for sections), a "facts" sidecar (used for Wikidata facts), or
    none of those. Exactly one of the three aggregations is filled
    in; the other two stay at their zero defaults.

    ``scan_failed`` is ``True`` when the parquet could not be opened
    (corrupt, truncated, or otherwise unreadable). The file's bytes
    still count toward :attr:`AugmentationStats.augmentation_parquet_bytes`,
    but its rows are not aggregated and the cache entry is rebuilt
    on the next read attempt.
    """

    relative_path: str
    fingerprint: str
    file_size_bytes: int
    kind: str
    scan_failed: bool = False
    rows: int = 0
    non_empty: int = 0
    empty_or_null: int = 0
    total_chars: int = 0
    total_words: int = 0
    total_tokens_estimate: int = 0
    document_ids: frozenset[str] = field(default_factory=frozenset)
    section_ids: frozenset[str] = field(default_factory=frozenset)
    qids: frozenset[str] = field(default_factory=frozenset)
    languages: Mapping[str, int] = field(default_factory=dict)
    # Fact-only fields. Empty mappings for non-fact files.
    fact_rows: int = 0
    fact_ids: frozenset[str] = field(default_factory=frozenset)
    subject_qids: frozenset[str] = field(default_factory=frozenset)
    property_ids: frozenset[str] = field(default_factory=frozenset)
    property_labels: Mapping[str, str] = field(default_factory=dict)
    property_counts: Mapping[str, int] = field(default_factory=dict)
    with_property_en_label: int = 0
    with_value_en_label: int = 0
    with_qualifiers: int = 0
    with_references: int = 0
    unavailable_qualifiers: int = 0
    unavailable_references: int = 0
    value_type_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AugmentationStats:
    """Private aggregation container for the augmentation sidecars.

    Scanned from the local finalized Parquet files under
    ``<processed>/{wikipedia,wikivoyage,wikidata}/...`` plus the core
    parquet count to derive the coverage classification. Never re-exported
    by :mod:`osm_polygon_wikidata_only.hf.dataset_stats`.
    """

    core_region_count: int
    fully_augmented_count: int
    partial_augmented_count: int
    not_augmented_count: int
    orphan_sidecar_stems: tuple[str, ...]
    wikipedia_documents: ProjectTextStats
    wikipedia_sections: ProjectTextStats
    wikivoyage_documents: ProjectTextStats
    wikivoyage_sections: ProjectTextStats
    wikidata_facts: WikidataFactStats
    core_parquet_bytes: int
    augmentation_parquet_bytes: int
    total_parquet_bytes: int
    unreadable_file_count: int
    combined_languages: CombinedLanguageStats = field(default_factory=CombinedLanguageStats)
