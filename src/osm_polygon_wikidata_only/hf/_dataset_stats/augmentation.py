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

import logging
from pathlib import Path
from typing import Any

from .augmentation_merge import (
    merge_project_text,
    merge_wikidata_facts,
)
from .augmentation_scan import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    KIND_DOCUMENT,
    KIND_FACT,
    KIND_SECTION,
    SECTION_COLUMNS,
    scan_one_file,
)
from .cache import (
    file_fingerprint as _file_fingerprint,
)
from .cache import (
    load_cache_index,
    write_cache_index,
)
from .cache import (
    relative_path as _relative_path,
)
from .cache import (
    scan_paths as _scan_paths,
)
from .combined_languages import compute_combined_language_stats
from .models import (
    AugmentationStats,
    PerFileSummary,
)
from .scanning import sorted_parquets
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

# Core sub-directories whose parquet sizes count toward core_parquet_bytes.
CORE_SUBDIRS: tuple[str, ...] = ("polygons", "polygon_articles")


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
        return summary if summary is not None else scan_one_file(processed_dir, parquet_path)
    return scan_one_file(processed_dir, parquet_path)


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
        "wikipedia_documents": merge_project_text(
            by_subdir["wikipedia/documents"], subdir_present=present["wikipedia/documents"]
        ),
        "wikipedia_sections": merge_project_text(
            by_subdir["wikipedia/sections"], subdir_present=present["wikipedia/sections"]
        ),
        "wikivoyage_documents": merge_project_text(
            by_subdir["wikivoyage/documents"], subdir_present=present["wikivoyage/documents"]
        ),
        "wikivoyage_sections": merge_project_text(
            by_subdir["wikivoyage/sections"], subdir_present=present["wikivoyage/sections"]
        ),
    }
    facts = merge_wikidata_facts(
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
