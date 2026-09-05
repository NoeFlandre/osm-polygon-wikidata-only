"""Persistence phase: write three parquets and the manifest entry.

This module owns the focused, durable-write phase of one processed
PBF. It is called by :mod:`pipeline.processor` (the documented
facade) in the same thread.

The phase:

* Picks three local parquet paths under
  ``data_root.processed/{polygons,articles,links}/<stem>.parquet``.
* Writes the three parquet files via
  :func:`osm_polygon_wikidata_only.io.parquet.write_*` into a
  temporary ``.parquet.tmp`` sibling, then atomically renames each
  one via :func:`os.replace`.
* If any write or replace fails, the temporary files are deleted
  in the ``finally`` block so a half-published PBF cannot survive.
* Computes manifest stats with :func:`accumulate_stats` from the existing rows.
* Calls :func:`osm_polygon_wikidata_only.io.manifest.upsert_entry`
  to atomically merge the canonical entry into
  ``processed_pbfs.json``.

This module never performs HTTP work, never reads the PBF, and
never updates the augmentation manifest.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only import __version__
from osm_polygon_wikidata_only.augmentation.integrity import (
    INTEGRITY_CONTRACT_VERSION,
    PolygonArticlesIntegrityResult,
    enforce_polygon_articles_integrity,
)
from osm_polygon_wikidata_only.config.paths import (
    PROCESSED_ARTICLES,
    PROCESSED_LINKS,
    PROCESSED_POLYGONS,
    DataRoot,
)
from osm_polygon_wikidata_only.domain.models import (
    Article,
    ManifestStats,
    Polygon,
    PolygonArticleLink,
)
from osm_polygon_wikidata_only.io.manifest import (
    manifest_path,
    update_entry,
    upsert_entry,
)
from osm_polygon_wikidata_only.io.parquet import (
    write_articles,
    write_polygon_articles,
    write_polygons,
)
from osm_polygon_wikidata_only.pipeline.stats import accumulate_stats

LOGGER = logging.getLogger(__name__)
# Lifecycle log message ("Built N unique articles and M polygon-article
# links") previously emitted under :mod:`pipeline.processor`. Keep
# emitting it under the same legacy logger name.
PROCESSOR_LOGGER = logging.getLogger("osm_polygon_wikidata_only.pipeline.processor")


@dataclass(frozen=True, slots=True)
class PersistenceOutcome:
    """The three parquet paths, the canonical manifest path, and the
    manifest entry that was written for *stem*. The caller adds the
    ``write_parquet`` and ``manifest`` slots of
    ``stage_timings_s`` from ``write_parquet_duration_s`` and
    ``manifest_duration_s``."""

    polygons_path: Path
    articles_path: Path
    links_path: Path
    manifest_path: Path
    manifest_entry: dict[str, Any]
    write_parquet_duration_s: float
    manifest_duration_s: float
    integrity_result: PolygonArticlesIntegrityResult | None = None


def _local_path(processed_dir: Path, subdir: str, stem: str) -> Path:
    return processed_dir / subdir / f"{stem}.parquet"


def _remote_path(subdir: str, stem: str) -> str:
    return f"{subdir}/{stem}.parquet"


def _persistence_paths(data_root: DataRoot, stem: str) -> tuple[Path, Path, Path]:
    """Return the canonical polygon, article, and link paths."""
    return (
        _local_path(data_root.processed, PROCESSED_POLYGONS, stem),
        _local_path(data_root.processed, PROCESSED_ARTICLES, stem),
        _local_path(data_root.processed, PROCESSED_LINKS, stem),
    )


def _temporary_paths(final_paths: tuple[Path, Path, Path]) -> tuple[Path, Path, Path]:
    """Return temporary siblings corresponding to final parquet paths."""
    return (
        final_paths[0].with_suffix(final_paths[0].suffix + ".tmp"),
        final_paths[1].with_suffix(final_paths[1].suffix + ".tmp"),
        final_paths[2].with_suffix(final_paths[2].suffix + ".tmp"),
    )


def _write_parquet_to_temporary(
    polygons: list[Polygon],
    articles: list[Article],
    links: list[PolygonArticleLink],
    temporary_paths: tuple[Path, Path, Path],
) -> None:
    """Write row collections to their temporary parquet siblings."""
    write_polygons(temporary_paths[0], [dict(p.__dict__) for p in polygons])
    write_articles(temporary_paths[1], [dict(a.__dict__) for a in articles])
    write_polygon_articles(temporary_paths[2], [dict(link.__dict__) for link in links])


def _replace_persistence_files(
    temporary_paths: tuple[Path, Path, Path],
    final_paths: tuple[Path, Path, Path],
) -> None:
    """Atomically install each completed parquet sibling."""
    for temporary, final in zip(temporary_paths, final_paths, strict=True):
        os.replace(temporary, final)


def _write_persistence_files(
    polygons: list[Polygon],
    articles: list[Article],
    links: list[PolygonArticleLink],
    final_paths: tuple[Path, Path, Path],
) -> float:
    """Write all parquet artifacts and remove temporary siblings on failure."""
    temporary_paths = _temporary_paths(final_paths)
    write_started = time.perf_counter()
    try:
        _write_parquet_to_temporary(polygons, articles, links, temporary_paths)
        _replace_persistence_files(temporary_paths, final_paths)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return time.perf_counter() - write_started


def _write_manifest_entry(
    data_root: DataRoot,
    stem: str,
    source_pbf: str,
    stats: ManifestStats,
) -> tuple[Path, dict[str, Any], float]:
    """Write the canonical processed-manifest entry and return its timing."""
    manifest_started = time.perf_counter()
    path = manifest_path(data_root.processed_manifests)
    entry = upsert_entry(
        path,
        source_pbf=source_pbf,
        region=stem,
        polygons_path=_remote_path(PROCESSED_POLYGONS, stem),
        articles_path=_remote_path(PROCESSED_ARTICLES, stem),
        polygon_articles_path=_remote_path(PROCESSED_LINKS, stem),
        stats=stats,
        extraction_version=__version__,
    )
    return path, entry, time.perf_counter() - manifest_started


def _integrity_metadata(result: PolygonArticlesIntegrityResult) -> dict[str, Any]:
    """Serialize the deterministic integrity result for the manifest."""
    return {
        "contract_version": INTEGRITY_CONTRACT_VERSION,
        "shard": result.shard,
        "original_row_count": result.original_row_count,
        "retained_row_count": result.retained_row_count,
        "rejected_row_count": result.rejected_row_count,
        "rewritten": result.rewritten,
        "rejections": [record.to_dict() for record in result.rejections],
    }


def _enforce_integrity_if_needed(
    data_root: DataRoot,
    stem: str,
    source_pbf: str,
    links_path: Path,
    polygons_path: Path,
    manifest_path_value: Path,
    entry: dict[str, Any],
) -> tuple[PolygonArticlesIntegrityResult | None, dict[str, Any]]:
    """Run reject-only link integrity enforcement when both artifacts exist."""
    if not links_path.is_file() or not polygons_path.is_file():
        return None, entry
    result = enforce_polygon_articles_integrity(data_root, stem)
    if result.rejected_row_count <= 0:
        return result, entry
    updated = update_entry(
        manifest_path_value,
        source_pbf=source_pbf,
        integrity=_integrity_metadata(result),
    )
    PROCESSOR_LOGGER.warning(
        "Integrity pass dropped %d polygon_articles row(s) for shard %r "
        "with mismatched wikidata; canonical polygons unchanged.",
        result.rejected_row_count,
        stem,
    )
    return result, updated


def run_persistence_phase(
    polygons: list[Polygon],
    articles: list[Article],
    links: list[PolygonArticleLink],
    *,
    data_root: DataRoot,
    stem: str,
    source_pbf: str,
) -> PersistenceOutcome:
    """Write the three parquet files and the manifest entry for *stem*.

    On write failure, all temporary ``*.parquet.tmp`` siblings are
    removed and the manifest is left untouched -- no half-published
    PBF.
    """
    polygons_path, articles_path, links_path = _persistence_paths(data_root, stem)
    write_duration = _write_persistence_files(
        polygons,
        articles,
        links,
        (polygons_path, articles_path, links_path),
    )
    mpath, entry, manifest_duration = _write_manifest_entry(
        data_root,
        stem,
        source_pbf,
        accumulate_stats(polygons, articles, links),
    )

    PROCESSOR_LOGGER.info(
        "Built %d unique articles and %d polygon-article links",
        len(articles),
        len(links),
    )

    integrity_result, entry = _enforce_integrity_if_needed(
        data_root,
        stem,
        source_pbf,
        links_path,
        polygons_path,
        mpath,
        entry,
    )

    return PersistenceOutcome(
        polygons_path=polygons_path,
        articles_path=articles_path,
        links_path=links_path,
        manifest_path=mpath,
        manifest_entry=entry,
        write_parquet_duration_s=write_duration,
        manifest_duration_s=manifest_duration,
        integrity_result=integrity_result,
    )


__all__ = [
    "PersistenceOutcome",
    "run_persistence_phase",
]
