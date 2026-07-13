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
* Streams manifest stats from the article, polygon and link rows
  through :class:`StreamingStats` -- O(N) memory, not O(N^2).
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
from osm_polygon_wikidata_only.config.paths import (
    PROCESSED_ARTICLES,
    PROCESSED_LINKS,
    PROCESSED_POLYGONS,
    DataRoot,
)
from osm_polygon_wikidata_only.domain.models import Article, Polygon, PolygonArticleLink
from osm_polygon_wikidata_only.io.manifest import (
    manifest_path,
    upsert_entry,
)
from osm_polygon_wikidata_only.io.parquet import (
    write_articles,
    write_polygon_articles,
    write_polygons,
)
from osm_polygon_wikidata_only.pipeline.stats import StreamingStats

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


def _local_path(processed_dir: Path, subdir: str, stem: str) -> Path:
    return processed_dir / subdir / f"{stem}.parquet"


def _remote_path(subdir: str, stem: str) -> str:
    return f"{subdir}/{stem}.parquet"


def _polygon_row(p: Polygon) -> dict[str, Any]:
    return dict(p.__dict__)


def _article_row(a: Article) -> dict[str, Any]:
    return dict(a.__dict__)


def _link_row(link: PolygonArticleLink) -> dict[str, Any]:
    return dict(link.__dict__)


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
    polygons_path = _local_path(data_root.processed, PROCESSED_POLYGONS, stem)
    articles_path = _local_path(data_root.processed, PROCESSED_ARTICLES, stem)
    links_path = _local_path(data_root.processed, PROCESSED_LINKS, stem)

    temporary_paths = [
        path.with_suffix(path.suffix + ".tmp")
        for path in (polygons_path, articles_path, links_path)
    ]
    write_started = time.perf_counter()
    try:
        write_polygons(temporary_paths[0], [_polygon_row(p) for p in polygons])
        write_articles(temporary_paths[1], [_article_row(a) for a in articles])
        write_polygon_articles(temporary_paths[2], [_link_row(link) for link in links])
        for temporary, final in zip(
            temporary_paths,
            (polygons_path, articles_path, links_path),
            strict=True,
        ):
            os.replace(temporary, final)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    write_duration = time.perf_counter() - write_started

    manifest_started = time.perf_counter()
    stats = StreamingStats()
    for polygon in polygons:
        stats.add_polygon(polygon)
    for article in articles:
        stats.add_article(article)
    for link in links:
        stats.add_link(link)
    final_stats = stats.finalize()

    mpath = manifest_path(data_root.processed_manifests)
    entry = upsert_entry(
        mpath,
        source_pbf=source_pbf,
        region=stem,
        polygons_path=_remote_path(PROCESSED_POLYGONS, stem),
        articles_path=_remote_path(PROCESSED_ARTICLES, stem),
        polygon_articles_path=_remote_path(PROCESSED_LINKS, stem),
        stats=final_stats,
        extraction_version=__version__,
    )
    manifest_duration = time.perf_counter() - manifest_started

    PROCESSOR_LOGGER.info(
        "Built %d unique articles and %d polygon-article links",
        len(articles),
        len(links),
    )

    return PersistenceOutcome(
        polygons_path=polygons_path,
        articles_path=articles_path,
        links_path=links_path,
        manifest_path=mpath,
        manifest_entry=entry,
        write_parquet_duration_s=write_duration,
        manifest_duration_s=manifest_duration,
    )


__all__ = [
    "PersistenceOutcome",
    "run_persistence_phase",
]
