"""Single-PBF processor.

This is the per-PBF orchestrator that ties together the PBF reader,
the polygon extractor, the enrichment clients, the parquet writers,
and the manifest updater. The CLI calls :func:`process_pbf` once per
input file.
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
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.models import Article, Polygon, PolygonArticleLink
from osm_polygon_wikidata_only.enrichment.article_linker import LinkSummary, fetch_qids
from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaClient
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import (
    manifest_path,
    upsert_entry,
)
from osm_polygon_wikidata_only.io.parquet import (
    write_articles,
    write_polygon_articles,
    write_polygons,
)
from osm_polygon_wikidata_only.io.pbf_reader import (  # noqa: F401  (kept for re-export)
    PBFReader,
    region_from_filename,
)
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from . import rows as row_operations
from .completeness import IncompleteEnrichmentError
from .extractor import candidate_to_polygon, polygon_to_dict
from .stats import StreamingStats

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PbfStem:
    """A parsed PBF filename.

    ``stem`` is the part of the filename before ``.osm.pbf``
    (e.g. ``monaco-latest``). ``region`` is the part before
    ``-latest.osm.pbf`` (e.g. ``monaco``). The remote parquet paths
    use ``stem`` so the layout is stable across reruns.
    """

    path: Path
    stem: str
    region: str

    @classmethod
    def from_path(cls, path: Path) -> PbfStem:
        name = path.name
        stem = name[: -len(".osm.pbf")] if name.endswith(".osm.pbf") else path.stem
        # Region is the stem without the trailing "-latest" (or the
        # whole stem if there's no "-latest" suffix).
        region = stem[: -len("-latest")] if stem.endswith("-latest") else stem
        return cls(path=path, stem=stem, region=region)


@dataclass(slots=True)
class ProcessResult:
    """What :func:`process_pbf` returns: row counts + manifest entry."""

    polygons_path: Path
    articles_path: Path
    polygon_articles_path: Path
    manifest_path: Path
    polygon_count: int
    article_count: int
    link_count: int
    manifest_entry: dict[str, Any]
    stage_timings_s: dict[str, float]


def _local_path(processed_dir: Path, subdir: str, stem: str) -> Path:
    return processed_dir / subdir / f"{stem}.parquet"


def _remote_path(subdir: str, stem: str) -> str:
    return f"{subdir}/{stem}.parquet"


def process_pbf(
    pbf_path: Path,
    *,
    data_root: DataRoot,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    settings: Settings,
    cache: JsonFileCache | None = None,
) -> ProcessResult:
    """Process one PBF end-to-end.

    Steps:

    1. Stream polygonal candidates from the PBF.
    2. Convert to :class:`Polygon` rows.
    3. Enrich each polygon with Wikidata + Wikipedia data.
    4. Deduplicate articles; build polygon-article links.
    5. Write the three parquet files.
    6. Upsert the manifest entry.
    """
    stem = PbfStem.from_path(pbf_path)
    timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    extracted_at = utc_now_iso()
    LOGGER.info("Processing %s (region=%s)", pbf_path.name, stem.region)

    # Step 1-2: extract polygons.
    polygons: list[Polygon] = []
    # Late import via the module so tests can monkeypatch
    # ``osm_polygon_wikidata_only.io.pbf_reader.PBFReader``.
    import osm_polygon_wikidata_only.io.pbf_reader as _pbf_reader_mod

    reader = _pbf_reader_mod.PBFReader(pbf_path)

    def add_candidate(candidate: object) -> None:
        if settings.limit is not None and len(polygons) >= settings.limit:
            return
        polygon = candidate_to_polygon(
            candidate,  # type: ignore[arg-type]
            source_pbf_stem=stem.stem,
            region=stem.region,
            source_pbf=pbf_path.name,
            extracted_at=extracted_at,
        )
        if polygon is not None:
            polygons.append(polygon)

    stream_candidates = getattr(reader, "iter_polygon_candidates", None)
    if callable(stream_candidates):
        stream_candidates(add_candidate)
    else:
        for candidate in reader.collect_polygon_candidates():
            add_candidate(candidate)
    LOGGER.info("Extracted %d polygons from %s", len(polygons), pbf_path.name)
    timings["extract"] = time.perf_counter() - stage_started

    # Step 3-4: enrich.
    stage_started = time.perf_counter()
    summaries: dict[str, LinkSummary] = {}
    unique_qids = sorted({p.wikidata for p in polygons if p.wikidata})
    if unique_qids:
        # Fetch unique QIDs once, then reuse the summaries for each polygon.
        unique_summaries = fetch_qids(
            unique_qids,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            languages=settings.languages,
            fetch_full_text=settings.fetch_full_text,
            max_articles_per_qid=settings.max_articles_per_qid,
            batch_size=settings.enrichment_batch_size,
            site_workers=settings.enrichment_site_workers,
        )
        for s in unique_summaries:
            summaries[s.qid] = s
    failures = [
        f"{summary.qid}:{site} ({summary.statuses.get(site, 'unknown')}): {error}"
        for summary in summaries.values()
        for site, error in summary.errors.items()
        if summary.statuses.get(site) != "empty_text"
    ]
    if failures:
        raise IncompleteEnrichmentError(
            "Incomplete Wikipedia enrichment; rerun the same command to resume: "
            + "; ".join(failures)
        )
    LOGGER.info(
        "Fetched summaries for %d unique QIDs (%d QIDs total)",
        len(summaries),
        len(unique_qids),
    )
    enriched: list[Polygon] = [
        row_operations.enrich_polygon(
            p,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            settings=settings,
            summaries=summaries,
        )
        for p in polygons
    ]
    timings["enrich"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()

    articles, links = row_operations.build_articles_and_links(enriched, summaries)
    LOGGER.info(
        "Built %d unique articles and %d polygon-article links",
        len(articles),
        len(links),
    )
    timings["build_rows"] = time.perf_counter() - stage_started

    # Step 5: write parquet.
    stage_started = time.perf_counter()
    polygons_path = _local_path(data_root.processed, PROCESSED_POLYGONS, stem.stem)
    articles_path = _local_path(data_root.processed, PROCESSED_ARTICLES, stem.stem)
    links_path = _local_path(data_root.processed, PROCESSED_LINKS, stem.stem)

    temporary_paths = [
        path.with_suffix(path.suffix + ".tmp")
        for path in (polygons_path, articles_path, links_path)
    ]
    try:
        write_polygons(temporary_paths[0], [polygon_to_dict(p) for p in enriched])
        write_articles(temporary_paths[1], [_article_to_dict(a) for a in articles])
        write_polygon_articles(temporary_paths[2], [_link_to_dict(link) for link in links])
        for temporary, final in zip(
            temporary_paths, (polygons_path, articles_path, links_path), strict=True
        ):
            os.replace(temporary, final)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    timings["write_parquet"] = time.perf_counter() - stage_started

    # Step 6: manifest.
    stage_started = time.perf_counter()
    stats = StreamingStats()
    for p in enriched:
        stats.add_polygon(p)
    for a in articles:
        stats.add_article(a)
    for link in links:
        stats.add_link(link)
    final_stats = stats.finalize()

    mpath = manifest_path(data_root.processed_manifests)
    entry = upsert_entry(
        mpath,
        source_pbf=pbf_path.name,
        region=stem.region,
        polygons_path=_remote_path(PROCESSED_POLYGONS, stem.stem),
        articles_path=_remote_path(PROCESSED_ARTICLES, stem.stem),
        polygon_articles_path=_remote_path(PROCESSED_LINKS, stem.stem),
        stats=final_stats,
        extraction_version=__version__,
    )
    timings["manifest"] = time.perf_counter() - stage_started

    return ProcessResult(
        polygons_path=polygons_path,
        articles_path=articles_path,
        polygon_articles_path=links_path,
        manifest_path=mpath,
        polygon_count=len(enriched),
        article_count=len(articles),
        link_count=len(links),
        manifest_entry=entry,
        stage_timings_s=timings,
    )


def _article_to_dict(a: Article) -> dict[str, Any]:
    return dict(a.__dict__)


def _link_to_dict(link: PolygonArticleLink) -> dict[str, Any]:
    return dict(link.__dict__)


__all__ = ["IncompleteEnrichmentError", "PbfStem", "ProcessResult", "process_pbf"]
