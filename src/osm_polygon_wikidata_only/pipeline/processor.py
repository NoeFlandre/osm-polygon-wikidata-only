"""Single-PBF processor -- the documented facade.

This is the per-PBF orchestrator that ties together the four focused
phases of one PBF:

1. :func:`pipeline.extractor.extract_pbf` -- stream and parse the
   PBF candidates into an immutable :class:`ExtractedPbf`.
2. :func:`pipeline.enrichment_phase.run_enrichment_phase` -- fetch
   unique-QID Wikidata + Wikipedia summaries with the heartbeat
   lifecycle and raise :class:`IncompleteEnrichmentError` on
   non-fatal-elided failures.
3. :func:`pipeline.row_construction.enrich_polygon` + :func:`pipeline.row_construction.build_articles_and_links`
   -- derive per-polygon coverage fields and build deterministic
   articles and polygon-article links.
4. :func:`pipeline.persistence.run_persistence_phase` -- write the
   three parquet files and the manifest entry.

The CLI calls :func:`process_pbf` once per input file. The other
four helpers (``PbfStem``, ``ExtractedPbf``, ``extract_pbf`` and
``process_extracted_pbf``) are also re-exported here for callers
that already imported them from :mod:`pipeline.processor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.models import Polygon
from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaClient
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.pbf_reader import (  # noqa: F401  (re-export)
    PBFReader,
    region_from_filename,
)
from osm_polygon_wikidata_only.pipeline.completeness import IncompleteEnrichmentError

from . import rows as row_operations
from .enrichment_phase import run_enrichment_phase
from .extractor import ExtractedPbf, PbfStem, extract_pbf
from .persistence import run_persistence_phase

LOGGER_NAME = "osm_polygon_wikidata_only.pipeline.processor"
LOGGER = logging.getLogger(LOGGER_NAME)


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

    Orchestrates the four focused phases and aggregates the canonical
    ``stage_timings_s`` keys:

    * ``extract`` -- PBF streaming + candidate conversion
    * ``enrich`` -- batch fetch of unique Wikidata + Wikipedia QIDs
    * ``build_rows`` -- per-polygon coverage + article and link rows
    * ``write_parquet`` -- three atomic parquet writes
    * ``manifest`` -- single upsert_entry into processed_pbfs.json

    ``cache`` is reserved for compatibility with future incremental
    implementations and is currently unused.
    """
    extracted = extract_pbf(pbf_path, settings=settings)
    return process_extracted_pbf(
        extracted,
        data_root=data_root,
        wikidata_client=wikidata_client,
        wikipedia_client=wikipedia_client,
        settings=settings,
        cache=cache,
    )


def process_extracted_pbf(
    extracted: ExtractedPbf,
    *,
    data_root: DataRoot,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    settings: Settings,
    cache: JsonFileCache | None = None,
) -> ProcessResult:
    """Enrich and publish a previously extracted PBF in the calling thread."""
    del cache  # Reserved for compatibility with the synchronous facade.
    stem = extracted.stem
    pbf_path = stem.path
    polygons = list(extracted.polygons)
    timings = {"extract": extracted.extraction_duration_s}

    # Step 1: enrichment.
    enrichment = run_enrichment_phase(
        polygons,
        region=stem.region,
        wikidata_client=wikidata_client,
        wikipedia_client=wikipedia_client,
        settings=settings,
    )
    timings["enrich"] = enrichment.enrichment_duration_s

    # Step 2: per-polygon coverage + article/link rows.
    row_started = _perf_counter()
    summaries = enrichment.summaries
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
    articles, links = row_operations.build_articles_and_links(enriched, summaries)
    timings["build_rows"] = _perf_counter() - row_started

    # Step 3: durable writes.
    persistence = run_persistence_phase(
        enriched,
        articles,
        links,
        data_root=data_root,
        stem=stem.stem,
        source_pbf=pbf_path.name,
    )
    timings["write_parquet"] = persistence.write_parquet_duration_s
    timings["manifest"] = persistence.manifest_duration_s

    return ProcessResult(
        polygons_path=persistence.polygons_path,
        articles_path=persistence.articles_path,
        polygon_articles_path=persistence.links_path,
        manifest_path=persistence.manifest_path,
        polygon_count=len(enriched),
        article_count=len(articles),
        link_count=len(links),
        manifest_entry=persistence.manifest_entry,
        stage_timings_s=timings,
    )


def _perf_counter() -> float:
    import time

    return time.perf_counter()


__all__ = [
    "ExtractedPbf",
    "IncompleteEnrichmentError",
    "PbfStem",
    "ProcessResult",
    "extract_pbf",
    "process_extracted_pbf",
    "process_pbf",
]
