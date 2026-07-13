"""Enrichment phase: per-region Wikidata + Wikipedia fetch.

This module owns the focused enrichment phase of one extracted PBF.
It is called by :mod:`pipeline.processor` (the documented facade)
in the same thread.

The phase:

* Picks the unique set of Wikidata QIDs across every polygon and
  issues exactly one batch fetch per PBF (so a polygon repeated
  three times still costs one upstream call).
* Wraps the batch fetch with the established heartbeat lifecycle
  so a long enrichment still emits operator-visible progress.
* Translates any non-fatal, non-empty fetch_status into an
  :class:`IncompleteEnrichmentError` before any parquet is written.
* Records the wall-clock time and the unique-QID count for the
  ``enrich`` entry of ``stage_timings_s``.

This module never writes parquet, never updates the manifest, and
never reaches into the data root -- those belong to
:mod:`pipeline.row_construction` and :mod:`pipeline.persistence`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.models import Polygon
from osm_polygon_wikidata_only.enrichment.article_linker import LinkSummary, fetch_qids
from osm_polygon_wikidata_only.enrichment.progress import EnrichmentProgress
from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaClient
from osm_polygon_wikidata_only.pipeline.completeness import (
    NON_FATAL_FETCH_STATUSES,
    IncompleteEnrichmentError,
)
from osm_polygon_wikidata_only.pipeline.heartbeat import EnrichmentHeartbeat

LOGGER = logging.getLogger(__name__)
# Lifecycle log messages ("Starting enrichment for X" and
# "Fetched summaries for N unique QIDs") previously emitted
# under :mod:`pipeline.processor`. Keep emitting them under
# that same legacy logger name.
PROCESSOR_LOGGER = logging.getLogger("osm_polygon_wikidata_only.pipeline.processor")


@dataclass(frozen=True, slots=True)
class EnrichmentOutcome:
    """Per-QID summaries plus the canonical ``stage_timings_s`` slot."""

    summaries: dict[str, LinkSummary]
    enrichment_duration_s: float
    unique_qids: tuple[str, ...]


def unique_qids(polygons: list[Polygon]) -> tuple[str, ...]:
    """Sorted tuple of distinct QIDs across every polygon."""
    return tuple(sorted({p.wikidata for p in polygons if p.wikidata}))


def run_enrichment_phase(
    polygons: list[Polygon],
    *,
    region: str,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    settings: Settings,
) -> EnrichmentOutcome:
    """Run the enrichment phase for one PBF.

    The heartbeat lifecycle is invoked here so an exceptionally long
    enrichment still emits progress logs. Any non-fatal fetch status
    *outside* of :data:`NON_FATAL_FETCH_STATUSES` raises
    :class:`IncompleteEnrichmentError` -- the upstream caller is
    expected to abort the write pipeline before any parquet hit disk.
    """
    started = time.perf_counter()
    qids = unique_qids(polygons)
    PROCESSOR_LOGGER.info(
        "Starting enrichment for %s: %d unique Wikidata QIDs",
        region,
        len(qids),
    )
    summaries: dict[str, LinkSummary] = {}
    if qids:
        progress = EnrichmentProgress(total_qids=len(qids))
        with EnrichmentHeartbeat(
            region=region,
            snapshot=progress.snapshot,
            log=PROCESSOR_LOGGER.info,
        ):
            unique_summaries = fetch_qids(
                list(qids),
                wikidata_client=wikidata_client,
                wikipedia_client=wikipedia_client,
                languages=settings.languages,
                fetch_full_text=settings.fetch_full_text,
                max_articles_per_qid=settings.max_articles_per_qid,
                batch_size=settings.enrichment_batch_size,
                site_workers=settings.enrichment_site_workers,
                progress=progress,
            )
        for s in unique_summaries:
            summaries[s.qid] = s
    _raise_on_non_fatal_failures(summaries)
    PROCESSOR_LOGGER.info(
        "Fetched summaries for %d unique QIDs (%d QIDs total)",
        len(summaries),
        len(qids),
    )
    return EnrichmentOutcome(
        summaries=summaries,
        enrichment_duration_s=time.perf_counter() - started,
        unique_qids=qids,
    )


def _raise_on_non_fatal_failures(summaries: dict[str, LinkSummary]) -> None:
    failures = [
        f"{summary.qid}:{site} ({summary.statuses.get(site, 'unknown')}): {error}"
        for summary in summaries.values()
        for site, error in summary.errors.items()
        if summary.statuses.get(site) not in NON_FATAL_FETCH_STATUSES
    ]
    if failures:
        raise IncompleteEnrichmentError(
            "Incomplete Wikipedia enrichment; rerun the same command to resume: "
            + "; ".join(failures)
        )


__all__ = ["EnrichmentOutcome", "run_enrichment_phase", "unique_qids"]
