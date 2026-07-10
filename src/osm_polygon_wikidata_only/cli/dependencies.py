"""Typed construction of runtime paths and enrichment clients."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot, resolve_data_root
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    CachedWikidataClient,
    HttpWikidataClient,
    WikidataClient,
)
from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
    CachedWikipediaClient,
    HttpWikipediaClient,
    WikipediaClient,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler

LOGGER = logging.getLogger(__name__)


def resolve_cli_data_root(args: argparse.Namespace) -> DataRoot:
    """Resolve the external data root from parsed CLI arguments."""
    return resolve_data_root(explicit=args.data_root, repo_root=Path.cwd())


def build_clients(
    settings: Settings, *, data_root: DataRoot
) -> tuple[WikidataClient, WikipediaClient, JsonFileCache | None]:
    """Build clients that share one global Wikimedia scheduler."""
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=settings.wikimedia_max_in_flight,
        requests_per_minute=settings.wikimedia_requests_per_minute,
    )
    wikidata: WikidataClient = HttpWikidataClient(settings, scheduler=scheduler)
    wikipedia: WikipediaClient = HttpWikipediaClient(settings, scheduler=scheduler)
    if not settings.cache_enabled:
        return wikidata, wikipedia, None
    try:
        wikidata = CachedWikidataClient(
            HttpWikidataClient(settings, scheduler=scheduler),
            JsonFileCache(data_root.cache_wikidata),
        )
        wikipedia = CachedWikipediaClient(
            HttpWikipediaClient(settings, scheduler=scheduler),
            JsonFileCache(data_root.cache_wikipedia),
        )
        return wikidata, wikipedia, JsonFileCache(data_root.cache)
    except OSError as error:
        LOGGER.debug("Cache disabled: %s", error)
        return wikidata, wikipedia, None


__all__ = ["build_clients", "resolve_cli_data_root"]
