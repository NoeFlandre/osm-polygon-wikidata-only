"""Typed construction of runtime paths and enrichment clients."""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot, resolve_data_root
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    CachedWikidataClient,
    HttpWikidataClient,
    WikidataClient,
)
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WIKIMEDIA_REQUESTS_PER_MINUTE,
    WikimediaConfigurationError,
    WikimediaSession,
    load_wikimedia_credentials,
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
    settings: Settings,
    *,
    data_root: DataRoot,
    environ: Mapping[str, str] | None = None,
) -> tuple[WikidataClient, WikipediaClient, JsonFileCache | None]:
    """Build clients that share one global Wikimedia scheduler."""
    source = os.environ if environ is None else environ
    credentials = load_wikimedia_credentials(source)
    ceiling = _request_rate_ceiling(source, authenticated=credentials is not None)
    if credentials is None:
        ceiling = min(ceiling, settings.wikimedia_requests_per_minute)
    initial_rate = min(settings.wikimedia_requests_per_minute, ceiling)
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=settings.wikimedia_max_in_flight,
        requests_per_minute=initial_rate,
        max_requests_per_minute=ceiling,
        minimum_requests_per_minute=min(60, initial_rate),
    )
    session = WikimediaSession(
        scheduler=scheduler,
        timeout_s=settings.request_timeout_s,
        user_agent=settings.user_agent,
        credentials=credentials,
    )
    LOGGER.info(
        "Wikimedia API mode: %s (rate ceiling: %.0f requests/minute)",
        f"authenticated as {credentials.username}" if credentials is not None else "anonymous",
        ceiling,
    )
    wikidata: WikidataClient = HttpWikidataClient(
        settings,
        scheduler=scheduler,
        session=session,
    )
    wikipedia: WikipediaClient = HttpWikipediaClient(
        settings,
        scheduler=scheduler,
        session=session,
    )
    if not settings.cache_enabled:
        return wikidata, wikipedia, None
    try:
        wikidata = CachedWikidataClient(
            wikidata,
            JsonFileCache(data_root.cache_wikidata),
        )
        wikipedia = CachedWikipediaClient(
            wikipedia,
            JsonFileCache(data_root.cache_wikipedia),
        )
        return wikidata, wikipedia, JsonFileCache(data_root.cache)
    except OSError as error:
        LOGGER.debug("Cache disabled: %s", error)
        return wikidata, wikipedia, None


def _request_rate_ceiling(environ: Mapping[str, str], *, authenticated: bool) -> float:
    raw_value = environ.get(WIKIMEDIA_REQUESTS_PER_MINUTE)
    if raw_value is None:
        return 1_200 if authenticated else 180
    try:
        value = float(raw_value.strip())
    except ValueError as error:
        raise WikimediaConfigurationError(
            f"{WIKIMEDIA_REQUESTS_PER_MINUTE} must be a positive number"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise WikimediaConfigurationError(
            f"{WIKIMEDIA_REQUESTS_PER_MINUTE} must be a positive number"
        )
    return value if authenticated else 180


__all__ = ["build_clients", "resolve_cli_data_root"]
