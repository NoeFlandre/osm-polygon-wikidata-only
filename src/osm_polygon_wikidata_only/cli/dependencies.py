"""Typed construction of runtime paths and enrichment clients."""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import replace
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

# Authenticated bot sessions get a much higher rate ceiling (1200 rpm
# by default) than anonymous traffic (180 rpm). To actually reach that
# ceiling, the global concurrency cap and per-host pacing have to be
# loosened for authenticated runs, otherwise the scheduler is the
# bottleneck even when the API would happily accept more traffic.
AUTH_MAX_IN_FLIGHT = 8
AUTH_MIN_HOST_INTERVAL_S = 0.05
AUTH_MIN_REQUESTS_PER_MINUTE = 200


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
    authenticated = credentials is not None
    ceiling = _request_rate_ceiling(source, authenticated=authenticated)
    if not authenticated:
        ceiling = min(ceiling, settings.wikimedia_requests_per_minute)
    effective = _effective_settings(settings, authenticated=authenticated, ceiling=ceiling)
    minimum_rate = min(
        AUTH_MIN_REQUESTS_PER_MINUTE if authenticated else 60.0,
        effective.wikimedia_requests_per_minute,
    )
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=effective.wikimedia_max_in_flight,
        requests_per_minute=effective.wikimedia_requests_per_minute,
        max_requests_per_minute=ceiling,
        minimum_requests_per_minute=minimum_rate,
    )
    session = WikimediaSession(
        scheduler=scheduler,
        timeout_s=effective.request_timeout_s,
        user_agent=effective.user_agent,
        credentials=credentials,
    )
    LOGGER.info(
        "Wikimedia API mode: %s (rate ceiling: %.0f requests/minute, in-flight=%d, "
        "host interval: %.2fs)",
        f"authenticated as {credentials.username}" if credentials is not None else "anonymous",
        ceiling,
        effective.wikimedia_max_in_flight,
        effective.wikipedia_min_interval_s,
    )
    wikidata: WikidataClient = HttpWikidataClient(
        effective,
        scheduler=scheduler,
        session=session,
    )
    wikipedia: WikipediaClient = HttpWikipediaClient(
        effective,
        scheduler=scheduler,
        session=session,
    )
    if not effective.cache_enabled:
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


def _effective_settings(settings: Settings, *, authenticated: bool, ceiling: float) -> Settings:
    """Pick per-call settings based on whether the user is authenticated.

    Anonymous traffic keeps the polite conservative defaults. Bot
    password sessions are explicitly allowed higher concurrency and
    tighter host pacing so the 1200 rpm ceiling is reachable.
    """
    if not authenticated:
        return settings
    return replace(
        settings,
        wikimedia_max_in_flight=max(settings.wikimedia_max_in_flight, AUTH_MAX_IN_FLIGHT),
        wikipedia_min_interval_s=min(settings.wikipedia_min_interval_s, AUTH_MIN_HOST_INTERVAL_S),
        wikidata_min_interval_s=min(settings.wikidata_min_interval_s, AUTH_MIN_HOST_INTERVAL_S),
        wikimedia_requests_per_minute=ceiling,
    )


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
