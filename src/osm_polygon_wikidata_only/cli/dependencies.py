"""Typed construction of runtime paths and enrichment clients."""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot, resolve_data_root
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    CachedWikidataClient,
    HttpWikidataClient,
    WikidataClient,
)
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WIKIMEDIA_MAX_IN_FLIGHT,
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
from osm_polygon_wikidata_only.utils.request_scheduler import (
    SYSTEMIC_ACTIVE_HOST_WINDOW_S,
    SYSTEMIC_HOST_FRACTION,
    SYSTEMIC_MINIMUM_HOSTS,
    AdaptiveRequestScheduler,
)

LOGGER = logging.getLogger(__name__)

# Authenticated bot sessions get a much higher rate ceiling (1200 rpm
# by default) than anonymous traffic (180 rpm). To actually reach that
# ceiling, the global concurrency cap and per-host pacing have to be
# loosened for authenticated runs, otherwise the scheduler is the
# bottleneck even when the API would happily accept more traffic.
#
# Concurrency math: 1200 rpm = 20 rps. At ~0.3s average API latency the
# required in-flight count is ~6 (20 x 0.3); 8 leaves headroom while
# staying well under the 16 hard cap enforced by the scheduler. This is
# a *client-side* choice, not a guaranteed server allowance, and is
# subordinate to the global rate ceiling and per-host cooldowns.
ANON_MAX_IN_FLIGHT = 3
AUTH_MAX_IN_FLIGHT_DEFAULT = 8
MAX_IN_FLIGHT_HARD_LIMIT = 16
AUTH_MIN_HOST_INTERVAL_S = 0.05
AUTH_MIN_REQUESTS_PER_MINUTE = 200


@dataclass(frozen=True, slots=True)
class WikimediaRuntime:
    """One process-wide request budget shared by every Wikimedia client."""

    settings: Settings
    scheduler: AdaptiveRequestScheduler
    session: WikimediaSession
    wikidata: WikidataClient
    wikipedia: WikipediaClient
    cache: JsonFileCache | None


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
    runtime = build_wikimedia_runtime(settings, data_root=data_root, environ=environ)
    return runtime.wikidata, runtime.wikipedia, runtime.cache


def build_wikimedia_runtime(
    settings: Settings,
    *,
    data_root: DataRoot,
    environ: Mapping[str, str] | None = None,
) -> WikimediaRuntime:
    """Build the single Wikimedia transport and all core clients."""
    source = os.environ if environ is None else environ
    credentials = load_wikimedia_credentials(source)
    authenticated = credentials is not None
    ceiling = _request_rate_ceiling(source, authenticated=authenticated)
    max_in_flight = _max_in_flight(source, authenticated=authenticated)
    if not authenticated:
        ceiling = min(ceiling, settings.wikimedia_requests_per_minute)
    effective = _effective_settings(
        settings,
        authenticated=authenticated,
        ceiling=ceiling,
        max_in_flight=max_in_flight,
    )
    minimum_rate = min(
        AUTH_MIN_REQUESTS_PER_MINUTE if authenticated else 60.0,
        effective.wikimedia_requests_per_minute,
    )
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=effective.wikimedia_max_in_flight,
        requests_per_minute=effective.wikimedia_requests_per_minute,
        max_requests_per_minute=ceiling,
        minimum_requests_per_minute=minimum_rate,
        active_host_window_s=SYSTEMIC_ACTIVE_HOST_WINDOW_S,
        minimum_systemic_hosts=SYSTEMIC_MINIMUM_HOSTS,
        systemic_host_fraction=SYSTEMIC_HOST_FRACTION,
    )
    session = WikimediaSession(
        scheduler=scheduler,
        timeout_s=effective.request_timeout_s,
        user_agent=effective.user_agent,
        credentials=credentials,
    )
    if credentials is not None:
        LOGGER.info(
            "Wikimedia API mode: credentials configured for %s; "
            "verification occurs per host; "
            "rate ceiling=%.0f rpm; "
            "in-flight=%d; "
            "authenticated host interval=%.2fs; "
            "anonymous intervals: Wikipedia=%.2fs, Wikidata=%.2fs, augmentation=%.2fs. "
            "The ceiling is a client-side limit, not a guaranteed server allowance.",
            credentials.username,
            ceiling,
            effective.wikimedia_max_in_flight,
            effective.wikimedia_authenticated_min_interval_s,
            effective.wikipedia_min_interval_s,
            effective.wikidata_min_interval_s,
            effective.augmentation_min_interval_s,
        )
    else:
        LOGGER.info(
            "Wikimedia API mode: anonymous (rate ceiling: %.0f requests/minute, "
            "in-flight=%d, host interval: %.2fs)",
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
        return WikimediaRuntime(effective, scheduler, session, wikidata, wikipedia, None)
    try:
        wikidata = CachedWikidataClient(
            wikidata,
            JsonFileCache(data_root.cache_wikidata),
        )
        wikipedia = CachedWikipediaClient(
            wikipedia,
            JsonFileCache(data_root.cache_wikipedia),
        )
        return WikimediaRuntime(
            effective,
            scheduler,
            session,
            wikidata,
            wikipedia,
            JsonFileCache(data_root.cache),
        )
    except OSError as error:
        LOGGER.debug("Cache disabled: %s", error)
        return WikimediaRuntime(effective, scheduler, session, wikidata, wikipedia, None)


def _effective_settings(
    settings: Settings, *, authenticated: bool, ceiling: float, max_in_flight: int
) -> Settings:
    """Pick per-call settings based on whether the user is authenticated.

    Anonymous traffic keeps the polite conservative defaults. Bot
    password sessions are explicitly allowed higher concurrency and
    tighter host pacing so the configured rpm ceiling is reachable.
    """
    if not authenticated:
        return settings
    return replace(
        settings,
        wikimedia_max_in_flight=max_in_flight,
        wikimedia_requests_per_minute=ceiling,
    )


def _max_in_flight(environ: Mapping[str, str], *, authenticated: bool) -> int:
    """Resolve the process-wide concurrency bound from configuration.

    Defaults to a conservative authenticated value (8) that can reach the
    1200 rpm ceiling at typical latency, or 3 for anonymous traffic. The
    hard upper bound (16) matches the scheduler's own validation.
    """
    raw_value = environ.get(WIKIMEDIA_MAX_IN_FLIGHT)
    default = AUTH_MAX_IN_FLIGHT_DEFAULT if authenticated else ANON_MAX_IN_FLIGHT
    if raw_value is None:
        return default
    try:
        value = int(raw_value.strip())
    except ValueError as error:
        raise WikimediaConfigurationError(
            f"{WIKIMEDIA_MAX_IN_FLIGHT} must be an integer between 1 and {MAX_IN_FLIGHT_HARD_LIMIT}"
        ) from error
    if not 1 <= value <= MAX_IN_FLIGHT_HARD_LIMIT:
        raise WikimediaConfigurationError(
            f"{WIKIMEDIA_MAX_IN_FLIGHT} must be between 1 and {MAX_IN_FLIGHT_HARD_LIMIT}"
        )
    return value


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


__all__ = [
    "WikimediaRuntime",
    "build_clients",
    "build_wikimedia_runtime",
    "resolve_cli_data_root",
]
