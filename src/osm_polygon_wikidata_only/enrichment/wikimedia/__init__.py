"""Internal Wikimedia transport package.

Re-exports the shared :func:`read_wikimedia_json` helper alongside the
public names from :mod:`enrichment.wikimedia_auth` so call sites can
use one import path for the whole Wikimedia transport boundary. The
``wikimedia_auth.py`` definition site is preserved; only its public
symbols are re-exported here.

Internal transport exceptions and the throttle callback alias are
implementation details of the helper and are *not* re-exported here.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WIKIMEDIA_BOT_PASSWORD,
    WIKIMEDIA_BOT_USERNAME,
    WIKIMEDIA_MAX_IN_FLIGHT,
    WIKIMEDIA_REQUESTS_PER_MINUTE,
    WikimediaAuthenticationError,
    WikimediaAuthSnapshot,
    WikimediaConfigurationError,
    WikimediaCredentials,
    WikimediaHttpSession,
    WikimediaSession,
    load_wikimedia_credentials,
)

from .transport import read_wikimedia_json

__all__ = [
    "WIKIMEDIA_BOT_PASSWORD",
    "WIKIMEDIA_BOT_USERNAME",
    "WIKIMEDIA_MAX_IN_FLIGHT",
    "WIKIMEDIA_REQUESTS_PER_MINUTE",
    "WikimediaAuthSnapshot",
    "WikimediaAuthenticationError",
    "WikimediaConfigurationError",
    "WikimediaCredentials",
    "WikimediaHttpSession",
    "WikimediaSession",
    "load_wikimedia_credentials",
    "read_wikimedia_json",
]
