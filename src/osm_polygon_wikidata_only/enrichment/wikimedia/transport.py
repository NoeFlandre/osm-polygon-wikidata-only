"""Shared Wikimedia JSON transport helper.

Responsibility:
    Execute exactly one ``session.read`` against a built
    :class:`urllib.request.Request`, decompress ``gzip`` bodies,
    JSON-decode the response, validate that the top-level value is
    an object, and forward HTTP 429/503 throttling to an optional
    callback before re-raising.

Out of scope (intentionally retained by each caller):
    * Request construction and ``User-Agent`` handling.
    * Retry loops / retry budgets (``utils.retry.with_retries``).
    * Cache behaviour (``io.cache.JsonFileCache``).
    * :class:`FetchResult` conversion (Wikipedia client only).
    * Wikipedia Action API fallback / Wikidata missing-entity
      semantics.
    * Logging: the helper emits no logs; callers decide whether to
      warn on throttling.

The helper raises an internal ``_NonObjectJsonError`` marker when the
decoded JSON is not a top-level object, so each caller can wrap it
with the exact URL-bearing message format its previous inline code
emitted (:mod:`enrichment.wikipedia_client`,
:mod:`enrichment.wikidata_client`, :mod:`augmentation.mediawiki`).
The marker is an implementation detail of the helper and is **not**
re-exported from :mod:`enrichment.wikimedia`.
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaHttpSession,
)
from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

ThrottleCallback = Callable[[str, float], None]
"""Callback used to report HTTP throttling: ``(host, delay_seconds)``.

The shape mirrors :meth:`AdaptiveRequestScheduler.report_host_throttled`
so production callers wire the scheduler in directly. The status code
is intentionally *not* exposed; callers that need it for logging can
re-read ``error.code`` from the re-raised HTTPError.
"""

THROTTLE_STATUS_CODES: frozenset[int] = frozenset({429, 503})


class _NonObjectJsonError(ValueError):
    """Internal marker: decoded JSON body is not a top-level object.

    Carries ``value_type`` (the runtime name of the decoded type) so
    callers can rebuild their preferred error messages without parsing
    ``str(exc)``. Not part of the public surface.
    """

    def __init__(self, value_type: str) -> None:
        self.value_type = value_type
        super().__init__(f"Expected JSON object, got {value_type}")


def read_wikimedia_json(
    request: urllib.request.Request,
    session: WikimediaHttpSession,
    *,
    host: str,
    anonymous_interval_s: float,
    authenticated_interval_s: float,
    throttle_callback: ThrottleCallback | None,
    default_throttle_s: float = 60.0,
) -> dict[str, Any]:
    """Read, decompress, JSON-decode, and validate a Wikimedia JSON object.

    On HTTP 429/503 the optional ``throttle_callback`` is invoked
    exactly once with ``(host, delay)``, then the original
    :class:`urllib.error.HTTPError` is re-raised. Non-throttle HTTP
    errors propagate without invoking the callback.
    """
    try:
        raw, encoding = session.read(
            request,
            min_interval_anonymous_s=anonymous_interval_s,
            min_interval_authenticated_s=authenticated_interval_s,
        )
    except urllib.error.HTTPError as error:
        _maybe_report_throttle(
            error,
            host=host,
            callback=throttle_callback,
            default_throttle_s=default_throttle_s,
        )
        raise

    if encoding == "gzip":
        raw = gzip.decompress(raw)

    parsed: object = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise _NonObjectJsonError(type(parsed).__name__)
    return parsed


def _maybe_report_throttle(
    error: urllib.error.HTTPError,
    *,
    host: str,
    callback: ThrottleCallback | None,
    default_throttle_s: float,
) -> None:
    """Invoke ``callback`` exactly once when ``error`` is HTTP 429/503."""
    if error.code not in THROTTLE_STATUS_CODES:
        return
    delay = retry_after_seconds(error, default_s=default_throttle_s)
    if callback is not None:
        callback(host, delay)


__all__ = ["ThrottleCallback", "read_wikimedia_json"]
