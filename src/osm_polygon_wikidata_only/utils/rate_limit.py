"""Small polite rate-limiting helpers for Wikimedia HTTP calls."""

from __future__ import annotations

import logging
import threading
import time
import urllib.error
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

LOGGER = logging.getLogger(__name__)

_LOCK = threading.Lock()
_NEXT_ALLOWED: dict[str, float] = {}


def wait_for_host(host: str, *, min_interval_s: float) -> None:
    """Ensure at least min_interval_s between requests to the same host."""
    if min_interval_s <= 0:
        return

    with _LOCK:
        now = time.monotonic()
        next_allowed = _NEXT_ALLOWED.get(host, now)
        sleep_s = max(0.0, next_allowed - now)
        _NEXT_ALLOWED[host] = max(now, next_allowed) + min_interval_s

    if sleep_s > 0:
        time.sleep(sleep_s)


def retry_after_seconds(
    error: urllib.error.HTTPError,
    *,
    default_s: float = 60.0,
    max_s: float = 600.0,
) -> float:
    """Parse HTTP Retry-After header, falling back to default_s."""
    value = error.headers.get("Retry-After") if error.headers is not None else None
    if not value:
        return default_s

    try:
        return min(max_s, max(0.0, float(value)))
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return min(max_s, max(0.0, (dt - datetime.now(UTC)).total_seconds()))
    except (TypeError, ValueError, OverflowError):
        return default_s


def sleep_after_429(error: urllib.error.HTTPError, *, default_s: float = 60.0) -> None:
    """Sleep after a 429 before letting the retry wrapper retry."""
    sleep_s = retry_after_seconds(error, default_s=default_s)
    LOGGER.warning("Rate limited by Wikimedia; sleeping %.1fs before retrying", sleep_s)
    time.sleep(sleep_s)
