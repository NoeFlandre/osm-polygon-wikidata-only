"""HTTP retry helpers shared by the Wikimedia clients.

Responsibility:
    Parse HTTP ``Retry-After`` headers (numeric seconds or HTTP-date)
    into a non-negative delay, clamped to a caller-defined ceiling.

Invariants:
    * The parser never raises; malformed or absent headers return
      ``default_s``.
    * Results are always in ``[0.0, max_s]``; HTTP-date headers in the
      past produce ``0.0`` (no negative sleep).
    * Naive HTTP-dates (no timezone) are interpreted as UTC.
"""

from __future__ import annotations

import urllib.error
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

__all__ = ["retry_after_seconds"]


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
