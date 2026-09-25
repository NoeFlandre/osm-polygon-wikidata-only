"""Characterization tests for :func:`retry_after_seconds`.

These tests freeze the behavior of the Retry-After header parser used
by the Wikimedia mediawiki, wikidata, and wikipedia clients. They are
written before the function is migrated from
``utils.rate_limit`` to ``utils.http_retry`` so the public contract
is locked before any structural change.
"""

from __future__ import annotations

import urllib.error
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from tests.helpers import http_error


def _limited(retry_after: str | None) -> urllib.error.HTTPError:
    return http_error(429, retry_after=retry_after, msg="limited")


def test_numeric_header_used_directly() -> None:
    """A numeric Retry-After value (in seconds) is parsed as ``float``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _limited("12.5")
    assert retry_after_seconds(error) == 12.5


def test_numeric_header_is_clamped() -> None:
    """Values above ``max_s`` are clamped to ``max_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _limited("9999")
    assert retry_after_seconds(error, max_s=600.0) == 600.0


def test_numeric_header_does_not_go_negative() -> None:
    """Negative numeric values clamp at 0.0."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _limited("-7")
    assert retry_after_seconds(error) == 0.0


def test_http_date_header_used_directly(monkeypatch: Any) -> None:
    """An HTTP-date in the future returns the seconds until that instant."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    fixed_now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.utils.http_retry.datetime",
        _FrozenDatetime(fixed_now),
    )
    future = fixed_now + timedelta(seconds=42)
    error = _limited(future.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    assert retry_after_seconds(error) == 42.0


def test_missing_header_returns_default() -> None:
    """A missing Retry-After header falls back to ``default_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _limited(None)
    assert retry_after_seconds(error) == 60.0


def test_malformed_header_returns_default() -> None:
    """A header that is neither a number nor a date falls back to ``default_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _limited("not-a-date")
    assert retry_after_seconds(error, default_s=17.0) == 17.0


def test_empty_header_returns_default() -> None:
    """An empty Retry-After falls back to ``default_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _limited("")
    assert retry_after_seconds(error, default_s=11.5) == 11.5


def test_past_date_clamps_at_zero(monkeypatch: Any) -> None:
    """An HTTP-date in the past returns 0.0 (no negative sleep)."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    fixed_now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.utils.http_retry.datetime",
        _FrozenDatetime(fixed_now),
    )
    past = fixed_now - timedelta(seconds=300)
    error = _limited(past.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    assert retry_after_seconds(error) == 0.0


def test_naive_http_date_is_assumed_utc(monkeypatch: Any) -> None:
    """An HTTP-date without an explicit timezone is assumed UTC."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    fixed_now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.utils.http_retry.datetime",
        _FrozenDatetime(fixed_now),
    )
    future = fixed_now + timedelta(seconds=120)
    error = _limited(future.strftime("%a, %d %b %Y %H:%M:%S"))
    assert retry_after_seconds(error) == 120.0


class _FrozenDatetime:
    """A drop-in ``datetime`` module replacement that pins ``now(UTC)``.

    The Retry-After parser uses ``datetime.now(UTC)`` to compute HTTP-date
    deltas, so freezing ``now`` keeps every test deterministic.
    """

    def __init__(self, fixed: datetime) -> None:
        self._fixed = fixed

    def now(self, tz: timezone | None = None) -> datetime:
        if tz is None:
            return self._fixed.replace(tzinfo=None)
        return self._fixed.astimezone(tz)
