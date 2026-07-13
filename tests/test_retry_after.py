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
from email.message import Message
from typing import Any


def _http_error(
    retry_after: str | None,
    *,
    error_code: int = 429,
) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.test", error_code, "limited", headers, None)


def test_numeric_header_used_directly() -> None:
    """A numeric Retry-After value (in seconds) is parsed as ``float``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _http_error("12.5")
    assert retry_after_seconds(error) == 12.5


def test_numeric_header_is_clamped() -> None:
    """Values above ``max_s`` are clamped to ``max_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _http_error("9999")
    assert retry_after_seconds(error, max_s=600.0) == 600.0


def test_numeric_header_does_not_go_negative() -> None:
    """Negative numeric values clamp at 0.0."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _http_error("-7")
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
    error = _http_error(future.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    assert retry_after_seconds(error) == 42.0


def test_missing_header_returns_default() -> None:
    """A missing Retry-After header falls back to ``default_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _http_error(None)
    assert retry_after_seconds(error) == 60.0


def test_malformed_header_returns_default() -> None:
    """A header that is neither a number nor a date falls back to ``default_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _http_error("not-a-date")
    assert retry_after_seconds(error, default_s=17.0) == 17.0


def test_empty_header_returns_default() -> None:
    """An empty Retry-After falls back to ``default_s``."""
    from osm_polygon_wikidata_only.utils.http_retry import retry_after_seconds

    error = _http_error("")
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
    error = _http_error(past.strftime("%a, %d %b %Y %H:%M:%S GMT"))
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
    error = _http_error(future.strftime("%a, %d %b %Y %H:%M:%S"))
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
