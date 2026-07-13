"""Tests for the shared :mod:`enrichment.wikimedia.transport` helper."""

from __future__ import annotations

import gzip
import json
import logging
import urllib.error
from email.message import Message
from typing import Any

import pytest


class _StubSession:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.reads: list[tuple[Any, float, float]] = []
        self._responses = list(responses or [])

    def read(
        self,
        request: Any,
        *,
        min_interval_anonymous_s: float,
        min_interval_authenticated_s: float,
    ) -> tuple[bytes, str]:
        self.reads.append((request, min_interval_anonymous_s, min_interval_authenticated_s))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.test", code, "error", headers, None)


def _req(url: str) -> Any:
    import urllib.request

    return urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})


def _read(session: _StubSession, **kwargs: Any) -> Any:
    from osm_polygon_wikidata_only.enrichment.wikimedia.transport import (
        read_wikimedia_json,
    )

    return read_wikimedia_json(session=session, **kwargs)


def test_returns_plain_object() -> None:
    session = _StubSession([(b'{"ok": true, "value": 1}', "identity")])
    parsed = _read(
        session,
        request=_req("https://example.test/api"),
        host="example.test",
        anonymous_interval_s=1.0,
        authenticated_interval_s=0.5,
        throttle_callback=None,
    )
    assert parsed == {"ok": True, "value": 1}
    assert len(session.reads) == 1


def test_decompresses_gzip() -> None:
    payload = gzip.compress(b'{"pages": [{"id": 42}]}')
    session = _StubSession([(payload, "gzip")])
    parsed = _read(
        session,
        request=_req("https://example.test/api"),
        host="example.test",
        anonymous_interval_s=1.0,
        authenticated_interval_s=0.5,
        throttle_callback=None,
    )
    assert parsed == {"pages": [{"id": 42}]}


def test_rejects_malformed_body() -> None:
    session = _StubSession([(b"{not-json", "identity")])
    with pytest.raises(json.JSONDecodeError):
        _read(
            session,
            request=_req("https://example.test/api"),
            host="example.test",
            anonymous_interval_s=1.0,
            authenticated_interval_s=0.5,
            throttle_callback=None,
        )


def test_raises_private_non_object_marker_and_facade_does_not_export_it() -> None:
    import osm_polygon_wikidata_only.enrichment.wikimedia as facade
    from osm_polygon_wikidata_only.enrichment.wikimedia.transport import (
        _NonObjectJsonError,
    )

    assert "NonObjectJsonError" not in facade.__all__
    session = _StubSession([(b"[1, 2, 3]", "identity")])
    with pytest.raises(_NonObjectJsonError) as exc_info:
        _read(
            session,
            request=_req("https://example.test/api"),
            host="example.test",
            anonymous_interval_s=1.0,
            authenticated_interval_s=0.5,
            throttle_callback=None,
        )
    assert exc_info.value.value_type == "list"


def test_429_invokes_callback_with_host_and_delay_only() -> None:
    session = _StubSession([_http_error(429, retry_after="5")])
    calls: list[tuple[str, float]] = []
    with pytest.raises(urllib.error.HTTPError):
        _read(
            session,
            request=_req("https://example.test/api"),
            host="example.test",
            anonymous_interval_s=1.0,
            authenticated_interval_s=0.5,
            throttle_callback=lambda host, delay: calls.append((host, delay)),
            default_throttle_s=60.0,
        )
    assert calls == [("example.test", 5.0)]


def test_503_invokes_callback_with_host_and_delay_only() -> None:
    session = _StubSession([_http_error(503, retry_after="9.5")])
    calls: list[tuple[str, float]] = []
    with pytest.raises(urllib.error.HTTPError):
        _read(
            session,
            request=_req("https://example.test/api"),
            host="example.test",
            anonymous_interval_s=1.0,
            authenticated_interval_s=0.5,
            throttle_callback=lambda host, delay: calls.append((host, delay)),
            default_throttle_s=60.0,
        )
    assert calls == [("example.test", 9.5)]


def test_non_throttle_http_error_does_not_invoke_callback() -> None:
    session = _StubSession([_http_error(404)])
    calls: list[tuple[str, float]] = []
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _read(
            session,
            request=_req("https://example.test/api"),
            host="example.test",
            anonymous_interval_s=1.0,
            authenticated_interval_s=0.5,
            throttle_callback=lambda host, delay: calls.append((host, delay)),
        )
    assert exc_info.value.code == 404
    assert calls == []


def test_callback_absence_is_safe_on_429() -> None:
    session = _StubSession([_http_error(429)])
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _read(
            session,
            request=_req("https://example.test/api"),
            host="example.test",
            anonymous_interval_s=1.0,
            authenticated_interval_s=0.5,
            throttle_callback=None,
        )
    assert exc_info.value.code == 429


def test_helper_does_not_log_throttle(caplog: pytest.LogCaptureFixture) -> None:
    session = _StubSession([_http_error(429, retry_after="7")])
    with caplog.at_level(logging.DEBUG, logger="osm_polygon_wikidata_only"):
        with pytest.raises(urllib.error.HTTPError):
            _read(
                session,
                request=_req("https://example.test/api"),
                host="example.test",
                anonymous_interval_s=1.0,
                authenticated_interval_s=0.5,
                throttle_callback=lambda *_a: None,
            )
    assert caplog.records == []
