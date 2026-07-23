from __future__ import annotations

import gzip
import io
import urllib.error
import urllib.request

import httpx
import pytest

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import WikimediaSession
from osm_polygon_wikidata_only.enrichment.wikimedia_http import PooledWikimediaOpener
from osm_polygon_wikidata_only.enrichment.wikipedia_client import HttpWikipediaClient
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler
from osm_polygon_wikidata_only.utils.retry import is_transient_network_error


def test_pooled_opener_preserves_method_url_headers_and_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b'{"ok":true}')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    opener = PooledWikimediaOpener(client=client)
    request = urllib.request.Request(
        "https://en.wikipedia.org/w/api.php?action=query",
        data=b"payload=value",
        headers={"User-Agent": "test-agent", "Accept": "application/json"},
        method="POST",
    )

    with opener.open(request, timeout=12.5) as response:
        assert response.read() == b'{"ok":true}'

    assert len(seen) == 1
    sent = seen[0]
    assert sent.method == "POST"
    assert str(sent.url) == request.full_url
    assert sent.headers["user-agent"] == "test-agent"
    assert sent.headers["accept"] == "application/json"
    assert sent.content == b"payload=value"


def test_pooled_opener_keeps_cookies_between_requests() -> None:
    cookies_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cookies_seen.append(request.headers.get("cookie", ""))
        return httpx.Response(200, headers={"set-cookie": "session=verified; Path=/"})

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))
    request = urllib.request.Request("https://en.wikipedia.org/w/api.php")

    with opener.open(request, timeout=1.0):
        pass
    with opener.open(request, timeout=1.0):
        pass

    assert cookies_seen == ["", "session=verified"]


def test_pooled_opener_returns_decoded_body_without_stale_content_encoding() -> None:
    compressed = gzip.compress(b'{"query":{"ok":true}}')

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=compressed,
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
        )

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with opener.open(
        urllib.request.Request("https://www.wikidata.org/w/api.php"), timeout=1.0
    ) as response:
        assert response.read() == b'{"query":{"ok":true}}'
        assert response.headers.get("Content-Encoding", "") == ""
        assert response.headers.get("Content-Type") == "application/json"


def test_pooled_opener_translates_http_status_to_urllib_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            content=b"throttled",
            headers={"Retry-After": "17"},
        )

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(urllib.error.HTTPError) as caught:
        opener.open(urllib.request.Request("https://en.wikipedia.org/w/api.php"), timeout=1.0)

    error = caught.value
    assert error.code == 429
    assert error.reason == "Too Many Requests"
    assert error.headers.get("Retry-After") == "17"
    assert error.read() == b"throttled"


def test_pooled_opener_translates_transport_errors_to_urllib_url_error() -> None:
    failure = httpx.UnsupportedProtocol("unsupported URL scheme")

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(urllib.error.URLError) as caught:
        opener.open(urllib.request.Request("https://en.wikipedia.org/w/api.php"), timeout=1.0)

    assert caught.value.reason is failure


def test_pooled_opener_normalizes_read_timeout_as_retryable_timeout() -> None:
    """HTTPX timeouts must retain urllib's retryable timeout semantics."""
    failure = httpx.ReadTimeout("read operation timed out")

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(urllib.error.URLError) as caught:
        opener.open(urllib.request.Request("https://en.wikipedia.org/w/api.php"), timeout=1.0)

    assert isinstance(caught.value.reason, TimeoutError)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection failed"),
        httpx.ReadError("read failed"),
        httpx.WriteError("write failed"),
        httpx.CloseError("close failed"),
        httpx.RemoteProtocolError("server disconnected without a response"),
        httpx.ProxyError("proxy unavailable"),
    ],
)
def test_pooled_opener_normalizes_transient_httpx_transport_errors(
    failure: httpx.TransportError,
) -> None:
    """Remote transport failures must reach the shared retry classifier."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(urllib.error.URLError) as caught:
        opener.open(urllib.request.Request("https://nap.wikipedia.org/w/api.php"), timeout=1.0)

    assert isinstance(caught.value.reason, ConnectionError)
    assert caught.value.__cause__ is failure
    assert is_transient_network_error(caught.value)


def test_pooled_opener_keeps_local_protocol_errors_non_retryable() -> None:
    """Client-side protocol mistakes must not enter an infinite retry loop."""
    failure = httpx.LocalProtocolError("invalid request")

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(urllib.error.URLError) as caught:
        opener.open(urllib.request.Request("https://en.wikipedia.org/w/api.php"), timeout=1.0)

    assert caught.value.reason is failure
    assert not is_transient_network_error(caught.value)


def test_wikipedia_client_retries_remote_disconnect_without_returning_http_error() -> None:
    """The production client must survive the recovery crash from the incident."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.",
                request=request,
            )
        return httpx.Response(
            200,
            request=request,
            content=b'{"query":{"pages":{"-1":{"title":"Castelsardo","missing":""}}}}',
        )

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))
    scheduler = AdaptiveRequestScheduler(max_in_flight=1, requests_per_minute=1200)
    session = WikimediaSession(
        scheduler=scheduler,
        timeout_s=1.0,
        user_agent="test-agent",
        opener_factory=lambda: opener,
    )
    client = HttpWikipediaClient(
        Settings(
            request_max_retries=2,
            request_base_delay_s=0.0,
            wikipedia_min_interval_s=0.0,
            wikimedia_authenticated_min_interval_s=0.0,
        ),
        scheduler=scheduler,
        session=session,
    )

    result = client.fetch_article("nap", "napwiki", "Castelsardo")

    assert attempts == 2
    assert result.status == "article_not_found"
    assert result.error == "page missing"


def test_pooled_opener_close_is_idempotent() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200)))
    opener = PooledWikimediaOpener(client=client)

    opener.close()
    opener.close()

    with pytest.raises(RuntimeError, match="closed"):
        opener.open(urllib.request.Request("https://en.wikipedia.org/w/api.php"), timeout=1.0)


def test_http_error_body_is_independent_of_closed_httpx_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"temporarily unavailable")

    opener = PooledWikimediaOpener(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(urllib.error.HTTPError) as caught:
        opener.open(urllib.request.Request("https://en.wikipedia.org/w/api.php"), timeout=1.0)

    assert isinstance(caught.value.fp, io.BytesIO)
    assert caught.value.read() == b"temporarily unavailable"


def test_default_session_uses_isolated_cookie_openers_on_one_bounded_pool() -> None:
    scheduler = AdaptiveRequestScheduler(max_in_flight=8, requests_per_minute=1200)
    session = WikimediaSession(
        scheduler=scheduler,
        timeout_s=10.0,
        user_agent="test-agent",
    )

    first = session._host_session("en.wikipedia.org")
    second = session._host_session("fr.wikipedia.org")

    assert first.opener is not second.opener
    assert isinstance(first.opener, PooledWikimediaOpener)
    assert isinstance(second.opener, PooledWikimediaOpener)
    assert first.opener.pool_identity is second.opener.pool_identity
    session.close()
