"""Bounded persistent HTTP transport for Wikimedia API requests."""

from __future__ import annotations

import io
import threading
import urllib.error
import urllib.request
from email.message import Message
from types import TracebackType
from typing import cast

import httpx


class _BufferedResponse:
    """Small urllib-compatible response backed by already-read bytes."""

    def __init__(self, body: bytes, headers: Message) -> None:
        self._body = body
        self.headers = headers

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _BufferedResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class _SharedTransportView(httpx.BaseTransport):
    """Delegate requests without letting a per-host client close the pool."""

    def __init__(self, transport: httpx.BaseTransport) -> None:
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._transport.handle_request(request)

    def close(self) -> None:
        return None


def _decoded_headers(headers: httpx.Headers) -> Message:
    """Convert headers after HTTPX decoded the response body.

    HTTPX transparently decompresses ``response.content``. Removing the
    corresponding content headers prevents callers from attempting a second
    decompression or trusting the compressed byte length.
    """
    converted = Message()
    for name, value in headers.multi_items():
        if name.lower() not in {"content-encoding", "content-length"}:
            converted.add_header(name, value)
    return converted


class PooledWikimediaOpener:
    """Thread-safe urllib-shaped opener using one bounded HTTPX pool."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        max_connections: int = 8,
        pool_identity: object | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )
        self._close_lock = threading.Lock()
        self._closed = False
        self._pool_identity = pool_identity if pool_identity is not None else self._client

    @property
    def pool_identity(self) -> object:
        """Opaque identity used to verify that host clients share one pool."""
        return self._pool_identity

    def open(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _BufferedResponse:
        """Execute a urllib request while preserving its observable contract."""
        if self._closed:
            raise RuntimeError("Wikimedia HTTP transport is closed")
        content = cast(bytes | None, request.data)
        try:
            response = self._client.request(
                request.get_method(),
                request.full_url,
                headers=dict(request.header_items()),
                content=content,
                timeout=timeout,
            )
        except httpx.TimeoutException as error:
            # Preserve urllib-compatible timeout semantics so the shared
            # retry classifier recognizes HTTPX read/connect/write/pool
            # timeouts as transient instead of aborting the pipeline.
            raise urllib.error.URLError(TimeoutError(str(error))) from error
        except httpx.TransportError as error:
            raise urllib.error.URLError(error) from error

        body = response.content
        headers = _decoded_headers(response.headers)
        if response.is_error:
            raise urllib.error.HTTPError(
                request.full_url,
                response.status_code,
                response.reason_phrase,
                headers,
                io.BytesIO(body),
            )
        return _BufferedResponse(body, headers)

    def close(self) -> None:
        """Release pooled sockets; safe to call more than once."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._client.close()


class PooledWikimediaTransport:
    """Own one bounded connection pool and isolated per-host cookie clients."""

    def __init__(self, *, max_connections: int) -> None:
        self._transport = httpx.HTTPTransport(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
                keepalive_expiry=30.0,
            )
        )
        self._openers: list[PooledWikimediaOpener] = []
        self._lock = threading.Lock()
        self._closed = False

    def opener(self) -> PooledWikimediaOpener:
        """Create a cookie-isolated client backed by the shared pool."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Wikimedia HTTP transport is closed")
            client = httpx.Client(
                transport=_SharedTransportView(self._transport),
                follow_redirects=True,
            )
            opener = PooledWikimediaOpener(
                client=client,
                pool_identity=self._transport,
            )
            self._openers.append(opener)
            return opener

    def close(self) -> None:
        """Close every host client and then its shared connection pool."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            openers = tuple(self._openers)
            self._openers.clear()
        for opener in openers:
            opener.close()
        self._transport.close()


__all__: list[str] = []
