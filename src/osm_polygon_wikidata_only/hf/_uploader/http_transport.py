"""Bounded Hugging Face HTTP transport for reliable CLI operation.

Some networks advertise IPv6 connectivity while silently dropping outbound
IPv6 TCP connections. Hugging Face's default shared client currently has no
timeout, so such a route can block before authentication forever. Dataset Hub
traffic is therefore bound to IPv4 and uses finite inactivity timeouts.
Wikimedia traffic is unaffected.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import httpx

_CONNECT_TIMEOUT_S = 10.0
_READ_TIMEOUT_S = 120.0
_WRITE_TIMEOUT_S = 120.0
_POOL_TIMEOUT_S = 30.0

_lock = threading.Lock()
_configured = False


def build_hf_http_client(
    *,
    event_hook: Callable[[httpx.Request], Any] | None = None,
) -> httpx.Client:
    """Build the process-wide Hub client with bounded IPv4 transport."""
    if event_hook is None:
        from huggingface_hub.utils._http import hf_request_event_hook

        event_hook = hf_request_event_hook
    return httpx.Client(
        transport=httpx.HTTPTransport(
            local_address="0.0.0.0",
            retries=2,
        ),
        event_hooks={"request": [event_hook]},
        follow_redirects=True,
        timeout=httpx.Timeout(
            connect=_CONNECT_TIMEOUT_S,
            read=_READ_TIMEOUT_S,
            write=_WRITE_TIMEOUT_S,
            pool=_POOL_TIMEOUT_S,
        ),
    )


def configure_hf_http_transport(*, _set_factory: Any = None) -> None:
    """Install the bounded client factory once for this process."""
    global _configured
    with _lock:
        if _configured:
            return
        if _set_factory is None:
            from huggingface_hub import set_client_factory

            _set_factory = set_client_factory
        _set_factory(build_hf_http_client)
        _configured = True


__all__ = ["build_hf_http_client", "configure_hf_http_transport"]
