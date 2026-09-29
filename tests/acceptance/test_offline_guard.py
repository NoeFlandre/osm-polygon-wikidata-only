"""Prove the acceptance socket guard really blocks the network (#115)."""

from __future__ import annotations

import socket

import pytest
from conftest import NetworkAccessError  # ty: ignore[unresolved-import]


def test_outbound_connection_is_refused() -> None:
    with pytest.raises(NetworkAccessError):
        socket.create_connection(("huggingface.co", 443), timeout=1)


def test_raw_socket_connect_and_dns_are_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkAccessError):
            sock.connect(("93.184.216.34", 80))
        with pytest.raises(NetworkAccessError):
            sock.connect_ex(("93.184.216.34", 80))
    with pytest.raises(NetworkAccessError):
        socket.getaddrinfo("query.wikidata.org", 443)


def test_loopback_lookup_is_still_allowed() -> None:
    assert socket.getaddrinfo("127.0.0.1", 0)
