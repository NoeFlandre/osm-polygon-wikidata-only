"""Offline contracts for the autouse socket guard."""

from __future__ import annotations

import socket
from typing import cast

import pytest

from tests._support import _socket_address_is_allowed


def test_autouse_guard_blocks_resolution_before_calling_resolver() -> None:
    with pytest.raises(socket.gaierror, match="outbound network access is disabled"):
        socket.getaddrinfo(cast(str, object()), 443)


@pytest.mark.local_network
def test_local_network_marker_never_enables_hostname_resolution() -> None:
    with pytest.raises(socket.gaierror, match="outbound network access is disabled"):
        socket.getaddrinfo("localhost", 443)


def test_autouse_guard_blocks_connect_before_opening_a_socket() -> None:
    with pytest.raises(OSError, match="outbound network access is disabled"):
        socket.socket.connect(cast(socket.socket, object()), ("203.0.113.1", 443))


def test_loopback_connections_also_require_the_local_network_marker() -> None:
    with pytest.raises(OSError, match="outbound network access is disabled"):
        socket.socket.connect(cast(socket.socket, object()), ("127.0.0.1", 443))


def test_autouse_guard_blocks_datagrams_before_opening_a_socket() -> None:
    with pytest.raises(OSError, match="outbound network access is disabled"):
        socket.socket.sendto(cast(socket.socket, object()), b"test", ("203.0.113.1", 53))


@pytest.mark.local_network
def test_local_network_marker_allows_loopback_addresses() -> None:
    assert _socket_address_is_allowed(("127.0.0.1", 443), allow_local_network=True)
    assert _socket_address_is_allowed(("::1", 443, 0, 0), allow_local_network=True)
    assert not _socket_address_is_allowed(("203.0.113.1", 443), allow_local_network=True)
    assert not _socket_address_is_allowed(("127.0.0.1", 443), allow_local_network=False)
    assert not _socket_address_is_allowed(("localhost", 443), allow_local_network=True)


@pytest.mark.local_network
def test_local_network_marker_allows_numeric_loopback_resolution() -> None:
    assert socket.getaddrinfo("127.0.0.1", 443, type=socket.SOCK_STREAM)
