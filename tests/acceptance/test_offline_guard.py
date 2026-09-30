"""Prove the acceptance socket guard really blocks the network (#115)."""

from __future__ import annotations

import socket

import pytest
from conftest import NetworkAccessError


def test_outbound_connection_is_refused() -> None:
    with pytest.raises(NetworkAccessError):
        socket.create_connection(("huggingface.co", 443), timeout=1)


def test_loopback_connection_helper_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSocket:
        def __init__(self, *_args: object) -> None:
            self.connected_to: object | None = None

        def connect(self, address: object) -> None:
            self.connected_to = address

        def close(self) -> None:
            pass

    monkeypatch.setattr(socket, "socket", FakeSocket)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443))
        ],
    )

    connection = socket.create_connection(("127.0.0.1", 443))

    assert isinstance(connection, FakeSocket)
    assert connection.connected_to == ("127.0.0.1", 443)


def test_raw_socket_connect_and_dns_are_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkAccessError):
            sock.connect(("93.184.216.34", 80))
        with pytest.raises(NetworkAccessError):
            sock.connect_ex(("93.184.216.34", 80))
    with pytest.raises(NetworkAccessError):
        socket.getaddrinfo("query.wikidata.org", 443)


@pytest.mark.parametrize(
    ("resolver", "args"),
    [
        ("gethostbyname", ("93.184.216.34",)),
        ("gethostbyname_ex", ("93.184.216.34",)),
        ("gethostbyaddr", ("invalid-ip",)),
        ("getnameinfo", (("93.184.216.34", 443), socket.NI_NUMERICHOST)),
        ("getfqdn", ("invalid-ip",)),
    ],
)
def test_remote_name_resolution_helpers_are_refused(
    resolver: str, args: tuple[object, ...]
) -> None:
    with pytest.raises(NetworkAccessError):
        getattr(socket, resolver)(*args)


def test_udp_sendto_to_remote_address_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(NetworkAccessError):
            sock.sendto(b"blocked", ("203.0.113.1", 53))


@pytest.mark.skipif(not hasattr(socket.socket, "sendmsg"), reason="socket.sendmsg is unavailable")
def test_udp_sendmsg_to_remote_address_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(NetworkAccessError):
            sock.sendmsg([b"blocked"], [], 0, ("203.0.113.1", 53))


@pytest.mark.local_network
def test_loopback_lookup_is_still_allowed() -> None:
    assert socket.getaddrinfo("127.0.0.1", 0)
    assert socket.gethostbyname("127.0.0.1") == "127.0.0.1"
