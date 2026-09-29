"""Shared acceptance-test configuration.

Every acceptance scenario is an offline, executable specification: it runs
against local fixtures, in-memory clients and a fake Hugging Face Hub. The
autouse ``block_network`` fixture turns any attempt to open an internet socket
into an immediate failure, so a scenario that silently reaches live SPARQL,
MediaWiki or the Hub fails loudly instead of passing by accident (#115).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any

import pytest


class NetworkAccessError(RuntimeError):
    """Raised when an acceptance scenario tries to use the network."""


def _is_local_address(address: Any) -> bool:
    if isinstance(address, str | bytes):
        # AF_UNIX socket paths never leave the machine.
        return True
    if isinstance(address, tuple) and address:
        return _is_loopback_host(address[0])
    return False


def _is_loopback_host(host: Any) -> bool:
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False
    normalized = host.rstrip(".").lower()
    if normalized in {"", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _refuse(*args: object, **kwargs: object) -> None:
    del kwargs
    raise NetworkAccessError(f"acceptance scenarios must stay offline; attempted {args!r}")


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any non-loopback connection, DNS lookup or connection helper."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not _is_local_address(address):
            _refuse(address)
        original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_local_address(address):
            _refuse(address)
        return original_connect_ex(self, address)

    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex
    original_gethostbyaddr = socket.gethostbyaddr
    original_getnameinfo = socket.getnameinfo
    original_getfqdn = socket.getfqdn
    original_create_connection = socket.create_connection
    original_sendto = socket.socket.sendto

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_loopback_host(host):
            _refuse(host)
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host: Any) -> Any:
        if not _is_loopback_host(host):
            _refuse(host)
        return original_gethostbyname(host)

    def guarded_gethostbyname_ex(host: Any) -> Any:
        if not _is_loopback_host(host):
            _refuse(host)
        return original_gethostbyname_ex(host)

    def guarded_gethostbyaddr(host: Any) -> Any:
        if not _is_loopback_host(host):
            _refuse(host)
        return original_gethostbyaddr(host)

    def guarded_getnameinfo(address: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_local_address(address):
            _refuse(address)
        return original_getnameinfo(address, *args, **kwargs)

    def guarded_getfqdn(name: Any = "") -> str:
        host = name or socket.gethostname()
        if not _is_loopback_host(host):
            _refuse(host)
        return original_getfqdn(name)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if not _is_local_address(address):
            _refuse(address)
        return original_create_connection(address, *args, **kwargs)

    def guarded_sendto(self: socket.socket, *args: Any, **kwargs: Any) -> int:
        address = kwargs.get("address")
        if address is None and args:
            address = args[-1]
        if address is not None and not _is_local_address(address):
            _refuse(address)
        return original_sendto(self, *args, **kwargs)

    original_sendmsg = getattr(socket.socket, "sendmsg", None)
    if original_sendmsg is not None:

        def guarded_sendmsg(self: socket.socket, *args: Any, **kwargs: Any) -> int:
            address = kwargs.get("address")
            if address is None and len(args) >= 4:
                address = args[3]
            if address is not None and not _is_local_address(address):
                _refuse(address)
            return original_sendmsg(self, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "sendmsg", guarded_sendmsg)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket, "gethostbyname_ex", guarded_gethostbyname_ex)
    monkeypatch.setattr(socket, "gethostbyaddr", guarded_gethostbyaddr)
    monkeypatch.setattr(socket, "getnameinfo", guarded_getnameinfo)
    monkeypatch.setattr(socket, "getfqdn", guarded_getfqdn)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket.socket, "sendto", guarded_sendto)
    yield


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test collected from this directory as ``acceptance``."""

    for item in items:
        if "tests/acceptance/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.acceptance)
