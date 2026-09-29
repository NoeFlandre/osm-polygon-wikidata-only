"""Shared acceptance-test configuration.

Every acceptance scenario is an offline, executable specification: it runs
against local fixtures, in-memory clients and a fake Hugging Face Hub. The
autouse ``block_network`` fixture turns any attempt to open an internet socket
into an immediate failure, so a scenario that silently reaches live SPARQL,
MediaWiki or the Hub fails loudly instead of passing by accident (#115).
"""

from __future__ import annotations

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
        host = address[0]
        return host in {"localhost", "127.0.0.1", "::1", ""}
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

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_local_address((host,)):
            _refuse(host)
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    yield


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test collected from this directory as ``acceptance``."""

    for item in items:
        if "tests/acceptance/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.acceptance)
