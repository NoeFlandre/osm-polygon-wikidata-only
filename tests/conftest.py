"""Shared pytest configuration for the root test suite."""

from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.database import DirectoryBasedExampleDatabase, ExampleDatabase

from tests._support import (
    _socket_address_is_allowed,
    skip_landmass_drawing,
    write_count_map_placeholder,
    write_coverage_map_placeholder,
)


def _dev_example_database() -> ExampleDatabase | None:
    """Keep the local example database under ``TMPDIR`` rather than the checkout."""

    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir:
        return None
    return DirectoryBasedExampleDatabase(str(Path(tmpdir) / "hypothesis-examples"))


# Hypothesis profiles, selected with ``HYPOTHESIS_PROFILE``. ``ci`` is the
# default under CI (``CI`` set) and is deterministic; ``dev`` is the default
# locally; ``nightly`` runs a deeper randomized pass.
settings.register_profile(
    "ci",
    max_examples=100,
    derandomize=True,
    database=None,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", deadline=None, database=_dev_example_database())
settings.register_profile("nightly", max_examples=1000, deadline=None, database=None)
settings.load_profile(
    os.environ.get("HYPOTHESIS_PROFILE") or ("ci" if os.environ.get("CI") else "dev")
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".pytest_cache" / "matplotlib"),
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``requires_just`` when the ``just`` runner is absent."""

    if shutil.which("just") is not None:
        return
    skip_just = pytest.mark.skip(reason="just executable is not installed")
    for item in items:
        if item.get_closest_marker("requires_just") is not None:
            item.add_marker(skip_just)


@pytest.fixture(autouse=True)
def _block_outbound_sockets(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Block external name resolution and socket traffic for every test.

    Tests that need a local server or client may opt in with ``local_network``;
    even then, only loopback addresses are permitted.
    """
    allow_local = request.node.get_closest_marker("local_network") is not None
    denied = "outbound network access is disabled (only marked loopback is allowed)"

    def guard_resolution(host: Any) -> None:
        if not _socket_address_is_allowed(host, allow_local_network=allow_local):
            raise socket.gaierror(denied)

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        original = getattr(socket, name)

        def guarded_resolution(
            host: Any, *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            guard_resolution(host)
            return _original(host, *args, **kwargs)

        monkeypatch.setattr(socket, name, guarded_resolution)

    for name in ("gethostbyaddr", "getnameinfo"):
        original = getattr(socket, name)

        def guarded_reverse_resolution(
            address: Any,
            *args: Any,
            _original: Any = original,
            _name: str = name,
            **kwargs: Any,
        ) -> Any:
            guard_resolution(address)
            if _name == "gethostbyaddr":
                host = address[0] if isinstance(address, (tuple, list)) else address
                return str(host), [], [str(host)]

            if args:
                flags = args[0]
                args = (flags | socket.NI_NUMERICHOST) & ~socket.NI_NAMEREQD, *args[1:]
            else:
                flags = kwargs.get("flags", 0)
                kwargs["flags"] = (flags | socket.NI_NUMERICHOST) & ~socket.NI_NAMEREQD
            return _original(address, *args, **kwargs)

        monkeypatch.setattr(socket, name, guarded_reverse_resolution)

    for name in ("connect", "connect_ex", "bind", "sendto"):
        original = getattr(socket.socket, name)

        def guarded_address_call(
            sock: socket.socket,
            *args: Any,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if getattr(sock, "family", None) != socket.AF_UNIX:
                address = kwargs.get("address")
                if address is None and args:
                    address = args[-1]
                if not _socket_address_is_allowed(address, allow_local_network=allow_local):
                    raise OSError(denied)
            return _original(sock, *args, **kwargs)

        monkeypatch.setattr(socket.socket, name, guarded_address_call)

    for name in ("send", "sendall", "sendmsg"):
        original = getattr(socket.socket, name, None)
        if original is None:
            continue

        def guarded_connected_send(
            sock: socket.socket,
            *args: Any,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if getattr(sock, "family", None) != socket.AF_UNIX:
                try:
                    peer = sock.getpeername()
                except OSError:
                    peer = None
                if not _socket_address_is_allowed(peer, allow_local_network=allow_local):
                    raise OSError(denied)
            return _original(sock, *args, **kwargs)

        monkeypatch.setattr(socket.socket, name, guarded_connected_send)


@pytest.fixture(scope="session")
def _map_render_stub_functions() -> dict[str, object]:
    """Return fast renderer substitutes for orchestration-only test cases."""
    return {
        "draw_landmasses": skip_landmass_drawing,
        "generate_coverage_map": write_coverage_map_placeholder,
        "render_count_map": write_count_map_placeholder,
    }


@pytest.fixture(autouse=True)
def _stub_map_renderers_for_orchestration(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub map imports in selected orchestration tests, not visual tests."""
    if request.node.get_closest_marker("map_orchestration") is None:
        return
    if request.node.get_closest_marker("real_map") is not None:
        return
    replacements = request.getfixturevalue("_map_render_stub_functions")
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(("osm_polygon_wikidata_only", "tests")):
            continue
        for attribute, replacement in replacements.items():
            if callable(getattr(module, attribute, None)):
                monkeypatch.setattr(module, attribute, replacement)
