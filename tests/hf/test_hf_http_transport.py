"""Hugging Face transport must never hang indefinitely on broken IPv6."""

from __future__ import annotations

import pytest


def test_hf_client_uses_ipv4_and_bounded_network_timeouts() -> None:
    from osm_polygon_wikidata_only.hf._uploader.http_transport import (
        build_hf_http_client,
    )

    client = build_hf_http_client(event_hook=lambda _request: None)
    try:
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 120.0
        assert client.timeout.write == 120.0
        assert client.timeout.pool == 30.0
        assert client._transport._pool._local_address == "0.0.0.0"  # type: ignore[attr-defined]
    finally:
        client.close()


def test_hf_transport_configuration_is_process_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.hf._uploader import http_transport

    installed: list[object] = []
    monkeypatch.setattr(http_transport, "_configured", False)

    http_transport.configure_hf_http_transport(_set_factory=installed.append)
    http_transport.configure_hf_http_transport(_set_factory=installed.append)

    assert len(installed) == 1
    assert callable(installed[0])


def test_token_verification_configures_transport_before_whoami(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.hf._uploader import token

    events: list[str] = []
    monkeypatch.setattr(
        token,
        "configure_hf_http_transport",
        lambda: events.append("configured"),
    )

    token.verify_hf_token(
        "hf_test",
        _whoami=lambda _value: events.append("whoami") or {"name": "tester"},
    )

    assert events == ["configured", "whoami"]


def test_api_construction_configures_transport_before_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.hf._uploader import operations

    events: list[str] = []
    monkeypatch.setattr(
        operations,
        "configure_hf_http_transport",
        lambda: events.append("configured"),
    )

    class Api:
        def __init__(self, *, token: str) -> None:
            events.append("client")
            self.token = token

    operations._build_hf_api("hf_test", api_factory=Api)

    assert events == ["configured", "client"]
