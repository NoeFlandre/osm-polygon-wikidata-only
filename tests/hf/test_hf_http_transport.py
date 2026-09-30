"""Hugging Face transport must never hang indefinitely on broken IPv6."""

from __future__ import annotations

import sys
from types import ModuleType

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
        pool = getattr(client._transport, "_pool")
        assert getattr(pool, "_local_address") == "0.0.0.0"
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


def test_token_helpers_cover_no_token_default_client_and_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.hf._uploader import token
    from osm_polygon_wikidata_only.hf._uploader.errors import UploadError

    monkeypatch.setattr(token, "resolve_hf_token", lambda _explicit: None)
    assert token.verify_hf_token(None) is None

    calls: list[str] = []

    class Api:
        def __init__(self, *, token: str) -> None:
            calls.append(token)

        def whoami(self) -> dict[str, str]:
            return {"name": "verified"}

    hub = ModuleType("huggingface_hub")
    setattr(hub, "HfApi", Api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert token._whoami_client(None)("secret") == {"name": "verified"}
    assert calls == ["secret"]

    monkeypatch.setattr(token, "resolve_hf_token", lambda _explicit: "secret")
    monkeypatch.setattr(token, "configure_hf_http_transport", lambda: None)

    def rejected(_token: str) -> object:
        raise RuntimeError("revoked")

    with pytest.raises(UploadError, match="rejected HF_TOKEN"):
        token.verify_hf_token("secret", _whoami=rejected)


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
