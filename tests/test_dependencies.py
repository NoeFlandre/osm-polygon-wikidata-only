"""Tests for CLI dependency composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli import dependencies
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    CachedWikidataClient,
    HttpWikidataClient,
)
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaConfigurationError,
    WikimediaSession,
)
from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
    CachedWikipediaClient,
    HttpWikipediaClient,
)


def data_root(tmp_path: Path) -> DataRoot:
    root = DataRoot(tmp_path)
    root.ensure()
    return root


def test_build_clients_keeps_anonymous_scheduler_fixed(tmp_path: Path) -> None:
    wikidata, wikipedia, cache = dependencies.build_clients(
        Settings(cache_enabled=False),
        data_root=data_root(tmp_path),
        environ={},
    )

    assert isinstance(wikidata, HttpWikidataClient)
    assert isinstance(wikipedia, HttpWikipediaClient)
    assert cache is None
    assert wikidata._scheduler is wikipedia._scheduler
    for _ in range(200):
        wikidata._scheduler.report_success()
    assert wikidata._scheduler.current_requests_per_minute == 180


def test_build_clients_shares_authenticated_session_and_ramps_to_default_ceiling(
    tmp_path: Path,
) -> None:
    wikidata, wikipedia, _ = dependencies.build_clients(
        Settings(cache_enabled=False),
        data_root=data_root(tmp_path),
        environ={
            "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
            "WIKIMEDIA_BOT_PASSWORD": "secret-value",
        },
    )

    assert isinstance(wikidata, HttpWikidataClient)
    assert isinstance(wikipedia, HttpWikipediaClient)
    assert isinstance(wikidata._session, WikimediaSession)
    assert wikidata._session is wikipedia._session
    assert wikidata._scheduler is wikipedia._scheduler
    for _ in range(2_000):
        wikidata._scheduler.report_success()
    assert wikidata._scheduler.current_requests_per_minute == 1_200


def test_build_clients_applies_authenticated_rate_ceiling_override(tmp_path: Path) -> None:
    wikidata, _, _ = dependencies.build_clients(
        Settings(cache_enabled=False),
        data_root=data_root(tmp_path),
        environ={
            "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
            "WIKIMEDIA_BOT_PASSWORD": "secret-value",
            "WIKIMEDIA_REQUESTS_PER_MINUTE": "300",
        },
    )

    assert isinstance(wikidata, HttpWikidataClient)
    for _ in range(1_000):
        wikidata._scheduler.report_success()
    assert wikidata._scheduler.current_requests_per_minute == 300


@pytest.mark.parametrize("value", ["", "fast", "0", "-1"])
def test_build_clients_rejects_invalid_rate_ceiling(tmp_path: Path, value: str) -> None:
    with pytest.raises(WikimediaConfigurationError) as captured:
        dependencies.build_clients(
            Settings(cache_enabled=False),
            data_root=data_root(tmp_path),
            environ={
                "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
                "WIKIMEDIA_BOT_PASSWORD": "secret-value",
                "WIKIMEDIA_REQUESTS_PER_MINUTE": value,
            },
        )

    assert "WIKIMEDIA_REQUESTS_PER_MINUTE" in str(captured.value)
    assert "secret-value" not in str(captured.value)


def test_build_clients_rejects_partial_credentials(tmp_path: Path) -> None:
    with pytest.raises(WikimediaConfigurationError, match="WIKIMEDIA_BOT_PASSWORD"):
        dependencies.build_clients(
            Settings(cache_enabled=False),
            data_root=data_root(tmp_path),
            environ={"WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline"},
        )


def test_cache_enabled_builds_each_http_client_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"wikidata": 0, "wikipedia": 0}
    real_wikidata = dependencies.HttpWikidataClient
    real_wikipedia = dependencies.HttpWikipediaClient

    def build_wikidata(*args: object, **kwargs: object) -> HttpWikidataClient:
        counts["wikidata"] += 1
        return real_wikidata(*args, **kwargs)  # type: ignore[arg-type]

    def build_wikipedia(*args: object, **kwargs: object) -> HttpWikipediaClient:
        counts["wikipedia"] += 1
        return real_wikipedia(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dependencies, "HttpWikidataClient", build_wikidata)
    monkeypatch.setattr(dependencies, "HttpWikipediaClient", build_wikipedia)

    wikidata, wikipedia, cache = dependencies.build_clients(
        Settings(cache_enabled=True),
        data_root=data_root(tmp_path),
        environ={},
    )

    assert isinstance(wikidata, CachedWikidataClient)
    assert isinstance(wikipedia, CachedWikipediaClient)
    assert cache is not None
    assert counts == {"wikidata": 1, "wikipedia": 1}
