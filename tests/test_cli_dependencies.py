"""Tests for construction of the shared Wikimedia runtime."""

from __future__ import annotations

import logging

import pytest

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import WikimediaConfigurationError
from osm_polygon_wikidata_only.io.cache import JsonFileCache

_AUTH_ENV = {
    "WIKIMEDIA_BOT_USERNAME": "User@pipeline",
    "WIKIMEDIA_BOT_PASSWORD": "secret",
}


def test_authenticated_runtime_shares_budget_with_augmentation_client(tmp_path) -> None:
    runtime = build_wikimedia_runtime(Settings(), data_root=DataRoot(tmp_path), environ=_AUTH_ENV)
    augmentation = AugmentationWikimediaClient(
        runtime.settings,
        JsonFileCache(tmp_path / "augmentation"),
        scheduler=runtime.scheduler,
        session=runtime.session,
    )

    # Conservative authenticated default that can actually reach the
    # 1200 rpm ceiling at typical API latency (~20 rps x ~0.3s ~= 6,
    # with headroom), while staying well under the 16 hard cap.
    assert runtime.scheduler.max_in_flight == 8
    assert runtime.scheduler.current_requests_per_minute == 1_200
    assert augmentation._scheduler is runtime.scheduler
    assert augmentation._session is runtime.session


def test_anonymous_runtime_keeps_conservative_concurrency(tmp_path) -> None:
    runtime = build_wikimedia_runtime(Settings(), data_root=DataRoot(tmp_path), environ={})

    assert runtime.scheduler.max_in_flight == 3
    assert runtime.scheduler.current_requests_per_minute == 180.0


def test_authenticated_concurrency_is_overridable_and_validated(tmp_path) -> None:
    runtime = build_wikimedia_runtime(
        Settings(),
        data_root=DataRoot(tmp_path),
        environ={**_AUTH_ENV, "WIKIMEDIA_MAX_IN_FLIGHT": "12"},
    )
    assert runtime.scheduler.max_in_flight == 12

    with pytest.raises(WikimediaConfigurationError):
        build_wikimedia_runtime(
            Settings(),
            data_root=DataRoot(tmp_path),
            environ={**_AUTH_ENV, "WIKIMEDIA_MAX_IN_FLIGHT": "99"},
        )


def test_startup_log_distinguishes_configured_credentials_from_verified(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="osm_polygon_wikidata_only.cli.dependencies")
    build_wikimedia_runtime(Settings(), data_root=DataRoot(tmp_path), environ=_AUTH_ENV)

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "credentials configured" in rendered
    assert "verification" in rendered
    # Must not claim the whole run is already authenticated.
    assert "authenticated as" not in rendered
    assert "secret" not in rendered
