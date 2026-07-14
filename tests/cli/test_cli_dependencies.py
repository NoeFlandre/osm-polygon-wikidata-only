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


# ---------------------------------------------------------------------------
# Integration: production scheduler uses proportional systemic backoff
# ---------------------------------------------------------------------------


def test_production_scheduler_proportional_195_hosts_throttle_3(tmp_path) -> None:
    """build_wikimedia_runtime() scheduler must NOT globally reduce when only 3
    of ~195 hosts are throttled.

    threshold = min(195, max(5, ceil(195 * 0.10))) = 20
    """
    runtime = build_wikimedia_runtime(Settings(), data_root=DataRoot(tmp_path), environ=_AUTH_ENV)
    scheduler = runtime.scheduler

    # Activate 195 hosts so the proportional denominator is large.
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    for host in hosts:
        scheduler.pace_host(host)

    # Throttle 3 distinct hosts.
    for i in range(3):
        scheduler.report_host_throttled(hosts[i], 2.0)

    assert scheduler.current_requests_per_minute == 1200.0, (
        f"3 of 195 hosts throttled must NOT reduce global rate; "
        f"got {scheduler.current_requests_per_minute}"
    )


def test_production_scheduler_proportional_195_hosts_throttle_7(tmp_path) -> None:
    """Throttling 7 of 195 hosts must also NOT reduce the global rate."""
    runtime = build_wikimedia_runtime(Settings(), data_root=DataRoot(tmp_path), environ=_AUTH_ENV)
    scheduler = runtime.scheduler

    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    for host in hosts:
        scheduler.pace_host(host)

    for i in range(7):
        scheduler.report_host_throttled(hosts[i], 2.0)

    assert scheduler.current_requests_per_minute == 1200.0, (
        f"7 of 195 hosts throttled must NOT reduce global rate; "
        f"got {scheduler.current_requests_per_minute}"
    )


def test_production_scheduler_systemic_20_of_195_reduces(tmp_path) -> None:
    """Throttling 20 of 195 hosts MUST trigger exactly one global reduction."""
    runtime = build_wikimedia_runtime(Settings(), data_root=DataRoot(tmp_path), environ=_AUTH_ENV)
    scheduler = runtime.scheduler

    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    for host in hosts:
        scheduler.pace_host(host)

    for i in range(20):
        scheduler.report_host_throttled(hosts[i], 1.0)

    assert scheduler.current_requests_per_minute == 600.0, (
        f"20 of 195 hosts throttled must trigger exactly one halving; "
        f"got {scheduler.current_requests_per_minute}"
    )


def test_standalone_augmentation_scheduler_uses_proportional_policy(tmp_path) -> None:
    """The standalone AugmentationWikimediaClient scheduler must also
    use proportional systemic backoff."""
    client = AugmentationWikimediaClient(
        Settings(),
        JsonFileCache(tmp_path / "augmentation"),
        environ=_AUTH_ENV,
    )
    scheduler = client._scheduler

    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    for host in hosts:
        scheduler.pace_host(host)

    # 5 throttled out of 195: no global reduction.
    for i in range(5):
        scheduler.report_host_throttled(hosts[i], 2.0)

    assert scheduler.current_requests_per_minute == scheduler._max_requests_per_minute, (
        f"standalone augmentation scheduler should use proportional policy; "
        f"5 of 195 throttled should not reduce; got {scheduler.current_requests_per_minute}"
    )
