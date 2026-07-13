"""Tests for CLI dependency composition."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
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
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler


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


def test_build_clients_logs_anonymous_mode_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=dependencies.LOGGER.name)

    dependencies.build_clients(
        Settings(cache_enabled=False),
        data_root=data_root(tmp_path),
        environ={},
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == dependencies.LOGGER.name and "Wikimedia API mode" in record.getMessage()
    ]
    assert messages == [
        "Wikimedia API mode: anonymous (rate ceiling: 180 requests/minute, in-flight=3, "
        "host interval: 0.50s)"
    ]


def test_build_clients_logs_authenticated_mode_once_without_password(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=dependencies.LOGGER.name)

    dependencies.build_clients(
        Settings(cache_enabled=False),
        data_root=data_root(tmp_path),
        environ={
            "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
            "WIKIMEDIA_BOT_PASSWORD": "secret-value",
        },
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == dependencies.LOGGER.name and "Wikimedia API mode" in record.getMessage()
    ]
    assert messages == [
        "Wikimedia API mode: credentials configured for NoeFlandre@pipeline; "
        "verification occurs per host; "
        "rate ceiling=1200 rpm; "
        "in-flight=8; "
        "authenticated host interval=0.05s; "
        "anonymous intervals: Wikipedia=0.50s, Wikidata=1.20s, augmentation=0.50s. "
        "The ceiling is a client-side limit, not a guaranteed server allowance."
    ]
    assert "secret-value" not in caplog.text


def test_build_clients_preserves_lower_anonymous_settings_rate(tmp_path: Path) -> None:
    wikidata, _, _ = dependencies.build_clients(
        Settings(cache_enabled=False, wikimedia_requests_per_minute=90),
        data_root=data_root(tmp_path),
        environ={},
    )

    assert isinstance(wikidata, HttpWikidataClient)
    for _ in range(200):
        wikidata._scheduler.report_success()
    assert wikidata._scheduler.current_requests_per_minute == 90


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


def test_authenticated_clients_use_full_rate_budget_with_safe_global_concurrency(
    tmp_path: Path,
) -> None:
    """Authenticated bot sessions must exploit their higher rate ceiling.

    Authenticated bot sessions keep the full 1200 rpm pacing ceiling,
    tighter host interval, and a conservative authenticated concurrency
    default (8) sized to reach ~20 rps at typical API latency. Concurrency
    is a client-side choice subordinate to the global rate ceiling and
    per-host cooldowns; it is not a guaranteed server allowance.
    """
    wikidata, wikipedia, _ = dependencies.build_clients(
        Settings(cache_enabled=False),
        data_root=data_root(tmp_path),
        environ={
            "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
            "WIKIMEDIA_BOT_PASSWORD": "secret-value",
        },
    )

    assert isinstance(wikidata, dependencies.HttpWikidataClient)
    assert isinstance(wikipedia, dependencies.HttpWikipediaClient)
    assert wikidata._scheduler.max_in_flight == 8
    assert wikidata._scheduler.current_requests_per_minute == 1_200
    # Settings must continue to represent anonymous host pacing.
    assert wikidata._settings.wikidata_min_interval_s == pytest.approx(1.2)
    assert wikipedia._settings.wikipedia_min_interval_s == pytest.approx(0.5)
    # The scheduler must start at (or near) the ceiling, not at the
    # conservative anonymous 180 rpm.
    assert wikidata._scheduler.current_requests_per_minute >= 600


def test_anonymous_clients_preserve_conservative_throttling(tmp_path: Path) -> None:
    wikidata, wikipedia, _ = dependencies.build_clients(
        Settings(cache_enabled=False),
        data_root=data_root(tmp_path),
        environ={},
    )

    assert isinstance(wikidata, dependencies.HttpWikidataClient)
    assert isinstance(wikipedia, dependencies.HttpWikipediaClient)
    # Anonymous sessions must keep the polite defaults.
    assert wikidata._scheduler.max_in_flight == 3
    assert wikidata._settings.wikidata_min_interval_s == pytest.approx(1.2)
    assert wikipedia._settings.wikipedia_min_interval_s == pytest.approx(0.5)
    assert wikidata._scheduler.current_requests_per_minute == pytest.approx(180.0)


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


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"Content-Type": "application/json"}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MockOpener:
    def __init__(self, login_success: bool = True):
        self.login_success = login_success
        self.requests = []

    def open(self, request, timeout=None):
        import json
        import urllib.parse

        self.requests.append(request)
        parsed_url = urllib.parse.urlparse(request.full_url)
        params = urllib.parse.parse_qs(
            request.data.decode() if request.data is not None else parsed_url.query
        )
        action = params.get("action", [""])[0]
        if action == "query" and params.get("meta") == ["tokens"]:
            body = json.dumps({"query": {"tokens": {"logintoken": "mock-token"}}}).encode()
            return FakeResponse(body)
        if action == "login":
            res = "Success" if self.login_success else "Failed"
            body = json.dumps({"login": {"result": res}}).encode()
            return FakeResponse(body)

        if "wbgetentities" in request.full_url or action == "wbgetentities":
            body = json.dumps(
                {
                    "entities": {
                        "Q5": {
                            "id": "Q5",
                            "labels": {"en": {"value": "Human"}},
                            "sitelinks": {"enwiki": {"title": "Human"}},
                        }
                    }
                }
            ).encode()
            return FakeResponse(body)

        body = json.dumps(
            {
                "query": {
                    "pages": {
                        "123": {
                            "pageid": 123,
                            "ns": 0,
                            "title": "Andorra",
                            "revisions": [{"revid": 456, "timestamp": "2026-07-13T00:00:00Z"}],
                            "extracts": "Andorra is a microstate.",
                            "fullurl": "https://en.wikipedia.org/wiki/Andorra",
                        }
                    }
                }
            }
        ).encode()
        return FakeResponse(body)


def test_effective_settings_preserve_anonymous_intervals(tmp_path: Path) -> None:
    environ = {
        "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret-value",
    }
    custom_settings = Settings(
        cache_enabled=False,
        wikipedia_min_interval_s=0.5,
        wikidata_min_interval_s=1.2,
        augmentation_min_interval_s=0.5,
        wikimedia_authenticated_min_interval_s=0.05,
    )
    runtime = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )

    assert runtime.settings.wikipedia_min_interval_s == 0.5
    assert runtime.settings.wikidata_min_interval_s == 1.2
    assert runtime.settings.augmentation_min_interval_s == 0.5
    assert runtime.settings.wikimedia_authenticated_min_interval_s == 0.05


@pytest.fixture
def mock_opener_setter(monkeypatch: pytest.MonkeyPatch):
    current_opener = [None]
    original_init = WikimediaSession.__init__

    def mock_init(self, *, scheduler, timeout_s, user_agent, credentials=None, opener_factory=None):
        op = current_opener[0] if current_opener[0] is not None else MockOpener()
        original_init(
            self,
            scheduler=scheduler,
            timeout_s=timeout_s,
            user_agent=user_agent,
            credentials=credentials,
            opener_factory=lambda: op,
        )

    monkeypatch.setattr(WikimediaSession, "__init__", mock_init)
    return current_opener


def test_verified_wikipedia_host_pacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_opener_setter
) -> None:
    environ = {
        "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret-value",
    }
    custom_settings = Settings(
        cache_enabled=False,
        wikipedia_min_interval_s=0.5,
        wikimedia_authenticated_min_interval_s=0.05,
    )
    opener = MockOpener(login_success=True)
    mock_opener_setter[0] = opener

    observed_pacing = []
    monkeypatch.setattr(
        AdaptiveRequestScheduler,
        "pace_host",
        lambda self, host, *, min_interval_s=0.0: observed_pacing.append((host, min_interval_s)),
    )
    runtime = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )
    runtime.wikipedia.fetch_article("en", "enwiki", "Andorra", fetch_full_text=False)

    assert len(observed_pacing) > 0
    hosts_paced = [host for host, _ in observed_pacing]
    assert "en.wikipedia.org" in hosts_paced
    intervals = [interval for host, interval in observed_pacing if host == "en.wikipedia.org"]
    assert len(intervals) > 0
    assert all(i == 0.05 for i in intervals)


def test_rejected_wikipedia_host_pacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_opener_setter
) -> None:
    environ = {
        "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret-value",
    }
    custom_settings = Settings(
        cache_enabled=False,
        wikipedia_min_interval_s=0.5,
        wikimedia_authenticated_min_interval_s=0.05,
    )
    opener = MockOpener(login_success=False)
    mock_opener_setter[0] = opener

    observed_pacing = []
    monkeypatch.setattr(
        AdaptiveRequestScheduler,
        "pace_host",
        lambda self, host, *, min_interval_s=0.0: observed_pacing.append((host, min_interval_s)),
    )
    runtime = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )
    runtime.wikipedia.fetch_article("en", "enwiki", "Andorra", fetch_full_text=False)

    assert len(observed_pacing) > 0
    intervals = [interval for host, interval in observed_pacing if host == "en.wikipedia.org"]
    assert len(intervals) > 0
    assert all(i == 0.5 for i in intervals)


def test_verified_wikidata_host_pacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_opener_setter
) -> None:
    environ = {
        "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret-value",
    }
    custom_settings = Settings(
        cache_enabled=False,
        wikidata_min_interval_s=1.2,
        wikimedia_authenticated_min_interval_s=0.05,
    )
    opener = MockOpener(login_success=True)
    mock_opener_setter[0] = opener

    observed_pacing = []
    monkeypatch.setattr(
        AdaptiveRequestScheduler,
        "pace_host",
        lambda self, host, *, min_interval_s=0.0: observed_pacing.append((host, min_interval_s)),
    )
    runtime = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )
    runtime.wikidata.get_entity("Q5")

    assert len(observed_pacing) > 0
    intervals = [interval for host, interval in observed_pacing if host == "www.wikidata.org"]
    assert len(intervals) > 0
    assert all(i == 0.05 for i in intervals)


def test_rejected_wikidata_host_pacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_opener_setter
) -> None:
    environ = {
        "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret-value",
    }
    custom_settings = Settings(
        cache_enabled=False,
        wikidata_min_interval_s=1.2,
        wikimedia_authenticated_min_interval_s=0.05,
    )
    opener = MockOpener(login_success=False)
    mock_opener_setter[0] = opener

    observed_pacing = []
    monkeypatch.setattr(
        AdaptiveRequestScheduler,
        "pace_host",
        lambda self, host, *, min_interval_s=0.0: observed_pacing.append((host, min_interval_s)),
    )
    runtime = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )
    runtime.wikidata.get_entity("Q5")

    assert len(observed_pacing) > 0
    intervals = [interval for host, interval in observed_pacing if host == "www.wikidata.org"]
    assert len(intervals) > 0
    assert all(i == 1.2 for i in intervals)


def test_augmentation_client_pacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_opener_setter
) -> None:
    environ = {
        "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret-value",
    }
    custom_settings = Settings(
        cache_enabled=False,
        augmentation_min_interval_s=0.5,
        wikimedia_authenticated_min_interval_s=0.05,
    )

    # Case A: Verified
    opener_verified = MockOpener(login_success=True)
    mock_opener_setter[0] = opener_verified

    observed_pacing = []
    monkeypatch.setattr(
        AdaptiveRequestScheduler,
        "pace_host",
        lambda self, host, *, min_interval_s=0.0: observed_pacing.append((host, min_interval_s)),
    )
    runtime_verified = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )
    augmentation_verified = AugmentationWikimediaClient(
        runtime_verified.settings,
        JsonFileCache(tmp_path / "cache_v"),
        environ=environ,
        scheduler=runtime_verified.scheduler,
        session=runtime_verified.session,
    )
    augmentation_verified.get_json(
        "https://en.wikipedia.org/w/api.php?action=query", key="test_v.json"
    )

    assert len(observed_pacing) > 0
    intervals_v = [interval for host, interval in observed_pacing if host == "en.wikipedia.org"]
    assert len(intervals_v) > 0
    assert all(i == 0.05 for i in intervals_v)

    # Case B: Rejected
    opener_rejected = MockOpener(login_success=False)
    mock_opener_setter[0] = opener_rejected

    observed_pacing.clear()
    runtime_rejected = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )
    augmentation_rejected = AugmentationWikimediaClient(
        runtime_rejected.settings,
        JsonFileCache(tmp_path / "cache_r"),
        environ=environ,
        scheduler=runtime_rejected.scheduler,
        session=runtime_rejected.session,
    )
    augmentation_rejected.get_json(
        "https://en.wikipedia.org/w/api.php?action=query", key="test_r.json"
    )

    assert len(observed_pacing) > 0
    intervals_r = [interval for host, interval in observed_pacing if host == "en.wikipedia.org"]
    assert len(intervals_r) > 0
    assert all(i == 0.5 for i in intervals_r)


def test_no_credentials_pacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_opener_setter
) -> None:
    custom_settings = Settings(
        cache_enabled=False,
        wikipedia_min_interval_s=0.5,
        wikidata_min_interval_s=1.2,
        augmentation_min_interval_s=0.5,
        wikimedia_authenticated_min_interval_s=0.05,
    )
    opener = MockOpener()
    mock_opener_setter[0] = opener

    observed_pacing = []
    monkeypatch.setattr(
        AdaptiveRequestScheduler,
        "pace_host",
        lambda self, host, *, min_interval_s=0.0: observed_pacing.append((host, min_interval_s)),
    )
    runtime = dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ={},
    )
    runtime.wikipedia.fetch_article("en", "enwiki", "Andorra", fetch_full_text=False)
    runtime.wikidata.get_entity("Q5")

    augmentation = AugmentationWikimediaClient(
        runtime.settings,
        JsonFileCache(tmp_path / "cache_no_creds"),
        environ={},
        scheduler=runtime.scheduler,
        session=runtime.session,
    )
    augmentation.get_json(
        "https://es.wikipedia.org/w/api.php?action=query", key="test_no_creds.json"
    )

    assert len(observed_pacing) > 0
    by_host = {host: min_i for host, min_i in observed_pacing}
    assert by_host["en.wikipedia.org"] == 0.5
    assert by_host["www.wikidata.org"] == 1.2
    assert by_host["es.wikipedia.org"] == 0.5


def test_startup_logs_contain_separate_intervals(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="osm_polygon_wikidata_only.cli.dependencies")
    environ = {
        "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret-value",
    }
    custom_settings = Settings(
        cache_enabled=False,
        wikipedia_min_interval_s=0.5,
        wikidata_min_interval_s=1.2,
        augmentation_min_interval_s=0.5,
        wikimedia_authenticated_min_interval_s=0.05,
    )
    dependencies.build_wikimedia_runtime(
        custom_settings,
        data_root=DataRoot(tmp_path),
        environ=environ,
    )
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "credentials configured for NoeFlandre@pipeline" in rendered
    assert (
        "verification occurs per host" in rendered
        or "verification is performed per host" in rendered
    )
    assert "authenticated host interval=0.05s" in rendered
    assert "anonymous intervals: Wikipedia=0.50s, Wikidata=1.20s, augmentation=0.50s" in rendered
    assert "client-side" in rendered
