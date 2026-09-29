from __future__ import annotations

from types import SimpleNamespace

import pytest

from osm_polygon_wikidata_only.pipeline.sync_heartbeat import SyncHeartbeat


@pytest.mark.parametrize("include_auth", [False, True])
def test_sync_heartbeat_logs_once_then_stops(include_auth: bool) -> None:
    class StopAfterOneLog:
        def __init__(self) -> None:
            self.results = iter((False, True))

        def wait(self, _timeout: float) -> bool:
            return next(self.results)

        def set(self) -> None:
            raise AssertionError("run should not change the stop signal")

    scheduler = SimpleNamespace(
        requests_last_minute=2,
        current_requests_per_minute=30.0,
        maximum_requests_per_minute=60.0,
        utilization_percent=3.3,
        in_flight=1,
        max_in_flight=4,
        throttle_events=0,
        throttled_hosts_last_minute=0,
        cooling_down_hosts=0,
        cooldown_remaining_s=0.0,
    )
    messages: list[str] = []
    auth = lambda: SimpleNamespace(  # noqa: E731
        credentials_configured=True,
        authenticated_hosts=1,
        anonymous_hosts=0,
    )
    heartbeat = SyncHeartbeat(
        region="region",
        region_index=1,
        region_total=1,
        augmentation_snapshot=lambda: SimpleNamespace(phase="documents", completed=1, total=2),
        scheduler_snapshot=lambda: scheduler,
        auth_snapshot=auth if include_auth else None,
        log=messages.append,
        interval_s=1.0,
        clock=lambda: 120.0,
    )
    heartbeat._stop = StopAfterOneLog()  # type: ignore[assignment]

    heartbeat.run()

    assert len(messages) == 1
    assert "Sync progress 1/1 region" in messages[0]
    assert ("authenticated hosts 1" in messages[0]) is include_auth
