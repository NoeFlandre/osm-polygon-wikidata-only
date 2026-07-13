from __future__ import annotations

from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import WikimediaAuthSnapshot
from osm_polygon_wikidata_only.pipeline.sync_heartbeat import format_sync_progress
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot


def test_format_sync_progress_reports_in_flight_and_rolling_throttle_count() -> None:
    progress = AugmentationProgress()
    progress.start("Wikipedia sections", total=6153)
    progress.advance(5403)
    scheduler = RequestSchedulerSnapshot(
        requests_last_minute=934,
        current_requests_per_minute=1200,
        maximum_requests_per_minute=1200,
        utilization_percent=77.833,
        in_flight=3,
        max_in_flight=12,
        throttle_events=2,
        throttled_hosts_last_minute=1,
        cooling_down_hosts=0,
        cooldown_remaining_s=0.0,
    )

    message = format_sync_progress(
        region="finland-latest",
        region_index=1,
        region_total=358,
        elapsed_s=600,
        augmentation=progress.snapshot(),
        scheduler=scheduler,
    )

    assert message == (
        "Sync progress 1/358 finland-latest: 10m elapsed; "
        "Wikipedia sections 5403/6153; "
        "requests 934/1200 rpm (78%); "
        "in-flight 3/12; "
        "429s last minute 2 across 1 host"
    )


def test_format_sync_progress_includes_active_cooldown_and_cooling_hosts() -> None:
    progress = AugmentationProgress()
    progress.start("Wikivoyage documents", total=20)
    scheduler = RequestSchedulerSnapshot(
        requests_last_minute=100,
        current_requests_per_minute=600,
        maximum_requests_per_minute=1200,
        utilization_percent=8.3,
        in_flight=0,
        max_in_flight=12,
        throttle_events=1,
        throttled_hosts_last_minute=1,
        cooling_down_hosts=1,
        cooldown_remaining_s=3.2,
    )

    message = format_sync_progress(
        region="finland-latest",
        region_index=1,
        region_total=358,
        elapsed_s=60,
        augmentation=progress.snapshot(),
        scheduler=scheduler,
    )

    assert message == (
        "Sync progress 1/358 finland-latest: 1m elapsed; "
        "Wikivoyage documents 0/20; "
        "requests 100/1200 rpm (8%); "
        "in-flight 0/12; "
        "active ceiling 600 rpm; "
        "429s last minute 1 across 1 host; "
        "cooldowns 1 host; "
        "cooldown 4s"
    )


def test_format_sync_progress_reports_authenticated_and_anonymous_hosts() -> None:
    progress = AugmentationProgress()
    progress.start("Article sections", total=100)
    scheduler = RequestSchedulerSnapshot(
        requests_last_minute=200,
        current_requests_per_minute=1200,
        maximum_requests_per_minute=1200,
        utilization_percent=16.667,
        in_flight=4,
        max_in_flight=8,
        throttle_events=0,
        throttled_hosts_last_minute=0,
        cooling_down_hosts=0,
        cooldown_remaining_s=0.0,
    )
    auth = WikimediaAuthSnapshot(
        credentials_configured=True, authenticated_hosts=18, anonymous_hosts=3
    )

    message = format_sync_progress(
        region="belgium-latest",
        region_index=1,
        region_total=357,
        elapsed_s=600,
        augmentation=progress.snapshot(),
        scheduler=scheduler,
        auth=auth,
    )

    assert "authenticated hosts 18, anonymous hosts 3" in message
    assert "in-flight 4/8" in message
