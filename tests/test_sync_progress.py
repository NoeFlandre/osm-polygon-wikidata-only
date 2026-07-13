from __future__ import annotations

from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.pipeline.sync_heartbeat import format_sync_progress
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot


def test_format_sync_progress_is_concise_and_explains_rate_usage() -> None:
    progress = AugmentationProgress()
    progress.start("Wikipedia sections", total=6153)
    progress.advance(5403)
    scheduler = RequestSchedulerSnapshot(
        requests_last_minute=934,
        current_requests_per_minute=1200,
        maximum_requests_per_minute=1200,
        utilization_percent=77.833,
        in_flight=3,
        throttle_events=2,
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
        "Wikipedia sections 5403/6153; requests 934/1200 rpm (78%); "
        "in-flight 3; 429s 2"
    )


def test_format_sync_progress_includes_active_cooldown() -> None:
    progress = AugmentationProgress()
    progress.start("Wikivoyage documents", total=20)
    scheduler = RequestSchedulerSnapshot(
        requests_last_minute=100,
        current_requests_per_minute=600,
        maximum_requests_per_minute=1200,
        utilization_percent=8.3,
        in_flight=0,
        throttle_events=1,
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

    assert message.endswith("429s 1; active ceiling 600 rpm; cooldown 4s")
