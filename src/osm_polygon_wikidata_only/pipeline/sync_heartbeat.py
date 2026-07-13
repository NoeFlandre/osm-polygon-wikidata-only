"""Low-noise progress reporting for unified regional synchronization."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgressSnapshot
from osm_polygon_wikidata_only.pipeline.heartbeat import EnrichmentHeartbeat
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot


def format_sync_progress(
    *,
    region: str,
    region_index: int,
    region_total: int,
    elapsed_s: float,
    augmentation: AugmentationProgressSnapshot,
    scheduler: RequestSchedulerSnapshot,
) -> str:
    """Format one concise, factual operator heartbeat."""
    message = (
        f"Sync progress {region_index}/{region_total} {region}: "
        f"{int(max(0.0, elapsed_s) // 60)}m elapsed; "
        f"{augmentation.phase} {augmentation.completed}/{augmentation.total}; "
        f"requests {scheduler.requests_last_minute}/"
        f"{scheduler.maximum_requests_per_minute:.0f} rpm "
        f"({scheduler.utilization_percent:.0f}%); "
        f"in-flight {scheduler.in_flight}; 429s {scheduler.throttle_events}"
    )
    if scheduler.current_requests_per_minute < scheduler.maximum_requests_per_minute:
        message += f"; active ceiling {scheduler.current_requests_per_minute:.0f} rpm"
    if scheduler.cooldown_remaining_s > 0:
        message += f"; cooldown {math.ceil(scheduler.cooldown_remaining_s)}s"
    return message


class SyncHeartbeat(EnrichmentHeartbeat):
    """Reuse the established heartbeat lifecycle with sync-specific snapshots."""

    def __init__(
        self,
        *,
        region: str,
        region_index: int,
        region_total: int,
        augmentation_snapshot: Callable[[], AugmentationProgressSnapshot],
        scheduler_snapshot: Callable[[], RequestSchedulerSnapshot],
        log: Callable[[str], None],
        interval_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sync_region = region
        self._region_index = region_index
        self._region_total = region_total
        self._augmentation_snapshot = augmentation_snapshot
        self._scheduler_snapshot = scheduler_snapshot
        self._sync_log = log
        self._sync_clock = clock
        self._sync_started_at = clock()
        super().__init__(
            region=region,
            snapshot=lambda: None,  # type: ignore[arg-type,return-value]
            log=log,
            interval_s=interval_s,
            clock=clock,
        )

    def run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sync_log(
                format_sync_progress(
                    region=self._sync_region,
                    region_index=self._region_index,
                    region_total=self._region_total,
                    elapsed_s=self._sync_clock() - self._sync_started_at,
                    augmentation=self._augmentation_snapshot(),
                    scheduler=self._scheduler_snapshot(),
                )
            )


__all__ = ["SyncHeartbeat", "format_sync_progress"]
