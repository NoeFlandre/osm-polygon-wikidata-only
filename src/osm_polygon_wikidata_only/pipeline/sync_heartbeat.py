"""Low-noise progress reporting for unified regional synchronization."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgressSnapshot
from osm_polygon_wikidata_only.enrichment.progress import EnrichmentProgressSnapshot
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import WikimediaAuthSnapshot
from osm_polygon_wikidata_only.pipeline.heartbeat import EnrichmentHeartbeat
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot


def _plural_hosts(count: int) -> str:
    return "host" if count == 1 else "hosts"


def _base_progress_parts(
    region: str,
    region_index: int,
    region_total: int,
    elapsed_s: float,
    augmentation: AugmentationProgressSnapshot,
    scheduler: RequestSchedulerSnapshot,
) -> list[str]:
    """Build the always-present heartbeat fields."""
    return [
        f"Sync progress {region_index}/{region_total} {region}: "
        f"{int(max(0.0, elapsed_s) // 60)}m elapsed",
        f"{augmentation.phase} {augmentation.completed}/{augmentation.total}",
        f"requests {scheduler.requests_last_minute}/"
        f"{scheduler.maximum_requests_per_minute:.0f} rpm "
        f"({scheduler.utilization_percent:.0f}%)",
        f"in-flight {scheduler.in_flight}/{scheduler.max_in_flight}",
    ]


def _auth_progress_part(auth: WikimediaAuthSnapshot | None) -> str | None:
    """Format authentication host counts when they are meaningful."""
    if auth is None or not (
        auth.credentials_configured or auth.authenticated_hosts or auth.anonymous_hosts
    ):
        return None
    return f"authenticated hosts {auth.authenticated_hosts}, anonymous hosts {auth.anonymous_hosts}"


def _throttle_progress_parts(scheduler: RequestSchedulerSnapshot) -> list[str]:
    """Format adaptive request-ceiling and cooldown details."""
    parts: list[str] = []
    if scheduler.current_requests_per_minute < scheduler.maximum_requests_per_minute:
        parts.append(f"active ceiling {scheduler.current_requests_per_minute:.0f} rpm")
    if scheduler.throttle_events > 0:
        parts.append(
            f"429s last minute {scheduler.throttle_events} "
            f"across {scheduler.throttled_hosts_last_minute} "
            f"{_plural_hosts(scheduler.throttled_hosts_last_minute)}"
        )
    if scheduler.cooling_down_hosts > 0:
        parts.append(
            f"cooldowns {scheduler.cooling_down_hosts} "
            f"{_plural_hosts(scheduler.cooling_down_hosts)}"
        )
    if scheduler.cooldown_remaining_s > 0:
        parts.append(f"cooldown {math.ceil(scheduler.cooldown_remaining_s)}s")
    return parts


def format_sync_progress(
    *,
    region: str,
    region_index: int,
    region_total: int,
    elapsed_s: float,
    augmentation: AugmentationProgressSnapshot,
    scheduler: RequestSchedulerSnapshot,
    auth: WikimediaAuthSnapshot | None = None,
) -> str:
    """Format one concise, factual operator heartbeat."""
    parts = _base_progress_parts(
        region,
        region_index,
        region_total,
        elapsed_s,
        augmentation,
        scheduler,
    )
    auth_part = _auth_progress_part(auth)
    if auth_part is not None:
        parts.append(auth_part)
    parts.extend(_throttle_progress_parts(scheduler))
    return "; ".join(parts)


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
        auth_snapshot: Callable[[], WikimediaAuthSnapshot] | None = None,
        interval_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sync_region = region
        self._region_index = region_index
        self._region_total = region_total
        self._augmentation_snapshot = augmentation_snapshot
        self._scheduler_snapshot = scheduler_snapshot
        self._auth_snapshot = auth_snapshot
        self._sync_log = log
        self._sync_clock = clock
        self._sync_started_at = clock()
        super().__init__(
            region=region,
            snapshot=lambda: EnrichmentProgressSnapshot(0, 0, 0, 0, 0, "sync"),
            log=log,
            interval_s=interval_s,
            clock=clock,
        )

    def run(self) -> None:
        while not self._stop.wait(self._interval_s):
            auth = self._auth_snapshot() if self._auth_snapshot is not None else None
            self._sync_log(
                format_sync_progress(
                    region=self._sync_region,
                    region_index=self._region_index,
                    region_total=self._region_total,
                    elapsed_s=self._sync_clock() - self._sync_started_at,
                    augmentation=self._augmentation_snapshot(),
                    scheduler=self._scheduler_snapshot(),
                    auth=auth,
                )
            )


__all__ = ["SyncHeartbeat", "format_sync_progress"]
