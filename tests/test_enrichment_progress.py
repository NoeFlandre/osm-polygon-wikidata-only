"""Tests for enrichment progress snapshots and heartbeats."""

from __future__ import annotations

import threading

import pytest

from osm_polygon_wikidata_only.enrichment.progress import (
    EnrichmentProgress,
    EnrichmentProgressSnapshot,
)
from osm_polygon_wikidata_only.pipeline.heartbeat import EnrichmentHeartbeat


class FakeStopEvent:
    def __init__(self, wait_results: list[bool]) -> None:
        self._wait_results = iter(wait_results)
        self.waits: list[float] = []
        self.was_set = False

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        return next(self._wait_results)

    def set(self) -> None:
        self.was_set = True


class RecordingThread:
    def __init__(self) -> None:
        self.started = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def join(self) -> None:
        self.joined = True


def test_progress_tracker_starts_with_an_immutable_snapshot() -> None:
    progress = EnrichmentProgress(total_qids=3)

    snapshot = progress.snapshot()

    assert snapshot == EnrichmentProgressSnapshot(
        qids_completed=0,
        qids_total=3,
        sites_completed=0,
        sites_total=0,
        articles_attempted=0,
    )
    with pytest.raises(AttributeError):
        snapshot.qids_completed = 1  # type: ignore[misc]


def test_progress_tracker_records_qid_totals_and_completion() -> None:
    progress = EnrichmentProgress(total_qids=0)

    progress.set_qids_total(5)
    progress.advance_qids(2)

    snapshot = progress.snapshot()
    assert snapshot.qids_total == 5
    assert snapshot.qids_completed == 2


def test_progress_tracker_records_completed_site_and_articles() -> None:
    progress = EnrichmentProgress(total_qids=3)

    progress.set_sites_total(4)
    progress.complete_site(articles_attempted=7)

    assert progress.snapshot() == EnrichmentProgressSnapshot(
        qids_completed=0,
        qids_total=3,
        sites_completed=1,
        sites_total=4,
        articles_attempted=7,
    )


def test_snapshot_is_not_changed_by_later_updates() -> None:
    progress = EnrichmentProgress(total_qids=2)
    before = progress.snapshot()

    progress.advance_qids()

    assert before.qids_completed == 0
    assert progress.snapshot().qids_completed == 1


def test_concurrent_site_updates_never_lose_counts() -> None:
    progress = EnrichmentProgress(total_qids=0)
    progress.set_sites_total(100)
    threads = [threading.Thread(target=progress.complete_site, args=(1,)) for _ in range(100)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    snapshot = progress.snapshot()
    assert snapshot.sites_completed == 100
    assert snapshot.articles_attempted == 100


@pytest.mark.parametrize(
    "operation",
    [
        lambda progress: progress.set_qids_total(-1),
        lambda progress: progress.advance_qids(-1),
        lambda progress: progress.set_sites_total(-1),
        lambda progress: progress.complete_site(-1),
    ],
)
def test_progress_tracker_rejects_negative_counts(operation: object) -> None:
    progress = EnrichmentProgress(total_qids=0)

    with pytest.raises(ValueError):
        operation(progress)  # type: ignore[operator]


def test_heartbeat_logs_one_snapshot_after_each_two_minute_wait() -> None:
    progress = EnrichmentProgress(total_qids=143)
    progress.advance_qids(143)
    progress.set_sites_total(64)
    for _ in range(18):
        progress.complete_site(articles_attempted=41)
    stop_event = FakeStopEvent([False, False, True])
    times = iter([0.0, 120.0, 240.0])
    messages: list[str] = []
    heartbeat = EnrichmentHeartbeat(
        region="antarctica",
        snapshot=progress.snapshot,
        log=messages.append,
        stop_event=stop_event,
        clock=lambda: next(times),
    )

    heartbeat.run()

    assert stop_event.waits == [120.0, 120.0, 120.0]
    assert len(messages) == 2
    assert "antarctica" in messages[0]
    assert "2m elapsed" in messages[0]
    assert "Wikidata 143/143 QIDs" in messages[0]
    assert "Wikipedia 18/64 sites, 738 articles attempted" in messages[0]
    assert "4m elapsed" in messages[1]


def test_heartbeat_emits_nothing_when_stopped_before_first_interval() -> None:
    messages: list[str] = []
    heartbeat = EnrichmentHeartbeat(
        region="tiny",
        snapshot=EnrichmentProgress(total_qids=0).snapshot,
        log=messages.append,
        stop_event=FakeStopEvent([True]),
    )

    heartbeat.run()

    assert messages == []


def test_heartbeat_context_starts_and_stops_thread_when_body_raises() -> None:
    stop_event = FakeStopEvent([True])
    thread = RecordingThread()

    def thread_factory(**_: object) -> RecordingThread:
        return thread

    heartbeat = EnrichmentHeartbeat(
        region="tiny",
        snapshot=EnrichmentProgress(total_qids=0).snapshot,
        log=lambda _: None,
        stop_event=stop_event,
        thread_factory=thread_factory,
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        with heartbeat:
            raise RuntimeError("pipeline failed")

    assert thread.started
    assert stop_event.was_set
    assert thread.joined


def test_heartbeat_contains_snapshot_errors_without_failing_pipeline() -> None:
    stop_event = FakeStopEvent([False])
    debug_messages: list[str] = []

    def broken_snapshot() -> EnrichmentProgressSnapshot:
        raise RuntimeError("snapshot unavailable")

    heartbeat = EnrichmentHeartbeat(
        region="tiny",
        snapshot=broken_snapshot,
        log=lambda _: None,
        debug=debug_messages.append,
        stop_event=stop_event,
    )

    heartbeat.run()

    assert stop_event.was_set
    assert len(debug_messages) == 1
    assert "snapshot unavailable" in debug_messages[0]
