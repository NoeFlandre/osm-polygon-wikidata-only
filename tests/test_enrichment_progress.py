"""Tests for enrichment progress snapshots and heartbeats."""

from __future__ import annotations

import threading

import pytest

from osm_polygon_wikidata_only.enrichment.progress import (
    EnrichmentProgress,
    EnrichmentProgressSnapshot,
)


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
