"""Thread-safe progress state for Wikimedia enrichment."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class EnrichmentProgressSnapshot:
    """Immutable counters describing completed enrichment work."""

    qids_completed: int
    qids_total: int
    sites_completed: int
    sites_total: int
    articles_attempted: int
    phase: str


class EnrichmentProgress:
    """Collect enrichment counters safely across site worker threads."""

    def __init__(self, *, total_qids: int) -> None:
        _require_non_negative(total_qids)
        self._qids_completed = 0
        self._qids_total = total_qids
        self._sites_completed = 0
        self._sites_total = 0
        self._articles_attempted = 0
        self._phase = "wikidata"
        self._lock = threading.Lock()

    def set_qids_total(self, total: int) -> None:
        _require_non_negative(total)
        with self._lock:
            self._qids_total = total

    def advance_qids(self, count: int = 1) -> None:
        _require_non_negative(count)
        with self._lock:
            self._qids_completed += count

    def start_wikipedia(self, total_sites: int) -> None:
        _require_non_negative(total_sites)
        with self._lock:
            self._sites_total = total_sites
            self._phase = "wikipedia"

    def complete_site(self, articles_attempted: int) -> None:
        _require_non_negative(articles_attempted)
        with self._lock:
            self._sites_completed += 1
            self._articles_attempted += articles_attempted

    def snapshot(self) -> EnrichmentProgressSnapshot:
        with self._lock:
            return EnrichmentProgressSnapshot(
                qids_completed=self._qids_completed,
                qids_total=self._qids_total,
                sites_completed=self._sites_completed,
                sites_total=self._sites_total,
                articles_attempted=self._articles_attempted,
                phase=self._phase,
            )


def _require_non_negative(value: int) -> None:
    if value < 0:
        raise ValueError("progress counts must be non-negative")


__all__ = ["EnrichmentProgress", "EnrichmentProgressSnapshot"]
