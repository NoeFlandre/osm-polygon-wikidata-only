"""Thread-safe progress state for long regional augmentation work."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AugmentationProgressSnapshot:
    phase: str
    completed: int
    total: int


class AugmentationProgress:
    """Track one augmentation phase without coupling workers to logging."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = "Starting augmentation"
        self._completed = 0
        self._total = 0

    def start(self, phase: str, *, total: int) -> None:
        with self._lock:
            self._phase = phase
            self._completed = 0
            self._total = max(0, total)

    def advance(self, amount: int = 1) -> None:
        with self._lock:
            self._completed = min(self._total, self._completed + max(0, amount))

    def complete(self) -> None:
        with self._lock:
            self._completed = self._total

    def snapshot(self) -> AugmentationProgressSnapshot:
        with self._lock:
            return AugmentationProgressSnapshot(self._phase, self._completed, self._total)


__all__ = ["AugmentationProgress", "AugmentationProgressSnapshot"]
