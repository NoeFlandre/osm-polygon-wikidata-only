"""Bounded progress reporter for the startup local-validation phase.

The startup augmentation-state validation phase iterates over every
input stem and may take several minutes for large datasets. This
module provides :class:`LocalValidationProgress` -- a thin
coordinator that:

* Emits exactly one begin log line.
* Emits periodic progress lines at a bounded cadence so the
  operator has visibility without producing a noisy log line for
  every input. A line is emitted when *either* the elapsed clock
  time since the previous emission reaches
  ``progress_interval_s`` *or* the iteration reaches one of a
  bounded number of completion checkpoints (roughly one-eighth
  of the input size), whichever fires first. Both triggers are
  subject to the same ``quiet_threshold`` so trivially small
  inputs stay compact.
* Emits exactly one completion log line with the total elapsed
  time.
* Avoids periodic logs for trivially small inputs (those with
  fewer than ``quiet_threshold`` stems) so the log stays compact
  for small developer runs.
* Never logs paths, hashes, stems, or any other identifying
  value: the begin line carries only the count, and the
  completion line carries only the count and elapsed time.
* Calls the supplied validator exactly once per stem.

The clock is injectable so tests are deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


def _format_minutes(seconds: float) -> str:
    """Render a compact elapsed-time string suitable for log output.

    The example style in the task uses ``42s`` or ``2m 11s``. This
    format matches that style and is used by both the periodic and
    completion log lines.
    """
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rem = divmod(seconds, 60)
    return f"{minutes}m {rem:02d}s"


@dataclass(frozen=True, slots=True)
class LocalValidationProgress:
    """Coordinator for the startup local-validation phase.

    Validators are pure callables that map a stem to its validated
    status. The clock is injectable so test runs are deterministic.
    """

    validator: Callable[[str], bool]
    stems: list[str]
    log: Callable[[str], None]
    clock: Callable[[], float]
    progress_interval_s: float = 30.0
    quiet_threshold: int = 25
    phase_label: str = "regions"

    def run(self) -> dict[str, bool]:
        """Validate every stem and return a mapping of stem -> status.

        Logging is bounded: one begin line, at most one periodic
        progress line per ``progress_interval_s`` of injected clock
        time (suppressed entirely when the input is smaller than
        ``quiet_threshold``), and one completion line. Each stem is
        visited exactly once.
        """
        total = len(self.stems)
        self.log(f"Validating finalized local state: {total} {self.phase_label}")

        results: dict[str, bool] = {}
        started_at = self.clock()
        last_logged_at = started_at
        emit_periodic = total >= self.quiet_threshold
        # Roughly cap periodic emission at ~8 progress events for any
        # single run, so even a clock that never advances produces a
        # bounded number of progress lines.
        checkpoint = max(1, total // 8) if emit_periodic else 0

        for index, stem in enumerate(self.stems, start=1):
            results[stem] = self.validator(stem)
            if not emit_periodic:
                continue
            now = self.clock()
            elapsed_since_last = now - last_logged_at
            if elapsed_since_last >= self.progress_interval_s or (
                checkpoint and index % checkpoint == 0 and index < total
            ):
                self.log(
                    f"Local validation progress: {index}/{total} {self.phase_label}; "
                    f"{_format_minutes(now - started_at)} elapsed"
                )
                last_logged_at = now

        elapsed = self.clock() - started_at
        self.log(
            f"Local validation complete: {total} {self.phase_label} in {_format_minutes(elapsed)}"
        )
        return results


__all__ = ["LocalValidationProgress"]
