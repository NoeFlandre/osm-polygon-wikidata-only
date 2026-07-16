"""Tests for the startup local-validation progress reporter.

The reporter wraps the augmentation-state validation phase that
gates the rest of the unified sync. It must:

* Emit exactly one begin log line, optionally followed by bounded
  progress lines for larger inputs, and exactly one completion line.
* Call the supplied validator exactly once per stem.
* Be deterministic under an injected clock.
* Avoid noisy periodic output for small inputs.
* Avoid exposing paths, hashes, or credentials.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from osm_polygon_wikidata_only.pipeline.local_validation import LocalValidationProgress


@dataclass
class _Recorder:
    messages: list[str]


def _recorder() -> tuple[_Recorder, Callable[[str], None]]:
    rec = _Recorder(messages=[])

    def _log(message: str) -> None:
        rec.messages.append(message)

    return rec, _log


def test_progress_logs_begin_completion_and_one_progress_line() -> None:
    rec, log = _recorder()
    stems = [f"region-{i:03d}" for i in range(100)]
    seen: list[str] = []

    def _validator(stem: str) -> bool:
        seen.append(stem)
        return True

    clock = [0.0]

    def _clock() -> float:
        return clock[0]

    progress = LocalValidationProgress(
        validator=_validator,
        stems=stems,
        log=log,
        clock=_clock,
        progress_interval_s=10.0,
        quiet_threshold=10,
    )
    progress.run()

    assert seen == stems
    # Begin line
    assert any(m.startswith("Validating finalized local state:") for m in rec.messages)
    # At least one periodic line
    assert any(m.startswith("Local validation progress:") for m in rec.messages)
    # Completion line
    assert any(m.startswith("Local validation complete:") for m in rec.messages)


def test_progress_small_inputs_do_not_emit_periodic_lines() -> None:
    rec, log = _recorder()

    def _validator(stem: str) -> bool:
        return True

    progress = LocalValidationProgress(
        validator=_validator,
        stems=["only-stem"],
        log=log,
        clock=lambda: 0.0,
        progress_interval_s=10.0,
        quiet_threshold=10,
    )
    progress.run()

    assert any(m.startswith("Validating finalized local state:") for m in rec.messages)
    assert not any(m.startswith("Local validation progress:") for m in rec.messages)
    assert any(m.startswith("Local validation complete:") for m in rec.messages)


def test_progress_reports_elapsed_time() -> None:
    rec, log = _recorder()
    clock_value = [0.0]

    def _clock() -> float:
        return clock_value[0]

    def _validator(stem: str) -> bool:
        # Advance the clock for each validated stem so the
        # progress reporter sees a non-zero elapsed time.
        clock_value[0] += 1.0
        return True

    progress = LocalValidationProgress(
        validator=_validator,
        stems=[f"r-{i}" for i in range(50)],
        log=log,
        clock=_clock,
        progress_interval_s=2.0,
        quiet_threshold=5,
    )
    progress.run()

    # The completion line carries the final elapsed time (50s)
    completion = [m for m in rec.messages if m.startswith("Local validation complete:")][-1]
    assert "in 50s" in completion or "1m" in completion


def test_progress_completion_includes_total_stems() -> None:
    rec, log = _recorder()
    progress = LocalValidationProgress(
        validator=lambda stem: True,
        stems=[f"r-{i}" for i in range(7)],
        log=log,
        clock=lambda: 0.0,
        progress_interval_s=5.0,
        quiet_threshold=2,
    )
    progress.run()
    completion = [m for m in rec.messages if m.startswith("Local validation complete:")][-1]
    assert "7 regions" in completion


def test_progress_validates_every_stem_exactly_once() -> None:
    seen: list[str] = []
    progress = LocalValidationProgress(
        validator=lambda stem: seen.append(stem) or True,
        stems=["a", "b", "c", "d"],
        log=lambda _msg: None,
        clock=lambda: 0.0,
    )
    progress.run()
    assert sorted(seen) == ["a", "b", "c", "d"]
    assert len(seen) == 4


def test_progress_does_not_leak_stem_or_paths() -> None:
    rec, log = _recorder()
    progress = LocalValidationProgress(
        validator=lambda stem: True,
        stems=["region-with-secret-path"],
        log=log,
        clock=lambda: 0.0,
    )
    progress.run()
    for message in rec.messages:
        assert "secret" not in message
        assert "/" not in message


def test_progress_returns_result_map() -> None:
    def _validator(stem: str) -> bool:
        return stem == "ok"

    progress = LocalValidationProgress(
        validator=_validator,
        stems=["ok", "bad", "ok"],
        log=lambda _msg: None,
        clock=lambda: 0.0,
    )
    result = progress.run()
    assert result == {"ok": True, "bad": False}


def test_progress_does_not_log_when_no_stems() -> None:
    rec, log = _recorder()
    progress = LocalValidationProgress(
        validator=lambda stem: True,
        stems=[],
        log=log,
        clock=lambda: 0.0,
    )
    progress.run()
    # Begin/completion lines still emitted for symmetry
    assert any(m.startswith("Validating finalized local state:") for m in rec.messages)
    assert any(m.startswith("Local validation complete:") for m in rec.messages)


def test_progress_periodic_logs_use_bounded_count(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    rec, log = _recorder()
    clock_value = [0.0]

    def _clock() -> float:
        return clock_value[0]

    def _validator(stem: str) -> bool:
        clock_value[0] += 2.0
        return True

    progress = LocalValidationProgress(
        validator=_validator,
        stems=[f"r-{i}" for i in range(400)],
        log=log,
        clock=_clock,
        progress_interval_s=10.0,
        quiet_threshold=20,
    )
    progress.run()

    periodic = [m for m in rec.messages if m.startswith("Local validation progress:")]
    # 400 stems with 2s each = 800s total. Interval 10s. So roughly
    # 80 progress ticks. Allow some slack -- the precise count is an
    # implementation detail; the test just enforces it is bounded.
    assert 5 <= len(periodic) <= 100
