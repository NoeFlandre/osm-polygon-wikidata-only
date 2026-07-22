from __future__ import annotations

import threading
import time

import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery import repair as repair_mod
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.checkpoints import (
    RecoveryBatchArtifacts,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.repair import (
    _execute_recovery_batches,
)
from osm_polygon_wikidata_only.utils import retry as retry_mod


class _MemoryCheckpointStore:
    def __init__(self) -> None:
        self.saved: dict[int, RecoveryBatchArtifacts] = {}

    def load(self, index: int, expected_qids: tuple[str, ...]) -> RecoveryBatchArtifacts | None:
        artifact = self.saved.get(index)
        return artifact if artifact is not None and artifact.qids == expected_qids else None

    def save(self, index: int, artifacts: RecoveryBatchArtifacts) -> None:
        self.saved[index] = artifacts


def _artifact(qids: tuple[str, ...]) -> RecoveryBatchArtifacts:
    return RecoveryBatchArtifacts(qids, (), (), ())


def test_recovery_keeps_three_independent_batches_active_and_returns_input_order() -> None:
    qids = tuple(f"Q{index}" for index in range(1, 77))
    store = _MemoryCheckpointStore()
    three_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    starts: list[tuple[str, ...]] = []

    def build(batch_qids: tuple[str, ...], _progress: object) -> RecoveryBatchArtifacts:
        nonlocal active, maximum_active
        with lock:
            starts.append(batch_qids)
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 3:
                three_started.set()
        assert release.wait(timeout=2), "test did not release active recovery batches"
        with lock:
            active -= 1
        return _artifact(batch_qids)

    result: list[RecoveryBatchArtifacts] = []

    def execute() -> None:
        result.extend(
            _execute_recovery_batches(
                stem="region-latest",
                affected_qids=qids,
                checkpoint_store=store,  # type: ignore[arg-type]
                build_batch=build,  # type: ignore[arg-type]
                emit=lambda _message: None,
            )
        )

    thread = threading.Thread(target=execute)
    thread.start()
    assert three_started.wait(timeout=2), "three-batch recovery window was not filled"
    release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert maximum_active == 3
    assert len(starts) == 4
    assert [artifact.qids for artifact in result] == [
        qids[0:25],
        qids[25:50],
        qids[50:75],
        qids[75:76],
    ]
    assert sorted(store.saved) == [0, 1, 2, 3]


def test_recovery_reuses_checkpoints_without_rebuilding_them() -> None:
    qids = tuple(f"Q{index}" for index in range(1, 52))
    store = _MemoryCheckpointStore()
    store.saved[0] = _artifact(qids[:25])
    built: list[tuple[str, ...]] = []

    def build(batch_qids: tuple[str, ...], _progress: object) -> RecoveryBatchArtifacts:
        built.append(batch_qids)
        return _artifact(batch_qids)

    messages: list[str] = []
    result = _execute_recovery_batches(
        stem="region-latest",
        affected_qids=qids,
        checkpoint_store=store,  # type: ignore[arg-type]
        build_batch=build,  # type: ignore[arg-type]
        emit=messages.append,
    )

    assert sorted(built) == sorted([qids[25:50], qids[50:51]])
    assert [artifact.qids for artifact in result] == [qids[:25], qids[25:50], qids[50:51]]
    assert any("batch 1/3 reused durable checkpoint" in message for message in messages)


def test_recovery_window_size_is_configurable_for_deterministic_benchmarks() -> None:
    qids = tuple(f"Q{index}" for index in range(1, 77))
    store = _MemoryCheckpointStore()
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def build(batch_qids: tuple[str, ...], _progress: object) -> RecoveryBatchArtifacts:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        with lock:
            active -= 1
        return _artifact(batch_qids)

    _execute_recovery_batches(
        stem="region-latest",
        affected_qids=qids,
        checkpoint_store=store,  # type: ignore[arg-type]
        build_batch=build,  # type: ignore[arg-type]
        emit=lambda _message: None,
        batch_window=1,
    )

    assert maximum_active == 1


def test_completed_batch_is_durable_while_slow_sibling_is_still_running() -> None:
    qids = tuple(f"Q{index}" for index in range(1, 51))
    store = _MemoryCheckpointStore()
    slow_started = threading.Event()
    release_slow = threading.Event()
    fast_saved = threading.Event()
    original_save = store.save

    def save(index: int, artifacts: RecoveryBatchArtifacts) -> None:
        original_save(index, artifacts)
        if index == 0:
            fast_saved.set()

    store.save = save  # type: ignore[method-assign]

    def build(batch_qids: tuple[str, ...], _progress: object) -> RecoveryBatchArtifacts:
        if batch_qids == qids[25:50]:
            slow_started.set()
            assert release_slow.wait(timeout=2), "test did not release slow sibling"
        return _artifact(batch_qids)

    thread = threading.Thread(
        target=lambda: _execute_recovery_batches(
            stem="region-latest",
            affected_qids=qids,
            checkpoint_store=store,  # type: ignore[arg-type]
            build_batch=build,  # type: ignore[arg-type]
            emit=lambda _message: None,
        )
    )
    thread.start()
    try:
        assert slow_started.wait(timeout=1)
        assert fast_saved.wait(timeout=1), "fast batch waited for its slow sibling"
        assert 0 in store.saved
        assert 1 not in store.saved
    finally:
        release_slow.set()
        thread.join(timeout=2)

    assert not thread.is_alive()


def test_interrupted_recovery_cancels_worker_retry_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retry_started = threading.Event()

    def build(batch_qids: tuple[str, ...], _progress: object) -> RecoveryBatchArtifacts:
        return retry_mod.with_retries(
            lambda: (_ for _ in ()).throw(OSError("offline")),
            attempts=2,
            base_delay=2,
            retry_on=(OSError,),
            on_retry=lambda *_args: retry_started.set(),
        )

    def interrupt(_futures: object) -> list[object]:
        assert retry_started.wait(timeout=1)
        raise KeyboardInterrupt

    monkeypatch.setattr(repair_mod, "as_completed", interrupt)
    started_at = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _execute_recovery_batches(
            stem="region-latest",
            affected_qids=tuple(f"Q{index}" for index in range(1, 51)),
            checkpoint_store=_MemoryCheckpointStore(),  # type: ignore[arg-type]
            build_batch=build,  # type: ignore[arg-type]
            emit=lambda _message: None,
        )

    assert time.monotonic() - started_at < 1
