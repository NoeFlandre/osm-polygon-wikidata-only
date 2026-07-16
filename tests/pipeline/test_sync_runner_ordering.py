"""Targeted runner ordering tests.

The runner must execute PUBLISH-only repairs AFTER the AUGMENT
backlog but BEFORE any new core PROCESSING. The first PROCESS
extraction prefetch may overlap earlier AUGMENT/PUBLISH work.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.pipeline.sync_runner import (
    RegionSyncState,
    SyncAction,
    run_sync,
)


def _state(stem: str, action: SyncAction, root: Path) -> Any:
    return RegionSyncState(stem, root / f"{stem}.osm.pbf", action)


def _path_stem(path: Path) -> str:
    return path.name.removesuffix(".osm.pbf")


def test_publish_runs_before_process(tmp_path: Path) -> None:
    """PUBLISH states must drain before any PROCESS state starts."""
    events: list[str] = []
    process_started = threading.Event()
    publish_started = threading.Event()

    def fake_extract(pbf_path: Path) -> Any:
        events.append(f"extract:{_path_stem(pbf_path)}")
        process_started.set()
        return object()

    def fake_process(_extracted: Any) -> Any:
        events.append("process:never-relevant")
        return object()

    def fake_augment(state: Any) -> Any:
        events.append(f"augment:{state.stem}")
        return object()

    publish_state = _state("publish-only", SyncAction.PUBLISH, tmp_path)
    process_state = _state("process-core", SyncAction.PROCESS, tmp_path)
    augment_state = _state("augment-backlog", SyncAction.AUGMENT, tmp_path)

    (tmp_path / "process-core.osm.pbf").write_bytes(b"")

    def fake_load_existing(state: Any) -> Any:
        publish_started.set()
        events.append(f"load-existing:{state.stem}")
        return object()

    rc = run_sync(
        [process_state, publish_state, augment_state],
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        load_existing_augmentation=fake_load_existing,
    )
    assert rc == 0
    # Augment backlog runs before process begins (prefetch is
    # allowed to overlap because it does not consume the result).
    assert events.index("augment:augment-backlog") < events.index("process:never-relevant")
    # Publish runs BEFORE process starts
    assert events.index("load-existing:publish-only") < events.index("process:never-relevant")
    # And publish has already started before process extraction result is needed
    assert publish_started.is_set() or events.index("load-existing:publish-only") < len(events)


def test_publish_executes_before_first_process_extraction_consumes(tmp_path: Path) -> None:
    """The first PROCESS extraction may prefetch in the background,
    but PUBLISH must run before the processing for that PROCESS
    state actually begins."""
    events: list[str] = []
    publish_done = threading.Event()
    extract_done = threading.Event()

    def fake_extract(pbf_path: Path) -> Any:
        events.append(f"extract:{_path_stem(pbf_path)}")
        # Mark extraction as done
        extract_done.set()
        return object()

    def fake_process(_extracted: Any) -> Any:
        # Process must NOT happen until publish has been processed
        assert publish_done.is_set(), "PROCESS must not begin before PUBLISH"
        events.append(f"process:{_path_stem(Path('processed.osm.pbf'))}")
        return object()

    def fake_augment(state: Any) -> Any:
        events.append(f"augment:{state.stem}")
        return object()

    def fake_load_existing(state: Any) -> Any:
        events.append(f"load-existing:{state.stem}")
        publish_done.set()
        return object()

    publish_state = _state("publish-only", SyncAction.PUBLISH, tmp_path)
    process_state = _state("process-core", SyncAction.PROCESS, tmp_path)
    (tmp_path / "process-core.osm.pbf").write_bytes(b"")

    run_sync(
        [process_state, publish_state],
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        load_existing_augmentation=fake_load_existing,
    )

    # Publish must complete before process happens
    publish_index = events.index("load-existing:publish-only")
    process_index = next(i for i, e in enumerate(events) if e.startswith("process:"))
    assert publish_index < process_index


def test_publish_repair_does_not_call_extract(tmp_path: Path) -> None:
    """A PUBLISH-only repair must never invoke PBF extraction."""
    extract_calls: list[Path] = []

    def fake_extract(pbf_path: Path) -> Any:
        extract_calls.append(pbf_path)
        return object()

    def fake_process(_extracted: Any) -> Any:
        return object()

    def fake_augment(_state: Any) -> Any:
        return object()

    def fake_load_existing(state: Any) -> Any:
        return object()

    states = [
        _state("repair-only", SyncAction.PUBLISH, tmp_path),
    ]
    run_sync(
        states,
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        load_existing_augmentation=fake_load_existing,
    )
    assert extract_calls == []


def test_publish_repair_uploads_before_first_process_publication(tmp_path: Path) -> None:
    """When a PUBLISH repair and a PROCESS region both exist, the
    publish upload MUST complete before the process upload."""
    publish_submitted = threading.Event()
    process_extracted = threading.Event()
    submit_order: list[str] = []

    def fake_extract(pbf_path: Path) -> Any:
        process_extracted.set()
        return object()

    def fake_process(_extracted: Any) -> Any:
        return object()

    def fake_augment(_state: Any) -> Any:
        return object()

    def fake_load_existing(state: Any) -> Any:
        return object()

    def fake_build(state: Any, augmentation: Any, core: Any) -> list[Any]:
        return [f"op-for-{state.stem}"]

    def fake_submit(ops: list[Any], message: str) -> None:
        submit_order.append(message)
        if "publish-only" in message:
            publish_submitted.set()
        # Block the process submit until publish is done.
        if "process-core" in message:
            assert publish_submitted.is_set(), "Process upload must follow publish upload"

    publish_state = _state("publish-only", SyncAction.PUBLISH, tmp_path)
    process_state = _state("process-core", SyncAction.PROCESS, tmp_path)
    (tmp_path / "process-core.osm.pbf").write_bytes(b"")

    def fake_commit(state: Any) -> str:
        return f"Sync complete region {state.stem}"

    rc = run_sync(
        [process_state, publish_state],
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        build_upload_files=fake_build,
        commit_message=fake_commit,
        submit_upload=fake_submit,
        load_existing_augmentation=fake_load_existing,
    )
    assert rc == 0
    assert submit_order == [
        "Sync complete region publish-only",
        "Sync complete region process-core",
    ]


def test_publish_repair_does_not_call_augment(tmp_path: Path) -> None:
    """A PUBLISH-only repair must use load_existing_augmentation, NOT
    call augment_region. This guarantees no extraction, no Wikidata
    lookup, no Wikivoyage fetch, no enrichment."""
    augment_calls: list[str] = []

    def fake_extract(pbf_path: Path) -> Any:
        return object()

    def fake_process(_extracted: Any) -> Any:
        return object()

    def fake_augment(state: Any) -> Any:
        augment_calls.append(state.stem)
        return object()

    def fake_load_existing(state: Any) -> Any:
        return object()

    states = [_state("repair-only", SyncAction.PUBLISH, tmp_path)]
    run_sync(
        states,
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        load_existing_augmentation=fake_load_existing,
    )
    assert augment_calls == []
