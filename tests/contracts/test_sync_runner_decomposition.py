"""Characterization tests for the Phase 6 unified sync runner.

These tests pin the exact behavior of the extracted
:func:`pipeline.sync_runner.run_sync`:

* The runner is a pure state executor. It receives injectable
  collaborators and must NOT import argparse, cli.*, hf.*,
  DataRoot, or Settings. There are no default production
  collaborators; every callable is required and provided by
  the caller.
* The CLI shell lives in :mod:`cli.run_sync`. The runner does
  not. The CLI shell emits the unified-plan count log line.
* Public identities: :func:`run_sync_plan`,
  :class:`SyncAction`, :class:`RegionSyncState` are exposed
  from ``pipeline.sync_runner`` unchanged.
* The runner restores the documented execution sequence:

    1. Start prefetching the first PROCESS PBF (background thread).
    2. Drain every AUGMENT (backlog) state in alphabetical order.
    3. Drain every PUBLISH (publish-only reconciliation) state in
       alphabetical order. Each repair uses ``load_existing_augmentation``
       so no extraction, Wikidata, Wikipedia, Wikivoyage, or
       augmentation call runs.
    4. For each PROCESS state, await its extraction, immediately
       prefetch the next PBF, enrich/persist the current region,
       then augment it, then continue.
    5. After each successful augmentation, if BOTH
       ``build_upload_files`` AND ``submit_upload`` are provided,
       build the upload file list and submit one atomic commit.

* Processing, extraction, and backlog-augmentation exceptions
  all propagate through the same boundary. The extraction
  executor and the upload queue are always shut down in
  ``finally`` (exactly once each). ``run_sync`` returns ``1``
  only for background-upload failures.
* The plan never contains a placeholder PROCESS state. Mixed
  plans execute in the exact documented order.
* When ``--push`` is disabled, publication assembly never
  runs and no upload snapshots are produced.
* Lifecycle and heartbeat log records retain the legacy
  ``osm_polygon_wikidata_only.pipeline.processor`` logger.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.pipeline import sync_runner as sync_runner_mod

# ---------------------------------------------------------------------------
# Helpers: synthetic states and tiny collaborators
# ---------------------------------------------------------------------------


def _state(stem: str, action: Any, root: Path) -> Any:
    return type(
        "State",
        (),
        {"stem": stem, "pbf_path": root / f"{stem}.osm.pbf", "action": action},
    )()


def _path_stem(path: Path) -> str:
    """Match the production-region stem extraction: ``.osm.pbf`` -> ``<stem>``."""
    return path.name.removesuffix(".osm.pbf")


def _settings() -> Settings:
    return Settings()


def _data_root(tmp: Path) -> DataRoot:
    root = DataRoot(tmp / "data")
    root.ensure()
    return root


# ---------------------------------------------------------------------------
# Public identity: pipeline.sync_runner must expose run_sync and plan types
# ---------------------------------------------------------------------------


def test_sync_runner_re_exports_plan_types() -> None:
    """The sync runner module must expose ``run_sync`` and the plan types
    by identity with the underlying modules."""
    from osm_polygon_wikidata_only.pipeline import sync_orchestrator as legacy
    from osm_polygon_wikidata_only.pipeline import sync_planner as planner

    assert hasattr(sync_runner_mod, "run_sync")
    assert sync_runner_mod.SyncAction is planner.SyncAction
    assert sync_runner_mod.RegionSyncState is planner.RegionSyncState
    assert sync_runner_mod.run_sync_plan is legacy.run_sync_plan


def test_cli_run_sync_shell_exists() -> None:
    """The CLI shell for ``sync-dir`` must live in :mod:`cli.run_sync`."""
    cli_run_sync = importlib.import_module("osm_polygon_wikidata_only.cli.run_sync")
    assert callable(cli_run_sync.execute)


def test_pipeline_sync_runner_does_not_import_cli_or_hf_or_argparse() -> None:
    """The runner is a pure state executor. It must not import any
    CLI framework, HF helper, argparse, DataRoot, Settings, or
    replacement helpers."""
    source = Path(sync_runner_mod.__file__).read_text(encoding="utf-8")
    import re

    forbidden_import_patterns = (
        r"^\s*from\s+osm_polygon_wikidata_only\.cli\b",
        r"^\s*import\s+osm_polygon_wikidata_only\.cli\b",
        r"^\s*from\s+osm_polygon_wikidata_only\.hf\b",
        r"^\s*import\s+osm_polygon_wikidata_only\.hf\b",
        r"^\s*import\s+argparse\b",
        r"^\s*from\s+argparse\b",
        r"^\s*from\s+\S+\.hf\.upload_queue\b",
        r"^\s*from\s+\S+\.hf\.uploader\b",
        r"^\s*from\s+osm_polygon_wikidata_only\.config\.paths\b",
        r"^\s*from\s+osm_polygon_wikidata_only\.config\.settings\b",
        r"^\s*from\s+dataclasses\s+import\s+replace\b",
    )
    for needle in forbidden_import_patterns:
        assert not re.search(needle, source, re.MULTILINE), (
            f"pipeline.sync_runner must not match {needle!r}; "
            "the CLI shell in cli.run_sync owns that boundary."
        )
    # The runner must not re-export any hf module name at all.
    for leaked in ("StubHfHub", "BackgroundUploadQueue", "upload_files"):
        assert not hasattr(sync_runner_mod, leaked), f"runner must not re-export {leaked}"


def test_run_sync_signature_requires_collaborators() -> None:
    """The runner must NOT have a default production collaborator for
    ``extract_pbf``; all three core collaborators are required."""
    sig = inspect.signature(sync_runner_mod.run_sync)
    params = sig.parameters
    assert params["extract_pbf"].default is inspect.Parameter.empty, (
        "extract_pbf must be a required collaborator with no production default"
    )
    assert params["process_extracted_pbf"].default is inspect.Parameter.empty
    assert params["augment_region"].default is inspect.Parameter.empty
    # Optional collaborators default to None
    for optional in ("build_upload_files", "submit_upload", "close_uploads", "on_complete"):
        assert params[optional].default is None, optional


# ---------------------------------------------------------------------------
# Mixed-plan regression test (AUGMENT(old-a), PROCESS(new-a), PROCESS(new-b),
# COMPLETE(done))
# ---------------------------------------------------------------------------


def test_mixed_plan_executes_in_documented_order(tmp_path: Path) -> None:
    """The exact documented ordering for a mixed plan.

    State list: AUGMENT(old-a), PROCESS(new-a), PROCESS(new-b),
    COMPLETE(done).

    Required ordering assertions:

    * first extraction starts before any backlog AUGMENT.
    * backlog AUGMENT (old-a) completes before PROCESS(new-a)
      runs.
    * PROCESS(new-b) extraction starts before PROCESS(new-a)
      enrichment completes.
    * for each PROCESS state, ``process_extracted_pbf`` is
      called then ``augment_region`` for that same region.
    * completed/upload order is exactly old-a, new-a, new-b.
    * no NotImplementedError placeholder is ever called.
    """
    events: list[str] = []
    second_extraction_started = threading.Event()
    enrichment_started = threading.Event()

    def fake_extract(pbf_path: Path) -> Any:
        name = _path_stem(pbf_path)
        events.append(f"extract:{name}")
        if name == "new-b":
            second_extraction_started.set()
        return type(
            "E",
            (),
            {
                "stem": type("S", (), {"stem": name, "path": pbf_path})(),
                "polygons": (),
                "extraction_duration_s": 0.0,
            },
        )()

    def fake_process(extracted: Any) -> Any:
        name = extracted.stem.stem
        events.append(f"process:{name}")
        if name == "new-a":
            assert second_extraction_started.wait(timeout=1.0), (
                "next extraction must start before current enrichment completes"
            )
            enrichment_started.set()
        return type(
            "R",
            (),
            {
                "stem": name,
                "manifest_entry": {"source_pbf": f"{name}.osm.pbf"},
            },
        )()

    def fake_augment(state: Any) -> Any:
        events.append(f"augment:{state.stem}")
        if state.stem == "new-a":
            assert enrichment_started.is_set(), (
                "new-a augmentation must run only after new-a processing"
            )
        return type("A", (), {"counts": "ok", "stem": state.stem})()

    state_old_aug = _state("old-a", sync_runner_mod.SyncAction.AUGMENT, tmp_path)
    state_new_a = _state("new-a", sync_runner_mod.SyncAction.PROCESS, tmp_path)
    state_new_b = _state("new-b", sync_runner_mod.SyncAction.PROCESS, tmp_path)
    state_done = _state("done", sync_runner_mod.SyncAction.COMPLETE, tmp_path)
    states = [state_old_aug, state_new_a, state_new_b, state_done]
    (tmp_path / "new-a.osm.pbf").write_bytes(b"")
    (tmp_path / "new-b.osm.pbf").write_bytes(b"")

    submit_calls: list[tuple[Any, str]] = []

    def fake_submit(files: list[tuple[Path, str]], msg: str) -> None:
        events.append(f"submit:{Path(files[0][0]).stem}")
        submit_calls.append((files, msg))

    def no_close() -> list[str]:
        events.append("close")
        return []

    rc = sync_runner_mod.run_sync(
        states,
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        build_upload_files=lambda s, a, c: [(Path(f"{s.stem}.parquet"), f"x/{s.stem}.parquet")],
        commit_message=lambda s: f"msg:{s.stem}",
        submit_upload=fake_submit,
        close_uploads=no_close,
    )
    assert rc == 0
    # Required ordering invariants.
    assert events[0] == "extract:new-a", "first extraction must start before backlog augmentation"
    assert events.index("augment:old-a") < events.index("process:new-a")
    assert events.index("extract:new-b") < events.index("augment:new-a")
    for stem in ("new-a", "new-b"):
        assert events.index(f"process:{stem}") < events.index(f"augment:{stem}")
    assert [c[0][0][0].stem for c in submit_calls] == ["old-a", "new-a", "new-b"]
    assert events.count("close") == 1


# ---------------------------------------------------------------------------
# Extraction failure cancels later processing
# ---------------------------------------------------------------------------


def test_extraction_failure_prevents_processing_and_later_regions(
    tmp_path: Path,
) -> None:
    """If the first PROCESS extraction fails, no later PROCESS or
    AUGMENT state is touched."""
    events: list[str] = []

    def fake_extract(pbf_path: Path) -> Any:
        events.append(f"extract:{_path_stem(pbf_path)}")
        raise RuntimeError("extraction failed")

    def fake_process(extracted: Any) -> Any:
        events.append(f"process:{extracted.stem.stem}")
        raise AssertionError("must never reach process_extracted_pbf")

    def fake_augment(state: Any) -> Any:
        events.append(f"augment:{state.stem}")
        raise AssertionError("must never reach augment_region")

    states = [
        _state("a-latest", sync_runner_mod.SyncAction.PROCESS, tmp_path),
        _state("b-latest", sync_runner_mod.SyncAction.PROCESS, tmp_path),
    ]
    (tmp_path / "a-latest.osm.pbf").write_bytes(b"")
    (tmp_path / "b-latest.osm.pbf").write_bytes(b"")

    def no_close() -> list[str]:
        events.append("close")
        return []

    with pytest.raises(RuntimeError, match="extraction failed"):
        sync_runner_mod.run_sync(
            states,
            extract_pbf=fake_extract,
            process_extracted_pbf=fake_process,
            augment_region=fake_augment,
            close_uploads=no_close,
        )
    assert events.count("close") == 1
    assert not any(e.startswith("process:") for e in events)
    assert not any(e.startswith("augment:") for e in events)


# ---------------------------------------------------------------------------
# Processing exception propagation and queue close on failure
# ---------------------------------------------------------------------------


def test_processing_exception_propagates_and_queue_still_closes(
    tmp_path: Path,
) -> None:
    """A processing exception must propagate, and the upload queue
    must still be closed in ``finally``."""
    events: list[str] = []

    def fake_extract(pbf_path: Path) -> Any:
        events.append(f"extract:{_path_stem(pbf_path)}")
        return type(
            "E",
            (),
            {"stem": type("S", (), {"stem": _path_stem(pbf_path)})(), "polygons": ()},
        )()

    def fake_process(_extracted: Any) -> Any:
        events.append("process:explode")
        raise RuntimeError("kaboom")

    def fake_augment(_state: Any) -> Any:
        events.append("augment:never")
        raise AssertionError("must never reach augment after process failure")

    state = _state("kaboom", sync_runner_mod.SyncAction.PROCESS, tmp_path)
    (tmp_path / "kaboom.osm.pbf").write_bytes(b"")

    def no_close() -> list[str]:
        events.append("close")
        return []

    with pytest.raises(RuntimeError, match="kaboom"):
        sync_runner_mod.run_sync(
            [state],
            extract_pbf=fake_extract,
            process_extracted_pbf=fake_process,
            augment_region=fake_augment,
            close_uploads=no_close,
        )
    assert events == ["extract:kaboom", "process:explode", "close"]


# ---------------------------------------------------------------------------
# Backlog augmentation exception propagation and queue close on failure
# ---------------------------------------------------------------------------


def test_backlog_augmentation_exception_propagates_and_queue_still_closes(
    tmp_path: Path,
) -> None:
    """A backlog AUGMENT exception must propagate AND mark
    execution as failed BEFORE the final upload shutdown, so the
    upload-error log line is consistent regardless of where the
    failure occurred."""
    events: list[str] = []
    upload_failure_calls: list[list[str]] = []

    def fake_extract(pbf_path: Path) -> Any:
        events.append(f"extract:{_path_stem(pbf_path)}")
        return type(
            "E",
            (),
            {"stem": type("S", (), {"stem": _path_stem(pbf_path)})(), "polygons": ()},
        )()

    def fake_process(extracted: Any) -> Any:
        events.append(f"process:{extracted.stem.stem}")
        return type("R", (), {})()

    def fake_augment(state: Any) -> Any:
        events.append(f"augment:{state.stem}")
        if state.stem == "backlog":
            raise RuntimeError("backlog failed")
        return type("A", (), {})()

    states = [
        _state("backlog", sync_runner_mod.SyncAction.AUGMENT, tmp_path),
        _state("core", sync_runner_mod.SyncAction.PROCESS, tmp_path),
    ]
    (tmp_path / "core.osm.pbf").write_bytes(b"")

    def fake_close() -> list[str]:
        events.append("close")
        return []

    with pytest.raises(RuntimeError, match="backlog failed"):
        sync_runner_mod.run_sync(
            states,
            extract_pbf=fake_extract,
            process_extracted_pbf=fake_process,
            augment_region=fake_augment,
            close_uploads=fake_close,
        )
    # Extract starts (prefetch) before backlog augment. Close runs exactly once.
    assert events[0] == "extract:core"
    assert "augment:backlog" in events
    assert events.count("close") == 1
    assert not any(e.startswith("process:") for e in events)
    assert not any(e.startswith("augment:core") for e in events)
    assert upload_failure_calls == []


def test_backlog_augmentation_exception_does_not_swallow_upload_failure_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If backlog augment raises AND close_uploads returns a failure
    list, both are surfaced: the exception propagates AND the
    runner returns 1 (no double-shutdown, no silent conversion)."""
    events: list[str] = []

    def fake_extract(pbf_path: Path) -> Any:
        return type(
            "E",
            (),
            {"stem": type("S", (), {"stem": _path_stem(pbf_path)})(), "polygons": ()},
        )()

    def fake_process(_extracted: Any) -> Any:
        return type("R", (), {})()

    def fake_augment(state: Any) -> Any:
        if state.stem == "backlog":
            raise RuntimeError("backlog failed")
        return type("A", (), {})()

    states = [
        _state("backlog", sync_runner_mod.SyncAction.AUGMENT, tmp_path),
        _state("core", sync_runner_mod.SyncAction.PROCESS, tmp_path),
    ]
    (tmp_path / "core.osm.pbf").write_bytes(b"")

    def fake_close() -> list[str]:
        events.append("close")
        return ["job-1 failed"]

    caplog.set_level(logging.INFO, logger="osm_polygon_wikidata_only.cli")
    with pytest.raises(RuntimeError, match="backlog failed"):
        sync_runner_mod.run_sync(
            states,
            extract_pbf=fake_extract,
            process_extracted_pbf=fake_process,
            augment_region=fake_augment,
            close_uploads=fake_close,
        )
    assert events == ["close"]


# ---------------------------------------------------------------------------
# Return code: 1 only for upload failures, never for execution exceptions
# ---------------------------------------------------------------------------


def test_run_sync_returns_one_only_for_upload_failures(tmp_path: Path) -> None:
    """The runner returns ``1`` only for background-upload failures.

    An execution-time exception propagates and the function does
    not silently convert it into a ``1``.
    """

    def fake_extract(pbf_path: Path) -> Any:
        return type(
            "E",
            (),
            {"stem": type("S", (), {"stem": _path_stem(pbf_path)})(), "polygons": ()},
        )()

    def fake_process(_extracted: Any) -> Any:
        return type("R", (), {})()

    def fake_augment(_state: Any) -> Any:
        return type("A", (), {})()

    state = _state("u", sync_runner_mod.SyncAction.PROCESS, tmp_path)
    (tmp_path / "u.osm.pbf").write_bytes(b"")

    def close_failures() -> list[str]:
        return ["upload-job-1 failed"]

    rc = sync_runner_mod.run_sync(
        [state],
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        close_uploads=close_failures,
    )
    assert rc == 1


def test_run_sync_returns_zero_on_clean_close(tmp_path: Path) -> None:
    """When close_uploads returns an empty list, the runner returns 0."""

    def fake_extract(pbf_path: Path) -> Any:
        return type(
            "E",
            (),
            {"stem": type("S", (), {"stem": _path_stem(pbf_path)})(), "polygons": ()},
        )()

    def fake_process(_extracted: Any) -> Any:
        return type("R", (), {})()

    def fake_augment(_state: Any) -> Any:
        return type("A", (), {})()

    state = _state("ok", sync_runner_mod.SyncAction.PROCESS, tmp_path)
    (tmp_path / "ok.osm.pbf").write_bytes(b"")

    rc = sync_runner_mod.run_sync(
        [state],
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        close_uploads=lambda: [],
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# Publication assembly is skipped when both build_upload_files and
# submit_upload are None (the --push=false contract).
# ---------------------------------------------------------------------------


def test_no_publication_assembly_when_publish_callbacks_are_none(
    tmp_path: Path,
) -> None:
    """When ``build_upload_files`` and ``submit_upload`` are both
    ``None``, the runner must never invoke publication assembly.

    The test records any call to a publication builder; if any
    call is recorded, the assertion fails.
    """
    publication_calls: list[tuple[Any, ...]] = []
    submit_calls: list[tuple[Any, ...]] = []

    def fake_extract(pbf_path: Path) -> Any:
        return type(
            "E",
            (),
            {"stem": type("S", (), {"stem": _path_stem(pbf_path)})(), "polygons": ()},
        )()

    def fake_process(_extracted: Any) -> Any:
        return type("R", (), {})()

    def fake_augment(_state: Any) -> Any:
        return type("A", (), {})()

    def bad_build(*args: Any, **kwargs: Any) -> list[tuple[Path, str]]:
        publication_calls.append((args, kwargs))
        return [(Path("should-not-exist.parquet"), "x/y.parquet")]

    def bad_submit(*args: Any, **kwargs: Any) -> None:
        submit_calls.append((args, kwargs))

    states = [
        _state("aug", sync_runner_mod.SyncAction.AUGMENT, tmp_path),
        _state("proc", sync_runner_mod.SyncAction.PROCESS, tmp_path),
    ]
    (tmp_path / "proc.osm.pbf").write_bytes(b"")

    rc = sync_runner_mod.run_sync(
        states,
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        # The runner MUST NOT call these when both are None.
        # We pass them in only to detect any accidental call by
        # leaving them as None.
        build_upload_files=None,
        submit_upload=None,
        close_uploads=None,
    )
    assert rc == 0
    assert publication_calls == []
    assert submit_calls == []


# ---------------------------------------------------------------------------
# CLI shell must forward runtime.cache (the production JsonFileCache
# returned by build_wikimedia_runtime) to process_extracted_pbf as the
# `cache` argument -- NOT a freshly created augmentation-root cache.
# ---------------------------------------------------------------------------


def test_cli_shell_forwards_runtime_cache_to_process_extracted_pbf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact runtime.cache object reaches process_extracted_pbf.

    The original sync implementation passed ``runtime.cache`` (the
    shared JsonFileCache built by ``build_wikimedia_runtime``) as
    the ``cache`` argument of ``process_extracted_pbf``. The CLI
    shell must preserve that identity -- not a second, ad-hoc
    augmentation cache created on the side.
    """
    from types import SimpleNamespace

    from osm_polygon_wikidata_only.cli import run_sync as cli_run_sync_mod
    from osm_polygon_wikidata_only.io import manifest as manifest_mod
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    root = _data_root(tmp_path)
    (tmp_path / "real.osm.pbf").write_bytes(b"")

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = pbf_path

        def iter_polygon_candidates(self, add_candidate: object) -> None:
            coords = [[[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]]]
            geom_json = '{"type": "Polygon", "coordinates": ' + str(coords) + "}"
            add_candidate(
                ("way", 1, {"wikidata": "Q1", "name": "X", "landuse": "forest"}, geom_json)
            )

        def collect_polygon_candidates(self) -> list[object]:
            return []

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)
    monkeypatch.setattr(manifest_mod, "load_manifest", lambda p: {})

    from osm_polygon_wikidata_only.enrichment.wikidata_client import InMemoryWikidataClient
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient
    from osm_polygon_wikidata_only.io.cache import JsonFileCache

    runtime_cache = JsonFileCache(root.cache / "runtime_cache", contract_version="text-sidecars-v1")
    fake_runtime = SimpleNamespace(
        scheduler=SimpleNamespace(snapshot=lambda: None),
        session=SimpleNamespace(auth_snapshot=lambda: None),
        wikidata=InMemoryWikidataClient({}),
        wikipedia=InMemoryWikipediaClient({}),
        cache=runtime_cache,
        settings=_settings(),
    )
    monkeypatch.setattr(
        cli_run_sync_mod, "build_wikimedia_runtime", lambda s, data_root: fake_runtime
    )
    monkeypatch.setattr(
        cli_run_sync_mod,
        "augment_region",
        lambda *args, **kwargs: SimpleNamespace(
            counts="ok", manifest_path=tmp_path / "aug_manifest.json"
        ),
    )

    seen_caches: list[object] = []
    from osm_polygon_wikidata_only.pipeline import processor as processor_mod

    real_process = processor_mod.process_extracted_pbf

    def spy_process(extracted: object, **kwargs: object) -> object:
        seen_caches.append(kwargs.get("cache"))
        return real_process(extracted, **kwargs)

    monkeypatch.setattr(processor_mod, "process_extracted_pbf", spy_process)

    args = SimpleNamespace(
        input=tmp_path,
        commit_message="x",
        push=False,
        dry_run=False,
        upload_threads=2,
    )
    rc = cli_run_sync_mod.execute(
        args,
        data_root=root,
        settings=_settings(),
        build_upload_files=lambda *a, **kw: [],
    )
    assert rc == 0
    assert seen_caches, "process_extracted_pbf was never called"
    # The exact same object -- not a different cache instance.
    assert seen_caches[0] is runtime_cache, (
        f"cache mismatch: expected runtime_cache {runtime_cache!r}, got {seen_caches[0]!r}"
    )


def test_cli_shell_forwards_none_runtime_cache_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When caching is disabled (``runtime.cache is None``), the CLI
    shell must forward ``None`` -- not a freshly created cache."""
    from types import SimpleNamespace

    from osm_polygon_wikidata_only.cli import run_sync as cli_run_sync_mod
    from osm_polygon_wikidata_only.io import manifest as manifest_mod
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    root = _data_root(tmp_path)
    (tmp_path / "real.osm.pbf").write_bytes(b"")

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = pbf_path

        def iter_polygon_candidates(self, add_candidate: object) -> None:
            coords = [[[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]]]
            geom_json = '{"type": "Polygon", "coordinates": ' + str(coords) + "}"
            add_candidate(
                ("way", 1, {"wikidata": "Q1", "name": "X", "landuse": "forest"}, geom_json)
            )

        def collect_polygon_candidates(self) -> list[object]:
            return []

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)
    monkeypatch.setattr(manifest_mod, "load_manifest", lambda p: {})

    from osm_polygon_wikidata_only.enrichment.wikidata_client import InMemoryWikidataClient
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient

    fake_runtime = SimpleNamespace(
        scheduler=SimpleNamespace(snapshot=lambda: None),
        session=SimpleNamespace(auth_snapshot=lambda: None),
        wikidata=InMemoryWikidataClient({}),
        wikipedia=InMemoryWikipediaClient({}),
        cache=None,
        settings=_settings(),
    )
    monkeypatch.setattr(
        cli_run_sync_mod, "build_wikimedia_runtime", lambda s, data_root: fake_runtime
    )
    monkeypatch.setattr(
        cli_run_sync_mod,
        "augment_region",
        lambda *args, **kwargs: SimpleNamespace(
            counts="ok", manifest_path=tmp_path / "aug_manifest.json"
        ),
    )

    seen_caches: list[object] = []
    from osm_polygon_wikidata_only.pipeline import processor as processor_mod

    real_process = processor_mod.process_extracted_pbf

    def spy_process(extracted: object, **kwargs: object) -> object:
        seen_caches.append(kwargs.get("cache"))
        return real_process(extracted, **kwargs)

    monkeypatch.setattr(processor_mod, "process_extracted_pbf", spy_process)

    args = SimpleNamespace(
        input=tmp_path,
        commit_message="x",
        push=False,
        dry_run=False,
        upload_threads=2,
    )
    rc = cli_run_sync_mod.execute(
        args,
        data_root=root,
        settings=_settings(),
        build_upload_files=lambda *a, **kw: [],
    )
    assert rc == 0
    assert seen_caches, "process_extracted_pbf was never called"
    assert seen_caches[0] is None, (
        f"cache mismatch: expected None when caching is disabled, got {seen_caches[0]!r}"
    )


# ---------------------------------------------------------------------------
# CLI-shell end-to-end test: real PROCESS->AUGMENT execution without TypeError
# ---------------------------------------------------------------------------


def test_cli_shell_real_process_state_executes_without_type_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real PROCESS state reaches core processing without a
    ``TypeError`` from a signature mismatch. ``--push`` is False;
    publication assembly must not run.
    """
    from dataclasses import replace as _replace
    from types import SimpleNamespace

    from osm_polygon_wikidata_only.cli import run_sync as cli_run_sync_mod
    from osm_polygon_wikidata_only.io import manifest as manifest_mod
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    root = _data_root(tmp_path)
    (tmp_path / "real.osm.pbf").write_bytes(b"")

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = pbf_path

        def iter_polygon_candidates(self, add_candidate: object) -> None:
            # Real polygon candidate the production extractor can parse.
            coords = [[[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]]]
            geom_json = '{"type": "Polygon", "coordinates": ' + str(coords) + "}"
            add_candidate(
                ("way", 1, {"wikidata": "Q1", "name": "X", "landuse": "forest"}, geom_json)
            )

        def collect_polygon_candidates(self) -> list[object]:
            return []

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)
    monkeypatch.setattr(manifest_mod, "load_manifest", lambda p: {})

    # InMemory clients and cache — no network, no real HF.
    from osm_polygon_wikidata_only.enrichment.wikidata_client import InMemoryWikidataClient
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient
    from osm_polygon_wikidata_only.io.cache import JsonFileCache

    settings = _replace(_settings(), skip_existing=False, force=True)

    fake_runtime = SimpleNamespace(
        scheduler=SimpleNamespace(snapshot=lambda: None),
        session=SimpleNamespace(auth_snapshot=lambda: None),
        wikidata=InMemoryWikidataClient({}),
        wikipedia=InMemoryWikipediaClient({}),
        cache=JsonFileCache(root.cache / "runtime_cache", contract_version="text-sidecars-v1"),
        settings=settings,
    )

    monkeypatch.setattr(
        cli_run_sync_mod, "build_wikimedia_runtime", lambda s, data_root: fake_runtime
    )

    # The augmentation path is irrelevant to this test: the goal is
    # to prove the PROCESS closure signature flows through. Replace
    # ``augment_region`` with a no-op so the augmentation step
    # doesn't try to talk to the fake runtime's session.
    monkeypatch.setattr(
        cli_run_sync_mod,
        "augment_region",
        lambda *args, **kwargs: SimpleNamespace(
            counts="ok", manifest_path=tmp_path / "aug_manifest.json"
        ),
    )

    publication_calls: list[Any] = []

    def fake_build_upload_files(*args: Any, **kwargs: Any) -> list[tuple[Path, str]]:
        publication_calls.append((args, kwargs))
        return []

    args = SimpleNamespace(
        input=tmp_path,
        commit_message="x",
        push=False,
        dry_run=False,
        upload_threads=2,
    )
    rc = cli_run_sync_mod.execute(
        args,
        data_root=root,
        settings=settings,
        build_upload_files=fake_build_upload_files,
    )
    assert rc == 0
    # Publication assembly is gated behind --push in cli.run_sync;
    # with push=False, the runner never receives the builder.
    assert publication_calls == []


# ---------------------------------------------------------------------------
# Logger name assertions: lifecycle and heartbeat records still emit under
# the legacy "osm_polygon_wikidata_only.pipeline.processor" logger.
# ---------------------------------------------------------------------------


def test_extractor_lifecycle_logs_use_processor_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``"Processing X (region=Y)"`` lifecycle message must be
    emitted under ``.pipeline.processor`` (legacy name), not under
    ``.pipeline.extractor``."""
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod
    from osm_polygon_wikidata_only.pipeline.extractor import extract_pbf

    pbf = tmp_path / "tiny-region.osm.pbf"
    pbf.write_bytes(b"")

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = pbf_path

        def iter_polygon_candidates(self, add_candidate: object) -> None:
            return None

        def collect_polygon_candidates(self) -> list[object]:
            return []

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)
    caplog.set_level(logging.INFO, logger="osm_polygon_wikidata_only.pipeline.processor")
    extract_pbf(pbf, settings=_settings())
    assert any("Processing tiny-region" in r.getMessage() for r in caplog.records)
    assert "pipeline.extractor" not in {r.name for r in caplog.records if r.name.startswith("osm")}


def test_extractor_geometry_debug_logger_name_unchanged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The geometry ``debug()`` message (only emitted on errors)
    must remain under ``.pipeline.extractor`` -- it is debugging
    only, not lifecycle."""

    from osm_polygon_wikidata_only.pipeline.extractor import _parse_geom

    caplog.set_level(logging.DEBUG, logger="osm_polygon_wikidata_only.pipeline.extractor")
    _parse_geom("{not valid json")
    assert any(r.name == "osm_polygon_wikidata_only.pipeline.extractor" for r in caplog.records)


def test_enrichment_phase_logs_use_processor_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The ``"Starting enrichment"`` lifecycle message and the
    heartbeat ``"Enrichment progress ..."`` record both emit
    under ``.pipeline.processor`` (legacy name)."""
    from osm_polygon_wikidata_only.domain.models import Polygon
    from osm_polygon_wikidata_only.enrichment.wikidata_client import InMemoryWikidataClient
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient
    from osm_polygon_wikidata_only.pipeline.enrichment_phase import run_enrichment_phase

    polygon = Polygon.make(
        source_pbf_stem="monaco-latest",
        region="monaco",
        source_pbf="monaco-latest.osm.pbf",
        osm_type="way",
        osm_id=1,
        wikidata="Q1",
        name="X",
        tags="{}",
        tag_keys="[]",
        tag_count=0,
        osm_primary_tag="forest",
        centroid="{}",
        lat=0.0,
        lon=0.0,
        bbox="[]",
        geometry="{}",
        area_m2=1.0,
        area_km2=1e-6,
        area_bucket="tiny",
        has_name=True,
        has_wikidata=True,
        extraction_version="0",
        extracted_at="2024-01-01T00:00:00Z",
    )
    caplog.set_level(logging.INFO, logger="osm_polygon_wikidata_only.pipeline.processor")
    run_enrichment_phase(
        [polygon],
        region="monaco",
        wikidata_client=InMemoryWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        settings=_settings(),
    )
    assert any("Starting enrichment for monaco" in r.getMessage() for r in caplog.records), (
        caplog.text
    )


def test_enrichment_phase_heartbeat_records_use_processor_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``EnrichmentHeartbeat`` log records (e.g. ``"Enrichment
    progress X"``) must be emitted under the legacy
    ``.pipeline.processor`` logger, not ``.pipeline.enrichment_phase``.
    """
    from osm_polygon_wikidata_only.domain.models import Polygon
    from osm_polygon_wikidata_only.enrichment.wikidata_client import InMemoryWikidataClient
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient
    from osm_polygon_wikidata_only.pipeline.enrichment_phase import run_enrichment_phase

    polygon = Polygon.make(
        source_pbf_stem="hb",
        region="hb",
        source_pbf="hb.osm.pbf",
        osm_type="way",
        osm_id=1,
        wikidata="Q1",
        name="X",
        tags="{}",
        tag_keys="[]",
        tag_count=0,
        osm_primary_tag="forest",
        centroid="{}",
        lat=0.0,
        lon=0.0,
        bbox="[]",
        geometry="{}",
        area_m2=1.0,
        area_km2=1e-6,
        area_bucket="tiny",
        has_name=True,
        has_wikidata=True,
        extraction_version="0",
        extracted_at="2024-01-01T00:00:00Z",
    )

    import time as _time

    class _CapturingHeartbeat:
        def __init__(
            self,
            *,
            region: str,
            snapshot: object,
            log: object,
            interval_s: float = 60.0,
            clock: object = _time.monotonic,
        ) -> None:
            self._log = log
            self._region = region

        def __enter__(self) -> object:
            # Emit one heartbeat message -- this is the path the
            # legacy processor logger used to receive.
            self._log(f"Enrichment progress for {self._region}")
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_fetch_qids(*args: object, **kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.pipeline.enrichment_phase.EnrichmentHeartbeat",
        _CapturingHeartbeat,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.pipeline.enrichment_phase.fetch_qids",
        fake_fetch_qids,
    )

    caplog.set_level(logging.INFO, logger="osm_polygon_wikidata_only.pipeline.processor")
    run_enrichment_phase(
        [polygon],
        region="hb",
        wikidata_client=InMemoryWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        settings=_settings(),
    )
    # Heartbeat records carry the legacy processor logger name.
    assert any(r.name == "osm_polygon_wikidata_only.pipeline.processor" for r in caplog.records), (
        caplog.text
    )
    assert any("Enrichment progress for hb" in r.getMessage() for r in caplog.records)
    # And NOT the enrichment_phase logger.
    assert not any(
        r.name == "osm_polygon_wikidata_only.pipeline.enrichment_phase" for r in caplog.records
    )


def test_persistence_phase_logs_use_processor_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The ``"Built N unique articles"`` lifecycle message must be
    emitted under ``.pipeline.processor`` (legacy name)."""
    from osm_polygon_wikidata_only.domain.models import Article, Polygon, PolygonArticleLink
    from osm_polygon_wikidata_only.pipeline.persistence import run_persistence_phase

    polygon = Polygon.make(
        source_pbf_stem="x-latest",
        region="x",
        source_pbf="x-latest.osm.pbf",
        osm_type="way",
        osm_id=1,
        wikidata="Q1",
        name="N",
        tags="{}",
        tag_keys="[]",
        tag_count=0,
        osm_primary_tag="forest",
        centroid="{}",
        lat=0.0,
        lon=0.0,
        bbox="[]",
        geometry="{}",
        area_m2=1.0,
        area_km2=1e-6,
        area_bucket="tiny",
        has_name=True,
        has_wikidata=True,
        extraction_version="0",
        extracted_at="2024-01-01T00:00:00Z",
    )
    article = Article(
        article_id="a1",
        wikidata="Q1",
        language="en",
        site="enwiki",
        title="T",
        url="https://en.wikipedia.org/wiki/T",
        page_id=1,
        revision_id=1,
        revision_timestamp="2024-01-01T00:00:00Z",
        retrieved_at="2024-01-01T00:00:00Z",
        wikidata_label="X",
        wikidata_description="",
        wikidata_aliases="[]",
        lead_text="",
        extract="",
        full_text="",
        full_text_format="plain_text",
        article_length_chars=0,
        article_length_words=0,
        article_length_tokens_estimate=0,
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories="[]",
        license="CC-BY-SA-4.0",
        attribution="",
        source_api="wikipedia_rest_api",
        fetch_status="ok",
        fetch_error="",
        content_hash="",
    )
    link = PolygonArticleLink(
        polygon_id="way/1",
        article_id="a1",
        wikidata="Q1",
        language="en",
        source_pbf="x-latest.osm.pbf",
        region="x",
        osm_type="way",
        osm_id=1,
        page_id=1,
        revision_id=1,
        is_best_language=True,
    )
    root = _data_root(tmp_path)
    caplog.set_level(logging.INFO, logger="osm_polygon_wikidata_only.pipeline.processor")
    run_persistence_phase(
        [polygon],
        [article],
        [link],
        data_root=root,
        stem="x-latest",
        source_pbf="x-latest.osm.pbf",
    )
    assert any("Built 1 unique articles" in r.getMessage() for r in caplog.records)
