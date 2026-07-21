# Resumable Wikidata Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long Wikidata recovery work observable and resumable with at most one small batch of repeated work.

**Architecture:** Recovery keeps finalized dataset files unchanged while it builds content-addressed, schema-validated Parquet checkpoints for batches of 25 QIDs. A periodic heartbeat reports the active batch and stage. After every batch is durable, the existing transaction mechanism consolidates the checkpoints into the canonical regional files and the existing publisher uploads the region once.

**Tech Stack:** Python 3.12, PyArrow/Parquet, existing atomic I/O and recovery transaction utilities, pytest.

---

### Task 1: Durable batch checkpoint store

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/_wikidata_recovery/checkpoints.py`
- Test: `tests/pipeline/test_wikidata_recovery_checkpoints.py`

- [ ] Write failing tests for atomic save/load, exact schema validation, plan-key invalidation, deterministic batch paths, and cleanup.
- [ ] Run `uv run pytest tests/pipeline/test_wikidata_recovery_checkpoints.py -q` and verify the missing module/API fails.
- [ ] Implement a private store using one directory per content-addressed plan and three Parquet files plus metadata per batch.
- [ ] Re-run the focused tests and commit the green implementation.

### Task 2: Recovery progress and heartbeat

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/_wikidata_recovery/progress.py`
- Test: `tests/pipeline/test_wikidata_recovery_progress.py`

- [ ] Write failing deterministic tests with an injected clock/stop signal for stage snapshots, elapsed time, ETA, and heartbeat lifecycle.
- [ ] Run the focused tests and verify failure for the missing API.
- [ ] Implement a thread-safe progress snapshot and low-noise 60-second heartbeat.
- [ ] Re-run the focused tests and commit the green implementation.

### Task 3: Batch the recovery computation

**Files:**
- Modify: `src/osm_polygon_wikidata_only/pipeline/_wikidata_recovery/repair.py`
- Modify: `src/osm_polygon_wikidata_only/cli/run_sync.py`
- Test: `tests/pipeline/test_wikidata_recovery_repair.py`
- Test: `tests/pipeline/test_sync_recovery_integration.py`

- [ ] Write failing tests proving a completed first batch is reused after a second-batch failure, stale checkpoints are ignored, stage logs are emitted, and publication is not submitted before final convergence.
- [ ] Run the focused tests and verify the expected behavioral failures.
- [ ] Process affected QIDs in deterministic groups of 25, persist each group, merge checkpoint results once, use the existing final atomic transaction, and delete checkpoints only after convergence.
- [ ] Wire the CLI logger into recovery and retain the existing upload-after-return behavior.
- [ ] Run all focused recovery and sync tests and commit the green implementation.

### Task 4: Documentation and complete verification

**Files:**
- Modify: `docs/architecture.md`

- [ ] Document checkpoint location, invalidation, batch size, heartbeat fields, interruption semantics, and final publication behavior.
- [ ] Run `uv run pytest -q` and the coverage gate.
- [ ] Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`, and `git diff --check`.
- [ ] Commit, push `main`, and report the exact restart command without stopping the user's currently running process.
