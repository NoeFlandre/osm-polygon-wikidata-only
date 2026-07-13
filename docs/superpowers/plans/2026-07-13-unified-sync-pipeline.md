# Unified Sync Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable `sync-dir` command that produces every existing core and augmentation artifact using one optimized Wikimedia runtime.

**Architecture:** A pure state planner determines missing work. One shared runtime supplies a single authenticated 1,200-RPM scheduler with three global in-flight slots to every Wikimedia client. The coordinator drains augmentation backlog, then processes and augments missing regions while preserving all legacy commands and outputs.

**Tech Stack:** Python 3.12, argparse, concurrent.futures, PyArrow, pytest, Ruff, Mypy.

---

### Task 1: Shared Wikimedia runtime

**Files:**
- Modify: `src/osm_polygon_wikidata_only/cli/dependencies.py`
- Modify: `src/osm_polygon_wikidata_only/augmentation/mediawiki.py`
- Test: `tests/test_cli_dependencies.py`
- Test: `tests/test_augmentation.py`

- [ ] Write failing tests asserting authenticated runtime configuration is exactly 1,200 RPM and three in-flight slots, and core/augmentation clients share the same scheduler/session.
- [ ] Run the focused tests and confirm failures due to the absent runtime API.
- [ ] Implement `WikimediaRuntime` and inject its scheduler/session into `AugmentationWikimediaClient`.
- [ ] Make legacy core and augmentation construction delegate to the runtime factory.
- [ ] Run focused tests, Ruff, and Mypy; commit the green checkpoint.

### Task 2: Region state planner

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/sync_planner.py`
- Create: `tests/test_sync_planner.py`

- [ ] Write failing table-driven tests for complete, augmentation-only backlog, unprocessed, stale augmentation, force, and deterministic ordering.
- [ ] Run the focused tests and confirm the planner is missing.
- [ ] Implement immutable `RegionSyncState` records and the pure planner.
- [ ] Run focused tests and commit the green checkpoint.

### Task 3: Unified coordinator

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/sync_orchestrator.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/orchestrator.py`
- Test: `tests/test_sync_orchestrator.py`

- [ ] Write failing tests proving backlog regions run first, newly processed regions are immediately augmented, complete regions are skipped, and Wikipedia full text is reused from core Parquet.
- [ ] Run focused tests and confirm the coordinator is missing.
- [ ] Implement the coordinator with one bounded extraction prefetch and injected callbacks for deterministic testing.
- [ ] Preserve local atomic writes and existing manifests.
- [ ] Run focused tests and commit the green checkpoint.

### Task 4: CLI, lock, and uploads

**Files:**
- Modify: `src/osm_polygon_wikidata_only/cli/parser.py`
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Create: `src/osm_polygon_wikidata_only/io/run_lock.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_run_lock.py`

- [ ] Write failing tests for `sync-dir`, one runtime construction, lock exclusion, `--skip-existing`, dry-run atomic uploads, and legacy command parsing.
- [ ] Run focused tests and confirm failures.
- [ ] Add the canonical command and a data-root-scoped exclusive lock.
- [ ] Wire coordinator results through the existing background upload and README mechanisms.
- [ ] Run focused tests and commit the green checkpoint.

### Task 5: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `tests/test_documentation.py`

- [ ] Document `sync-dir` as canonical and legacy commands as advanced compatibility entry points.
- [ ] Document the 1,200-RPM authenticated ceiling, three global slots, backlog recovery, and restart semantics.
- [ ] Run `uv run pytest -q`, `uv run ruff check .`, `uv run mypy src`, and `git diff --check`.
- [ ] Stop the two legacy processes only after local verification, run a small live smoke test, then start the canonical command against the real backlog.
- [ ] Verify clean Git state, merge to `main`, and push.
