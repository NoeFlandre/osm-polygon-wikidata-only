# Sync Application Deep Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hide unified sync callback wiring and upload lifecycle behind a small `SyncApplication.run()` façade while preserving `cli.run_sync.execute(...)` and every existing execution contract.

**Architecture:** Keep `cli.run_sync.execute` as the composition root: it performs argument handling, migration, runtime construction, reconciliation, and planning. Move the post-plan execution coordinator into `cli.sync_application.SyncApplication`, which receives a frozen service bundle and mutable run context. The application owns callback construction, runner invocation, upload draining, metadata refresh, and final reconciliation logging; no public CLI signature or publication order changes.

**Tech Stack:** Python 3.12, dataclasses, pytest, pytest-cov, Ruff, ty, existing `pipeline.sync_runner`, bounded upload queue.

---

### Task 1: Add characterization tests for the new façade

**Files:**
- Create: `tests/cli/test_sync_application.py`

- [x] **Step 1: Write tests before production code.**

  Add tests that specify these behaviors:

  1. `cli.run_sync.execute` constructs and calls `SyncApplication.run()` after planning.
  2. A mixed recovery/augment/publish/process plan preserves runner ordering and passes the same bound collaborators.
  3. Push-disabled execution never assembles or submits publication operations.
  4. Push-enabled execution submits one operation list per completed state and uses the existing commit-message factory.
  5. A recovery result updates repaired-stem and map-refresh tracking before publication assembly.
  6. Upload-queue failures force return code `1` and do not report successful reconciliation.
  7. Metadata refresh runs only after a successful runner and drained queue, and its durable marker is cleared only after upload succeeds.
  8. Exceptions from the runner propagate while the queue is still closed in `finally`.
  9. The application result exposes the final return code and reconciliation flags without exposing internal callbacks.

  Use tiny in-memory states, fake runner/client/queue services, and temporary directories. Do not call Wikimedia, Hugging Face, or PBF code.

- [x] **Step 2: Run the new suite to verify RED.**

  Run:

  ```bash
  UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache uv run pytest --no-cov -q tests/cli/test_sync_application.py
  ```

  Expected: failures because `SyncApplication` and its context/service contract do not exist yet, while the existing `execute` function does not delegate to it.

### Task 2: Implement the deep module behind explicit boundaries

**Files:**
- Create: `src/osm_polygon_wikidata_only/cli/sync_application.py`
- Modify: `src/osm_polygon_wikidata_only/cli/run_sync.py`

- [x] **Step 1: Add the minimal public records.**

  Define frozen/slotted records:

  ```python
  @dataclass(frozen=True, slots=True)
  class SyncApplicationServices:
      extract_pbf: Callable[[Path], Any]
      process_extracted_pbf: Callable[[Any], Any]
      augment_region: Callable[[RegionSyncState], Any]
      load_existing_augmentation: Callable[[RegionSyncState], Any]
      recover_region: Callable[[RegionSyncState], Any]
      run_sync: Callable[..., int]
      plan_link_migration: Callable[..., Any]
      apply_link_migration: Callable[..., Any]
      audit_wikidata_integrity: Callable[..., Any]
      ensure_recovery_audit_unblocked: Callable[[Any], None]
      repair_wikidata_region: Callable[..., Any]
      prepare_local_retirement: Callable[..., Any]
      add_pending_publications: Callable[..., Any]
      record_region_recovery_receipt: Callable[..., Any]
      assemble_region_upload: Callable[..., list[Any]]
      assemble_metadata_only_upload: Callable[..., list[Any]]
      load_existing_core_for_publication: Callable[..., Any]
      commit_message: Callable[[RegionSyncState], str]
      log_remote_reconciliation_summary: Callable[..., None]
      load_metadata_refresh_marker: Callable[..., Any]
      clear_metadata_refresh_marker: Callable[..., Any]
      augmentation_progress: Callable[[], Any]
      sync_heartbeat: Callable[..., Any]
      logger: Any
  ```

  Add a mutable `SyncApplicationContext` containing the planned states, runtime, augmentation client, upload queue, push flag, pending/reconciliation sets, and repair flags. Add a small `SyncApplicationResult` containing `return_code`, `core_will_be_repaired`, `core_repaired`, and `metadata_repaired`.

- [x] **Step 2: Move the existing callback/lifecycle logic mechanically.**

  Implement `SyncApplication.run()` with the current ordering and exception semantics: build bound extract/process callbacks, recover, augment, publish-only repair, process, assemble/submit, close uploads, refresh metadata, and emit the existing summary. Preserve the exact callback arguments and commit order. The class must not import argparse or construct network clients.

- [x] **Step 3: Make `cli.run_sync.execute` a thin adapter.**

  Keep all current planning and runtime construction in `execute`. Replace its nested callback block and runner/finalization block with construction of `SyncApplicationContext` and `SyncApplicationServices`, then return `SyncApplication(...).run()`. Pass module-level functions into the service bundle so existing monkeypatch-based tests retain their patch points.

- [x] **Step 4: Run the new façade tests.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache uv run pytest --no-cov -q tests/cli/test_sync_application.py
  ```

  Expected: all new tests pass.

### Task 3: Run the full no-regression gates

**Files:**
- Modify: `docs/development.md` only if the new module boundary needs a concise developer note.

- [x] **Step 1: Run focused CLI and pipeline tests.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache uv run pytest --no-cov -q tests/cli tests/contracts/test_sync_runner_decomposition.py tests/pipeline/test_sync_reconciliation.py tests/pipeline/test_sync_runner_ordering.py
  ```

- [x] **Step 2: Run the complete project gate.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache just check
  ```

- [x] **Step 3: Run strict CRAP and mutation gates.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache just quality-advanced
  ```

  Expected: every configured CRAP score remains below 6 and every configured mutant is killed.

- [x] **Step 4: Inspect the diff and verify repository state.**

  ```bash
  git diff --check
  git status --short --branch
  git diff --stat
  ```

  Do not commit or push until the user explicitly asks for this refactor to be published.
