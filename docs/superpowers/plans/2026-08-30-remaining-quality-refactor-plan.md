# Remaining Quality Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Lower the remaining tested complexity-B hotspots while preserving the Grid5000 preflight and Hugging Face inventory contracts.

**Architecture:** Keep the existing orchestration and external boundaries. Extract deterministic GPU-output parsing, shared Hub-client selection, and remote metadata validation into private helpers with no new dependency or public API.

**Tech Stack:** Python 3.12, pytest/pytest-cov, Ruff, ty, Radon, crap4py, Just, mutmut.

---

### Task 1: Baseline and scope the safe refactor

**Files:**
- Inspect: `src/osm_polygon_wikidata_only/grid5000/sentence_job.py`
- Inspect: `src/osm_polygon_wikidata_only/hf/remote_inventory.py`
- Inspect: `tests/grid5000/test_sentence_job.py`
- Inspect: `tests/hf/test_reconciliation.py`

- [x] **Step 1: Run the full tracked baseline.**

Run:

```bash
PATH=/usr/bin:/bin MPLBACKEND=Agg MPLCONFIGDIR=/tmp/osm-polygon-wikidata-only-next-baseline-mpl COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-next-baseline-coverage .venv/bin/pytest --ignore=tests/unit --cov=osm_polygon_wikidata_only --cov-report=json:/tmp/osm-polygon-wikidata-only-next-baseline-coverage.json -q
```

Expected result: `2614 passed, 2 skipped`, with total coverage at least 93%.

- [x] **Step 2: Measure complexity and static quality.**

Run `.venv/bin/radon cc -n B -s src scripts`, `.venv/bin/ruff check src tests scripts --exclude tests/unit`, `.venv/bin/ruff format --check src tests scripts --exclude tests/unit`, and `.venv/bin/ty check src scripts`. The remaining actionable B-level functions are `_query_gpus`, `RemoteInventory.fetch_paths`, and `_remote_file_info`.

### Task 2: Isolate GPU-output parsing

**Files:**
- Modify: `tests/grid5000/test_sentence_job.py`
- Modify: `src/osm_polygon_wikidata_only/grid5000/sentence_job.py`
- Modify: `Justfile`
- Modify: `tests/test_packaging.py`

- [x] **Step 1: Add a failing parser-seam test.**

Add a test importing `_parse_gpu_output` and assert that it turns `"GPU A, uuid-a, 40960 MiB\n\nGPU B, uuid-b, 24576 MiB\n"` into two ordered `GpuIdentity` values. Run the focused test and observe the expected import failure because the helper does not exist.

- [x] **Step 2: Implement the smallest delegation.**

Add:

```python
def _parse_gpu_output(stdout: str | None) -> tuple[GpuIdentity, ...]:
    return tuple(_parse_gpu_line(line) for line in (stdout or "").splitlines() if line.strip())
```

and make `_query_gpus()` call it while retaining the existing return-code and empty-output errors.

- [x] **Step 3: Cover the subprocess environment and measure CRAP.**

Test `_job_environment()` and `_run_command()` with a fake subprocess boundary, then run the sentence-job test file with coverage and Radon. Require every function in `sentence_job.py` to remain below CRAP 6.

### Task 3: Simplify remote inventory boundaries

**Files:**
- Modify: `tests/hf/test_reconciliation.py`
- Modify: `src/osm_polygon_wikidata_only/hf/remote_inventory.py`
- Modify: `Justfile`
- Modify: `tests/test_packaging.py`

- [x] **Step 1: Add failing helper-seam tests.**

Add tests for `_resolve_hub_client()` building one API after resolving one token, and `_remote_file_fields()` accepting only a string path plus integer size. Run those tests before implementation and observe the expected import failure.

- [x] **Step 2: Extract client and metadata helpers.**

Add `_resolve_hub_client()`, `_remote_file_fields()`, and `_remote_sha256()`. Replace duplicated client-selection branches in `RemoteInventory.fetch()` and `fetch_paths()`, and make `_remote_file_info()` compose the metadata helpers. Preserve explicit-hub short-circuiting, legacy fallback, request arguments, filtering, and translated errors.

- [x] **Step 3: Measure the inventory scope.**

Run `tests/hf/test_reconciliation.py` with coverage for `remote_inventory.py`, then Radon and the CRAP scorer. Require every measured function below 6.

### Task 4: Publish quality catalog updates

**Files:**
- Modify: `Justfile`
- Modify: `docs/development.md`
- Modify: `tests/test_packaging.py`
- Create: `docs/superpowers/specs/2026-08-30-remaining-quality-refactor-design.md`
- Create: `docs/superpowers/plans/2026-08-30-remaining-quality-refactor-plan.md`

- [x] **Step 1: Add a failing packaging contract.**

Assert that `crap-job` and `crap-inventory` recipes reference their focused tests and source files and that `crap-all` includes both recipes. Run the packaging test before adding the recipes and observe the missing-recipe failure.

- [x] **Step 2: Add bounded recipes and documentation.**

Add the two coverage/Radon/CRAP recipes to `Justfile`, include them in `crap-all`, `quality-strength`, and `quality-advanced`, and document that subprocess/network boundaries are CRAP-gated but intentionally outside the deterministic mutmut scope.

### Task 5: Final verification and publication

**Files:**
- No additional production files.

- [x] **Step 1: Run the complete quality checks.**

Run `UV_NO_SYNC=1 just crap-all`, the full tracked regression suite with `--ignore=tests/unit`, Ruff, format, ty, acceptance tests, architecture/docs checks, and `git diff --check`.

- [x] **Step 2: Verify mutation-scope stability.**

Confirm `pyproject.toml` has no diff and that no configured mutmut source or selected test changed. Reuse the validated `2410/2410` killed result for the unchanged deterministic scope; do not treat partial exploratory results for external boundaries as a gate.

- [x] **Step 3: Commit and push exact scope.**

Stage only the design/plan, the two production modules, their tests, `Justfile`, `docs/development.md`, and packaging tests. Commit on `main`, push `origin/main`, and verify local and remote heads match while leaving all pre-existing untracked files untouched.
