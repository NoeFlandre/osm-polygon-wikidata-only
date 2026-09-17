# Language Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing V1/V2 language-split command so V2 reruns remove only owned stale shards and a failed write cannot damage the previous valid release.

**Architecture:** Keep the shared inventory contract, CLI facade, and V1 generator stable. Change the V2 generator to preflight source/output disjointness, stage all generated shards and the manifest on the destination filesystem, validate them, and install with explicit backups and rollback. Regression tests cover documentation, mutation scope, stale ownership, overlap rejection, determinism, and injected failures.

**Tech Stack:** Python 3.12, PyArrow Parquet streaming, pytest, Ruff, ty, mutmut, MkDocs, Just, Git worktree.

---

## File map

- Modify `src/osm_polygon_wikidata_only/v2/language_splits.py`: V2 preflight, staging, stale ownership, installation, rollback, and cleanup.
- Modify `tests/v2/test_language_splits.py`: V2 regression and failure-injection tests.
- Modify `tests/hf/test_language_split_release.py` only if facade behavior lacks a regression for V1 preservation, deterministic planning, or preflight-before-generation.
- Modify `tests/quality/test_mutation_scope.py` only if the existing scope contract does not cover the facade and its tests.
- Modify `tests/test_mkdocs.py` only if the existing navigation contract does not cover `docs/language-splits.md`.
- Modify `pyproject.toml` only if the existing mutation configuration omits `hf/language_split_release.py` or its focused test module.
- Modify `mkdocs.yml` only if the existing navigation omits `language-splits.md`.
- The committed design is `docs/superpowers/specs/2026-09-17-language-release-hardening-design.md`.

All commands use the HDD worktree. Use a task-scoped cache such as `/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening` and keep pytest basetemp there. Do not read or rewrite the 19–37 GB processed trees in unit fixtures.

### Task 1: Establish a fresh baseline

**Files:** None.

- [ ] **Step 1: Verify the worktree and base revision.**

  Run:

  ```bash
  git status --short --branch
  git rev-parse HEAD
  git diff --check
  ```

  Expected: clean `codex/language-release-hardening`, revision `d514be28fc5be84de00bc9a476e38d25ca7a0b55`, and no whitespace errors.

- [ ] **Step 2: Run the repository baseline recipe.**

  Run:

  ```bash
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" just baseline
  ```

  Record the exit code and complete test count. If it fails, preserve the output as a pre-existing baseline failure and do not attribute it to this branch.

- [ ] **Step 3: Run the focused language/documentation baseline.**

  Run:

  ```bash
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" \
  uv run python -m pytest \
    tests/hf/test_language_split_release.py \
    tests/v2/test_language_splits.py \
    tests/quality/test_mutation_scope.py \
    tests/test_mkdocs.py \
    --no-cov -p no:cacheprovider -q \
    --basetemp="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/baseline"
  ```

  Record failures before adding tests.

### Task 2: Add RED tests for the five release-path defects

**Files:**
- Modify: `tests/v2/test_language_splits.py`
- Modify: `tests/hf/test_language_split_release.py` if a facade regression is missing
- Modify: `tests/quality/test_mutation_scope.py` if scope coverage is missing
- Modify: `tests/test_mkdocs.py` if navigation coverage is missing

- [ ] **Step 1: Add a byte-snapshot helper and the stale-shard regression.**

  Seed the existing temporary V2 fixture, run `build_v2_language_splits`, record every file below `language_splits/` plus `manifests/language_splits.json`, then add an obsolete generated shard through the previous manifest and an unrelated operator-owned Parquet file. Change the source so the generated shard is no longer expected, rerun, and assert the obsolete owned shard and empty generated directories disappear while the operator file remains byte-identical.

- [ ] **Step 2: Add overlap rejection before mutation.**

  Call `build_v2_language_splits(root, output_root=root / "wikipedia")` and `build_v2_language_splits(root, output_root=root / "manifests/processed_pbfs.json")` against a fixture with a sentinel file. Assert `V2LanguageSplitError`, an error naming source/output overlap, unchanged sentinel bytes, no generated output directory, and no transaction directory. Also cover an output root equal to the processed root.

- [ ] **Step 3: Add replacement-failure rollback.**

  Generate a valid release, snapshot all generated shards and the manifest, monkeypatch the module’s replacement primitive to raise after at least one new file has been installed, rerun generation, and assert the snapshot is byte-for-byte identical and no `.tmp`, `.backup`, or staging directory remains.

- [ ] **Step 4: Add backup-failure rollback.**

  Generate a valid release, inject a failure after one existing target has been moved to a backup, rerun generation, and assert the previous release and manifest are restored byte-for-byte with no backup left behind.

- [ ] **Step 5: Add the documentation and mutation-scope contracts if absent.**

  Assert `mkdocs.yml` maps `Language-split release` to `language-splits.md`; assert the mutation configuration contains both `src/osm_polygon_wikidata_only/hf/language_split_release.py` and `src/osm_polygon_wikidata_only/v2/language_splits.py`, plus both corresponding test modules.

- [ ] **Step 6: Run only the new tests to prove RED.**

  Run:

  ```bash
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" \
  uv run python -m pytest \
    tests/v2/test_language_splits.py \
    tests/hf/test_language_split_release.py \
    tests/quality/test_mutation_scope.py \
    tests/test_mkdocs.py \
    --no-cov -p no:cacheprovider -q \
    --basetemp="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/red"
  ```

  Expected: the new transaction/overlap assertions fail against the current in-place V2 writer; existing tests must still collect.

- [ ] **Step 7: Commit the RED tests.**

  ```bash
  git add tests/v2/test_language_splits.py tests/hf/test_language_split_release.py tests/quality/test_mutation_scope.py tests/test_mkdocs.py
  git commit -m "test(language-splits): cover safe rerun and rollback contracts"
  ```

### Task 3: Implement V2 preflight and staged generation

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/language_splits.py`
- Test: `tests/v2/test_language_splits.py`

- [ ] **Step 1: Add destination validation before side effects.**

  Resolve `processed_root` and `output_root`, require the output to be inside the processed root, reject a non-directory output path, build the validated V2 inventory, and reject any output root that contains a source file, source directory, or manifest path. Do this before `mkdir`, stale discovery, or writer creation. Keep `batch_size < 1` rejection unchanged.

- [ ] **Step 2: Add a same-filesystem staging root.**

  Create one hidden temporary directory beside the final output root. Route every Parquet writer to the staging root, not the final output. Validate staged schema including metadata, row count, language bucket, and SHA-256 before installation. Write the manifest into the staging root only after all files pass conservation and source-fingerprint checks.

- [ ] **Step 3: Run the focused RED tests.**

  Run the Task 2 pytest command. Expected: overlap tests pass; rollback and staging tests still fail only where installation behavior is not implemented.

- [ ] **Step 4: Commit the minimal staged-generation implementation.**

  ```bash
  git add src/osm_polygon_wikidata_only/v2/language_splits.py
  git commit -m "fix(language-splits): stage V2 releases before install"
  ```

### Task 4: Implement manifest-owned stale cleanup and rollback

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/language_splits.py`
- Test: `tests/v2/test_language_splits.py`

- [ ] **Step 1: Read only the previous manifest’s owned shard paths.**

  Parse the prior V2 language manifest defensively. Accept only string `.parquet` paths that resolve below the current output root and match the generated shard shape. Treat a missing, malformed, or foreign-output manifest as owning no files. Never use an unrestricted `rglob` for deletion.

- [ ] **Step 2: Install with backups and rollback.**

  Back up the union of previous owned paths, current staged paths, and the existing manifest using same-directory `os.replace`. Install staged files in deterministic path order and install the manifest last. On `BaseException`, remove newly installed paths, restore backups in deterministic order, and re-raise. In all cases remove temporary staging and backup files; after success prune only empty directories belonging to owned generated paths.

- [ ] **Step 3: Run the focused tests to prove GREEN.**

  ```bash
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" \
  uv run python -m pytest tests/v2/test_language_splits.py tests/hf/test_language_split_release.py tests/quality/test_mutation_scope.py tests/test_mkdocs.py --no-cov -p no:cacheprovider -q --basetemp="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/green"
  ```

  Expected: all focused tests pass, including stale-shard removal, operator-file preservation, overlap refusal, partial-install rollback, backup rollback, and V1 facade compatibility.

- [ ] **Step 4: Commit the transaction implementation.**

  ```bash
  git add src/osm_polygon_wikidata_only/v2/language_splits.py tests/v2/test_language_splits.py
  git commit -m "fix(language-splits): make V2 installation recoverable"
  ```

### Task 5: Refactor and verify deterministic contracts

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/language_splits.py` only for behavior-preserving extraction
- Test: `tests/v2/test_language_splits.py`, `tests/hf/test_language_split_release.py`

- [ ] **Step 1: Add repeated-run byte determinism assertions.**

  Run the V1 and V2 generators twice against the same fixture and compare every output byte and manifest payload. Assert V1 remains under `processed/language_splits/`, V2 remains under `processed_v2/language_splits/`, and no polygon table is copied into either output.

- [ ] **Step 2: Refactor only after green.**

  Keep helpers single-purpose: path safety, manifest ownership, staging, install, restore, and cleanup. Preserve public names and error classes. Do not change source schemas, language normalization, output naming, or row ordering.

- [ ] **Step 3: Run focused quality checks.**

  ```bash
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" uv run ruff check src/osm_polygon_wikidata_only/v2/language_splits.py tests/v2/test_language_splits.py tests/hf/test_language_split_release.py tests/quality/test_mutation_scope.py
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" uv run ruff format --check src/osm_polygon_wikidata_only/v2/language_splits.py tests/v2/test_language_splits.py tests/hf/test_language_split_release.py tests/quality/test_mutation_scope.py
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" uv run ty check src tests
  UV_CACHE_DIR="/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/cache/language-release-hardening/uv" uv run python scripts/quality/architecture.py
  ```

- [ ] **Step 4: Commit the refactor and regression tests.**

  ```bash
  git add src/osm_polygon_wikidata_only/v2/language_splits.py tests/v2/test_language_splits.py tests/hf/test_language_split_release.py
  git commit -m "test(language-splits): lock deterministic release behavior"
  ```

### Task 6: Run the complete hardening quality gate

**Files:** None unless a failing check identifies a real defect; never weaken thresholds or configuration.

- [ ] **Step 1: Run all required Just recipes.**

  Run each recipe from the HDD worktree with the task-scoped UV cache, recording exit status and full output:

  ```bash
  just baseline
  just ruff
  just typecheck
  just tests
  just property-tests
  just acceptance-tests
  just architecture-checks
  just crap-all
  just mutation
  just smoke-test
  just docs
  just docker-check
  just diff-review
  just quality-gauntlet
  ```

- [ ] **Step 2: Inspect the final diff and mutation report.**

  Run `git diff origin/main...HEAD --check`, inspect every changed file, and require zero unreviewed mutation survivors, no CRAP threshold regression, no documentation warnings, and no Docker smoke failure.

- [ ] **Step 3: Commit any narrowly scoped fixes with a new RED→GREEN cycle.**

  If a check fails, reproduce it with the smallest focused test first, identify the root cause, add or adjust the regression test, and rerun the failing check before changing production code.

### Task 7: Review and hand off the hardening branch

**Files:** Final diff only.

- [ ] **Step 1: Verify branch scope.**

  Confirm only the design/plan documents, V2 hardening implementation, and language-release regression tests changed. Confirm no processed data, raw PBF, cache outside the task directory, or unrelated repository was modified.

- [ ] **Step 2: Run the final focused and repository checks again after the last commit.**

  Use the exact commands from Tasks 5 and 6 and record fresh exit codes.

- [ ] **Step 3: Push the feature branch and open a PR linked to issues #24–#28.**

  Push `codex/language-release-hardening`, verify the remote branch SHA, and create a non-draft PR describing the staged V2 transaction, manifest-owned cleanup, overlap refusal, rollback tests, and quality evidence. Do not merge or publish Hugging Face artifacts until the PR diff and checks have been independently reviewed.

