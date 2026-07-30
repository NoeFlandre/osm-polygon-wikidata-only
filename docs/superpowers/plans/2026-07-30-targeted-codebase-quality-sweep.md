# Targeted Codebase Quality Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mypy with a strict ty gate and decompose only the codebase's oversized mixed-responsibility modules while preserving every observable pipeline and dataset contract.

**Architecture:** Existing import paths remain stable facades. Private focused packages receive moved implementation, and characterization tests pin identities, signatures, schemas, ordering, and output before each extraction.

**Tech Stack:** Python 3.12, uv, pytest, PyArrow, Ruff, ty, Hatchling.

---

### Task 1: Establish the baseline and migrate the type checker

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/development.md`
- Modify: `docs/architecture.md`
- Test: `tests/test_packaging.py`
- Test: `tests/test_documentation.py`

- [ ] **Step 1: Add failing configuration contracts**

Add tests that parse `pyproject.toml` and CI/documentation text and assert:

```python
assert "ty" in dev_dependencies
assert not any(dep.startswith("mypy") for dep in dev_dependencies)
assert "tool.mypy" not in pyproject
assert "uv run ty check src scripts" in ci_text
assert "mypy" not in current_public_docs
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/test_packaging.py tests/test_documentation.py -q
```

Expected: failures identifying the current mypy dependency/configuration and
missing ty gate.

- [ ] **Step 3: Perform the minimal tooling migration**

Pin `ty==0.0.64` in `[dependency-groups].dev`, remove mypy and
`[tool.mypy]`, update the lockfile, CI, README, and current architecture and
development guides. Historical plan/spec documents are excluded from the
current-documentation assertion.

- [ ] **Step 4: Resolve all ty diagnostics**

Run:

```bash
uv run ty check src scripts
```

Fix the 20 baseline diagnostics with explicit validated narrowing. In
particular, convert JSON values only after `isinstance` checks, type fallback
publication values before calling `assemble_region_upload`, and give heartbeat
fallback callbacks their declared snapshot return type. Do not add global
ignores.

- [ ] **Step 5: Verify and commit**

Run the focused tests, Ruff, and ty, then commit:

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml README.md docs src tests
git commit -m "chore: replace mypy with ty quality gate"
```

### Task 2: Extract the link-migration implementation behind its facade

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/_link_migration/__init__.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/_link_migration/models.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/_link_migration/planning.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/_link_migration/conversion.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/_link_migration/transaction.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/link_migration.py`
- Test: `tests/contracts/test_link_migration_ownership.py`

- [ ] **Step 1: Add failing identity and ownership tests**

Assert that every existing public symbol exported by `pipeline.link_migration`
is identical to its focused owner and that the focused dependency graph is
acyclic:

```python
assert facade.MigrationPlan is models.MigrationPlan
assert facade.plan_link_migration is planning.plan_link_migration
assert facade.apply_link_migration is transaction.apply_link_migration
```

- [ ] **Step 2: Run the test and confirm RED because the package is absent**

- [ ] **Step 3: Move code without altering signatures or algorithms**

Move frozen models/enums to `models.py`, canonical row construction to
`conversion.py`, classification/planning to `planning.py`, and journaled apply
logic to `transaction.py`. Keep `link_migration.py` as an explicit re-export
facade with its existing `__all__`.

- [ ] **Step 4: Run link-migration, contract, recovery, and full tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(pipeline): split link migration by responsibility"
```

### Task 3: Focus Wikidata recovery repair

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/_wikidata_recovery/row_merge.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/_wikidata_recovery/validation.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/_wikidata_recovery/manifest_stage.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/_wikidata_recovery/repair.py`
- Test: `tests/contracts/test_wikidata_recovery_ownership.py`

- [ ] **Step 1: Add failing ownership tests**

Pin helper ownership while preserving `repair_wikidata_region` in `repair.py`.
The test must compare helper identities and assert no new public facade names.

- [ ] **Step 2: Confirm RED**

- [ ] **Step 3: Extract pure row conversion/merge, validation, and manifest staging**

Keep network/client coordination and checkpoint orchestration in `repair.py`.
Pass dependencies explicitly; do not create module globals or change batch
sizes, progress messages, checkpoint keys, or replacement order.

- [ ] **Step 4: Run recovery, resumability, integration, and full tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(recovery): separate repair validation and staging"
```

### Task 4: Split publication assembly behind the existing facade

**Files:**
- Create: `src/osm_polygon_wikidata_only/hf/_publication/snapshots.py`
- Create: `src/osm_polygon_wikidata_only/hf/_publication/validation.py`
- Create: `src/osm_polygon_wikidata_only/hf/_publication/loading.py`
- Create: `src/osm_polygon_wikidata_only/hf/_publication/assembly.py`
- Modify: `src/osm_polygon_wikidata_only/hf/publication.py`
- Test: `tests/contracts/test_publication_ownership.py`

- [ ] **Step 1: Add RED identity and operation-order tests**

Pin facade identities and the exact `PublicationOp` ordering for core,
augmentation, region, metadata-only, and containment-retirement uploads.

- [ ] **Step 2: Confirm RED**

- [ ] **Step 3: Extract focused implementation**

Move README/map/manifest snapshot functions, validation, existing-artifact
loading, and operation assembly into separate modules. Preserve explicit facade
re-exports, legacy deletion guards, README-last behavior, and atomic snapshot
semantics.

- [ ] **Step 4: Run publication contracts, golden tests, CLI integration, and full suite**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(hf): separate publication assembly concerns"
```

### Task 5: Reduce the sync composition root

**Files:**
- Create: `src/osm_polygon_wikidata_only/cli/_sync/__init__.py`
- Create: `src/osm_polygon_wikidata_only/cli/_sync/preflight.py`
- Create: `src/osm_polygon_wikidata_only/cli/_sync/publication.py`
- Create: `src/osm_polygon_wikidata_only/cli/_sync/recovery.py`
- Modify: `src/osm_polygon_wikidata_only/cli/run_sync.py`
- Test: `tests/contracts/test_sync_cli_ownership.py`

- [ ] **Step 1: Add RED ownership and signature tests**

Pin `run_sync.execute` and CLI signatures while specifying ownership for
pre-publication migration, recovery-audit checks, upload construction, and
post-upload cleanup.

- [ ] **Step 2: Confirm RED**

- [ ] **Step 3: Extract helpers and flatten `execute`**

Move cohesive helpers only. Keep `execute` as the composition root and retain
the same injected collaborator parameters, call ordering, exclusive lock,
logging, error propagation, queue draining, and return codes.

- [ ] **Step 4: Run CLI, sync, reconciliation, checkpoint, and full tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(cli): focus unified sync composition root"
```

### Task 6: Split augmentation statistics scanning from aggregation

**Files:**
- Create: `src/osm_polygon_wikidata_only/hf/_dataset_stats/augmentation_scan.py`
- Create: `src/osm_polygon_wikidata_only/hf/_dataset_stats/augmentation_cache.py`
- Create: `src/osm_polygon_wikidata_only/hf/_dataset_stats/augmentation_merge.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_dataset_stats/augmentation.py`
- Test: `tests/contracts/test_dataset_stats_ownership.py`

- [ ] **Step 1: Add RED ownership and deterministic-output tests**

Pin focused helper ownership and assert byte-identical serialized cache
summaries plus equal `AugmentationStats` for fixture inputs.

- [ ] **Step 2: Confirm RED**

- [ ] **Step 3: Extract scanning, cache serialization, and merge logic**

Keep `compute_augmentation_stats` as coordinator. Preserve sorted enumeration,
cache fingerprints/version, malformed-file behavior, warning logger identity,
and all displayed dataset-card values.

- [ ] **Step 4: Run statistics, dataset-card golden, publication, and full tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(stats): separate augmentation scanning and merging"
```

### Task 7: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/development.md`
- Modify: package `__init__.py` files created above
- Test: `tests/test_documentation.py`

- [ ] **Step 1: Add RED documentation contracts**

Assert the current package map names each owner, commands use uv/Ruff/ty, and
public documentation contains no private machine paths or current mypy
instructions.

- [ ] **Step 2: Confirm RED and update documentation minimally**

- [ ] **Step 3: Run fresh full verification**

```bash
uv sync --frozen
uv run pytest -q
uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q
uv run ruff check .
uv run ruff format --check .
uv run ty check src scripts
uv build
git diff --check
```

- [ ] **Step 4: Audit the diff and commit**

Verify no data/schema/manifest/cache/public-path behavior changed, then commit:

```bash
git commit -m "docs: publish maintained codebase architecture"
```

- [ ] **Step 5: Push**

Confirm `HEAD` is on `main`, the worktree is clean, and `origin/main` is the
expected base before:

```bash
git push origin main
```
