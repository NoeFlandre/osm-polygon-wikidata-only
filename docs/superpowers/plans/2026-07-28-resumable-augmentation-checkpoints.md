# Resumable Augmentation Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable Seagate-backed augmentation checkpoints so interruption repeats at most the active bounded section batch.

**Architecture:** A private checkpoint store owns content-addressed, atomic phase artifacts beneath `DataRoot.cache`. The existing orchestrator remains the policy owner and composes exact existing phase helpers with checkpoint load/save operations; canonical outputs and manifests retain their current transaction boundary.

**Tech Stack:** Python 3.12, PyArrow Parquet, atomic filesystem replacement, pytest, uv, Ruff, mypy.

---

### Task 1: Checkpoint store contract

**Files:**
- Create: `src/osm_polygon_wikidata_only/augmentation/checkpoints.py`
- Create: `tests/augmentation/test_augmentation_checkpoints.py`

- [ ] Write failing tests proving deterministic plan keys, Seagate/data-root cache confinement, stem validation, atomic phase round trips, exact schema validation, corrupt checkpoint rejection, and clear-only-this-region behavior.
- [ ] Run `uv run pytest tests/augmentation/test_augmentation_checkpoints.py -q` and verify failures are caused by the missing module.
- [ ] Implement `augmentation_plan_key`, `AugmentationCheckpointStore`, and typed artifacts. Use exact document, section, and fact schemas; write temporary directories inside the plan root and publish them with `os.replace`.
- [ ] Run the focused tests and verify green.

### Task 2: Bounded section batches

**Files:**
- Modify: `src/osm_polygon_wikidata_only/augmentation/steps.py`
- Modify: `tests/augmentation/test_augmentation_split.py`

- [ ] Write a failing test for a section-batch helper that preserves existing parse semantics, progress advancement, project partitioning, and deterministic sorting.
- [ ] Run the focused test and verify RED.
- [ ] Extract `fetch_document_sections_batch` from the existing helper; keep `fetch_document_sections` behavior and signature compatible.
- [ ] Run both existing and new augmentation split tests and verify green.

### Task 3: Orchestrator resume integration

**Files:**
- Modify: `src/osm_polygon_wikidata_only/augmentation/orchestrator.py`
- Create: `tests/augmentation/test_augmentation_resume.py`

- [ ] Write failing fresh-process tests that interrupt after completed section batches and after each completed phase.
- [ ] Prove RED: the baseline repeats every phase and has no durable augmentation checkpoint.
- [ ] Build the plan key from exact core hashes, ordered QIDs, and Wikipedia document identities.
- [ ] Load or compute entities, Wikivoyage documents, section batches of 50, and facts. Advance progress factually for reused work.
- [ ] Preserve checkpoints on every exception and `KeyboardInterrupt`; clear them only after the manifest update succeeds.
- [ ] Assert byte-equivalent tables/rows and identical counts between uninterrupted and resumed runs.
- [ ] Run focused augmentation and resume tests and verify green.

### Task 4: Documentation and complete verification

**Files:**
- Modify: `docs/architecture.md`

- [ ] Document the external-data-root checkpoint location, invalidation key, bounded-loss guarantee, and successful cleanup.
- [ ] Run `uv run pytest tests/augmentation -q`.
- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check .`.
- [ ] Run `uv run mypy src`.
- [ ] Run `git diff --check`.
- [ ] Inspect the final diff for unrelated changes and confirm the running sync process was never stopped.
