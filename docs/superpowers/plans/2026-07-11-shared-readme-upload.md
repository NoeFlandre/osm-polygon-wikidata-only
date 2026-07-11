# Shared README Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the dataset README current in every main-pipeline and augmentation upload without conflicting implementations.

**Architecture:** Extract canonical README snapshot rendering from the main upload callback into one CLI helper. Call it from both upload paths immediately before constructing each atomic upload file list.

**Tech Stack:** Python 3.12, pytest, PyArrow, Hugging Face Hub uploader.

---

### Task 1: Shared README snapshot

**Files:**
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Test: `tests/test_cli.py`

- [ ] Write a failing test proving a rendered snapshot contains the canonical dataset card.
- [ ] Run `uv run pytest tests/test_cli.py -q` and confirm the missing helper failure.
- [ ] Extract the existing manifest aggregation, dataset statistics, and card rendering into `_write_readme_snapshot`.
- [ ] Run `uv run pytest tests/test_cli.py -q` and confirm it passes.

### Task 2: Augmentation atomic upload

**Files:**
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Test: `tests/test_cli.py`

- [ ] Write a failing CLI test proving augmentation dry-run upload contains `README.md`.
- [ ] Run the focused test and confirm it fails because README is absent.
- [ ] Generate the snapshot immediately before augmentation upload and append it as `README.md`.
- [ ] Run focused and full quality gates.

### Task 3: Minimal user documentation

**Files:**
- Modify: `README.md`

- [ ] Document the five additive sidecar paths and the two augmentation commands.
- [ ] State that each pushed augmentation atomically refreshes the canonical dataset README.
- [ ] Run documentation tests, commit, and push `main`.
