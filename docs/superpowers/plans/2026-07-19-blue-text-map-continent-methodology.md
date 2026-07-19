# Blue Text Map and Continent Methodology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distinguish the first dataset-card map in blue and make every continent statistic understandable and automatically reproducible.

**Architecture:** Keep `coverage_map` as the shared renderer with backward-compatible orange defaults. Pass an explicit blue palette only from `geographic_text_presence`, and keep all continent methodology prose colocated with the data-derived table renderer.

**Tech Stack:** Python 3.12, Matplotlib, PyArrow, pytest, Ruff, mypy.

---

### Task 1: Blue combined-text map

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/coverage_map.py`
- Modify: `src/osm_polygon_wikidata_only/hf/geographic_text_presence.py`
- Test: `tests/hf/test_public_dataset_card_geography.py`
- Regenerate: `assets/geographic_text_presence.png`

- [ ] Add a failing test that captures the combined renderer call and requires blue fill and dark-blue edge colors while asserting the generic renderer defaults remain orange.
- [ ] Run the focused test and confirm the expected color assertion fails.
- [ ] Add optional point-color and edge-color keyword parameters to the shared renderer, preserving its current defaults, and pass the fixed blue palette from the combined renderer.
- [ ] Run the focused tests and confirm they pass.
- [ ] Regenerate the real combined-text PNG from finalized Parquet data.

### Task 2: Public continent methodology

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/continent_stats.py`
- Test: `tests/hf/test_public_dataset_card_geography.py`

- [ ] Add failing assertions requiring centroid assignment, Natural Earth resolution, all metric definitions, the coverage formula, distinct counting, multi-continent semantics, `Unassigned`, and automatic recomputation language.
- [ ] Run the focused test and confirm the methodology assertions fail.
- [ ] Expand `render_continent_stats` with concise public-facing methodology and metric definitions while leaving computed rows unchanged.
- [ ] Run the focused tests and confirm they pass.

### Task 3: End-to-end regeneration and publication

**Files:**
- Regenerate: dataset README snapshot and `assets/geographic_text_presence.png`

- [ ] Run `uv run pytest -q`, coverage, Ruff check/format, mypy, and `git diff --check`.
- [ ] Generate the real README and verify forbidden internal wording remains absent.
- [ ] Commit, fast-forward `main`, and push GitHub.
- [ ] Assemble and upload the metadata-only publication atomically through the existing uploader.
- [ ] Download the live README and maps and verify the first map is blue, the second remains orange, and the methodology is present.
