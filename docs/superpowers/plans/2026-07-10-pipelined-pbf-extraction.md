# Pipelined PBF Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap extraction of the next PBF with enrichment and publication of the current PBF without changing artifacts, ordering, failure semantics, or Wikimedia scheduling.

**Architecture:** Split the existing single-PBF processor into a pure orchestration facade around `extract_pbf` and `process_extracted_pbf`. The directory orchestrator owns one extraction worker and one prefetched result, while all enrichment and durable writes remain sequential on the calling thread.

**Tech Stack:** Python 3.12, `concurrent.futures`, pytest, Ruff, strict mypy.

---

### Task 1: Characterize overlap and ordering

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] Add a gated orchestrator test proving extraction of PBF B starts while PBF A is processing.
- [ ] Assert callbacks and returned results remain in input order.
- [ ] Run the focused test and confirm it fails because extraction is currently inside sequential `process_pbf`.

### Task 2: Separate extraction from processing

**Files:**
- Modify: `src/osm_polygon_wikidata_only/pipeline/processor.py`
- Modify: `tests/test_pipeline.py`

- [ ] Introduce an immutable `ExtractedPbf` carrying path metadata, polygons, and extraction duration.
- [ ] Extract the existing reader/conversion block unchanged into `extract_pbf`.
- [ ] Move enrichment through manifest publication into `process_extracted_pbf`.
- [ ] Keep `process_pbf` as the compatible synchronous facade composing both functions.
- [ ] Run processor and end-to-end tests and confirm identical rows, schemas, and manifests.

### Task 3: Pipeline directory orchestration

**Files:**
- Modify: `src/osm_polygon_wikidata_only/pipeline/orchestrator.py`
- Modify: `tests/test_pipeline.py`

- [ ] Filter skipped inputs before scheduling extraction.
- [ ] Use one extraction worker and prefetch at most one next PBF.
- [ ] Start the next extraction before processing the current extracted value.
- [ ] Keep processing, callbacks, results, manifest writes, and API clients on the calling thread.
- [ ] Verify extraction failures propagate and prevent processing later PBFs.

### Task 4: Complete verification

**Files:**
- Verify all changed source, tests, and documentation.

- [ ] Run the focused pipeline and end-to-end suites.
- [ ] Run the full branch-aware coverage suite.
- [ ] Run Ruff lint/format, strict mypy, and package builds.
- [ ] Review the final diff for output-contract changes.
