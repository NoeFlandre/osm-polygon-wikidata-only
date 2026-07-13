# Sync Progress Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add concise progress and Wikimedia request-budget telemetry to long-running unified synchronization without changing its outputs or request behavior.

**Architecture:** Extend the existing shared adaptive scheduler with immutable telemetry snapshots. Track augmentation phase progress in a small thread-safe object and render both snapshots through the existing heartbeat pattern while each region is augmented.

**Tech Stack:** Python 3.12, dataclasses, threading, pytest, existing logging and scheduler infrastructure.

---

### Task 1: Scheduler telemetry

**Files:**
- Modify: `src/osm_polygon_wikidata_only/utils/request_scheduler.py`
- Test: `tests/test_utils.py`

- [ ] Add failing deterministic-clock tests for rolling request count, utilization, in-flight requests, cooldown, and throttle events.
- [ ] Run `uv run pytest tests/test_utils.py -q` and confirm the telemetry API is missing.
- [ ] Add an immutable scheduler snapshot and update counters inside existing lock boundaries.
- [ ] Re-run the focused tests.

### Task 2: Augmentation progress

**Files:**
- Create: `src/osm_polygon_wikidata_only/augmentation/progress.py`
- Modify: `src/osm_polygon_wikidata_only/augmentation/orchestrator.py`
- Test: `tests/test_augmentation.py`

- [ ] Add failing tests for phase totals and thread-safe advancement.
- [ ] Add a small progress tracker and optional `progress` argument to `augment_region`.
- [ ] Update phase boundaries and worker completions without changing ordering or outputs.
- [ ] Re-run augmentation tests.

### Task 3: Unified sync heartbeat

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/sync_heartbeat.py`
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Test: `tests/test_sync_progress.py`

- [ ] Add failing formatting and lifecycle tests for concise 60-second messages.
- [ ] Implement a heartbeat that reports region position, phase progress, measured/max RPM, utilization, in-flight requests, 429 count, and cooldown.
- [ ] Wrap augmentation work in the heartbeat and retain existing completion logs.
- [ ] Run focused tests, then the complete suite and quality gates.
