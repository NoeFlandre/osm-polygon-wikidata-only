# Transient Wikimedia Network Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent long-running syncs from aborting during temporary Wikimedia DNS, connectivity, throttling, or server outages.

**Architecture:** Extend the shared retry helper with optional unbounded attempts and a retry predicate. Centralize transient network classification, then use the production-unbounded setting across all existing Wikimedia transports while preserving finite test overrides.

**Tech Stack:** Python 3.12 standard networking modules, pytest, Ruff, mypy.

---

### Task 1: Retry semantics and classification

**Files:**
- Modify: `src/osm_polygon_wikidata_only/utils/retry.py`
- Test: `tests/test_utils.py`

- [ ] Add failing tests for unbounded recovery, permanent failure rejection, DNS/timeout/connection/temporary-HTTP classification, permanent HTTP rejection, and interrupt propagation.
- [ ] Run the focused tests and confirm failures describe the missing behavior.
- [ ] Implement optional unbounded attempts and the centralized transient classifier with capped exponential backoff.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Production Wikimedia integration and observability

**Files:**
- Modify: `src/osm_polygon_wikidata_only/config/settings.py`
- Modify: `src/osm_polygon_wikidata_only/augmentation/mediawiki.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikidata/transport.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikipedia/transport.py`
- Test: `tests/augmentation/test_augmentation.py`
- Test: `tests/enrichment/test_wikimedia_transport_clients.py`

- [ ] Add failing integration tests proving the production default recovers after more than eight DNS failures on each client path, permanent failures do not loop, and retry warnings are sparse and sanitized.
- [ ] Set the production retry attempt default to unbounded while preserving explicit integer overrides.
- [ ] Pass the transient predicate and sparse warning callback through every Wikimedia retry call.
- [ ] Run focused client tests and confirm they pass.

### Task 3: Verification and delivery

- [ ] Run the full test suite, coverage, Ruff check/format, mypy, and `git diff --check`.
- [ ] Commit, fast-forward verified work to `main`, and push.
- [ ] Provide the safe restart command and explain that cached Sweden responses are reused.
