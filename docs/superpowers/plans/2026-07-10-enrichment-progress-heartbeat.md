# Enrichment Progress Heartbeat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit one factual enrichment progress message every two minutes while preserving quiet short runs and the existing once-per-run authentication-mode log.

**Architecture:** A thread-safe tracker in `enrichment` records QID, Wikipedia-site, and article-attempt counters without logging. A small pipeline heartbeat reads immutable snapshots on a daemon thread and logs them every 120 seconds; `process_pbf` owns its lifecycle around `fetch_qids`. Existing enrichment callers remain compatible because tracking is optional.

**Tech Stack:** Python 3.12, dataclasses, threading, logging, pytest, Ruff, mypy, uv.

---

## File structure

- Create `src/osm_polygon_wikidata_only/enrichment/progress.py`: immutable snapshot and thread-safe progress tracker.
- Create `src/osm_polygon_wikidata_only/pipeline/heartbeat.py`: two-minute daemon heartbeat lifecycle and message formatting.
- Modify `src/osm_polygon_wikidata_only/enrichment/article_linker.py`: optional tracker instrumentation at completed batch boundaries.
- Modify `src/osm_polygon_wikidata_only/pipeline/processor.py`: immediate QID summary and scoped heartbeat composition.
- Create `tests/test_enrichment_progress.py`: tracker and heartbeat unit tests.
- Modify `tests/test_enrichment.py`: batched and compatibility-path instrumentation tests.
- Modify `tests/test_pipeline.py`: processor composition and logging tests.
- Modify `tests/test_dependencies.py`: authentication-mode log and redaction regression tests.
- Modify `README.md` and `docs/architecture.md`: explain heartbeat cadence and counters.

### Task 1: Thread-safe progress model

**Files:**
- Create: `tests/test_enrichment_progress.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/progress.py`

- [ ] **Step 1: Write the failing tracker tests**

Specify that `EnrichmentProgress(total_qids=3).snapshot()` initially returns an immutable `EnrichmentProgressSnapshot(qids_completed=0, qids_total=3, sites_completed=0, sites_total=0, articles_attempted=0)`. Add focused tests for `set_qids_total(5)`, `advance_qids(2)`, `set_sites_total(4)`, and `complete_site(articles_attempted=7)`. Verify snapshots are values unaffected by later updates and concurrent `complete_site(1)` calls never lose increments.

```python
def test_progress_tracker_records_completed_site_and_articles() -> None:
    progress = EnrichmentProgress(total_qids=3)
    progress.set_sites_total(4)
    progress.complete_site(articles_attempted=7)
    assert progress.snapshot() == EnrichmentProgressSnapshot(
        qids_completed=0,
        qids_total=3,
        sites_completed=1,
        sites_total=4,
        articles_attempted=7,
    )
```

- [ ] **Step 2: Run tracker tests and verify RED**

Run: `uv run pytest tests/test_enrichment_progress.py -q`

Expected: collection fails because `enrichment.progress` does not exist.

- [ ] **Step 3: Implement the minimal tracker**

Create frozen `EnrichmentProgressSnapshot` and `EnrichmentProgress` with `set_qids_total`, `advance_qids`, `set_sites_total`, `complete_site`, and `snapshot`. Protect all mutable integer fields with one `threading.Lock`; reject negative totals/increments with `ValueError`; cap neither completed count nor attempted count because updates reflect completed work and tests should expose caller mistakes rather than hide them.

- [ ] **Step 4: Run tracker tests and verify GREEN**

Run: `uv run pytest tests/test_enrichment_progress.py -q`

Expected: all tracker tests pass.

- [ ] **Step 5: Run focused static checks and commit**

Run: `uv run ruff check tests/test_enrichment_progress.py src/osm_polygon_wikidata_only/enrichment/progress.py && uv run mypy src/osm_polygon_wikidata_only/enrichment/progress.py`

Commit: `feat: add enrichment progress tracking`

### Task 2: Instrument enrichment batch boundaries

**Files:**
- Modify: `tests/test_enrichment.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/article_linker.py`

- [ ] **Step 1: Write failing batched-path progress tests**

Using the existing in-memory or recording batch clients, call `fetch_qids(..., progress=EnrichmentProgress(total_qids=0), batch_size=1)` for two QIDs whose filtered sitelinks form two Wikipedia site groups. Assert the returned summaries are unchanged and the final snapshot reports two completed QIDs, two total QIDs, two completed sites, two total sites, and the exact number of distinct titles attempted.

- [ ] **Step 2: Run the batched progress test and verify RED**

Run: `uv run pytest tests/test_enrichment.py -q`

Expected: `fetch_qids` rejects the new `progress` keyword.

- [ ] **Step 3: Implement minimal batched instrumentation**

Add `progress: EnrichmentProgress | None = None`. Replace the Wikidata list comprehension with an explicit chunk loop so `advance_qids(len(chunk))` happens only after a completed client call. After request grouping call `set_sites_total(len(requests))`. Make `fetch_site` return its distinct-title count; after each executor result is received call `complete_site(articles_attempted=title_count)`. Do not emit logs or change result ordering.

- [ ] **Step 4: Run the batched progress test and verify GREEN**

Run: `uv run pytest tests/test_enrichment.py -q`

Expected: the new test and all existing enrichment tests pass.

- [ ] **Step 5: Write the failing compatibility-path progress test**

Call `fetch_qids` with non-batch protocol clients and a tracker. Assert one QID advancement after each completed `link_qid`, the correct total, zero unavailable site/article counters, and unchanged summaries.

- [ ] **Step 6: Run compatibility test and verify RED**

Run: `uv run pytest tests/test_enrichment.py -q`

Expected: QID completion remains zero in the compatibility path.

- [ ] **Step 7: Implement compatibility-path instrumentation and verify GREEN**

Set the tracker QID total from the materialized request list and call `advance_qids()` after each successful `link_qid`. Run: `uv run pytest tests/test_enrichment.py tests/test_enrichment_progress.py -q`.

Expected: all focused tests pass.

- [ ] **Step 8: Run static checks and commit**

Run: `uv run ruff check tests/test_enrichment.py src/osm_polygon_wikidata_only/enrichment && uv run mypy src/osm_polygon_wikidata_only/enrichment`

Commit: `feat: report enrichment batch progress`

### Task 3: Two-minute heartbeat lifecycle

**Files:**
- Modify: `tests/test_enrichment_progress.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/heartbeat.py`

- [ ] **Step 1: Write failing heartbeat behavior tests**

Define a fake stop event whose `wait(120)` results are `[False, False, True]`, a fake monotonic clock returning `0`, `120`, and `240`, and a recording log callback. Call `heartbeat.run()` directly and assert exactly two messages, both containing the region and current factual snapshot, with `2m` and `4m` elapsed. Assert every wait receives `120`.

Add a context-lifecycle test with an injected recording thread factory proving `__enter__` starts once, `__exit__` sets the stop event and joins once, including when the context body raises. Add a snapshot-provider failure test proving the heartbeat contains the error and stops without propagating into pipeline work.

- [ ] **Step 2: Run heartbeat tests and verify RED**

Run: `uv run pytest tests/test_enrichment_progress.py -q`

Expected: import fails because `pipeline.heartbeat` does not exist.

- [ ] **Step 3: Implement the minimal heartbeat**

Create `ENRICHMENT_HEARTBEAT_INTERVAL_S = 120` and `EnrichmentHeartbeat`. Constructor dependencies are `region`, `snapshot`, `log`, `interval_s`, `clock`, `stop_event`, and `thread_factory`. `run()` loops on `stop_event.wait(interval_s)`, formats integer elapsed minutes and all counters, and catches snapshot/log failures with one DEBUG message. The context manager starts a daemon named `enrichment-progress`, then stops and joins it in `__exit__` without suppressing body exceptions.

- [ ] **Step 4: Run heartbeat tests and verify GREEN**

Run: `uv run pytest tests/test_enrichment_progress.py -q`

Expected: tracker and heartbeat tests pass.

- [ ] **Step 5: Run static checks and commit**

Run: `uv run ruff check tests/test_enrichment_progress.py src/osm_polygon_wikidata_only/pipeline/heartbeat.py && uv run mypy src/osm_polygon_wikidata_only/pipeline/heartbeat.py`

Commit: `feat: add two-minute enrichment heartbeat`

### Task 4: Compose processor and authentication-mode logging

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_dependencies.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/processor.py`

- [ ] **Step 1: Write failing processor logging/composition tests**

Patch `processor.fetch_qids` with a recording function and patch `processor.EnrichmentHeartbeat` with a recording context manager. Process a small fake PBF and assert:

- INFO contains `Starting enrichment for tiny: 2 unique Wikidata QIDs` immediately after extraction;
- one tracker is passed to `fetch_qids`;
- the heartbeat receives region `tiny` and that tracker's `snapshot` provider;
- the heartbeat context exits on a synthetic `IncompleteEnrichmentError` or client exception;
- output counts remain unchanged for a successful run.

- [ ] **Step 2: Run processor tests and verify RED**

Run: `uv run pytest tests/test_pipeline.py -q`

Expected: no start summary, heartbeat composition, or tracker injection exists.

- [ ] **Step 3: Implement minimal processor composition**

After computing `unique_qids`, log the immediate summary, construct `EnrichmentProgress(len(unique_qids))`, and wrap only `fetch_qids` in `EnrichmentHeartbeat(region=stem.region, snapshot=progress.snapshot, log=LOGGER.info)`. Pass `progress=progress`. Skip heartbeat construction for zero QIDs while still logging the zero-QID start summary.

- [ ] **Step 4: Run processor tests and verify GREEN**

Run: `uv run pytest tests/test_pipeline.py tests/test_enrichment_progress.py tests/test_enrichment.py -q`

Expected: all focused tests pass with unchanged dataset outputs.

- [ ] **Step 5: Write failing authentication-mode log tests**

Use `caplog` around `build_clients` for anonymous and authenticated environments. Assert exactly one `Wikimedia API mode` record per construction, containing `anonymous` or `authenticated as NoeFlandre@pipeline`, the rate ceiling, and never `secret-value`.

- [ ] **Step 6: Run dependency tests and verify the contract**

Run: `uv run pytest tests/test_dependencies.py -q`

Expected: tests pass if the existing once-per-run log already meets the revised spec. If a test fails, change only the log formatting needed to meet the accepted contract, then rerun until green.

- [ ] **Step 7: Run static checks and commit**

Run: `uv run ruff check tests/test_pipeline.py tests/test_dependencies.py src/osm_polygon_wikidata_only/pipeline && uv run mypy src/osm_polygon_wikidata_only/pipeline`

Commit: `feat: log enrichment heartbeat and API mode`

### Task 5: Reader documentation

**Files:**
- Modify: `tests/test_documentation.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: Write failing documentation assertions**

Require both public documents to mention the `two-minute` enrichment heartbeat, QID/site/article counters, no ETA, and unchanged request pacing. Retain existing Bot Password documentation assertions.

- [ ] **Step 2: Run documentation tests and verify RED**

Run: `uv run pytest tests/test_documentation.py -q`

Expected: assertions fail because heartbeat behavior is undocumented.

- [ ] **Step 3: Document behavior and verify GREEN**

Add a concise reliability bullet to README and an observability paragraph to architecture docs. Run: `uv run pytest tests/test_documentation.py -q`.

Expected: all documentation tests pass.

- [ ] **Step 4: Run static checks and commit**

Run: `uv run ruff check tests/test_documentation.py && uv run ruff format --check tests/test_documentation.py && git diff --check`

Commit: `docs: explain enrichment progress heartbeat`

### Task 6: Full verification and integration

**Files:**
- Modify only files required by failures found in the complete gate.

- [ ] **Step 1: Run complete tests and coverage**

Run: `uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q`

Expected: all tests pass and coverage remains at least 80%.

- [ ] **Step 2: Run all static and package checks**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
git diff --check
uv build
```

Expected: every command exits zero.

- [ ] **Step 3: Review accepted behavior**

Verify the final diff against every acceptance criterion in `docs/superpowers/specs/2026-07-10-enrichment-progress-heartbeat-design.md`: 120-second cadence, quiet short runs, domain counters, immediate cleanup, one authentication-mode line, no password, and no enrichment-output changes.

- [ ] **Step 4: Finish and integrate**

Use `superpowers:verification-before-completion`, perform the required code review inline because delegation is not authorized, and use `superpowers:finishing-a-development-branch`. Merge locally into `main`, rerun the full quality gate on the merge commit, then remove the worktree and feature branch.
