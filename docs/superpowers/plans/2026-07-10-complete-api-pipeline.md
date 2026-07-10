# Complete API Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an API-only, uncapped multilingual pipeline that fails closed on incomplete enrichment and overlaps each completed PBF upload with processing of the next PBF.

**Architecture:** A process-wide adaptive request scheduler governs every Wikimedia request, while the existing versioned filesystem cache becomes an atomic success journal. Enrichment computes and validates the full expected sitelink set before atomically publishing local artifacts. A bounded durable upload queue runs in a background worker and is drained before the CLI exits.

**Tech Stack:** Python 3.12, urllib/MediaWiki Action API, concurrent.futures, PyArrow, huggingface_hub, pytest, Ruff, mypy.

---

## File structure

- Create `src/osm_polygon_wikidata_only/utils/request_scheduler.py`: global concurrency, request budget, cooldown, and adaptive recovery.
- Create `src/osm_polygon_wikidata_only/enrichment/completeness.py`: expected-work validation and actionable incomplete-enrichment errors.
- Create `src/osm_polygon_wikidata_only/io/atomic.py`: reusable atomic JSON and file-replacement helpers.
- Create `src/osm_polygon_wikidata_only/hf/upload_queue.py`: bounded background uploader with durable job state.
- Modify `config/settings.py`: comprehensive defaults and scheduler/upload controls.
- Modify `cli/commands.py`: comprehensive CLI settings, upload queue wiring, and exit status.
- Modify Wikimedia clients: scheduler use, gzip/maxlag, versioned full-text cache keys, and distinguish transient failures.
- Modify `enrichment/article_linker.py`: uncapped expected-work collection and batch member validation.
- Modify `pipeline/processor.py`: fail-closed audit and temporary Parquet publication.
- Modify `pipeline/orchestrator.py`: callback after each locally complete PBF.
- Modify tests by responsibility; no production network or real large PBF fixtures.

### Task 1: Comprehensive defaults

**Files:**
- Modify: `src/osm_polygon_wikidata_only/config/settings.py`
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests** proving `_build_settings` returns `languages is None`, `fetch_full_text is True`, and `max_articles_per_qid is None` for a normal command, and that legacy narrowing flags cannot silently become production defaults.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q`; expect the default-language and cap assertions to fail.
- [ ] **Step 3: Change defaults** so `Settings.languages` and `Settings.max_articles_per_qid` remain `None`, argparse defaults to all languages/no cap, `--languages` remains an explicit narrowing override, and `--max-articles-per-qid` defaults to `None`. Keep `--all-languages` as a compatible explicit alias.
- [ ] **Step 4: Run** `uv run pytest tests/test_cli.py -q`; expect all CLI tests to pass.
- [ ] **Step 5: Commit** with `git commit -am "feat: default to complete multilingual enrichment"`.

### Task 2: Global adaptive scheduler

**Files:**
- Create: `src/osm_polygon_wikidata_only/utils/request_scheduler.py`
- Modify: `src/osm_polygon_wikidata_only/config/settings.py`
- Modify: `src/osm_polygon_wikidata_only/utils/rate_limit.py`
- Modify: `tests/test_utils.py`

- [ ] **Step 1: Write failing deterministic tests** using injected `clock`, `sleep`, and a guarded work function. Assert no more than three calls overlap globally, a cooldown blocks all hosts, token acquisition respects the configured requests-per-minute budget, and sustained success restores a reduced rate without exceeding the configured ceiling.
- [ ] **Step 2: Run** `uv run pytest tests/test_utils.py -q`; expect import failure for `request_scheduler`.
- [ ] **Step 3: Implement** `AdaptiveRequestScheduler(max_in_flight=3, requests_per_minute=180, clock=time.monotonic, sleep=time.sleep)` with a semaphore, locked next-token timestamp, `defer(delay_s)`, `report_throttled(delay_s)`, `report_success()`, and `run(callable)`. Adaptation only lowers or restores pacing; it never raises concurrency above three or rate above configuration.
- [ ] **Step 4: Replace host-only 429 coordination** with scheduler-global cooldown while retaining host pacing as an optional secondary limit. Add settings `wikimedia_max_in_flight=3` and `wikimedia_requests_per_minute=180`, validating both as positive and the concurrency as at most three.
- [ ] **Step 5: Run** `uv run pytest tests/test_utils.py -q`; expect scheduler tests to pass.
- [ ] **Step 6: Commit** `git add ... && git commit -m "feat: coordinate Wikimedia requests globally"`.

### Task 3: Atomic, versioned success cache

**Files:**
- Create: `src/osm_polygon_wikidata_only/io/atomic.py`
- Modify: `src/osm_polygon_wikidata_only/io/cache.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`
- Modify: `tests/test_io_cache.py`
- Modify: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests** that interrupt a cache write and preserve the previous valid entry, reject a cache entry with a different contract version, separate lead-only and full-text article keys, and never treat a cached transient error as successful work.
- [ ] **Step 2: Run** `uv run pytest tests/test_io_cache.py tests/test_enrichment.py -q`; expect atomicity/version tests to fail.
- [ ] **Step 3: Implement** `atomic_write_text(path, text)` with a same-directory temporary file, flush, `os.fsync`, and `os.replace`, cleaning the temporary file on failure.
- [ ] **Step 4: Extend `JsonFileCache`** with `contract_version`, store it in metadata, reject mismatches on read, and use `atomic_write_text`. Successful Wikimedia records use a long-lived versioned cache; transient errors remain diagnostics and do not satisfy completeness.
- [ ] **Step 5: Version article keys** with `wikipedia-v2/full-text/{site}/{safe_title}` and entity keys with `wikidata-v2/{qid}`. Cache reads require `status == "ok"` and a structurally valid payload.
- [ ] **Step 6: Run** the two test files and expect all tests to pass.
- [ ] **Step 7: Commit** with `git commit -m "feat: make enrichment checkpoints atomic"`.

### Task 4: Efficient API transport and complete batch validation

**Files:**
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/article_linker.py`
- Modify: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests** asserting requests include `maxlag`, `Accept-Encoding: gzip`, the identifying User-Agent, and maximum bounded QID/title batches. Add tests where a batch omits one member and prove that member is retried individually rather than returned as missing.
- [ ] **Step 2: Run** `uv run pytest tests/test_enrichment.py -q`; expect header, maxlag, and fallback assertions to fail.
- [ ] **Step 3: Route all HTTP calls through the scheduler**, decode gzip responses, pass `maxlag=5`, honor 429/503 `Retry-After`, and classify MediaWiki `maxlag` as transient. Keep existing parsers as the single source of row semantics.
- [ ] **Step 4: Validate batch membership** in cached and HTTP clients. Retry only missing/malformed members in progressively smaller batches, terminating at individual requests. Preserve requested QID and sorted sitelink order.
- [ ] **Step 5: Deduplicate article work globally per PBF** by `(language, site, title)` before batching, then fan successful results back to every QID without altering deterministic article/link ordering.
- [ ] **Step 6: Run** `uv run pytest tests/test_enrichment.py -q`; expect all enrichment tests to pass.
- [ ] **Step 7: Commit** with `git commit -m "perf: maximize safe Wikimedia batching"`.

### Task 5: Fail-closed completeness and atomic Parquet publication

**Files:**
- Create: `src/osm_polygon_wikidata_only/enrichment/completeness.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/processor.py`
- Modify: `src/osm_polygon_wikidata_only/io/parquet.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_io_parquet.py`

- [ ] **Step 1: Write failing tests** for `IncompleteEnrichmentError`: a transiently unresolved valid QID, missing expected article, failed full-text parse, or missing expected link must prevent final files and manifest updates. Verify existing final files remain byte-identical after failure.
- [ ] **Step 2: Run** `uv run pytest tests/test_pipeline.py tests/test_io_parquet.py -q`; expect incomplete runs to publish under current behavior.
- [ ] **Step 3: Implement completeness types** `ExpectedArticle(qid, language, site, title)` and `audit_complete(summaries, expected, articles, links)`. Terminal Wikidata `missing` remains a resolved no-entity outcome so its polygon is preserved; transport ambiguity is an error.
- [ ] **Step 4: Make processor fail closed** after enrichment and again after row construction. Include all unresolved keys and last diagnostics in `IncompleteEnrichmentError`. Never convert failed requests into an empty successful summary.
- [ ] **Step 5: Write all three Parquet files to PBF-scoped temporary paths**, validate them, then atomically replace final files as one publication phase before updating the local manifest. Clean temporary files after success or error.
- [ ] **Step 6: Run** the focused tests and expect all to pass, including unchanged output-order fixtures.
- [ ] **Step 7: Commit** with `git commit -m "feat: prevent incomplete PBF publication"`.

### Task 6: Durable non-blocking upload queue

**Files:**
- Create: `src/osm_polygon_wikidata_only/hf/upload_queue.py`
- Modify: `src/osm_polygon_wikidata_only/hf/uploader.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/orchestrator.py`
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Modify: `tests/test_hf.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests** with blocking events proving upload of PBF A overlaps processing PBF B, the queue is bounded, failed jobs persist, successful retry removes state, remote manifest publication follows artifact commit, shutdown drains jobs, and an unresolved upload makes `main()` return nonzero.
- [ ] **Step 2: Run** `uv run pytest tests/test_hf.py tests/test_cli.py tests/test_pipeline.py -q`; expect import/callback failures.
- [ ] **Step 3: Add `on_complete: Callable[[ProcessResult], None] | None`** to `orchestrate` and invoke it immediately after each successful local result.
- [ ] **Step 4: Implement `UploadQueue`** with one background worker and bounded `queue.Queue`. Persist each job atomically under `data_root/cache/upload_jobs`; submit the three PBF files plus the manifest snapshot; use one atomic Hub commit so referenced artifacts and manifest appear together; retry with bounded backoff; expose `close_and_wait() -> list[UploadFailure]`.
- [ ] **Step 5: Wire CLI `--push`** to create the queue before orchestration, submit from `on_complete`, continue to the next PBF immediately, drain in `finally`, and return `1` if processing or uploads remain failed. `--dry-run` uses one shared `StubHfHub`.
- [ ] **Step 6: Resume persisted upload jobs** before accepting newly completed results, deduplicating by source PBF and artifact content fingerprint.
- [ ] **Step 7: Run** all focused tests and expect them to pass.
- [ ] **Step 8: Commit** with `git commit -m "feat: upload completed PBFs in background"`.

### Task 7: Observability and production documentation

**Files:**
- Modify: `src/osm_polygon_wikidata_only/pipeline/processor.py`
- Modify: `src/osm_polygon_wikidata_only/hf/upload_queue.py`
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write a failing log-capture test** asserting each PBF reports expected articles, cache successes, request batches, cooldown/retry time, audit success, local publication, upload enqueue, and final upload status.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q`; expect missing structured stage messages.
- [ ] **Step 3: Add concise counters and timing logs** without logging article text or credentials. Ensure incomplete errors print unresolved identifiers and explicitly instruct rerunning the same command.
- [ ] **Step 4: Update README** to state comprehensive defaults, fail-closed/resume behavior, global unauthenticated scheduler, background uploads, and the canonical all-PBF command:

```bash
uv run osm-polygon-wikidata-only process-dir "$OSM_POLYGON_DATA_ROOT/raw" \
  --skip-existing \
  --push
```

- [ ] **Step 5: Run** `uv run pytest tests/test_cli.py -q`; expect the logging test to pass.
- [ ] **Step 6: Commit** with `git commit -am "docs: document complete resumable pipeline"`.

### Task 8: Full verification and main integration

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run focused completeness suite**: `uv run pytest tests/test_enrichment.py tests/test_pipeline.py tests/test_hf.py tests/test_cli.py -q`; expect all pass.
- [ ] **Step 2: Run full tests**: `uv run pytest -q`; expect all pass with no network calls.
- [ ] **Step 3: Run lint**: `uv run ruff check .`; expect no findings.
- [ ] **Step 4: Run formatter check**: `uv run ruff format --check .`; expect all files formatted.
- [ ] **Step 5: Run types**: `uv run mypy src`; expect success with no issues.
- [ ] **Step 6: Inspect** `git diff main...HEAD --check` and `git status --short`; expect no whitespace errors and a clean worktree.
- [ ] **Step 7: Use the finishing-development-branch workflow**, merge `codex/complete-api-pipeline` into `main` as requested, rerun the full verification gate on `main`, and only then provide the canonical processing command.
