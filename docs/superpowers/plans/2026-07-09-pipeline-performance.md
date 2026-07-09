# Pipeline Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate enrichment, PBF processing, row construction, and multi-PBF orchestration while preserving every dataset result.

**Architecture:** Concrete HTTP and cached clients receive optional batch methods; the linker uses them when both sides support them and otherwise retains the existing single-request path. The processor consumes the reader callback and reuses derived article data. All results are reassembled in their existing deterministic order.

**Tech Stack:** Python 3.12, urllib, concurrent.futures, osmium, PyArrow, pytest, ruff, mypy.

---

### Task 1: Coordinate rate-limit cooldowns

**Files:** `src/osm_polygon_wikidata_only/utils/rate_limit.py`, `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`, `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`, `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
def test_defer_host_moves_next_request_after_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: 10.0)
    rate_limit.defer_host("en.wikipedia.org", 30.0)
    assert rate_limit.next_wait_seconds("en.wikipedia.org") == 30.0
```

- [ ] **Step 2: Verify RED** — `uv run pytest tests/test_utils.py -q` must fail because the cooldown API is absent.
- [ ] **Step 3: GREEN** — add locked `defer_host`/`next_wait_seconds`; call `defer_host(host, retry_after_seconds(...))` before each existing 429 sleep.
- [ ] **Step 4: Verify GREEN** — `uv run pytest tests/test_utils.py -q` passes.

### Task 2: Batch Wikidata entities without bypassing cache

**Files:** `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`, `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests** for `HttpWikidataClient.get_entities(["Q1", "Q2"])` parsing two entities in input order and `CachedWikidataClient.get_entities` serving hits and batching only misses.
- [ ] **Step 2: Verify RED** — `uv run pytest tests/test_enrichment.py -q` fails because `get_entities` is missing.
- [ ] **Step 3: GREEN** — add `get_entities(qids)` to HTTP/cached concrete clients. Make a single `wbgetentities` URL with pipe-separated valid IDs; return `WikidataEntity | None` values in input order. Cache each result using current keys/statuses. Make `get_entity` delegate to it for one QID.
- [ ] **Step 4: Verify GREEN** — `uv run pytest tests/test_enrichment.py -q` passes.

### Task 3: Batch Wikipedia titles and fall back per title

**Files:** `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`, `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests** for `fetch_articles("en", "enwiki", ["Alpha", "Beta"])` returning both requested titles, missing-page status, cache hits, and malformed batch fallback invoking `fetch_article` for every title.
- [ ] **Step 2: Verify RED** — `uv run pytest tests/test_enrichment.py -q` fails because `fetch_articles` is absent.
- [ ] **Step 3: GREEN** — build a pipe-separated MediaWiki `titles` URL, map pages through normalized/redirect mappings, and parse one page with the existing parser. Add the cache-aware equivalent. If a batch cannot map every title, call the current individual method for every member; never omit a result.
- [ ] **Step 4: Verify GREEN** — `uv run pytest tests/test_enrichment.py -q` passes.

### Task 4: Deterministic concurrent linker batches

**Files:** `src/osm_polygon_wikidata_only/config/settings.py`, `src/osm_polygon_wikidata_only/enrichment/article_linker.py`, `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests** proving `fetch_qids(["Q2", "Q1", "Q2"])` keeps input summary order, site-sorted article order, `max_articles_per_qid` semantics, and individual fallback on a failed site batch.
- [ ] **Step 2: Verify RED** — `uv run pytest tests/test_enrichment.py -q` fails because it only makes single-item calls.
- [ ] **Step 3: GREEN** — add conservative batch size/site-worker settings. Detect both optional batch methods, group selected sitelinks by `(language, site)`, run bounded site jobs, then populate summaries by the saved original positions. Retain the existing serial path for all current in-memory/third-party clients.
- [ ] **Step 4: Verify GREEN** — `uv run pytest tests/test_enrichment.py -q` passes.

### Task 5: Remove local copies and duplicate article computations

**Files:** `src/osm_polygon_wikidata_only/pipeline/processor.py`, `src/osm_polygon_wikidata_only/pipeline/extractor.py`, `src/osm_polygon_wikidata_only/io/parquet.py`, `tests/test_pipeline.py`, `tests/test_io_parquet.py`

- [ ] **Step 1: Write failing tests** that assert `process_pbf` uses `iter_polygon_candidates`, preserves the limit boundary, and repeated QIDs yield unchanged article/link rows; add an iterator-only `write_table` test.
- [ ] **Step 2: Verify RED** — `uv run pytest tests/test_pipeline.py tests/test_io_parquet.py -q` fails because candidates are collected first and each repeated link rebuilds its article.
- [ ] **Step 3: GREEN** — convert candidates in the reader callback (with compatibility fallback for collect-only test readers), construct each `Article` once by ID, use shallow copies for flat dataclass rows, and remove the redundant `list(materialized)` before `from_pylist`.
- [ ] **Step 4: Verify GREEN** — `uv run pytest tests/test_pipeline.py tests/test_io_parquet.py -q` passes.

### Task 6: Reuse the manifest during an orchestrated run

**Files:** `src/osm_polygon_wikidata_only/io/manifest.py`, `src/osm_polygon_wikidata_only/pipeline/orchestrator.py`, `src/osm_polygon_wikidata_only/pipeline/processor.py`, `tests/test_pipeline.py`, `tests/test_io_manifest.py`

- [ ] **Step 1: Write a failing test** counting one `load_manifest` call while processing several PBFs and verifying later skip decisions see entries added during the run.
- [ ] **Step 2: Verify RED** — `uv run pytest tests/test_pipeline.py tests/test_io_manifest.py -q` fails because every skip/upsert reloads the manifest.
- [ ] **Step 3: GREEN** — make `upsert_entry` accept an optional mutable entries map. Load once in `orchestrate`, use it for skip checks, pass it to `process_pbf`, and mutate/save it after every successful PBF. Direct processing retains load-and-save behavior.
- [ ] **Step 4: Verify GREEN** — `uv run pytest tests/test_pipeline.py tests/test_io_manifest.py -q` passes.

### Task 7: Invariant benchmark and quality gate

**Files:** `tests/test_performance_invariants.py`, `README.md`

- [ ] **Step 1: Write a failing synthetic invariant test** comparing normalized serial and batched pipeline rows for repeated QIDs.
- [ ] **Step 2: Verify RED** — `uv run pytest tests/test_performance_invariants.py -q` fails before the preceding changes are complete.
- [ ] **Step 3: GREEN** — create only in-memory generated candidates/clients for the benchmark; document batch and worker settings plus retained host pacing and retries.
- [ ] **Step 4: Verify all** — `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src` passes without real PBF or network execution.
- [ ] **Step 5: Commit and integrate** — commit the implementation, fast-forward `main` to `codex/pipeline-performance`, and confirm the root worktree is clean.
