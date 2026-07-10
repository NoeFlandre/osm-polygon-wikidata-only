# Per-Host Throttling Circuit Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a single Wikimedia host's 429/503 from stalling the entire enrichment pipeline by adding a multi-host circuit breaker to the request scheduler.

**Architecture:** Add `report_host_throttled(host, delay)` to `AdaptiveRequestScheduler`. It records throttle events keyed by hostname in a rolling window and only escalates to the existing `report_throttled()` when `threshold` distinct hosts fail within `window_s` seconds. The two enrichment client call sites switch from `report_throttled` to `report_host_throttled`.

**Tech Stack:** Python 3.12, stdlib only (`threading`, `time`), pytest, ruff, mypy strict.

**Spec:** `docs/superpowers/specs/2026-07-10-per-host-throttling-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/osm_polygon_wikidata_only/utils/request_scheduler.py` | Global request scheduling | Add `report_host_throttled` method + 2 constructor params + event tracking state |
| `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py` | Wikipedia API client | Change 1 call site in `_http_get` |
| `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py` | Wikidata API client | Change 1 call site in `_http_get` |
| `tests/test_utils.py` | Scheduler unit tests | Add 4 new test functions |
| `tests/test_enrichment.py` | Client integration tests | Update 2 existing test functions |

---

### Task 1: Add `report_host_throttled` to `AdaptiveRequestScheduler`

**Files:**
- Modify: `src/osm_polygon_wikidata_only/utils/request_scheduler.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test for single-host no escalation**

Add this test to the end of `tests/test_utils.py`, after the last scheduler test (`test_scheduler_successes_restore_rate_after_throttling`):

```python
def test_report_host_throttled_single_host_does_not_escalate() -> None:
    """A 429 from one host must not trigger the global cooldown."""
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        clock=lambda: now[0],
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 60.0)

    # Rate unchanged, no cooldown applied.
    assert scheduler.current_requests_per_minute == 200
    assert scheduler.run(lambda: "ok") == "ok"
    assert sleeps == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_utils.py::test_report_host_throttled_single_host_does_not_escalate -v`
Expected: FAIL with `AttributeError: 'AdaptiveRequestScheduler' object has no attribute 'report_host_throttled'`

- [ ] **Step 3: Implement `report_host_throttled` and constructor params**

In `src/osm_polygon_wikidata_only/utils/request_scheduler.py`:

**3a.** Add two new parameters to `__init__` (after `successes_per_increase`, before `clock`):

```python
        successes_per_increase: int = 100,
        host_throttle_window_s: float = 10.0,
        host_throttle_threshold: int = 3,
        clock: Callable[[], float] = time.monotonic,
```

**3b.** Add validation for the new params (after the existing `successes_per_increase` validation):

```python
        if successes_per_increase <= 0:
            raise ValueError("successes_per_increase must be positive")
        if host_throttle_window_s <= 0:
            raise ValueError("host_throttle_window_s must be positive")
        if host_throttle_threshold < 1:
            raise ValueError("host_throttle_threshold must be at least 1")
```

**3c.** Add instance state (after `self._cooldown_until = 0.0`):

```python
        self._cooldown_until = 0.0
        self._host_throttle_window_s = host_throttle_window_s
        self._host_throttle_threshold = host_throttle_threshold
        self._host_throttle_events: dict[str, float] = {}
```

**3d.** Add the new method (after `report_throttled`, before `run`):

```python
    def report_host_throttled(self, host: str, delay_s: float) -> None:
        """Record a per-host throttle; escalate globally only when systemic."""
        with self._lock:
            now = self._clock()
            cutoff = now - self._host_throttle_window_s
            self._host_throttle_events = {
                h: t for h, t in self._host_throttle_events.items() if t > cutoff
            }
            self._host_throttle_events[host] = now
            if len(self._host_throttle_events) < self._host_throttle_threshold:
                return
        self.report_throttled(delay_s)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_utils.py::test_report_host_throttled_single_host_does_not_escalate -v`
Expected: PASS

- [ ] **Step 5: Write the multi-host escalation test**

Add to `tests/test_utils.py`:

```python
def test_report_host_throttled_escalates_when_threshold_hosts_fail() -> None:
    """Three distinct hosts within the window must trigger global backoff."""
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        successes_per_increase=1,
        host_throttle_threshold=3,
        clock=lambda: now[0],
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 60.0)
    scheduler.report_host_throttled("de.wikipedia.org", 60.0)
    # Two hosts: not enough.
    assert scheduler.current_requests_per_minute == 200

    scheduler.report_host_throttled("es.wikipedia.org", 60.0)
    # Third distinct host: escalation fires.
    assert scheduler.current_requests_per_minute == 100
    assert scheduler.run(lambda: "ok") == "ok"
    assert sleeps == [60.0]
```

- [ ] **Step 6: Run the escalation test**

Run: `uv run pytest tests/test_utils.py::test_report_host_throttled_escalates_when_threshold_hosts_fail -v`
Expected: PASS

- [ ] **Step 7: Write the window-pruning test**

Add to `tests/test_utils.py`:

```python
def test_report_host_throttled_prunes_events_outside_window() -> None:
    """Old events expire, so a slow trickle never escalates."""
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        host_throttle_window_s=10.0,
        host_throttle_threshold=3,
        clock=lambda: now[0],
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 5.0)
    now[0] += 11.0  # advance past the 10s window
    scheduler.report_host_throttled("de.wikipedia.org", 5.0)
    now[0] += 11.0
    scheduler.report_host_throttled("es.wikipedia.org", 5.0)

    # Each event was alone in its window; never escalated.
    assert scheduler.current_requests_per_minute == 200
    assert sleeps == []
```

- [ ] **Step 8: Run all scheduler tests together**

Run: `uv run pytest tests/test_utils.py -v -k "scheduler or host_throttle or defer_host"`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/osm_polygon_wikidata_only/utils/request_scheduler.py tests/test_utils.py
git commit -m "feat: add per-host throttling circuit breaker to scheduler"
```

---

### Task 2: Update `HttpWikipediaClient` call site

**Files:**
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`
- Test: `tests/test_enrichment.py`

- [ ] **Step 1: Update the existing test to mock `report_host_throttled`**

In `tests/test_enrichment.py`, find `test_http_wikipedia_client_reports_429_to_scheduler` and replace it with:

```python
def test_http_wikipedia_client_reports_429_to_host_throttle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000)
    recorded: list[tuple[str, float]] = []
    monkeypatch.setattr(
        scheduler,
        "report_host_throttled",
        lambda host, delay: recorded.append((host, delay)),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.enrichment.wikipedia_client.defer_host", lambda *_: None
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.enrichment.wikipedia_client.wait_for_host",
        lambda *_, **__: None,
    )
    client = HttpWikipediaClient(Settings(), scheduler=scheduler, session=ThrottledSession())

    with pytest.raises(urllib.error.HTTPError):
        client._http_get(client._build_url("en", "Alpha", fetch_full_text=True))

    assert recorded == [("en.wikipedia.org", 17.0)]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_enrichment.py::test_http_wikipedia_client_reports_429_to_host_throttle -v`
Expected: FAIL with `AssertionError: assert [] == [('en.wikipedia.org', 17.0)]` (the client still calls `report_throttled`, so the mock on `report_host_throttled` captures nothing)

- [ ] **Step 3: Update the call site in `HttpWikipediaClient._http_get`**

In `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`, in the `_http_get` method, change:

```python
                defer_host(host, delay)
                self._scheduler.report_throttled(delay)
```

to:

```python
                defer_host(host, delay)
                self._scheduler.report_host_throttled(host, delay)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_enrichment.py::test_http_wikipedia_client_reports_429_to_host_throttle -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py tests/test_enrichment.py
git commit -m "feat: route wikipedia 429 through per-host throttle"
```

---

### Task 3: Update `HttpWikidataClient` call site

**Files:**
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`
- Test: `tests/test_enrichment.py`

- [ ] **Step 1: Update the existing test to mock `report_host_throttled`**

In `tests/test_enrichment.py`, find `test_http_wikidata_client_reports_429_to_scheduler` and replace it with:

```python
def test_http_wikidata_client_reports_429_to_host_throttle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000)
    recorded: list[tuple[str, float]] = []
    monkeypatch.setattr(
        scheduler,
        "report_host_throttled",
        lambda host, delay: recorded.append((host, delay)),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.enrichment.wikidata_client.defer_host", lambda *_: None
    )
    client = HttpWikidataClient(Settings(), scheduler=scheduler, session=ThrottledSession())

    with pytest.raises(urllib.error.HTTPError):
        client._http_get(client._build_url("Q1"))

    assert recorded == [("www.wikidata.org", 17.0)]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_enrichment.py::test_http_wikidata_client_reports_429_to_host_throttle -v`
Expected: FAIL with `AssertionError: assert [] == [('www.wikidata.org', 17.0)]`

- [ ] **Step 3: Update the call site in `HttpWikidataClient._http_get`**

In `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`, in the `_http_get` method, change:

```python
                defer_host("www.wikidata.org", delay)
                self._scheduler.report_throttled(delay)
```

to:

```python
                defer_host("www.wikidata.org", delay)
                self._scheduler.report_host_throttled("www.wikidata.org", delay)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_enrichment.py::test_http_wikidata_client_reports_429_to_host_throttle -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/osm_polygon_wikidata_only/enrichment/wikidata_client.py tests/test_enrichment.py
git commit -m "feat: route wikidata 429 through per-host throttle"
```

---

### Task 4: Full quality gate

**Files:** No new files; verify everything together.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run ruff lint**

Run: `uv run ruff check src/ tests/`
Expected: "All checks passed"

- [ ] **Step 3: Run ruff format check**

Run: `uv run ruff format --check src/ tests/`
Expected: No formatting differences

- [ ] **Step 4: Run mypy strict**

Run: `uv run mypy src/`
Expected: "Success: no issues found"

- [ ] **Step 5: Fix any issues found in steps 1-4**

If any test, lint, format, or type issue arises, fix it and re-run the failing step.

- [ ] **Step 6: Commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: resolve quality gate issues for per-host throttling"
```

If no fixes were needed, skip this step.
