# Per-Host Throttling Circuit Breaker Design

## Goal

Prevent a single Wikimedia host's HTTP 429 or 503 from stalling the entire
enrichment pipeline. When the pipeline enriches QIDs across 310+ language
editions, one flaky host should only pause itself, not every other host.

## Problem

Both `HttpWikidataClient._http_get()` and `HttpWikipediaClient._http_get()`
call `self._scheduler.report_throttled(delay)` on any 429 or 503 response.
That method sets a **global** cooldown (`_cooldown_until`) that blocks every
Wikimedia host and halves the **global** request rate toward the minimum.

In production this manifests as a catastrophic stall: one language edition
returns a 429, the default `Retry-After` of 60 seconds pauses all workers for
over a minute, and the halved rate (1200 -> 600 -> 300 -> 200 minimum) takes
hundreds of successful requests to climb back. Observed throughput dropped
from ~2200 articles/min to zero for two full minutes, then crawled at
90-134/min for the rest of the run.

The existing `defer_host(host, delay)` call already blocks only the offending
host for the `Retry-After` duration. The additional `report_throttled()` call
is the overreaction.

## Scope

This change adds a multi-host circuit breaker to `AdaptiveRequestScheduler` and
updates the two call sites in the enrichment clients. It does not change the
global rate limiter, semaphore, adaptive ramp-up, `defer_host` behavior,
`report_throttled` semantics, retry logic, or the per-host minimum interval.

## Design

### New scheduler method: `report_host_throttled`

Add a method that records throttle events keyed by hostname in a rolling
window. Only when `threshold` **distinct** hosts have been throttled within
`window_s` seconds does it escalate to the existing `report_throttled(delay)`,
which applies the global cooldown and rate reduction.

```python
def report_host_throttled(self, host: str, delay_s: float) -> None:
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

Two new constructor parameters with defaults:

- `host_throttle_window_s: float = 10.0`
- `host_throttle_threshold: int = 3`

### Call-site changes

In both `HttpWikidataClient._http_get()` and `HttpWikipediaClient._http_get()`,
replace:

```python
self._scheduler.report_throttled(delay)
```

with:

```python
self._scheduler.report_host_throttled(host, delay)
```

For the Wikidata client, `host` is always `"www.wikidata.org"`. A single-host
429 will never reach the threshold of 3 distinct hosts, so it will never
escalate. The existing `defer_host("www.wikidata.org", delay)` already backs
off that host. This is correct: a Wikidata-only throttle should not slow down
Wikipedia requests to other hosts.

### What does not change

- `defer_host(host, delay)` continues to block only the offending host.
- `report_throttled(delay)` stays available for direct external callers and
  remains the escalation path.
- The global semaphore, rate limiter, and adaptive ramp-up are untouched.
- For a true systemic Wikimedia overload (many hosts return 429 within
  seconds), the circuit breaker still fires and applies the global backoff.

## Data Flow

A 429 from `fr.wikipedia.org` triggers:

1. `defer_host("fr.wikipedia.org", 60)` -- only `fr.wikipedia.org` is paused.
2. `scheduler.report_host_throttled("fr.wikipedia.org", 60)` -- records the
   event. If fewer than 3 distinct hosts have failed in the last 10 seconds,
   the global rate is unaffected and all other hosts continue at full speed.

If `de.wikipedia.org`, `es.wikipedia.org`, and `it.wikipedia.org` also return
429 within the same 10-second window, the third call escalates to
`report_throttled(60)`, applying the global cooldown and rate reduction. This
protects against a system-wide overload.

## Test Strategy

Tests use the injected `clock` and `sleep` callables already established in
`test_utils.py` for deterministic verification.

- **Single-host no escalation**: one `report_host_throttled` call does not
  change `current_requests_per_minute` or apply a cooldown.
- **Repeated same-host no escalation**: multiple calls for the same host within
  the window still do not escalate (1 distinct host).
- **Multi-host escalation**: `threshold` distinct hosts within the window
  triggers the global rate halving and cooldown, verifiable via `current_requests_per_minute`
  and the sleep observed by `run()`.
- **Window pruning**: events older than `window_s` are forgotten, so a slow
  trickle of single-host failures across a long period never escalates.
- **Client call-site tests**: verify `HttpWikipediaClient` and
  `HttpWikidataClient` call `report_host_throttled` instead of
  `report_throttled` on a 429. Existing client tests that assert global
  backoff behavior are updated to reflect per-host semantics.

## Acceptance Criteria

- A single host's 429 or 503 does not reduce the global request rate or apply
  a global cooldown.
- Three or more distinct hosts throttled within 10 seconds do apply the global
  cooldown and rate reduction.
- `defer_host` behavior for the offending host is unchanged.
- All existing scheduler, client, and pipeline tests pass.
- The full lint, type-check, format, and test suite is green.
