# Enrichment Progress Heartbeat Design

## Goal

Prevent long enrichment stages from appearing stale after polygon extraction by
emitting one concise, factual progress heartbeat every two minutes. Logs should
describe dataset work rather than individual HTTP requests and must remain quiet
when enrichment completes quickly.

## Scope

This change adds progress observation and logging around the existing Wikidata
and Wikipedia enrichment workflow. It does not change request scheduling,
concurrency, caching, retries, enrichment results, output ordering, schemas, or
CLI options.

## User Experience

Immediately after extraction, the processor logs the number of unique valid
Wikidata QIDs that enrichment will resolve. If enrichment remains active for 120
seconds, it logs a snapshot similar to:

```text
Enrichment progress for afghanistan: 4m elapsed; Wikidata 143/143 QIDs; Wikipedia 18/64 sites, 742 articles attempted
```

Further snapshots appear no more than once every 120 seconds. Existing completion
and stage-timing logs remain unchanged. Enrichment finishing before the first
interval produces only the immediate start and existing completion messages.

Dependency construction also emits exactly one INFO startup summary stating
whether Wikimedia access is `anonymous` or `authenticated as <username>`, plus
the configured rate ceiling. The password is never included. Authentication
mode is not repeated in every heartbeat because it does not change during a run.

The snapshot fields are factual counters:

- Wikidata QIDs completed and total unique valid QIDs.
- Wikipedia language sites completed and total sites discovered from resolved
  sitelinks after language filtering.
- Wikipedia article titles attempted across completed site batches.

Counters may remain unchanged between heartbeats during a slow API call or retry;
the elapsed time still confirms that the process is alive. The logger never emits
per-request messages or estimates completion time.

## Components

### Progress snapshot and tracker

A focused enrichment progress module owns an immutable snapshot value and a
thread-safe mutable tracker. The tracker records the enrichment phase, QID totals
and completions, Wikipedia site totals and completions, and attempted article
titles. It exposes atomic update methods and one snapshot method; it does not log
or know about pipeline regions.

### Enrichment instrumentation

`fetch_qids` accepts an optional tracker. Existing callers that omit it behave
identically.

In the batched path:

- Each completed Wikidata chunk advances the QID counter.
- After sitelinks are grouped, the tracker receives the total number of site
  groups.
- Each completed site group advances the site counter and adds the number of
  distinct titles attempted by that group.

In the non-batched compatibility path, each completed QID advances the QID
counter. Wikipedia site/article counters remain unavailable there rather than
being inferred inaccurately.

Progress updates do not affect result construction, exception propagation, or
ordering.

### Heartbeat

A reusable heartbeat context manager owns one daemon thread and an injected
snapshot provider. It waits 120 seconds before its first log, then logs every 120
seconds until stopped. Its stop event interrupts the wait immediately, so normal
completion and exceptions cannot leave a sleeping thread or delayed message.

The processor creates the tracker, logs the immediate start summary, and wraps
only the `fetch_qids` call in the heartbeat context. The heartbeat formats the
region, monotonic elapsed time, and latest snapshot at INFO level.

The interval is a module constant rather than a new CLI setting. Clock/wait
dependencies remain injectable at the heartbeat boundary for deterministic unit
tests.

## Error Handling

- Tracker updates use a lock and cannot expose partially updated snapshots.
- The heartbeat always stops in context-manager cleanup when enrichment returns
  or raises.
- Snapshot or logging failures are contained inside the heartbeat thread and
  logged once at DEBUG; they never fail enrichment.
- Enrichment exceptions and existing incomplete-enrichment behavior propagate
  unchanged.
- Zero-QID and zero-site work produce accurate zero totals without division or
  percentage calculations.

## Testing Strategy

Implementation follows red-green-refactor:

- Tracker tests first specify atomic initial state, QID progress, site totals,
  site completion, article counts, and immutable snapshots.
- `fetch_qids` tests first specify batched and compatibility-path counter updates
  without changing returned summaries.
- Heartbeat tests first use controllable events and fake snapshot providers to
  prove no early message, one message per interval, factual formatting, immediate
  shutdown, and cleanup after exceptions.
- Processor tests first prove the immediate start message, tracker injection,
  region-specific heartbeat construction, and preservation of existing outputs.
- Dependency logging tests first prove the once-per-run anonymous/authenticated
  summary, authenticated username, rate ceiling, and password redaction.
- The complete test, coverage, lint, formatting, typing, and package-build gates
  run before integration.

## Documentation

The README reliability section and architecture documentation describe the
two-minute heartbeat, its counters, and the fact that it does not imply an ETA or
alter request pacing.

## Acceptance Criteria

- An immediate INFO message follows extraction and states the unique valid QID
  count.
- Startup logs state anonymous or authenticated Wikimedia mode exactly once;
  authenticated logs may identify the username but never the password.
- A long enrichment emits at most one domain-progress snapshot every 120 seconds.
- A short enrichment emits no heartbeat snapshot.
- Snapshots report completed/total QIDs, completed/total Wikipedia sites, attempted
  article titles, region, and elapsed time when those counters are available.
- No individual HTTP request is logged.
- Heartbeat cleanup is immediate on success and failure.
- Enrichment outputs, deterministic ordering, caching, retries, and rate behavior
  are unchanged.
- All quality gates pass and the work is merged into `main`.
