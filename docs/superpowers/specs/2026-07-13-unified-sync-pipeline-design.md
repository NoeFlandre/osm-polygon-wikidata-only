# Unified Sync Pipeline Design

## Goal

Add one canonical `sync-dir <raw-dir>` command that converges every known region to the existing complete dataset contract: core OSM polygon tables, all-language Wikipedia full text, Wikipedia sections, all-language Wikivoyage documents and sections, and multilingual Wikidata facts with English labels. Existing `process-*` and `augment-*` commands remain compatible. No existing schema, row semantics, identifiers, paths, or artifacts change.

## Current state and constraints

The core manifest currently leads the augmentation manifest (130 versus 24 completed regions at design time). Two independently running commands create separate schedulers with eight slots each, despite Wikimedia rate limits being global across projects for an identity. This produces coordinated waves of HTTP 429 responses and leaves completed core regions waiting for augmentation.

The authenticated Bot Password budget is 1,200 requests per minute. The unified process will retain that full pacing ceiling while following Wikimedia's current recommendation of no more than three concurrent requests. Batching and useful CPU/network overlap—not excessive concurrency—will maximize throughput.

## Architecture

### State planner

A pure planner classifies each PBF stem from the raw directory and local manifests as:

- `complete`: core and current augmentation both exist;
- `augment`: core exists but augmentation is missing/stale;
- `process`: the raw PBF exists but core is missing/stale.

The planner orders `augment` work before `process` work so the existing backlog converges immediately. `--skip-existing` skips only `complete` regions. `--force` preserves its existing meaning and rebuilds selected work.

### One Wikimedia runtime

A `WikimediaRuntime` owns one `AdaptiveRequestScheduler`, one cookie-preserving `WikimediaSession`, and the core plus augmentation clients. In authenticated mode it uses:

- 1,200 requests/minute initial and maximum pacing;
- exactly three global in-flight requests;
- existing 50-item Wikidata/Wikipedia batches;
- shared global cooldown on 429/503 using `Retry-After`;
- multiplicative rate reduction after throttling and gradual recovery after successful windows;
- per-host pacing/cooldown for local overload.

The augmentation client receives this runtime rather than creating its own scheduler/session. Standalone legacy commands use the same factory, so their individual behavior and outputs remain compatible.

### Unified orchestration

`sync-dir` first augments every core-only region. It then processes each missing core region and immediately augments that region from the just-written core artifacts. Wikipedia documents are copied from the existing core article Parquet; they are never fetched again. Exact-revision HTML is fetched only for section splitting. Wikivoyage and Wikidata fact calls continue to use deterministic persistent caches.

A single-worker extraction prefetch may parse the next missing-core PBF while the current region performs network work. There is never more than one prefetched extraction, bounding memory and disk pressure. Network stages all use the one shared scheduler.

### Publication and recovery

Each stage remains atomic locally. For a core-only backlog region, its augmentation artifacts, augmentation manifest, and README are uploaded atomically as today. For a newly processed region, core output is durably written before augmentation; publication happens only through the existing resumable upload mechanisms. A failure leaves completed local artifacts and caches reusable on restart. No successful API response, extraction, or uploaded stage is intentionally repeated when `--skip-existing` is used.

An exclusive data-root sync lock prevents two `sync-dir` instances from duplicating work. It does not kill or mutate legacy processes; operators must stop old parallel commands before switching to `sync-dir`.

## Observability

The command logs initial counts for complete, augmentation-backlog, and unprocessed regions; each state transition; the shared authenticated rate ceiling and concurrency; throttle/recovery events; stage timings; and atomic upload completion. Progress makes it clear whether work is extraction, core enrichment, augmentation, or upload.

## Testing

TDD covers state planning, gap reconciliation, compatibility commands, shared scheduler identity, authenticated 1,200-RPM/three-slot configuration, no duplicate Wikipedia full-text requests, resumable skipping, lock behavior, failure recovery, and a small end-to-end fixture. Full pytest, Ruff, Mypy, and a live small-region smoke test are required before integration.

