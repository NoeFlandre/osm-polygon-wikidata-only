# Architecture

The project is intentionally layered so each concern can be tested in
isolation.

| Layer | Responsibility |
| --- | --- |
| `config` | Immutable runtime settings and external data-root resolution. |
| `domain` | Stable IDs, geometry/analysis helpers, flat dataset records, schemas. |
| `io` | PBF streaming, cache files, manifests, and Parquet persistence. |
| `enrichment` | Wikidata/Wikipedia clients, cache wrappers, batching, and linking. |
| `pipeline` | Extract, enrich, construct rows, write artifacts, and update manifests. |
| `hf` | Remote paths, dataset card rendering, atomic Hub uploads. |
| `cli` | Argument parsing and dependency wiring only. |

## Dependency direction

Dependencies point inward: CLI and pipeline orchestration compose I/O and
enrichment; enrichment depends on configuration, cache interfaces, and small
utilities; domain code is pure and does not import infrastructure. Stable
facade modules preserve documented imports while focused subpackages contain
models and implementation details.

The largest workflows are split by responsibility:

- `cli.parser` owns argparse and immutable settings conversion;
- `pipeline.rows` owns deterministic domain-row construction;
- `pipeline.processor` sequences extraction, enrichment, publication, and
  metrics;
- `enrichment.wikipedia.models` and `enrichment.wikidata.models` define the
  typed contracts used across clients and linkers.

Private implementation modules may evolve, but the supported imports in
[`docs/api.md`](api.md) are compatibility boundaries.

## Wikimedia request scheduling

One process-wide `AdaptiveRequestScheduler` is the single source of
truth for Wikimedia request pacing. The scheduler is hierarchical:

- A global **client-side rate ceiling** caps requests per minute. The
  default is `180` rpm for anonymous runs and `1200` rpm for runs
  authenticated via a Bot Password. This ceiling is *not* a guaranteed
  server allowance; the API may still throttle clients below it.
- A global **concurrency bound** (`max_in_flight`) caps simultaneous
  in-flight requests across every Wikimedia host. The default is `3`
  for anonymous runs and `8` for authenticated runs. `8` is enough to
  saturate the `1200` rpm ceiling at typical API latency, while
  staying well under the scheduler's hard cap of `16`.
- Each host keeps **independent per-host state**: a cooldown clock
  (set by `Retry-After` and back-pressure) and a minimum interval
  between requests. Per-host pacing happens *before* the global
  permit is acquired so a host stuck in a long cooldown cannot hold
  a scarce global permit and starve unrelated hosts.
- A single host's `429`/`503` cools down only that host. The global
  rate is reduced **only when throttling is systemic** — when
  several distinct hosts are throttled within a bounded window —
  and the systemic decision plus its suppression-timestamp update
  happen atomically, so a flurry of throttles does not repeatedly
  halve the global rate within seconds.

`WikimediaSession` is the single transport boundary. It owns the
per-host authentication state (login handshake performed lazily per
host, with the bot password verified against the host's API endpoint)
and exposes the per-host pacing decision: hosts that have *verified*
authentication are paced at the authenticated minimum interval;
hosts contacted anonymously or whose bot password was rejected are
paced at the per-kind anonymous interval. Authentication state is
telemetry-reported via `WikimediaAuthSnapshot`, which counts
`authenticated_hosts`, `anonymous_hosts`, and `pending_hosts`
(hosts whose login is currently in flight), so a host that might
still verify is never mislabelled as anonymous.

Long enrichment is observable without request-level noise. A thread-safe tracker
records completed and total QIDs, completed and total Wikipedia sites, and
articles attempted. The processor reads an immutable snapshot in a two-minute
heartbeat that also names the active Wikidata or Wikipedia phase, and stops its
daemon immediately when enrichment returns or raises. This is a liveness signal,
not an ETA, and it does not alter request pacing, ordering, retries, or output
construction.

A PBF is published locally only after every expected article succeeds. Its
three Parquet files, manifest snapshot, and generated Hugging Face dataset card
are then queued in one background upload commit while the next PBF begins.
Failed upload jobs persist under the external data root and resume on the next
invocation. The dataset and pipeline are maintained by Noé Flandre.

## Geographic coverage visualizations

Every successful core publication regenerates two deterministic PNGs
before the README snapshot is rendered, both using the same H3 resolution
3 layout, basemap, world extent, deterministic ordering, and atomic
write-through-temp-file:

- `assets/geographic_wikipedia_text_coverage.png` aggregates polygons
  into H3 cells and colours each cell by the fraction linked to at least
  one Wikipedia article with non-empty text. The denominator is the full
  set of dataset polygons (already conditional on an OSM `wikidata=*`
  tag); the numerator counts unique polygons, never polygon-article
  links. Cell opacity is constant for eligible cells; grey cells below
  twenty polygons are flagged as low-sample in the caption.
- `assets/geographic_polygon_count.png` colours each H3 cell by its raw
  polygon count on a logarithmic scale. Each dataset polygon is counted
  exactly once, conditioned on the upstream `wikidata=*` filter. Low
  counts remain visible; opacity is not used as a second data encoding.

Both assets are generated only by the legacy core upload path and the
canonical `sync-dir` core publication path; augmentation-only work
reuses the existing assets because their inputs do not affect the
metrics. Both images are embedded in the source `README.md` and the
generated Hugging Face dataset card.

## Compatibility contract

The CLI, Parquet schemas, manifest paths, deterministic ordering, and public
client classes are stable. New internals must be introduced behind existing
public functions or explicit capability protocols.

## Verification

Run `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`,
and `uv run mypy src` before merging a change.
