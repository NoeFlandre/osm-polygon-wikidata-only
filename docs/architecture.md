# Architecture

The project is intentionally layered so each concern can be tested in
isolation.

| Layer | Responsibility |
| --- | --- |
| `config` | Immutable runtime settings and external data-root resolution. |
| `domain` | Stable IDs, geometry/analysis helpers, flat dataset records, schemas. |
| `io` | PBF streaming, cache files, manifests, atomic writes, and Parquet persistence. |
| `enrichment` | Wikidata/Wikipedia clients, cache wrappers, batching, and linking. |
| `pipeline` | Extract, enrich, construct rows, write artifacts, update manifests, run sync. |
| `augmentation` | Augmentation orchestrator, focused pipeline steps, Wikimedia discovery, normalization. |
| `hf` | Remote paths, dataset card, dataset stats, geographic visualizations, publication, atomic Hub uploads. |
| `cli` | Argument parsing and dependency wiring only. |
| `utils` | Small utilities: JSON, time, retry, request scheduler. |

## Dependency direction

Dependencies point inward: CLI and pipeline orchestration compose I/O and
enrichment; enrichment depends on configuration, cache interfaces, and small
utilities; domain code is pure and does not import infrastructure. Stable
facade modules preserve documented imports while focused subpackages contain
models and implementation details.

The largest workflows are split by responsibility:

- `cli.parser` owns argparse and immutable settings conversion;
- `pipeline.rows` / `pipeline.row_construction` own deterministic
  domain-row construction;
- `pipeline.processor` sequences extraction, enrichment, publication, and
  metrics;
- `pipeline.sync_planner` owns deterministic sync-state planning;
  `pipeline.sync_runner` provides framework-free workflow orchestration
  using injected collaborators;
- `enrichment.wikidata.models` and `enrichment.wikipedia.models` define
  the typed contracts used across clients and linkers;
- `enrichment.wikimedia.transport` is the shared per-host throttle and
  `Retry-After` parsing boundary;
- `enrichment.wikidata.transport` and `enrichment.wikipedia.transport`
  own their respective `Http*` / `InMemory*` clients and JSON parsing;
- `hf.publication` owns the three pure upload-file assemblers
  (`assemble_core_upload`, `assemble_region_upload`,
  `assemble_augmentation_upload`) and the coverage refresh / README
  snapshot helpers; the same `write_readme_snapshot` recomputes both
  core and augmentation stats from finalized local Parquet inputs before
  each publication path (legacy core, unified sync, augmentation-only);
- `hf.coverage_map` and `hf.geographic_text_coverage` produce the
  deterministic PNG visualizations;
- `hf.dataset_stats` exposes the canonical `DatasetStats` /
  `compute_dataset_stats` / `render_stats_section` facade. The private
  augmentation scanner and private aggregation models live under
  `hf._dataset_stats.augmentation` and `hf._dataset_stats.models`
  and are NOT exported by the public facade;
- `hf.dataset_card` renders the multi-table README with the documented
  YAML configurations (one per core and augmentation table) and the
  augmentation schema descriptions sourced from
  `osm_polygon_wikidata_only.augmentation.schema_descriptions`.

Private implementation modules may evolve, but the supported imports in
[`docs/api.md`](api.md) are compatibility boundaries.

### Focused internal modules

Underscore-prefixed packages are private (do not import directly).
Other focused modules may also be implementation details behind
compatibility facades, even though they do not carry an underscore
prefix. All of the following may change without notice; import them
only through the public facades listed in
[`docs/api.md`](api.md).

Underscore-prefixed (private) packages:

- `osm_polygon_wikidata_only.hf._dataset_stats.{models,scanning,aggregation,rendering}`
- `osm_polygon_wikidata_only.hf._geographic.{models,parquet_inputs,h3_geometry,aggregation,basemap,rendering,coverage,polygon_count}`
- `osm_polygon_wikidata_only.hf._uploader.{errors,protocol,stub,token,authorization,operations}`

Other focused modules (not underscore-prefixed but still implementation
details behind facades):

- `osm_polygon_wikidata_only.enrichment.wikimedia.transport` — shared
  Wikimedia transport (per-host throttle, JSON read).
- `osm_polygon_wikidata_only.enrichment.wikidata.{models,cache,transport,parsing}`
  — Wikidata client split: typed contracts, on-disk cache, HTTP/InMemory
  client, QID/sitelink/entity parsing.
- `osm_polygon_wikidata_only.enrichment.wikipedia.{models,cache,transport,parsing}`
  — Wikipedia client split: typed contracts, on-disk cache, HTTP/InMemory
  client, response parsing.
- `osm_polygon_wikidata_only.augmentation.steps` — focused augmentation
  pipeline helpers (Wikidata fact builder, document fetcher,
  sidecar updater, augmentation manifest merge).
- `osm_polygon_wikidata_only.pipeline.sync_runner` and
  `osm_polygon_wikidata_only.pipeline.sync_orchestrator` — framework-free
  orchestration for the unified sync workflow. `sync_runner` performs
  the actual workflow execution (prefetch, augment-backlog, process,
  augment, complete) with injectable collaborators; it is not a pure
  state machine.
- `osm_polygon_wikidata_only.hf._uploader` (also listed above): the
  dependency graph is acyclic -- errors → protocol → stub / token →
  authorization / operations.

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

## Exception boundary policy

The codebase deliberately retains broad `except Exception` boundaries at
five well-defined sites, plus `except BaseException` at the two atomic
write helpers. The reasoning, in each case, is documented in the source
next to the boundary and pinned by focused tests:

| Site | Why `except Exception` (or `BaseException`) is retained |
| --- | --- |
| `io.atomic.atomic_write_text`, `hf._geographic.rendering.atomic_save_png` | `BaseException` is required so the temporary-file cleanup branch fires on `KeyboardInterrupt` and `SystemExit`. A narrow `Exception` would leak temp files on Ctrl-C. |
| `hf._uploader.operations` | `huggingface_hub` legitimately exposes a broad set of unstable exception types (`HfHubHTTPError`, `RepositoryNotFoundError`, `EntryNotFoundError`, `BadRequestError`, `OSError`, ...). The operations module translates every one of them into `UploadError` via `_translate_hf_error` so callers see a uniform exception type, with special handling for 401/403/404 and the auth-marker substring list. |
| `hf._uploader.token` | Same third-party rationale as `_uploader.operations`. `resolve_hf_token` swallows backend exceptions when probing `get_token`; `verify_hf_token` wraps `whoami` failures as `UploadError`. |
| `hf.upload_queue` | Different behavior. The worker does not translate every exception into `UploadError`; it records each failed job's detail (with the underlying exception appended to the message) into its `failures` list and lets the daemon worker survive to process the next queued job. The `except Exception` boundaries ensure a single bad upload cannot take down the queue thread. |
| `pipeline.heartbeat.EnrichmentHeartbeat.run` | Observational heartbeat failures are contained and logged at debug without disrupting the pipeline. A daemon thread must not propagate uncaught exceptions to the surrounding pipeline context. |
| `hf.publication.refresh_coverage_assets`, `hf.publication.snapshot_upload_manifests` | `ensure_world_land` performs network I/O via `urllib.request.urlretrieve` which raises a broad, unstable set of exception types (`URLError`, `HTTPError`, `ContentTooShortError`, `socket.timeout`, `OSError`). Documented fallback: render without continents + invoke `world_land_warning` (when not `None`). |
| `hf._geographic.parquet_inputs.read_required_columns` | PyArrow's metadata API raises across several unstable exception types depending on the corruption mode (`OSError`, `ArrowInvalid`, `ArrowKeyError`, `ArrowIOError`). When the metadata read fails, the implementation falls through with an empty `actual` column-name set and lets the subsequent column-pruned `pq.read_table` call determine the outcome: a valid parquet with the requested columns still loads successfully; missing columns are translated into `CoverageMapError`. |

## Compatibility contract

The CLI, Parquet schemas, manifest paths, deterministic ordering, and public
client classes are stable. New internals must be introduced behind existing
public functions or explicit capability protocols.

## Verification

Run `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`,
and `uv run mypy src` before merging a change.
