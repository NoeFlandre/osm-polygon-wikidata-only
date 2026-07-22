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
| `hf` | Remote paths, dataset card, dataset stats, geographic visualizations, publication, atomic Hub uploads. ALL remote paths published by this codebase are centralized in `hf.repo_layout`; the single exception is the named legacy migration constant `LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE` consumed only by the atomic migration commit that unifies the augmentation manifest under `manifests/`. |
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
  immutable `CorePublicationArtifacts` and `PublicationValidationError`
  definitions live in `hf._publication.models` and are re-exported by the
  facade;
- `hf.coverage_map`, `hf.geographic_text_presence`, and
  `hf.geographic_text_coverage` produce the deterministic PNG visualizations;
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
- `osm_polygon_wikidata_only.hf._publication.models`
- `osm_polygon_wikidata_only.hf._uploader.{errors,protocol,stub,token,authorization,operations,plan}`
- `osm_polygon_wikidata_only.pipeline._wikidata_recovery.storage`, which owns
  schema-checked recovery reads/writes and canonical local paths; recovery
  result/error models, including `RecoveryRepairResult`, live beside the audit
  models and remain available through the recovery facade

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
  the actual workflow execution (AUGMENT backlog, PUBLISH-only
  reconciliation repairs, PROCESS, COMPLETE) with injectable
  collaborators; it is not a pure state machine.
- `osm_polygon_wikidata_only.pipeline.local_validation` — bounded
  startup progress reporter for the local augmentation-state
  validation phase that gates the unified sync.
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
and uses one process-wide HTTP/1.1 connection pool bounded by the same
global concurrency limit. Cookies remain domain-scoped while live
connections are reused across requests to avoid repeated TCP/TLS setup.
The session also exposes the per-host pacing decision: hosts that have *verified*
authentication are paced at the authenticated minimum interval;
hosts contacted anonymously or whose bot password was rejected are
paced at the per-kind anonymous interval. Authentication state is
telemetry-reported via `WikimediaAuthSnapshot`, which counts
`authenticated_hosts`, `anonymous_hosts`, and `pending_hosts`
(hosts whose login is currently in flight), so a host that might
still verify is never mislabelled as anonymous.

Production requests retry classified transient failures without a fixed
attempt ceiling: temporary DNS/connectivity failures, timeouts, connection
resets, `429`, and retryable `5xx` responses wait with capped exponential
backoff until service returns or the user interrupts the process. Sparse
warnings confirm that the pipeline remains active. Permanent HTTP errors,
invalid payloads, authentication/configuration failures, and other
non-network exceptions still fail immediately; tests may configure a finite
attempt count for deterministic failure cases.

Long enrichment is observable without request-level noise. A thread-safe tracker
records completed and total QIDs, completed and total Wikipedia sites, and
articles attempted. The processor reads an immutable snapshot in a two-minute
heartbeat that also names the active Wikidata or Wikipedia phase, and stops its
daemon immediately when enrichment returns or raises. This is a liveness signal,
not an ETA, and it does not alter request pacing, ordering, retries, or output
construction.

A PBF is published locally only after every expected article succeeds. Its
core Parquet files (polygons, the temporary local `articles/` staging table,
and polygon links), manifest snapshot, and generated Hugging Face dataset card
are then queued in one background upload commit while the next PBF begins.
The remote upload atomically adds the canonical `wikipedia/documents/` table
and deletes the legacy `articles/` path in the same commit; the local staging
file is removed only after confirmed publication. Failed upload jobs persist
under the external data root and resume on the next invocation. The dataset
and pipeline are maintained by Noé Flandre.

## Geographic coverage visualizations

Every successful core publication regenerates four deterministic PNGs
before the README snapshot is rendered. The H3 maps share resolution 3,
the basemap and world extent, deterministic ordering, and atomic writes.
Antimeridian cells are clipped into closed local polygons so a renderer
cannot draw world-spanning closure lines:

- `assets/geographic_text_presence.png` is the first public map and plots
  each polygon with non-empty Wikipedia or Wikivoyage text exactly once.
- `assets/coverage_map.png` displays the global distribution of the dataset polygons as a scatter plot of centroids.
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

All four assets are generated by core publication paths. Augmentation-only
work regenerates the combined Wikipedia/Wikivoyage text-presence map because
new Wikivoyage documents can change that metric, while reusing the three
core-only maps. The generated dataset card also computes continent statistics
from polygon centroids and the bundled Natural Earth Admin-0 continent
classification. All four images are embedded in the generated Hugging
Face dataset card.

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

## Remote reconciliation & publication convergence

To handle publication-convergence defects (e.g., local processed stems missing from the remote repository after interrupted runs), the pipeline integrates a remote reconciliation phase:

- **Presence-based reconciliation**: The pipeline compares the set of expected remote canonical parquets for a region against the files actually present on the Hugging Face Hub. Gaps are determined strictly by path presence.
- **Input scoping**: Reconciliation is strictly scoped to the input PBF stems specified in the command arguments. Stems outside the command's input scope are not reconciled or validated to prevent unrelated local issues from aborting the pipeline.
- **Restart recovery & resumability**: If a local region is completely processed and augmented but some of its core or augmentation parquets are missing on the remote, the pipeline schedules a publication repair. Missing core files are recovered directly from finalized local parquet files without re-triggering expensive raw PBF extraction, enrichment, or Wikidata lookups.
- **Single remote inventory read**: The remote file inventory is fetched exactly once at the beginning of the command if `--push` is active, avoiding redundant API calls and rate-limiting.
- **Metadata refresh**: A repository-level metadata repair (updating the manifest, README, maps, and coverage charts) is enqueued at the end of the run if any repository-level assets are missing on the remote.

## Unified sync action priority

Before action planning, containment migration audits the small checked-in set
of known whole-file Geofabrik overlaps. Polygon identity containment is a hard
precondition. Missing non-core rows are unioned into staged parent tables;
original parent and child artifacts are copied to
`quarantine/containment-v1/` before active children are retired. Publication
uploads parent replacements, refreshed manifests, README and maps, plus child
deletions as one Hugging Face commit. The durable
`manifests/containment_retirements.json` prevents retired raw PBFs from
re-entering later `sync-dir` plans.

The unified sync (`sync-dir`) runs every region through one of five
mutually exclusive action buckets. Within each bucket, stems are
processed alphabetically; the planner produces a deterministic plan
that the runner drains in this exact order:

1. **RECOVERY** -- finalized, current regions eligible for an exhaustive
   QID-level integrity audit. The runner audits one region at a time; a healthy
   region stores or reuses its content-addressed receipt and advances without
   publication, while a damaged region is repaired and published before the
   next region begins. There is no global all-QID validation barrier. The audit
   uses column-pruned reads of polygons, links, and
   canonical Wikipedia documents, plus the small identity columns of Wikidata
   facts. It validates only missing relationships against
   authoritative Wikidata state, and reuses content-addressed receipts for
   unchanged healthy inputs. Affected QIDs are refetched in deterministic
   groups of 25. Up to three independent groups run concurrently, while the
   shared scheduler remains the sole authority for global and per-host request
   limits. Each completed group is stored immediately under
   `cache/wikidata_recovery/checkpoints/<stem>/<plan-hash>/` as schema-validated
   Parquet without waiting for slower groups. The plan hash covers regional input
   fingerprints, section content, affected QIDs, and relevant settings, so a
   checkpoint cannot be reused after its inputs change. Restarting repeats only
   unfinished groups; completed groups are reused without refetching or
   reparsing. Within each group, up to eight QIDs fetch Wikipedia documents
   concurrently and up to eight documents fetch section HTML concurrently;
   both use the existing shared scheduler and their results are flattened in
   deterministic input order. A 60-second heartbeat reports each active group and stage,
   documents, sections, facts, elapsed time, estimated remaining time,
   request-rate utilization, in-flight requests, rolling throttles, and cooling hosts.
   After all groups are durable, repaired core,
   documents, sections, facts, and both manifests are replaced as one durable
   journaled transaction before an atomic regional publication. Checkpoints
   are removed only after the post-repair audit converges; Hugging Face never
   receives a partial group. Orphan fact
   rows whose subject QID is absent from every regional polygon are pruned in
   that transaction without refetching or changing joinable facts. Such a
   facts-only repair refreshes manifests, statistics, and the README but reuses
   all maps; map rendering runs only when polygon, link, or document inputs
   changed. A no-op repair produces neither a publication nor regenerated
   artifacts. Transport or
   validation failures write neither a terminal receipt nor partial outputs;
   a blocked finalized shard is left unchanged and aborts the command before
   extraction begins; already completed regional repairs remain durable.
   The audit emits bounded checkpoints for its local scan and authoritative
   validation phases. Wikidata HTTP-200 API errors are inspected before entity
   parsing: transient codes such as `maxlag`, `readonly`, and `ratelimited`
   remain inside the existing retry loop, while permanent or structurally
   malformed responses fail closed with their API code and message. The same
   validation happens before augmentation responses enter the shared cache;
   legacy cached API-error payloads are evicted and fetched again automatically.
2. **AUGMENT backlog** -- the existing augmentation backlog. Regions
   whose core is finalized but whose augmentation is stale or
   missing are repaired first; each AUGMENT call performs
   Wikimedia sidecar work and, on success, enqueues an atomic
   remote publication for that region.
   A newly completed augmentation is audited in the same invocation, so a
   previously incomplete region cannot be left behind by the recovery phase.
3. **PUBLISH** -- safe, Wikimedia-free publish-only reconciliation
   repairs. Regions whose local core and augmentation artifacts
   are already finalized but missing from the remote are uploaded
   using `load_existing_augmentation` -- no extraction, no
   Wikidata lookup, no Wikipedia parse, no Wikivoyage fetch. The
   repair only enqueues a Hugging Face upload.
4. **PROCESS** -- new core processing. Regions whose local core
   artifacts are missing run PBF extraction, enrichment, and
   augmentation. The first PROCESS extraction is prefetched in
   a background thread so PUBLISH-only repairs above can overlap,
   and the runner may prefetch subsequent PBF extractions
   concurrently while enriching the current region (the
   one-PBF-ahead invariant).
5. **COMPLETE** -- no action required; convergence.

Maps / README "refreshed" claims in the final log line are
authoritative only when a successful core or metadata-only
publication actually refreshed them; success is reported only
after the background upload queue has drained successfully.
Upload failures remain retryable on the next invocation because
the durable pending-publications manifest and the upload-queue
state files survive the failure.

### Startup visibility

The local augmentation-state validation phase iterates over every
input stem and may take several minutes for large datasets. The
`pipeline.local_validation.LocalValidationProgress` coordinator
emits a single begin log line, bounded periodic progress lines
(suppressed for inputs smaller than 25 stems), and a single
completion log line with the total elapsed time. Each stem is
visited exactly once. The clock is injectable for deterministic
tests.

## Compatibility contract

The CLI, Parquet schemas, manifest paths, deterministic ordering, and public
client classes are stable. New internals must be introduced behind existing
public functions or explicit capability protocols.

## Verification

Run `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`,
and `uv run mypy src` before merging a change.
