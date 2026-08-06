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
| `v2` | Isolated Wikipedia-tag dataset contract, V1 reuse index, direct-page enrichment, storage, card, and publication runner. |
| `augmentation` | Augmentation orchestrator, focused pipeline steps, Wikimedia discovery, normalization. |
| `hf` | Remote paths, dataset card, dataset stats, geographic visualizations, publication, atomic Hub uploads. ALL remote paths published by this codebase are centralized in `hf.repo_layout`; the single exception is the named legacy migration constant `LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE` consumed only by the atomic migration commit that unifies the augmentation manifest under `manifests/`. |
| `cli` | Argument parsing and dependency wiring only. |
| `utils` | Small utilities: JSON, time, retry, request scheduler. |

## Project toolchain

uv owns dependency resolution, the lockfile, isolated command execution, and
package builds. pytest owns behavioral and contract verification; Ruff owns
linting and formatting; ty is the sole static type checker. A root `Justfile`
is the human-facing command catalog, and GitHub Actions delegates its coverage,
lint, format-check, type-check, and build steps to those same recipes.
Repository-local pre-commit hooks run the fast Ruff and ty subset.

The production CLI remains argparse-based because its help and parsing behavior
are public compatibility contracts. The independent, read-only remote audit is
an operator interface: Typer parses its options, Rich renders deterministic
tables, and tqdm displays its local-region scan only on an interactive
terminal. This keeps interactive presentation concerns out of pipeline
orchestration and dataset generation.

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
  facade; `hf._publication.artifacts` is the single schema-validation and
  finalized-core loading boundary;
- `hf.coverage_map`, `hf.geographic_text_presence`, and
  `hf.geographic_text_coverage` produce the deterministic PNG visualizations;
- `hf.dataset_stats` exposes the canonical `DatasetStats` /
  `compute_dataset_stats` / `render_stats_section` facade. The private
  augmentation scanner and private aggregation models live under
  `hf._dataset_stats.augmentation` and `hf._dataset_stats.models`
  and are NOT exported by the public facade. Deterministic cache encoding is
  isolated in `hf._dataset_stats.summary_codec`;
- `hf.dataset_card` renders the multi-table README with the documented
  YAML configurations (one per core and augmentation table) and the
  augmentation schema descriptions sourced from
  `osm_polygon_wikidata_only.augmentation.schema_descriptions`.

Private implementation modules may evolve, but the supported imports in
[`docs/api.md`](api.md) are compatibility boundaries.

## V2 Wikipedia-tag workflow

V2 is selected only with `sync-dir --dataset-version v2`. It does not call the
V1 processor or alter the V1 `processed/` tree. The runner scans each source
PBF with an opt-in reader that retains polygonal elements carrying either a
valid Wikidata tag or a valid multilingual Wikipedia reference. It then merges
those discoveries with the finalized V1 polygon rows.

The read-only V1 reuse index is keyed by normalized `(language, page_id)` and
title, with QID indexes for diagnostics. A direct Wikipedia tag first checks
that index. Only a page absent from V1 reaches the existing Wikipedia client,
wrapped in a V2-versioned JSON cache under `cache/v2/`. This preserves the
V1 corpus while avoiding duplicate HTTP work.

The V1 reuse index is persisted in `cache/v2/v1-index/`. It validates and
commits one V1 Parquet shard at a time, keeps only identity metadata in SQLite,
and loads matching document rows on demand. Each Parquet row group is a
separate transaction, so restarting resumes inside an interrupted shard rather
than rebuilding completed row groups. The runner builds this index in the
background while it extracts the next PBF, and its SQLite storage initialization
also runs in that background worker. Until storage is ready, lookups are
treated as provisional misses. Rows from committed shards can be reused
immediately. A title not found yet may be fetched speculatively while the
index continues; after indexing completes, a later V1 match always wins, and
only a title still absent from V1 keeps the fetched result. This removes idle
waiting without turning a partial index miss into the final decision.

V2 writes an isolated `processed_v2/` tree. Its `polygons` schema adds direct
tag references, structured rejections, and discovery-source provenance. Its
`wikipedia/documents` schema permits a null Wikidata QID for direct-only pages.
`polygon_document_links` is the single V2 relationship table for Wikipedia
and Wikivoyage, with `link_sources` identifying Wikidata sitelinks versus
direct Wikipedia tags. Existing V1 sidecars are copied by content and are
never deleted or rewritten in V1 storage. The V2
`wikipedia/sections/<stem>.parquet` artifact is required and uses the exact V1
22-column section schema. Existing section rows are reused; only Wikipedia
documents without a persisted section are fetched at their exact revision and
parsed with the shared V1 section parser.

Extraction and fetch progress is durable under
`<data-root>/cache/v2/checkpoints/`. Extraction writes schema-checked Parquet
chunks periodically and records the source PBF fingerprint, so an interrupted
scan reuses valid chunks and invalidates them if the input changes. Direct page
results are saved after each completed polygon, and section results are saved
after each completed document, including explicit empty-section markers. The
checkpoint contract also records the input references and full-text mode;
corrupt, stale, or mismatched state is discarded and retried. Checkpoints are
cleared only after the final region files are written and reconciled, so a
stopped run never exposes partial data as a completed region.

Each region is written through staged Parquet files and a journal-free
replacement transaction, then the V2 manifest is updated atomically. A
completed local region is skipped on a non-publishing rerun; when publishing,
the runner checks the optional remote inventory and uploads an existing local
region if the Hub is incomplete. Region commits are followed by a deterministic
V2 README and manifest publication. A failed upload leaves local artifacts
available for retry and never causes V1 rollback.

### Focused internal modules

Underscore-prefixed packages are private (do not import directly).
Other focused modules may also be implementation details behind
compatibility facades, even though they do not carry an underscore
prefix. All of the following may change without notice; import them
only through the public facades listed in
[`docs/api.md`](api.md).

Underscore-prefixed (private) packages:

- `osm_polygon_wikidata_only.hf._dataset_stats.{models,scanning,aggregation,rendering,summary_codec}`
- `osm_polygon_wikidata_only.hf._geographic.{models,parquet_inputs,h3_geometry,aggregation,basemap,rendering,coverage,polygon_count}`
- `osm_polygon_wikidata_only.hf._publication.{models,artifacts}`
- `osm_polygon_wikidata_only.hf._uploader.{errors,protocol,stub,token,authorization,operations,plan}`
- `osm_polygon_wikidata_only.pipeline._link_migration.models`,
  `pipeline._link_migration.conversion`, and
  `pipeline._link_migration.transaction`, which own immutable plans, pure
  legacy-link conversion, and crash-safe ordered replacement behind
  `pipeline.link_migration`;
- `osm_polygon_wikidata_only.pipeline._wikidata_recovery.storage`,
  `pipeline._wikidata_recovery.link_rows`, and
  `pipeline._wikidata_recovery.validation`, which own schema-checked recovery
  I/O, link conversion, referential integrity, and preservation checks;
  recovery result/error models, including `RecoveryRepairResult`, remain
  available through the recovery facade;
- `osm_polygon_wikidata_only.cli._sync.retirement`, which owns fail-closed
  canonical-add/legacy-delete pairing while `cli.run_sync` remains the
  composition root

Other focused modules (not underscore-prefixed but still implementation
details behind facades):

- `osm_polygon_wikidata_only.v2.sections` — exact-revision Wikipedia HTML
  adaptation, deterministic section ordering, and per-document checkpoint
  callbacks used by the V2 merge coordinator.
- `osm_polygon_wikidata_only.v2.index_scanning` — schema-checked V1 shard
  discovery, bounded row-group projection, and Parquet handle ownership used
  by the resumable V2 reuse index.
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
- `osm_polygon_wikidata_only.augmentation.checkpoints` — private durable
  phase and bounded-section-batch checkpoints for interrupted augmentation.
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
  for anonymous runs and `12` for authenticated runs. `12` provides
  enough headroom to approach the `1200` rpm ceiling when API latency
  is around half a second, while staying below the hard cap of `16`.
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
under the configured local data root and resume on the next invocation. The dataset
and pipeline are maintained by Noé Flandre.

## Augmentation interruption recovery

Normal regional augmentation checkpoints completed derived work beneath
`<data-root>/cache/augmentation_checkpoints/`; in production this is on the
operator-configured external data volume. The content key includes the exact
core hashes, QIDs, Wikipedia document identities, and checkpoint contract
version. Entity payloads, Wikivoyage documents, and Wikidata facts are saved at
complete phase boundaries. Article sections are saved in deterministic batches
of 50 documents, so an interruption repeats at most the active batch; the
existing request cache also preserves successful requests inside that batch.

Checkpoint directories are published atomically and loaded only when their
metadata, schemas, and input identities match exactly. Corrupt, incomplete, or
stale checkpoints are ignored. Canonical sidecars retain their existing
all-phases transaction boundary, and a region's checkpoint directory is removed
only after integrity enforcement and its augmentation manifest entry complete
successfully.

## Geographic coverage visualizations

Every successful publication that changes core or text inputs regenerates
three deterministic PNGs before the README snapshot is rendered. The H3
map uses resolution 3, the shared basemap and world extent, deterministic
ordering, and atomic writes. Antimeridian cells are clipped into closed
local polygons so a renderer cannot draw world-spanning closure lines:

- `assets/geographic_text_presence.png` is the first public map and plots
  each polygon with non-empty Wikipedia or Wikivoyage text exactly once.
- `assets/dataset_hero.png` is the static overview image shown at the top of
  the GitHub README and Hugging Face dataset card. It is published alongside
  each README snapshot and is separate from the generated maps below.
- `assets/coverage_map.png` displays the global distribution of the dataset polygons as a scatter plot of centroids.
- `assets/geographic_text_density.png` assigns qualifying polygon centroids
  to H3 cells and colours each cell by the raw number of unique polygons
  having non-empty Wikipedia or Wikivoyage text. A polygon is counted once
  even if both projects or several documents qualify. The logarithmic
  purple-to-yellow scale keeps both sparse and dense cells visible.

Core publication paths generate all three assets. Augmentation-only work
regenerates the two Wikipedia/Wikivoyage-sensitive maps while reusing the
all-polygons map. In the same atomic Hub commit, publication adds the new
text-density map and deletes the superseded Wikipedia-rate and all-polygon
H3 maps. The generated dataset card also computes continent statistics
from polygon centroids and the bundled Natural Earth Admin-0 continent
classification. All three images are embedded in the generated Hugging
Face dataset card.

## Final Trackio snapshot

The project publishes one static Trackio run named
`final-dataset-snapshot`. It contains the finished dataset's headline scale,
corpus, language, and regional metrics, a small storage/link table, and exactly
three plots: the text-coverage funnel, the top ten Wikipedia languages plus
`Other languages`, and dataset composition on a logarithmic scale. Automatic
CPU/GPU metrics, API calls, retries, cache counters, throughput, and processing
timeline data are disabled. Local Trackio state is written under the configured
data-root cache, and the static dashboard is published at
`https://huggingface.co/spaces/NoeFlandre/osm-polygon-wikidata-only-trackio`.

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

## Unified polygon-to-document links

`polygon_articles/<stem>.parquet` is the canonical many-to-many join from
polygons to both Wikipedia and Wikivoyage documents. Its `project` and
`document_id` columns identify the target corpus and revision; every row must
resolve to one polygon and one document with a matching Wikidata identifier.

`sync-dir` upgrades legacy Wikipedia-only link shards from finalized local
tables without PBF extraction or Wikimedia requests. Invalid historical
relationships are recorded in a deterministic rejection ledger, while valid
Wikipedia links and all valid Wikivoyage relationships are retained. The link
file, any normalized sidecars, both manifests, the rejection ledger, and
durable publication intent are committed through one resumable journal.
Regional data uploads drain first; maps, aggregate statistics, and the dataset
card are regenerated once at the end and only then is the metadata-refresh
marker cleared. Both legacy and canonical link schemas are accepted explicitly
during rollout, but every successful migration writes only the canonical
schema.

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
   validation phases. Up to three independent validation chunks overlap under
   the same shared scheduler, so one lagged entity cannot idle unrelated work.
   Wikidata HTTP-200 API errors are inspected before entity
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

Run `just check` before merging a change. It executes the frozen uv sync,
pytest coverage suite, Ruff lint and format checks, ty, the package build, and
the whitespace gate used by GitHub Actions.
