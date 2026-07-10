# Complete API Pipeline Design

## Goal

Process every eligible OSM polygon and every valid language-Wikipedia sitelink
through Wikimedia APIs as quickly as the unauthenticated service permits,
without publishing incomplete results. Every article includes full text. There
is no language allow-list and no per-QID article cap.

Completed PBF artifacts upload in the background while the next PBF is being
processed. The command waits for outstanding uploads before exiting and fails
if any artifact remains unpublished.

## Non-negotiable data contract

- Polygon eligibility and conversion remain unchanged: closed ways and
  multipolygon relations with a non-empty `wikidata` tag are processed in
  deterministic source order.
- All valid language-Wikipedia sitelinks returned for every valid QID are
  expected work. Commons, Wikidata, and other non-language wiki projects remain
  excluded by the existing language-Wikipedia predicate.
- Full article text is always fetched. Language filtering and article caps are
  removed from the production CLI defaults and execution path.
- A successful PBF contains every convertible polygon, every expected article,
  and every expected polygon-article link. An unresolved QID or article prevents
  completion rather than becoming a silently partial dataset.
- Parquet schemas, stable identifiers, row ordering, polygon filtering, and
  article parsing remain compatible unless a separately approved schema change
  is required later.
- No incomplete PBF is added to the completed manifest or uploaded.

## Architecture

### Work discovery and deduplication

The processor extracts all polygons using the existing reader and conversion
logic, then deduplicates their QIDs while retaining deterministic first-seen
order. Wikidata entities are fetched in maximum API-safe batches. Their
sitelinks are filtered only by the existing valid language-Wikipedia rule.

Expected article work is deduplicated by `(site, title)` before any Wikipedia
request. The processor retains the association from each QID to every expected
sitelink so one fetched article can satisfy repeated polygons and QIDs without
changing link construction.

### Global adaptive Wikimedia scheduler

All Wikidata and Wikipedia HTTP calls share one process-wide scheduler because
Wikimedia's unauthenticated API limits apply globally across projects. The
scheduler:

- permits no more than three requests in flight across all Wikimedia hosts;
- uses a global token budget below the documented unauthenticated identified
  client ceiling, with configuration for operational tuning;
- sends maximum safe title/QID batches to minimize request count;
- honors `Retry-After` through one shared global cooldown;
- treats HTTP 429, HTTP 503, MediaWiki `maxlag`, timeouts, and transient network
  failures as retryable;
- uses bounded exponential backoff with jitter and slowly increases throughput
  after sustained success;
- requests gzip compression, reuses HTTP connections, includes `maxlag`, and
  sends the identifying User-Agent on every request; and
- exposes deterministic hooks so timing, concurrency, and recovery are tested
  without sleeping or accessing the network.

Per-host pacing may remain as a secondary safety boundary, but it cannot be the
primary quota mechanism.

### Batch validation and fallback

Every batch response is validated against the complete requested member list.
A batch optimization is successful only for members with a structurally valid
terminal success response. Missing or malformed members return to the durable
pending queue and are retried in smaller batches, eventually individually.

Retries are bounded per command invocation. Exhausting that budget leaves the
work pending on disk and fails the affected PBF. It never turns missing work
into a successful empty result. A later identical command resumes pending work.

### Durable enrichment journal

A versioned, atomic journal under the external data root records:

- PBF identity and processing-policy fingerprint;
- ordered polygon/QID discovery metadata needed to validate a resume;
- the expected sitelink set for each resolved QID;
- successful Wikidata entity responses;
- successful full-text article responses keyed by site and title; and
- retry metadata and the last diagnostic for unresolved work.

Successful checkpoints are written using temporary-file replacement. Cached
terminal successes are reused across PBFs and reruns. Failure records are
diagnostic only and never satisfy completeness. Cache keys include the response
contract version and full-text policy so incompatible old entries cannot be
mistaken for complete records.

If the PBF identity or policy fingerprint changes, stale PBF-specific discovery
state is discarded safely while reusable compatible entity/article successes
remain available.

### Completeness audit and atomic local publication

Before publication, an audit proves:

1. every extracted valid QID has a resolved entity;
2. every valid language-Wikipedia sitelink is represented in the expected work
   set;
3. every expected work item has a successful, parseable full-text article;
4. every polygon has the same enrichment counters derivable from that complete
   set;
5. every expected polygon-article association has a deterministic link; and
6. row counts and identifiers contain no unexplained omissions or duplicates.

Parquet output is first written to PBF-scoped temporary paths. Only a passing
audit permits atomic replacement of the three final local files and an update
to the local completed manifest. A failed or interrupted run leaves existing
final artifacts untouched and retains only resumable journal/cache state.

### Background upload pipeline

After atomic local publication, the completed PBF is submitted to a bounded
background upload queue and processing immediately advances to the next PBF.
The queue is intentionally bounded so slow uploads eventually apply backpressure
instead of consuming unbounded disk, memory, or threads.

Each upload job persists its state and uploads the three Parquet artifacts. The
remote manifest is advanced only after all artifacts referenced by the relevant
manifest state have uploaded successfully. Uploads use retries and resumable
Hub behavior. Failed jobs remain durable for a subsequent invocation.

At normal shutdown the command drains the queue. Processing failures and
permanently failed uploads produce a nonzero exit status. Interrupt handling
stops accepting new work, preserves journal and upload state, and performs a
bounded orderly shutdown without claiming incomplete work as finished.

## Error reporting and observability

Logs distinguish API requests from articles so batching benefits are visible.
Per PBF they report discovered polygons, unique QIDs, expected sitelinks, cache
hits, API batches, retry/cooldown time, resolved articles, completeness status,
local publication, upload queue state, and stage timings.

Failure output identifies every unresolved QID or `(site, title)` and the last
error. It also prints that rerunning the same command resumes from checkpoints.
No error path is reduced to a warning followed by successful publication.

## TDD and verification

Implementation follows red-green-refactor in small changes. Tests first prove:

1. CLI defaults select all languages, full text, and no cap;
2. every valid language-Wikipedia sitelink becomes expected work;
3. repeated QIDs and articles are requested once without changing output links;
4. batch and single-item paths produce identical ordered results;
5. missing batch members are retried and cannot pass completeness;
6. the scheduler never exceeds three global in-flight requests and respects a
   shared cooldown;
7. adaptive recovery, `maxlag`, 429, 503, timeout, and retry behavior use a
   deterministic fake clock;
8. journal writes are atomic and an interrupted run requests only unresolved
   work after restart;
9. incompatible cache policy entries are rejected;
10. incomplete enrichment cannot replace Parquet, update the manifest, or queue
    an upload;
11. successful artifacts and row ordering remain equivalent to the established
    domain logic;
12. an upload overlaps processing of the next PBF;
13. upload retries and remote-manifest ordering prevent dangling references;
14. shutdown drains successful uploads and exits nonzero for unresolved ones;
    and
15. interrupt-and-resume behavior retains both enrichment and upload progress.

The automated suite uses in-memory clients, generated candidates, fake clocks,
temporary files, and a stub Hub. It performs no real PBF-scale run and sends no
network requests. The final gate is `pytest`, Ruff lint, Ruff formatting, and
mypy. A synthetic benchmark compares request count, cache reuse, row output,
and processing/upload overlap without setting a brittle wall-clock threshold.

## Operational defaults and scope

The normal CLI is the comprehensive mode: all languages, full text, and no cap.
This design assumes no Wikimedia authentication. Authentication can be added
later without weakening the scheduler or completeness contract.

Official Wikimedia dumps, Wikimedia Enterprise, schema expansion, polygon
filter changes, and multiple-PBF enrichment in parallel are out of scope.
Keeping one PBF's enrichment active at a time limits memory and makes completion
and resume boundaries explicit; background upload is the only cross-PBF
pipeline overlap in this change.
