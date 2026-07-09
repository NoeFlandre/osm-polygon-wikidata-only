# Pipeline Performance Design

## Goal

Make the PBF-to-Parquet pipeline substantially faster while preserving its
observable data contract: every current polygon candidate is processed, every
selected article is fetched and recorded, schemas and output order remain
deterministic, and cache/retry/rate-limit behavior remains safe.

## Non-negotiable invariants

- The PBF reader keeps its existing eligibility rule: only closed ways and
  multipolygon relations with non-empty `wikidata` tags are candidates.
- The processor keeps every candidate that can currently be converted to a
  polygon, including polygons whose QID has no entity or articles.
- Language filtering, `fetch_full_text`, and `max_articles_per_qid` retain
  their current semantics and output ordering.
- The three Parquet schemas, row values, deterministic row ordering, manifest
  stats, and public client APIs remain compatible.
- A failed or rate-limited batch never silently drops work: individual
  requests provide the existing per-item fallback behavior.

## Design

### Batched, ordered enrichment

Add optional batch capabilities to the concrete HTTP clients without changing
the existing single-item abstract-client contract used by callers and tests.

`fetch_qids` will detect those capabilities. It will partition QIDs into
bounded Wikidata batches and selected article titles into bounded batches per
Wikipedia site. It will reassemble results in the exact QID and sitelink order
used today. Non-batch clients retain the existing single-item code path.

The HTTP batch APIs use the same Action API fields and parsers as individual
requests. A malformed, transiently failed, or rate-limited batch falls back to
the established single-item path for every item in that batch. This makes a
batch an optimization only; it cannot turn a partial batch result into missing
polygons or articles.

### Safe concurrency and rate coordination

Use a small, bounded executor only for independent Wikipedia sites. Work for
one site remains ordered and is paced by the existing per-host limit. The
shared host limiter will gain a cooldown update API: a 429 response extends
the host's next-allowed time for all workers before retries begin. This avoids
parallel retry storms while continuing all pending work after the server's
requested delay.

The defaults preserve conservative Wikimedia pacing. New settings are opt-in
performance controls with values chosen to keep per-host traffic compliant;
setting concurrency to one reproduces serial execution.

### Local CPU and I/O path

- Process reader callbacks directly rather than first allocating a second list
  of raw candidates; retain polygon ordering and the `limit` boundary.
- Build reusable article metadata once per unique article rather than
  recalculating hashes, word counts, token estimates, and serialized metadata
  for every polygon link to that article.
- Avoid redundant list copies on the Parquet path while retaining the exact
  schemas and empty-table behavior.
- Load the manifest once during multi-PBF orchestration, update the in-memory
  view after each successful PBF, and preserve skip/force decisions.

## Failure handling

Individual requests keep their current retry policy. Batch requests use the
same retry and throttling infrastructure. If a batch cannot produce a complete
and valid mapping, only its members are retried individually. Per-article
failure statuses continue to be represented by `LinkSummary` rather than
raising or skipping the related polygon.

## Test and performance strategy

Follow RED-GREEN-REFACTOR for each change. Add unit tests that first prove:

1. batched and individual enrichment produce identical ordered summaries;
2. partial batch failure makes individual fallback requests for every missing
   member;
3. host cooldown applies across concurrent callers;
4. the streaming reader path preserves all candidates and the limit boundary;
5. repeated QIDs produce byte-equivalent article rows and unchanged link
   counts; and
6. cached manifest skip decisions match the current per-PBF behavior.

Add a synthetic benchmark fixture only. It measures extraction, enrichment,
and row-materialization throughput with in-memory clients; it never runs a
real PBF or sends real network traffic.

## Scope exclusions

This work does not change polygon eligibility, datasets, schemas, article
selection, source APIs, retry counts, cache keys, or publication behavior. It
does not run the production pipeline as part of development or verification.
