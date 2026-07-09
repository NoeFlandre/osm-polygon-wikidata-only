# Stage Performance Design

## Goal

Measure and improve extraction, enrichment, row construction, Parquet,
manifest, and upload stages without changing artifacts or selection logic.

## Design

`StageTimings` records named monotonic durations in `ProcessResult` and the
CLI logs them after each PBF. It is metadata only and never enters Parquet or
the manifest.

Batch requests are partitioned into conservative API-safe chunks. Wikidata
uses QID chunks; Wikipedia uses title chunks per site, retaining input and
article order after bounded concurrent site jobs complete. Existing single
request and cache fallback paths remain unchanged.

Directory orchestration loads the manifest once and passes the mutable mapping
to each completed PBF upsert. Tests use generated candidates and in-memory
clients to assert stage timing coverage, chunk boundaries, ordering, and row
equivalence without a real PBF or network.
