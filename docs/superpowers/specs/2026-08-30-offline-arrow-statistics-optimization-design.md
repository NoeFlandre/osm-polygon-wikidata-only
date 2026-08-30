# Offline Arrow Statistics Optimization Design

## Goal

Reduce the CPU time, allocations, and temporary Python objects created by the
local V1 dataset-statistics scan without changing any public API, output,
error policy, ordering, or dependency.

## Measured baseline

The current full scan of the mounted 19 GB processed tree completed in
103.252 seconds without profiler instrumentation. Its serialized
`DatasetStats` SHA-256 was
`878029edfc723db6f52b8d64120e19c785bd0df823a5c5a7ebd252b10c422ee4`.

The instrumented scan completed in 200.875 seconds. Cumulative profile time
was concentrated in:

- polygon-table aggregation: 103.423 seconds;
- article-table aggregation: 39.457 seconds;
- selected-column Parquet reads: 50.462 seconds;
- polygon-language JSON parsing: 39.727 seconds.

The 1,184,110 polygon rows contain 666,228 non-empty serialized language
lists but only 83,485 distinct non-empty values. Re-decoding every repeated
value therefore performs substantial avoidable work.

The local V2, Hugging Face, Grid5000, and throughput-focused regression scope
passed 768 tests in 364.91 seconds before implementation.

## Chosen approach

1. Keep file discovery, selected columns, malformed-file handling, and the
   `DatasetStats` API unchanged.
2. Use PyArrow's installed compute kernels for boolean counts, language-count
   thresholds, integer sums, and article-language value counts. These
   operations remain inside Arrow buffers instead of materializing millions
   of Python scalar objects.
3. Aggregate repeated serialized polygon-language lists by frequency and
   decode each distinct value once per scan. Keep the cache local to one
   `_StatsAccumulator`, cap it explicitly, and retain the existing handling
   of empty, malformed, non-list, and non-string values.
4. Retain the existing Python helper behavior as a compatibility fallback for
   unsupported Arrow types. The production schemas use the native fast path;
   unusual test or caller-created tables keep historical semantics.
5. Do not add dependencies, downloads, network calls, file-level concurrency,
   persistent caches, or changes to processed data.

## Alternatives rejected

### Parallel whole-file aggregation

Reading several full Parquet files concurrently could improve some machines,
but it can contend on the external drive, oversubscribe Arrow's own workers,
and increase peak memory. The measured bottleneck is Python scalar work, so
native kernels are the lower-risk first intervention.

### A new native JSON dependency

A faster JSON package would add installation and compatibility surface and
would violate the offline constraint. Frequency-aware decoding removes most
JSON calls with the standard library already in use.

### Plotting and test-only shortcuts

Map rendering is visible in test durations, but it is not the dominant
repeated production scan. Weakening deterministic image checks or caching
test outputs would improve test timing without improving the data pipeline
and is outside this change.

## Behavior and safety contracts

- The exact serialized `DatasetStats` result must remain byte-equivalent.
- Null booleans retain Python truthiness semantics, including a true
  `has_wikipedia` value paired with a null English flag.
- Null numeric values remain excluded from sums and threshold counts.
- Empty strings and `[]` remain ignored without JSON decoding.
- Malformed JSON, non-list JSON, and non-string language values remain empty.
- Duplicate language names within one serialized list retain their historical
  multiplicity.
- Recoverable Parquet failures continue to warn and skip exactly as before.
- Dataset-size accounting, deterministic file order, and bounded metadata
  concurrency remain unchanged.

## Verification

Follow RED-GREEN-REFACTOR. First add focused tests for repeated decode
elimination, null behavior, malformed values, duplicate language entries,
chunked Arrow inputs, and article totals. Then implement the smallest native
helpers and retain fallback paths.

Run focused tests, all dataset-statistics tests, Ruff, formatting, type checks,
CRAP below 6, and focused mutation testing. Finally run the complete tracked
test suite and repeat the same full-tree benchmark. Accept the optimization
only if the exact SHA-256 remains unchanged and wall time improves materially;
otherwise revert the optimization rather than trading maintainability for a
marginal result.
