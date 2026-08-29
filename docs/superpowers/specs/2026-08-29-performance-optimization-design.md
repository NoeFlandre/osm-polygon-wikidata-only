# Performance Optimization Design

## Goal

Reduce avoidable CPU, allocation, and peak-memory overhead in the paused V2
sentence workflow and local dataset-card/statistics scan without changing
output, statistics, or resumability.

## Evidence

On the current checkout, finalizing 8,192 sentence rows from 32 checkpoint
batches took a median of 1.062 seconds across three runs. Writing the same
validated checkpoint Parquet tables directly with Arrow took 0.331 seconds,
an estimated 3.2x improvement. Before this optimization, the implementation
deserialized every batch to Python dictionaries and immediately rebuilt an
Arrow table before writing the final sidecar.

The local statistics scan over the 19 GB processed tree completed in 1:52.42
with the current single-file reader. A read-only comparison using
`ParquetFile.read(columns=...)` completed in 1:45.89 and produced the same
serialized `DatasetStats` SHA-256
(`878029edfc723db6f52b8d64120e19c785bd0df823a5c5a7ebd252b10c422ee4`). On a
100-file representative polygon sample, the direct reader had a 0.248 second
median versus 0.485 seconds for `pq.read_table`, a 1.96x improvement.

For resumability checks on a 32-batch/8,192-row sentence checkpoint, reading
full Arrow tables had a 0.166 second median versus 0.053 seconds for validating
the Parquet footer and row count, a 3.14x improvement.

The card metric scan was also profiled on representative real V2 shards. A
single document pass reduced an exact 20-shard metric calculation from 21.925
seconds to 16.900 seconds (1.30x), while the polygon identity/QID/source pass
fell from 3.255 seconds to 1.537 seconds (2.12x). The same one-pass changes
preserve 88,849 polygon identities, 327 languages, and 60,025,344 document
words. Reusing an already-open Parquet handle for schema inspection reduced a
representative document helper from 1.183 seconds to 0.154 seconds with the
same result. On all 375 real link shards, bounded four-worker metadata reads
returned the same 2,468,604 rows in 3.313 seconds versus 30.772 seconds
sequentially (9.29x in that run).

The full V1 statistics scan still produced the exact serialized SHA-256
`878029edfc723db6f52b8d64120e19c785bd0df823a5c5a7ebd252b10c422ee4`. The
read-only V2 card scan completed over all 386 manifest regions in 340.556
seconds and returned 1,259,424 polygons, 2,332,127 documents, 12,666,253
sections, and 2,527,091 links.

## Chosen approach

1. Add checkpoint APIs that validate a batch's Parquet schema and row count
   from footer metadata, or read it as a schema-validated `pyarrow.Table` when
   row values are required.
2. Make checkpoint-validity discovery and resumed row accounting use footer
   metadata, so restart checks do not decode rows unnecessarily.
3. Keep `load_batch()` and `load_rows()` as Python-row compatibility APIs for
   callers that need them.
4. Make sentence output finalization write checkpoint Arrow tables directly,
   preserving deterministic batch order, the existing sentence schema,
   Snappy compression, atomic replacement, and final schema validation.
5. Cache the immutable sentence schema to avoid reconstructing it in hot
   loops.
6. Make the shared statistics `safe_table()` helper use `ParquetFile.read()`
   for one known file, avoiding dataset discovery while retaining the same
   selected columns and recoverable-error policy.
7. Scan card document and polygon metric columns once per file, while keeping
   footer row counts independent so files without metric columns retain their
   historical counts.
8. Reuse open Parquet handles for card and comparison schema checks, reuse
   collected row counts during card assembly, and parallelize only independent
   metadata reads with a bounded four-worker pool.

## Contracts preserved

- Sentence row values, ordering, null handling, schema metadata, and output
  compression remain unchanged.
- A missing, unreadable, or schema-invalid checkpoint remains invalid; final
  output still reads and validates each batch table before publication.
- Checkpoint metadata and contiguous-batch rules remain unchanged.
- Output is still written to a temporary file and atomically replaced only
  after validation succeeds.
- Unsupported languages remain unsplit according to the existing contract.
- No Grid5000 jobs, sentence uploads, or HF dataset-card changes are part of
  this work.

## Local pipeline measurement gate

The statistics reader change is allowed because the full-tree comparison
already matched the exact serialized `DatasetStats` result and the bounded
sample measured a speedup. Tests must retain column pruning and the existing
skip-on-`OSError`/`KeyError`/`ArrowInvalid` behavior. The only concurrency is a
bounded four-worker pool for independent Parquet metadata reads; it does not
skip validation, change file ordering, or add unchecked caching.

## Verification

- Add focused tests before implementation for table round-tripping,
  checkpoint validity, and output equivalence.
- Run the focused tests RED, implement the smallest change, then run them
  GREEN.
- Run the repository regression suite plus Ruff, type checking, CRAP with a
  score below 6, and the focused mutation gate with zero surviving mutants.
- Re-run the bounded finalization, card, and metadata benchmarks and record
  exact output-equivalence results.
