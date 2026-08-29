# Performance Optimization Design

## Goal

Reduce avoidable CPU, allocation, and peak-memory overhead in the paused V2
sentence workflow and local dataset-card/statistics scan without changing
output, statistics, or resumability.

## Evidence

On the current checkout, finalizing 8,192 sentence rows from 32 checkpoint
batches took a median of 1.062 seconds across three runs. Writing the same
validated checkpoint Parquet tables directly with Arrow took 0.331 seconds,
an estimated 3.2x improvement. The current implementation deserializes every
batch to Python dictionaries and immediately rebuilds an Arrow table before
writing the final sidecar.

The local statistics scan over the 19 GB processed tree completed in 1:52.42
with the current single-file reader. A read-only comparison using
`ParquetFile.read(columns=...)` completed in 1:45.89 and produced the same
serialized `DatasetStats` SHA-256
(`878029edfc723db6f52b8d64120e19c785bd0df823a5c5a7ebd252b10c422ee4`). On a
100-file representative polygon sample, the direct reader had a 0.248 second
median versus 0.485 seconds for `pq.read_table`, a 1.96x improvement.

## Chosen approach

1. Add a checkpoint API that reads one batch as a schema-validated
   `pyarrow.Table` without converting it to Python objects.
2. Make checkpoint-validity discovery use that table API, so restart checks do
   not materialize rows unnecessarily.
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

## Contracts preserved

- Sentence row values, ordering, null handling, schema metadata, and output
  compression remain unchanged.
- A missing, unreadable, or schema-invalid checkpoint remains invalid.
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
skip-on-`OSError`/`KeyError`/`ArrowInvalid` behavior. No concurrency increase,
skipped validation, or unchecked caching will be introduced as part of this
slice.

## Verification

- Add focused tests before implementation for table round-tripping,
  checkpoint validity, and output equivalence.
- Run the focused tests RED, implement the smallest change, then run them
  GREEN.
- Run the repository regression suite plus Ruff, type checking, CRAP with a
  score below 6, and the focused mutation gate with zero surviving mutants.
- Re-run the bounded finalization benchmark and record the measured result.
