# Offline Arrow Statistics Optimization Implementation Plan

> **Execution:** Follow `superpowers:executing-plans`,
> `superpowers:test-driven-development`, and
> `superpowers:verification-before-completion`. The user explicitly approved
> direct execution on the current `main` branch.

**Goal:** Remove avoidable Python scalar materialization and repeated JSON
decoding from the local dataset-statistics scan while preserving its exact
serialized output and all failure contracts.

**Architecture:** Keep file enumeration and Parquet reads unchanged. Move
schema-native reductions into small PyArrow compute helpers, retain Python
fallbacks for non-production Arrow types, and attach a capped language-list
decode cache to the scan-local accumulator. Tests protect both output
semantics and the absence of repeated decoding.

**Tech stack:** Python 3.12+, PyArrow 24 already installed, pytest, Ruff, ty,
radon/crap4py, and mutmut. No network or dependency installation.

---

## Task 1: Specify the native aggregation contract in failing tests

**Files:**

- Modify: `tests/hf/test_dataset_stats.py`

1. Add a test that creates two Arrow polygon tables containing the same
   serialized language list, including a duplicate language within that list.
   Patch `aggregation.json.loads` with a counting wrapper, aggregate both
   tables through one `_StatsAccumulator`, and assert:
   - the serialized value is decoded once for the whole scan;
   - every row still contributes its full language multiplicity;
   - malformed and canonical empty lists remain empty.
2. Add a chunked-table test covering null booleans, a true Wikipedia flag with
   a null English flag, null language counts, and threshold boundaries at 2,
   5, and 10. Assert every accumulator field matches the current Python
   semantics.
3. Add an article-table test with chunked language and integer columns,
   repeated languages, empty/null languages, null totals, and tied counts.
   Assert totals and deterministic language ordering.
4. Patch the legacy Python scalar helpers to fail in the native-schema tests.
   This makes the performance contract deterministic without a flaky timing
   threshold.
5. Run only the new tests and confirm RED because repeated values are decoded
   repeatedly and native tables still call Python scalar helpers.
6. Commit only the failing tests with
   `test: specify Arrow-native statistics aggregation`.

## Task 2: Implement native scalar reductions

**Files:**

- Modify: `src/osm_polygon_wikidata_only/hf/_dataset_stats/aggregation.py`

1. Import `pyarrow.compute` and add small helpers that:
   - sum a native boolean column after filling nulls with false;
   - count `left and not right` while preserving the historical null
     truthiness contract by filling both inputs with false;
   - count integer values at or above one threshold;
   - sum integer columns while ignoring nulls;
   - aggregate Arrow value-count structs into a `Counter[str]` in first-seen
     order.
2. Each helper must take the native path only for the expected Arrow type and
   delegate to the existing Python helper for unsupported types. Do not catch
   unrelated exceptions or broaden the malformed-file policy.
3. Update polygon and article table aggregation to use these helpers. Keep QID
   and region set behavior unchanged.
4. Run the native-schema tests GREEN, then run all
   `tests/hf/test_dataset_stats.py` and
   `tests/hf/test_dataset_stats_scanning.py` tests.

## Task 3: Decode repeated polygon language lists once per scan

**Files:**

- Modify: `src/osm_polygon_wikidata_only/hf/_dataset_stats/aggregation.py`
- Modify: `tests/hf/test_dataset_stats.py`

1. Add a private capped decode cache to `_StatsAccumulator`. It must be local
   to one scan and must not affect the serialized `DatasetStats` result.
2. Convert each Arrow language-list column to value counts. Iterate distinct
   values in first-seen order, decode through the accumulator cache, and
   multiply each decoded language by that serialized value's frequency.
3. Keep `str`, `bytes`, and `bytearray` parsing behavior. Do not decode empty
   values or `[]`; cache empty results for malformed and non-list JSON.
4. When the cap is reached, continue correctly without caching new values.
   Never evict in a way that changes output.
5. Run the repeated-decode test GREEN and add a cap-boundary test that proves
   correctness beyond the cache limit without allocating a production-sized
   fixture.
6. Run all dataset-statistics tests and refactor only after GREEN.

## Task 4: Validate quality and mutation resistance

**Files:**

- Modify `pyproject.toml` only if the installed mutmut CLI cannot target the
  changed module without a persistent source entry.

1. Run targeted Ruff lint and formatting over the changed production and test
   files, excluding unrelated untracked paths.
2. Run `ty check src scripts` from the existing virtual environment.
3. Generate focused branch coverage and CRAP reports for
   `_dataset_stats/aggregation.py`; every relevant function must score below
   6.
4. Run focused mutation testing for the changed deterministic helpers and
   require zero surviving, timed-out, suspicious, or untested mutants. Add
   behavioral tests to kill meaningful survivors; do not weaken mutations or
   exclude changed lines merely to satisfy the gate.
5. Commit the validated implementation and tests with a clear performance
   message, staging only explicit paths.

## Task 5: Benchmark, reprofile, and iterate

**Files:**

- Update the approved design document only with measured final evidence.

1. Run a repeated in-memory benchmark over representative real polygon and
   article tables, comparing the committed baseline implementation from Git
   with the working implementation in isolated processes. Verify equal
   accumulator payloads on every run.
2. Run the same complete local V1 scan used for the baseline:
   `/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/processed`.
3. Require the exact serialized SHA-256
   `878029edfc723db6f52b8d64120e19c785bd0df823a5c5a7ebd252b10c422ee4`.
4. Compare with the 103.252-second uninstrumented baseline. Keep the change
   only if the improvement is material and repeatable.
5. Reprofile the optimized scan. If another local bottleneck admits a
   similarly safe and material change under this approved architecture,
   repeat RED-GREEN-REFACTOR; otherwise stop rather than pursue marginal gains.
6. Record exact before/after evidence in the design document and commit it.

## Task 6: Complete offline regression verification and publish code

**Files:**

- No additional production files expected.

1. Run the complete tracked pytest suite with `tests/unit` ignored because it
   is a pre-existing untracked tree from another package.
2. Run all tracked Ruff, formatting, type, documentation, package-build,
   CRAP, and mutation gates using the existing environment and `/tmp` caches.
3. Run `git diff --check`, inspect the exact changed paths, and confirm every
   unrelated untracked file is untouched.
4. Commit any final validated documentation evidence with explicit staging.
5. Push the current `main` branch only after all local verification succeeds;
   the implementation and verification themselves perform no network or
   download operations.
6. Verify local `HEAD` equals `origin/main` after the push.
