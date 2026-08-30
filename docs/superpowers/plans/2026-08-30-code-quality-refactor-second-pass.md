# Code Quality Refactor Second Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the remaining measured complexity in V2 comparison, sentence checkpoint loading, and centroid filtering without changing any public API, output, ordering, or restart behavior.

**Architecture:** Keep the existing public facades and Parquet/file boundaries. Extract small private helpers for one-batch aggregation, one-row accounting, validated checkpoint indexes, and selected/all centroid iteration. Extend the existing deterministic CRAP scope for the comparison/checkpoint code and add a separate isolated geography scope for the Matplotlib-backed loader.

**Tech Stack:** Python 3.12, pytest/pytest-cov, Ruff, ty, Radon, crap4py, mutmut, uv, Just.

---

### Task 1: Record the second-pass baseline

**Files:**
- No production changes.
- Create: `docs/superpowers/plans/2026-08-30-code-quality-refactor-second-pass.md`

- [x] **Step 1: Run the tracked regression baseline.**

Run `PATH=/usr/bin:/bin MPLCONFIGDIR=/tmp/osm-polygon-wikidata-only-second-pass-baseline UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-second-pass-baseline-coverage /Users/noeflandre/.local/bin/uv run pytest --ignore=tests/unit --cov=osm_polygon_wikidata_only --cov-report=term-missing --cov-report=json:/tmp/osm-polygon-wikidata-only-second-pass-baseline-coverage.json -q`.

Expected baseline: `2571 passed, 2 skipped`, approximately `92.93%` total coverage. The unrelated untracked `tests/unit` tree remains excluded because it imports a package outside this repository.

- [x] **Step 2: Measure complexity and static quality.**

Run the repository Ruff/format/`ty` checks and `radon cc -n B -s src scripts`. Use the measured offenders rather than broad stylistic edits.

### Task 2: Simplify comparison aggregation

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/comparison.py`
- Modify: `tests/v2/test_comparison.py`
- Modify: `Justfile`
- Modify: `tests/test_packaging.py`

- [x] **Step 1: Add RED tests for source decoding and row predicates.**

Add tests that call the existing comparison behavior with malformed JSON, a non-list source payload, and invalid source member types, asserting the current `ValueError` message. Add a fixture with a missing comparison column and verify it is skipped. Run `uv run pytest --no-cov -q tests/v2/test_comparison.py`; new helper imports must fail before implementation.

- [x] **Step 2: Extract minimal batch/row helpers.**

Keep `_unique_values()` responsible only for opening files and iterating batches; move batch value extraction into `_values_from_batch()`. Keep `_polygon_sources()` and `_direct_documents_by_polygon()` responsible only for file/column iteration; move one-row filtering and accumulation into `_record_polygon_source()` and `_record_direct_document()`. Split `_source_list()` into JSON decoding and source-list shape validation while preserving its exact error text.

- [x] **Step 3: Verify green and add the comparison module to CRAP.**

Run the complete comparison test file, then extend `crap` with `tests/v2/test_comparison.py` and `src/osm_polygon_wikidata_only/v2/comparison.py`. Require every reported comparison function to remain below `6.00`.

### Task 3: Simplify sentence checkpoint row loading

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/sentence_checkpoints.py`
- Modify: `tests/v2/test_sentence_checkpoints.py`
- Modify: `Justfile`

- [x] **Step 1: Add RED tests for invalid and non-contiguous row requests.**

Add tests for `load_rows(batch_count=-1)`, a checkpoint containing batch `1` without batch `0`, and a corrupted batch that is discovered during row loading. Assert the existing error messages. Run the focused test file before implementation and confirm each new assertion fails for the intended missing behavior.

- [x] **Step 2: Extract checkpoint index validation and row materialization.**

Move requested-count selection into `_requested_batch_count()`, contiguous-index validation into `_validated_batch_indexes()`, and the ordered batch read/extension loop into `_load_checkpoint_rows()`. `SentenceCheckpoint.load_rows()` must retain its public signature and exact errors while delegating to those helpers.

- [x] **Step 3: Verify green and measure the checkpoint module.**

Run `tests/v2/test_sentence_checkpoints.py` and the related sentence-runner/checkpoint tests. Add the checkpoint test/source files to the bounded `crap` recipe and require all measured functions below `6.00`.

### Task 4: Simplify centroid filtering without changing map output

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/coverage_map.py`
- Modify: `tests/hf/test_coverage_map.py`
- Modify: `Justfile`

- [x] **Step 1: Add RED tests for both centroid iteration modes.**

Add focused tests for all-row iteration and selected-ID iteration, including `None` coordinates and non-matching IDs. Assert the existing parallel `(lon, lat)` output and skip policy. Run the tests before implementation.

- [x] **Step 2: Extract all-row and selected-row iterators.**

Make `_valid_centroid_rows()` dispatch to `_all_centroid_rows()` or `_selected_centroid_rows()` and preserve the current column names, order, float conversion, and null handling. Do not alter rendering, network caching, or public exports.

- [x] **Step 3: Add an isolated geography CRAP scope.**

Create a `crap-geography` recipe covering the coverage-map tests/source, using a temporary Matplotlib configuration and `Agg` backend. Include it in `crap-all`; require every measured geography function below `6.00`.

### Task 5: Simplify geographic Parquet schema handling

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/_geographic/parquet_inputs.py`
- Modify: `tests/hf/test_geographic_text_coverage.py`
- Modify: `Justfile`

- [x] **Step 1: Add RED tests for metadata fallback and missing-column translation.**

Add focused tests for the metadata-column reader returning the fallback marker when `pq.read_metadata()` fails and for the column-error formatter preserving the current `CoverageMapError` text. Run the geographic tests before adding the new private helpers.

- [x] **Step 2: Separate metadata discovery, batch streaming, and exception translation.**

Make `iter_required_columns()` coordinate three small helpers: one reads metadata with the existing broad fallback, one opens the Parquet file and streams validated columns, and one translates `ArrowInvalid`/`KeyError`/`OSError`. Keep the compatibility `read_required_columns()` wrapper and all error messages unchanged.

- [x] **Step 3: Measure the geographic Parquet scope.**

Add the geographic text-coverage tests/source to the separate `crap-geography-inputs` recipe because its Parquet loader is an independent deterministic boundary; require the isolated scope to remain below `6.00` without hiding a relevant function.

### Task 6: Simplify dataset-stat parsing and link metadata accounting

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/_dataset_stats/aggregation.py`
- Modify: `tests/hf/test_dataset_stats.py`
- Modify: `Justfile`

- [x] **Step 1: Add RED tests for language-list shapes.**

Cover non-string values, malformed JSON, JSON objects, valid lists, and canonical `"[]"` without weakening the existing no-decode fast path. Run the focused statistics tests before adding the decoder helper.

- [x] **Step 2: Extract language decoding and simplify link-file accounting.**

Move JSON decoding/type validation into a small helper and keep `_parse_language_list()` as the empty/type dispatch. Replace the duplicated link-file size loop with one deterministic sum while preserving unreadable-file and bounded-worker behavior.

- [x] **Step 3: Add a bounded dataset-stat CRAP scope.**

Measure `aggregation.py` with its existing statistics tests and require every reported function below `6.00`. Do not include unrelated rendering or network code.

### Task 7: Simplify SaT provider selection and setup

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/sat.py`
- Modify: `tests/v2/test_sat.py`
- Modify: `Justfile`
- Modify: `pyproject.toml`
- Modify: `tests/test_packaging.py`

- [x] **Step 1: Add RED tests for provider selection/setup seams.**

Add direct tests for provider selection with override, CPU fallback, GPU requirement, missing `SaT`, and positive model construction. New helper imports must fail before implementation.

- [x] **Step 2: Extract provider selection and model construction.**

Keep `SaT3lSegmenter`’s public constructor, error messages, provider order, pinned revision, and active-CUDA verification unchanged while delegating selection and dynamic module/model loading to small helpers.

- [x] **Step 3: Measure the SaT module.**

Add `sat.py` and `tests/v2/test_sat.py` to a bounded CRAP scope. Keep the SaT adapter outside the mutation target list because its dynamic optional dependency and provider inspection cross an external runtime boundary. Preserve the existing CPU/GPU runtime compatibility contract.

### Task 8: Final verification and publication

**Files:**
- Modify: `docs/superpowers/plans/2026-08-30-code-quality-refactor-second-pass.md`

- [x] **Step 1: Run all focused RED→GREEN→REFACTOR checks and static gates.**

Run the affected test files, Ruff, format, and `ty check src scripts` after each slice. Re-run the full tracked suite with `--ignore=tests/unit` after the final slice.

- [x] **Step 2: Run complete configured quality gates.**

Run all configured CRAP scopes through their direct offline equivalents when the `uv` wrapper cannot resolve the already-installed environment, acceptance tests, architecture checks, and the full tracked coverage suite. Require zero non-killed mutants in the unchanged mutation scope and a maximum CRAP below `6.00` in every configured scope.

- [x] **Step 3: Review and publish exact scope.**

Run `git diff --check`, stage only this plan, the listed source modules, their tests, `Justfile`, and the documentation updates. Commit on `main`, push `origin/main`, and verify matching local/remote heads while preserving all pre-existing untracked user files.
