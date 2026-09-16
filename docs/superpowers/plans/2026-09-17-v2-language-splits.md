# V2 Language Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream every validated V2 language-bearing row into deterministic Hugging Face-compatible language shards without changing V1 artifacts.

**Architecture:** `v2/language_splits.py` owns the V2-only command, streaming writer, output manifest, and atomic output files. It consumes the accepted `hf.language_splits` inventory/normalizer and writes one shard per source region, table, and language below `processed_v2/language_splits/`; polygons are validated by the inventory but never routed. The generated manifest is authoritative and contains only relative paths and deterministic hashes.

**Tech Stack:** Python 3.12, PyArrow Parquet row-group iteration/writers, repository deterministic JSON and atomic I/O helpers, `datasets.load_dataset` for the local loading acceptance test, pytest, Ruff, ty, architecture checker, Radon/CRAP.

---

### Task 1: Specify the V2 generator through real Parquet tests

**Files:**
- Create: `tests/v2/test_language_splits.py`
- Read-only reference: `src/osm_polygon_wikidata_only/hf/language_splits.py`

- [ ] **Step 1: Write the failing tests first.** Add fixture helpers that write a V2 manifest and the exact V2 polygon, document, section, and link schemas. The fixture must include two region files, a polygon whose `best_language` is `en` but whose document/link rows are `fr` and `de`, distinct document identities, and link languages `None`, whitespace, `en/fr`, `simple`, and `be_x_old`.

  Add tests named `test_v2_split_keeps_each_multilingual_row_in_its_own_partition`, `test_v2_split_routes_unusable_values_to_lang_unknown`, `test_v2_split_conserves_rows_and_preserves_schema`, `test_v2_split_preserves_sorted_source_and_row_order`, `test_v2_split_is_byte_stable_on_repeated_runs`, `test_v2_split_does_not_route_or_copy_the_polygon_table`, `test_v2_split_rejects_a_v1_processed_root_without_writing_output`, `test_v2_split_can_be_loaded_with_standard_datasets`, and `test_v2_split_module_runs_as_a_local_command`.

  Each test calls `build_v2_language_splits(processed_v2)` or `main([str(processed_v2)])` and asserts the public result/manifest. The loading test calls `datasets.load_dataset("parquet", data_files={"lang-fr": [str(path)]}, split="lang-fr")` using only the French document shard and checks its row count and complete column set.

- [ ] **Step 2: Run only the new test module to prove RED.**

  Run:

  ```bash
  MPLCONFIGDIR=/private/tmp/osm-polygon-wikidata-only-v2-focused/matplotlib \
  MPLBACKEND=Agg MPL_IGNORE_SYSTEM_FONTS=1 \
  .venv/bin/python -m pytest tests/v2/test_language_splits.py \
    --no-cov -p no:cacheprovider --assert=plain \
    --basetemp=/private/tmp/osm-polygon-wikidata-only-v2-focused/language-splits-red -q
  ```

  Expected result: collection fails because `osm_polygon_wikidata_only.v2.language_splits` does not yet exist. No production implementation is written before this failure is observed.

### Task 2: Implement bounded V2 row-level partitioning

**Files:**
- Create: `src/osm_polygon_wikidata_only/v2/language_splits.py`
- Do not modify: `src/osm_polygon_wikidata_only/hf/language_splits.py`
- Do not modify: shared CLI files, `pyproject.toml`, `Justfile`, `mkdocs.yml`, `docs/index.md`, or `docs/adr/0002-language-split-contract.md`

- [ ] **Step 1: Add the V2 result models and constants.** Define the public `V2LanguageSplitResult`, `V2LanguageSplitFile`, `build_v2_language_splits`, and `main` APIs, plus `LANGUAGE_SPLITS_DIRNAME`, `LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH`, `V2_LANGUAGE_SPLIT_CONTRACT_VERSION`, and `DEFAULT_BATCH_SIZE=65_536`. Model fields must serialize in fixed order through `to_dict`, while all paths stored in the manifest are relative to `processed_v2`.

- [ ] **Step 2: Add inventory-driven source selection.** Call `build_language_inventory(processed_v2, DatasetContract.V2)` exactly for the V2 root. Iterate `language_table_specs(DatasetContract.V2)` in contract order, use each table inventory's sorted `source_files`, and resolve only those files under `processed_v2`. Do not inspect polygon row values after validation and do not import V1 storage or release paths.

- [ ] **Step 3: Add the streaming shard writer.** For each source Parquet file, use `pq.ParquetFile.iter_batches(columns=None, batch_size=batch_size)`. Normalize only the batch language column with `normalize_language`, group ascending physical row indices by the returned partition, select those indices from the complete record batch, and write them to one `ParquetWriter` per language at `language_splits/<configuration>/<split>/<source_stem>.parquet`. Keep source-file order, batch order, and row order; use the validated table schema and Snappy compression; atomically replace each final shard and clean temporary files on all exceptions.

- [ ] **Step 4: Add conservation and manifest validation.** Compare observed per-table/per-language row counts with the accepted inventory buckets, reject any mismatch, validate every written Parquet schema including metadata, hash each output with the bounded file-hash helper, and write `processed_v2/manifests/language_splits.json` last using `atomic_write_json`. Include `wikipedia-tags-v2`, `language-splits-v1`, the source manifest digest, artifact fingerprint, table identity columns, counts, configuration/split names, source file, output path, and output hash. A source table with no rows produces no phantom language shard.

- [ ] **Step 5: Add the module command boundary.** Parse one positional `processed_v2` path and an optional positive `--batch-size`; call `build_v2_language_splits` and print the manifest path. The command must not upload, import, or invoke Hugging Face Hub publication.

- [ ] **Step 6: Run the new tests to prove GREEN.**

  Run the Task 1 command again. Expected result: every new test passes, with no temporary `.tmp` files left in the generated tree.

### Task 3: Refactor only after green and verify the focused contract

**Files:**
- Modify only: `src/osm_polygon_wikidata_only/v2/language_splits.py` and `tests/v2/test_language_splits.py` if the green implementation exposes duplication or a test clarity issue

- [ ] **Step 1: Run the focused V2 and accepted contract tests.**

  ```bash
  MPLCONFIGDIR=/private/tmp/osm-polygon-wikidata-only-v2-focused/matplotlib \
  MPLBACKEND=Agg MPL_IGNORE_SYSTEM_FONTS=1 \
  .venv/bin/python -m pytest tests/hf/test_language_splits.py tests/v2 \
    --no-cov -p no:cacheprovider --assert=plain \
    --basetemp=/private/tmp/osm-polygon-wikidata-only-v2-focused/pytest -q
  ```

  Expected result: all focused tests pass; the accepted shared contract file remains byte-identical to `0743b9e3…`.

- [ ] **Step 2: Run source-only lint and type checks.**

  ```bash
  UV_CACHE_DIR=/private/tmp/osm-polygon-wikidata-only-v2-uv-cache \
  .venv/bin/ruff check src/osm_polygon_wikidata_only/v2/language_splits.py tests/v2/test_language_splits.py
  UV_CACHE_DIR=/private/tmp/osm-polygon-wikidata-only-v2-uv-cache \
  .venv/bin/ruff format --check src/osm_polygon_wikidata_only/v2/language_splits.py tests/v2/test_language_splits.py
  UV_CACHE_DIR=/private/tmp/osm-polygon-wikidata-only-v2-uv-cache \
  .venv/bin/ty check src tests
  ```

- [ ] **Step 3: Run the relevant architecture and CRAP checks.** Run `scripts/quality/architecture.py` against `src`, generate focused coverage for the new V2 tests under `/private/tmp`, run Radon and `scripts/quality/crap_score.py` against the new module, and require the new functions to remain below CRAP 6.00.

### Task 4: Review, commit, push, and open the non-draft PR

**Files:**
- Review only the final diff; no shared integration file may be added to the change

- [ ] **Step 1: Verify scope and repository state.** Confirm only V2 module/tests and the V2 design/process documents changed; confirm V1 source/data paths are untouched; run `git diff --check` and inspect the exact diff.

- [ ] **Step 2: Commit with Conventional Commit syntax.** Use `feat(v2): add row-level language split generation` after all checks pass.

- [ ] **Step 3: Push the exact branch and verify the remote SHA.** Push `codex/language-splits-v2` to `origin`, then compare local `HEAD`, `origin/codex/language-splits-v2`, and GitHub's branch ref.

- [ ] **Step 4: Open the non-draft PR.** Create a pull request from `codex/language-splits-v2` to `main` with `Closes #11`, include the exact checks and their results, and confirm the PR is non-draft and linked to issue #11. Do not publish any artifact to Hugging Face.
