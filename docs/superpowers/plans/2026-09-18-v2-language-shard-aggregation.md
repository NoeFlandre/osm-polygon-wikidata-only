# V2 Language Shard Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with checkpoints.

**Goal:** Make both datasets' language partitions selectable in Dataset Viewer, replace V2's source-file-per-language output with deterministic bounded shards that fit one Hugging Face atomic commit, then publish and verify both targets.

**Architecture:** Keep V1 data files untouched. Update the V2 generator to stream each validated table in source order into one persistent Parquet writer per language, rotating at 100,000 rows and recording contributing source files. Update the release planner and publication card to declare every language-bearing table as a Viewer config with language splits, predict and describe `part-*` shards, and retain local rollback, remote stale-file ownership, and the 25,000-file fail-closed guard.

**Tech Stack:** Python 3.12, PyArrow Parquet, pytest, Ruff, ty, Hugging Face Hub API/CLI, GitHub CLI.

---

### Task 1: Add RED coverage for aggregated V2 shards

**Files:**
- Modify: `tests/v2/test_language_splits.py`
- Modify: `tests/hf/test_language_split_release.py`
- Modify: `tests/hf/test_language_split_publication.py`

- [ ] **Step 1: Add a V2 fixture with two source files and repeated languages.**

Use the existing fixture builders to create two source Parquet files for one V2 table. Assert that the input rows have distinct source order, repeated `en` rows, and an `unknown` value.

- [ ] **Step 2: Add failing assertions for the new output contract.**

Assert that generation produces paths shaped like:

```python
"language_splits/<configuration>/lang-en/part-00000-of-00002.parquet"
```

and that each manifest file record has `source_files`, `row_count`, and the new contract version. Assert the original schema and source-order rows are preserved, and force a small test shard limit to prove deterministic rotation.

- [ ] **Step 3: Add planner and publication preflight tests.**

Assert that `_expected_v2_files` derives shard paths from inventory row counts without source-file-specific names, and that a normal V2 plan is below `MAX_ATOMIC_PUBLICATION_FILES`. Keep the existing oversized-plan refusal test for a synthetic plan.

- [ ] **Step 4: Add failing Dataset Viewer front-matter assertions.**

Parse the managed card YAML and assert that each language-bearing configuration has a `<configuration>_by_language` entry, each non-empty language has a V1 `split: lang-<language>` or V2 `split: lang_<language>` entry with language-code dashes replaced by underscores, V1 uses exact files, and V2 uses the aggregated `part-*.parquet` glob.

- [ ] **Step 5: Run the focused tests to confirm RED.**

Run:

```bash
uv run pytest tests/v2/test_language_splits.py tests/hf/test_language_split_release.py tests/hf/test_language_split_publication.py -q
```

Expected: new aggregation assertions fail while the existing baseline failures, if any, remain separately identifiable.

- [ ] **Step 6: Commit the RED tests.**

```bash
git add tests/v2/test_language_splits.py tests/hf/test_language_split_release.py tests/hf/test_language_split_publication.py
git commit -m "test(v2): specify aggregated language shards"
```

### Task 2: Implement deterministic V2 aggregation

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/language_splits.py`
- Test: `tests/v2/test_language_splits.py`

- [ ] **Step 1: Version the V2 language manifest and define the shard limit.**

Add `V2_LANGUAGE_SPLIT_CONTRACT_VERSION = "v2-language-splits-v2"` and `DEFAULT_MAX_ROWS_PER_SHARD = 100_000`. Make the limit injectable in the generator for small deterministic tests, while the public release path uses the default.

- [ ] **Step 2: Replace source-file writers with per-language rotating writers.**

For each language table, keep writers keyed by normalized language. Stream source files in inventory order, split each language's record-batch indices at the row limit, write complete rows with the expected schema, and rotate to `part-{index:05d}-of-{count:05d}.parquet`. Close all writers after the table completes.

- [ ] **Step 3: Record shard provenance and validate each output.**

Change `V2LanguageSplitFile` to store `source_files: tuple[str, ...]`. Track the ordered source files contributing to each shard, validate schema and row count, hash the staged file, and serialize the new field. Keep row conservation keyed by `(table, language)`.

- [ ] **Step 4: Update staged installation and manifest parsing.**

Keep the existing rollback transaction. Update manifest path validation and sorting to accept the aggregated shard names, preserve unmanaged files, and remove only paths listed by the previous manifest. Write the new contract version and all shard metadata before installing the manifest last.

- [ ] **Step 5: Run the focused V2 tests to confirm GREEN.**

Run:

```bash
uv run pytest tests/v2/test_language_splits.py -q
```

Expected: all V2 generator tests pass, including aggregation, rotation, provenance, schema, conservation, rollback, overlap, and stale-file cases.

- [ ] **Step 6: Commit the generator.**

```bash
git add src/osm_polygon_wikidata_only/v2/language_splits.py tests/v2/test_language_splits.py
git commit -m "feat(v2): aggregate language rows into bounded shards"
```

### Task 3: Update release planning and publication contracts

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/language_split_release.py`
- Modify: `src/osm_polygon_wikidata_only/hf/language_split_publication.py`
- Modify: `tests/hf/test_language_split_release.py`
- Modify: `tests/hf/test_language_split_publication.py`

- [ ] **Step 1: Make V2 dry-run planning use inventory row counts.**

For each non-empty table/language bucket, calculate `ceil(row_count / DEFAULT_MAX_ROWS_PER_SHARD)` and emit deterministic `part-*` records with planned row counts. Do not scan every source file again just to construct the dry-run path list.

- [ ] **Step 2: Preserve the atomic publication guard.**

Count planned data files plus manifest and card. Keep the refusal message and ensure V2's real inventory passes before generation. Keep V1's existing paths and count behavior byte-compatible.

- [ ] **Step 3: Update managed card wording.**

Render V2 examples using `language_splits/<configuration>/lang-<language>/part-*.parquet`; keep V1's `lang-<language>-00000-of-00001.parquet` wording unchanged.

- [ ] **Step 4: Add deterministic Viewer config declarations to both cards.**

Carry a sorted mapping of each language-bearing configuration to its non-empty language buckets in `LanguagePublicationPlan`. Replace a marked YAML front-matter block while preserving all existing fields. Emit one config per language-bearing table and one split/path entry per language; use exact V1 paths and V2 shard globs. Keep V1 Viewer split names as `lang-<language>`. For V2, replace dashes in the language code with underscores for the Viewer split name (`lang_<language>`), while retaining `lang-<language>` in the storage path. Include the unknown partition and preserve unrelated configs.

- [ ] **Step 5: Update publication tests.**

Cover under-limit V2 plans, exact operation paths, stale old and new V2 shard cleanup, card YAML Viewer configs, card preservation/idempotence, and revision-bound verification. Assert that publication never includes paths outside the manifest-owned V2 namespace.

- [ ] **Step 6: Run publication-focused tests and commit.**

```bash
uv run pytest tests/hf/test_language_split_release.py tests/hf/test_language_split_publication.py -q
git add src/osm_polygon_wikidata_only/hf/language_split_release.py src/osm_polygon_wikidata_only/hf/language_split_publication.py tests/hf/test_language_split_release.py tests/hf/test_language_split_publication.py
git commit -m "feat(hf): plan and publish aggregated V2 shards"
```

Expected: focused release tests pass and the V2 plan contains fewer than 25,000 operations for the live inventory.

### Task 4: Update public documentation and run local quality gates

**Files:**
- Modify: `docs/language-splits.md`
- Modify: `tests/fixtures/golden/cli_help_publish-language-splits.txt` only if help output changes

- [ ] **Step 1: Document the V2 shard layout and contract version.**

Explain that both dataset cards declare language configs and Viewer-selectable language splits: V1 uses `lang-*`, while V2 uses underscore-safe `lang_*` names and retains dash-based storage directories. Explain that V2 uses deterministic `part-*` Parquet shards, preserves row-level language semantics and source provenance, and remains one atomic Hub commit. Remove the statement that V2 is currently impossible while retaining the 25,000-file safety refusal.

- [ ] **Step 2: Run focused and static checks.**

```bash
uv run pytest tests/v2/test_language_splits.py tests/hf/test_language_split_release.py tests/hf/test_language_split_publication.py -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src scripts
```

- [ ] **Step 3: Run repository quality checks available on this branch.**

Run the repository's targeted architecture, CRAP, mutation-scope, strict documentation, packaging, and Docker help checks. Record exact pass/fail results; do not claim the full gate if a pre-existing or environment-only check remains blocked.

- [ ] **Step 4: Commit documentation and final local tests.**

```bash
git add docs/language-splits.md tests/fixtures/golden/cli_help_publish-language-splits.txt
git commit -m "docs: document aggregated V2 language shards"
```

### Task 5: Review, merge, and release V2

**Files/remote targets:**
- GitHub PR in `NoeFlandre/osm-polygon-wikidata-only`
- Hugging Face dataset `NoeFlandre/osm-polygon-wikidata-and-wikipedia`
- GitHub issues/project item #13

- [ ] **Step 1: Review the final diff and run the merged-branch CI.**

Open a non-draft PR from `codex/v2-language-shard-aggregation`, request independent review, resolve findings, and wait for required CI checks to pass before merging.

- [ ] **Step 2: Fast-forward the clean canonical checkout after merge.**

Verify `/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/repo` is clean, fetch `origin/main`, and fast-forward it to the merged revision.

- [ ] **Step 3: Update and verify the V1 Viewer metadata.**

Run the exact V1 publication command against the already-published V1 files. It may commit only the README front matter/managed section; it must not rewrite V1 data. Verify `/splits` lists language configs and `/first-rows` succeeds for representative language splits.

- [ ] **Step 4: Generate and dry-run V2 from the HDD data root.**

Run the V2 language-split dry run and confirm the predicted operation count is below 25,000. Generate locally only after the plan is accepted; preserve the existing V1 output and all unrelated data.

- [ ] **Step 5: Apply one exact V2 Hub commit.**

Run:

```bash
uv run osm-polygon-wikidata-only publish-language-splits \
  --data-root "/Volumes/Seagate M3/projects/osm-polygon-wikidata-only" \
  --dataset-version v2 \
  --confirm-repo NoeFlandre/osm-polygon-wikidata-and-wikipedia \
  --apply
```

The command must either commit all V2 shards, manifest, and managed card together or fail before remote mutation.

- [ ] **Step 6: Independently verify both Hub revisions.**

At each returned immutable revision, verify the complete file inventory, manifest source fingerprint, shard count, total rows, schema/sample rows, representative LFS hashes, card preservation, and Dataset Viewer/API visibility. Rerun each exact command and require `no_op=true`, `committed=false`, and no changed files.

- [ ] **Step 7: Update the issue and Project 1.**

Record the source fingerprint, final Hub revision, file/row counts, no-op revision, and verification evidence on issue #13. Close #13 and set its Project 1 item to Done only after all checks pass. Remove the temporary worktree and scratch artifacts, then verify the canonical checkout is clean.
