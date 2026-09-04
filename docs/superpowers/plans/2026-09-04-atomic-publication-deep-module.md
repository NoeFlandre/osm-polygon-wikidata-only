# Atomic Publication Deep Module Implementation Plan

**Goal:** Turn `io/atomic.py` from a pair of fixed writers into a deep module
that owns the package's single durable-publication ritual, and delete the
eleven hand-rolled copies of that ritual without changing a byte on disk.

**Architecture:** One `atomic_replacement()` context manager owns
`mkdir` → `mkstemp` → close → *caller writes* → `os.replace` → cleanup. Every
existing writer becomes a payload-specific spelling of it. No public API of any
other module changes.

**Tech Stack:** Python 3.12, pytest/pytest-cov, Ruff, ty, Radon, crap4py, Just,
mutmut.

---

### Task 1: Baseline and scope

- [x] **Step 1: Run the tracked baseline.** `2646 passed, 2 skipped`.

- [x] **Step 2: Measure complexity.** `radon cc -n B -s src scripts` reports no
  remaining B-level function, so the actionable target is duplication rather
  than per-function complexity. `grep` for `os.replace` finds sixteen sites;
  eleven of them re-implement the same temporary-sibling ritual, including two
  byte-identical `_copy_file_atomically` functions in
  `grid5000/sentence_protocol.py` and `grid5000/sentence_controller.py`, a
  third copy as `_atomic_copy` in `pipeline/_wikidata_recovery/transaction.py`,
  and an `_atomic_overwrite_parquet` in `augmentation/integrity.py` that
  duplicates `io.atomic.atomic_write_parquet` exactly.

### Task 2: Extract the ritual

**Files:**
- Create: `tests/io/test_atomic.py`
- Modify: `src/osm_polygon_wikidata_only/io/atomic.py`
- Modify: `tests/io/test_atomic_write.py`

- [x] **Step 1: Add the failing contract tests.** Cover the hidden-sibling
  naming, missing parent directories, overwrite, cleanup on `RuntimeError` and
  on `KeyboardInterrupt`, a body that removed the sibling itself, descriptor
  closure, the deterministic JSON encoding, and byte copies. Observe the
  expected `ImportError`.

- [x] **Step 2: Implement `atomic_replacement`, `atomic_write_json`, and
  `atomic_copy_file`,** and re-express `atomic_write_text` and
  `atomic_write_parquet` on the context manager.

- [x] **Step 3: Re-point the two writer-level cleanup tests.** They injected
  failure through `atomic.os.fdopen`, which the shared ritual no longer calls;
  `atomic.os.fsync` keeps the same intent, and the mechanism itself is now
  pinned directly in `tests/io/test_atomic.py`.

### Task 3: Delete the duplicated rituals

**Files:**
- Modify: `grid5000/sentence_protocol.py`, `grid5000/sentence_controller.py`,
  `grid5000/sentence_job.py`
- Modify: `pipeline/containment_migration.py`, `pipeline/link_migration.py`,
  `pipeline/_link_migration/transaction.py`,
  `pipeline/_wikidata_recovery/transaction.py`
- Modify: `v2/checkpoints.py`, `v2/sentence_checkpoints.py`,
  `v2/sentence_runner.py`
- Modify: `hf/_geographic/rendering.py`, `io/cache.py`
- Modify: `augmentation/integrity.py`,
  `augmentation/wikipedia_document_migration.py`,
  `augmentation/rejection_ledger.py`
- Modify: `tests/grid5000/test_sentence_artifacts.py`,
  `tests/hf/test_geographic_text_coverage.py`,
  `tests/io/test_cache_recovery_contracts.py`

- [x] **Step 1: Replace the three byte-copy implementations** with
  `atomic_copy_file`.

- [x] **Step 2: Replace the seven `atomic_write_text(path, json_dumps(v) + "\n")`
  call sites** with `atomic_write_json`.

- [x] **Step 3: Convert the remaining inline rituals** -- streamed and buffered
  Parquet writes, the matplotlib PNG save, and the metadata-preserving install
  -- to `atomic_replacement`, and delete the three pass-through aliases that
  only forwarded to `atomic_write_parquet`.

- [x] **Step 4: Re-point the three tests that reached into a moved
  implementation.** The grid5000 temp-naming test is replaced by a behavioral
  test of `_install_checkpoint_tree`, since the naming contract now lives in
  `tests/io/test_atomic.py`; the coverage-map test patches `os.replace` where
  the ritual now lives; the cache test patches `atomic_write_json`.

- [x] **Step 5: Keep the two divergent formats intact.** Migration journals in
  `pipeline/link_migration.py` and `pipeline/_link_migration/transaction.py`
  keep their `indent=2, sort_keys=True` encoding,
  `pipeline/containment_migration.py` keeps its explicit `pq.write_table` call,
  and `_install_file` keeps `shutil.copy2`.

### Task 4: Gate the consolidated module

**Files:**
- Modify: `Justfile`, `docs/development.md`, `tests/test_packaging.py`

- [x] **Step 1: Add a failing packaging contract** for a `crap-atomic` recipe
  and its inclusion in `crap-all`.

- [x] **Step 2: Add the bounded recipe** and wire it into `crap-all`,
  `quality-strength`, and `quality-advanced`.

- [x] **Step 3: Decide the mutation scope from measurement, not assumption.**
  A trial run with `io/atomic.py` as the only mutmut source produced 39 mutants
  with 6 survivors, of which three are equivalent: `"UTF-8"` for `"utf-8"`
  (Python's codec registry is case-insensitive), `"SNAPPY"` for `"snappy"`, and
  an omitted `compression` (pyarrow already defaults to snappy). The gate
  refuses any survivor and has no allowlist, so the module stays out of the
  mutmut scope, consistent with the existing policy for file-system boundaries,
  and `test_file_boundary_refactors_stay_out_of_mutation_scope` records that
  with its reason. `crap-atomic` reports every function at or below CRAP 2.00
  with 100% branch coverage.

### Task 5: Verification

- [x] **Step 1:** Full tracked suite, Ruff, format, ty, acceptance,
  architecture, `crap-all`, `mutation`, smoke, and `git diff --check`.

- [x] **Step 2:** Confirm the mutmut configuration in `pyproject.toml` is
  unchanged, so the existing zero-survivor result still applies to the same
  deterministic scope.
