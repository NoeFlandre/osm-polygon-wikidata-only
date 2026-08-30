# Code Quality Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the highest measured maintenance risks in the sentence runner, card scanners, Grid5000 safety boundary, and quality-gate scripts while preserving every existing output, API, checkpoint, and publication contract.

**Architecture:** Keep the existing public facades and file layout. Extract small pure helpers for manifest decoding, metadata construction, checkpoint-batch writing, columnar card scanning, ledger safety, containment-audit payloads, and quality-report parsing; keep filesystem, Parquet, and subprocess effects at their current boundaries. Extend the existing bounded CRAP gate to the refactored modules so the quality claim is measured rather than inferred.

**Tech Stack:** Python 3.12, pytest/pytest-cov, Ruff, ty, Radon, crap4py, mutmut, uv, Just.

---

### Task 1: Establish and record the baseline

**Files:**
- Create: `docs/superpowers/plans/2026-08-30-code-quality-refactor.md`
- No production changes.

- [x] **Step 1: Run the tracked regression suite.**

Run `UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest -q --ignore=tests/unit` and record the result. The unrelated, untracked `tests/unit` directory is excluded because it imports a package outside this repository and must remain untouched.

- [x] **Step 2: Run baseline static and quality checks.**

Run Ruff, `ty`, coverage, `just crap-all`, Radon, and `just mutation`. Use the resulting function complexity and coverage data to choose only measurable refactor seams.

### Task 2: Make sentence-runner manifest handling single-purpose and typed

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/sentence_runner.py`
- Modify: `tests/v2/test_sentence_runner.py`
- Modify: `Justfile`

- [x] **Step 1: Add a RED test for the extracted summary decoder.**

Add this focused test and import `_summary_from_mapping` from the runner:

```python
def test_summary_decoder_preserves_manifest_values() -> None:
    summary = _summary_from_mapping(
        {
            "stem": "region-latest",
            "project": "wikipedia",
            "sections": 3,
            "split_sections": 2,
            "unsplit_sections": 1,
            "sentence_rows": 4,
            "supported_languages": ["en"],
            "unsupported_languages": ["xx"],
        },
        error="invalid summary",
    )

    assert summary == SentenceRegionSummary(
        stem="region-latest",
        project="wikipedia",
        sections=3,
        split_sections=2,
        unsplit_sections=1,
        sentence_rows=4,
        supported_languages=("en",),
        unsupported_languages=("xx",),
    )
```

Run `UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/v2/test_sentence_runner.py::test_summary_decoder_preserves_manifest_values`. It must fail at collection because the new helper does not exist yet.

- [x] **Step 2: Implement the decoder and turn the existing readers green.**

Add one `_summary_from_mapping(raw: Mapping[str, object], *, error: str)` helper that constructs `SentenceRegionSummary`, catches the existing `KeyError`, `TypeError`, and `ValueError` cases, and raises `ValueError(error)` with the existing exception chaining. Make `_summary_from_checkpoint()` and `_load_manifest_summaries()` delegate to it without changing their external error messages. Run the focused test and the complete `tests/v2/test_sentence_runner.py` file.

- [x] **Step 3: Add a RED test for shared manifest identity.**

Add a test that runs a one-region sentence split, reads the JSON manifest, and asserts the exact metadata keys and values for contract version, segmenter, model ID, model revision, and segmenter version. Import `_manifest_metadata` and assert it equals the manifest metadata. The test must first fail because `_manifest_metadata` is absent.

- [x] **Step 4: Extract and use manifest metadata/payload helpers.**

Implement `_manifest_metadata(segmenter)` once and use it in both writing and validation. Extract `_manifest_payload(...)` only if it removes duplication without changing JSON key order. Keep the current unsupported-language policy string and sorted region/language ordering byte-for-byte compatible. Run the focused manifest tests and the full sentence-runner file.

- [x] **Step 5: Reduce orchestration branching with tested helpers.**

Extract `_requested_stems(...)` for normalization/duplicate removal, `_write_checkpoint_batches(...)` for validated non-empty Arrow writes, and manifest payload/region decoders. Preserve the existing error messages, output schema, atomic replacement, and empty-batch behavior. Add focused tests for missing stems, empty selection, invalid stems, invalid manifests, and restart fast-path reuse.

- [x] **Step 6: Include the runner in its bounded CRAP scope.**

Extend the `crap` recipe to cover `src/osm_polygon_wikidata_only/v2/sentence_runner.py` and `tests/v2/test_sentence_runner.py`. Run the focused CRAP command and keep every reported sentence-runner function below 6.00; add tests or split a helper further if coverage/complexity leaves an offender.

### Task 3: Simplify and fully exercise the CRAP/mutation-report utilities

**Files:**
- Modify: `scripts/quality/crap_score.py`
- Modify: `scripts/quality/mutation_gate.py`
- Modify: `tests/quality/test_crap_score.py`
- Modify: `tests/quality/test_mutation_gate.py`
- Modify: `Justfile`

- [x] **Step 1: Add RED tests for the report-parsing seams.**

Add focused tests for class blocks being ignored, method names being qualified, unique suffix path matching, malformed Radon fields, malformed coverage line arrays, and the zero/non-killed mutation-result paths. Import the new small helpers only when each test is introduced so each production seam begins with a genuine failing test.

- [x] **Step 2: Extract minimal report adapters.**

Split `entries_from_reports()` into helpers that validate one Radon block, resolve its qualified name and line range, resolve one coverage file, and calculate one function's coverage. Replace repeated sorting expressions with one `_worst_first()` helper. Preserve `CrapEntry`, the standard formula, all current validation errors, and CLI output.

- [x] **Step 3: Simplify mutation-gate validation.**

Extract non-killed result selection from `ensure_all_killed()` so each function remains below complexity 6 while retaining the exact empty-report and non-killed error contracts. Keep the script standard-library-only.

- [x] **Step 4: Add a dedicated quality-tools CRAP recipe.**

Run the quality tests with coverage, Radon JSON, and `scripts/quality/crap_score.py --maximum 6`; include this recipe in `crap-all`. The recipe must report no function at or above 6.00.

### Task 4: Simplify adjacent measured boundaries

**Files:**
- Modify: `src/osm_polygon_wikidata_only/v2/card.py`
- Modify: `src/osm_polygon_wikidata_only/grid5000/sentence_controller.py`
- Modify: `scripts/audit_containment.py`
- Modify: `tests/v2/test_card.py`
- Modify: `tests/grid5000/test_sentence_controller.py`
- Add: `tests/quality/test_audit_containment.py`

- [x] **Step 1: Decompose the card scanner.**

Extract document/polygon column decoding and row accounting while preserving single-pass scans, metric values, and the historical missing-language failure.

- [x] **Step 2: Decompose ledger safety and the containment audit payload.**

Keep source-commit migration decisions and read-only audit output byte-compatible while making the pure decisions independently testable.

### Task 5: Final validation and publication

**Files:**
- No generated reports or mutation worktrees committed.

- [x] **Step 1: Run focused RED→GREEN→REFACTOR checks after each slice.**

Run the affected test files, Ruff, format check, and `ty check src scripts` after every refactor. Re-run the full tracked regression suite after the last slice.

- [x] **Step 2: Run the complete quality gates.**

Run the full repository tests with `--ignore=tests/unit`, coverage, acceptance tests, architecture checks, all bounded CRAP scopes, and mutation testing. Require zero surviving/unexecuted/timeout/suspicious mutants in the configured mutation scope and a maximum CRAP below 6 in every configured scope.

- [x] **Step 3: Review and publish only validated tracked changes.**

Run `git diff --check`, inspect the full diff and public contracts, stage only the plan, source, test, and Justfile paths changed by this work, commit with a Conventional Commit subject, and push `main`. Preserve all pre-existing untracked files.
