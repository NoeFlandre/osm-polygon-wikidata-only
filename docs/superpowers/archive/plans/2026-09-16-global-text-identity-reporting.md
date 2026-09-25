# Global Text-Identity Reporting Implementation Plan

> **Historical, archived.** This document records a completed plan or design and
> may not match the current code (paths, names, unchecked boxes). Do not execute
> it; see `docs/adr/` and the code for the current record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make V1 and V2 text maps and public reporting count each text-bearing OSM polygon once by canonical `(osm_type, osm_id)` identity across overlapping regional extracts.

**Architecture:** Add one deterministic, streaming polygon-identity index to the geographic reporting layer. It will retain the first sorted coordinate for each canonical identity, keep regional row counts separate, and expose the same identity join to text-presence maps, H3 aggregation, continent reporting, V1 statistics, and V2 card statistics. Existing polygon geometry/area accounting will remain on its row-based scanner and will not consume the new deduplicated view.

**Tech Stack:** Python 3, PyArrow Parquet batch readers, H3, pytest fixtures, Ruff, ty, architecture checks, CRAP/mutation quality commands, Markdown dataset-card renderers.

---

### Task 1: Add the canonical polygon identity index

**Files:**
- Create: `src/osm_polygon_wikidata_only/hf/_geographic/polygon_identities.py`
- Test: `tests/hf/test_polygon_identities.py`

- [ ] **Step 1: Write deterministic failing tests**

  Add fixtures containing two sorted regional files with different `polygon_id` values but the same `osm_type="way"` and `osm_id=7`, plus a distinct relation. Assert that `load_unique_polygon_records` returns two identities, keeps the first sorted coordinate, and maps every duplicate `polygon_id` back to the canonical identity. Add a fixture with missing/invalid coordinates and assert that the identity index still counts the identity while its coordinate is absent. Add a compatibility fixture without canonical columns and assert that legacy `source:way:id` identifiers remain readable for existing geographic tests.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run `uv run pytest tests/hf/test_polygon_identities.py -q`. Expected result: collection or import failure because the new index module and `load_unique_polygon_records` do not yet exist.

- [ ] **Step 3: Implement the smallest index API**

  Define the following concrete interface:

  ```python
  PolygonIdentity = tuple[str, int | str]

  @dataclass(frozen=True, slots=True)
  class PolygonRecord:
      identity: PolygonIdentity
      polygon_id: str
      wikidata: str
      lon: float | None
      lat: float | None

  @dataclass(frozen=True, slots=True)
  class PolygonIndex:
      records: dict[PolygonIdentity, PolygonRecord]
      by_polygon_id: dict[str, PolygonIdentity]

  def load_unique_polygon_records(paths: Iterable[Path]) -> PolygonIndex:
      """Read sorted polygon files and retain one deterministic record per identity."""
  ```

  Read only columns present in each Parquet schema. When `osm_type` and `osm_id` exist, require a parseable typed identity and use `(str(osm_type), int(osm_id))`; when old fixture schemas omit both columns, parse the final `:<osm_type>:<numeric-id>` suffix or use a namespaced legacy `(\"legacy\", polygon_id)` key. Insert records in sorted path/row order and use `setdefault` so overlapping regional copies cannot replace the first coordinate. Preserve null coordinates as `None` instead of changing area/statistics behavior.

- [ ] **Step 4: Run the focused tests and verify GREEN**

  Run `uv run pytest tests/hf/test_polygon_identities.py -q`. Expected result: all identity-index tests pass.

- [ ] **Step 5: Refactor only after green**

  Keep identity parsing, row streaming, and deterministic record selection in this focused module. Run `uv run ruff check src/osm_polygon_wikidata_only/hf/_geographic/polygon_identities.py tests/hf/test_polygon_identities.py` and keep the index free of renderer imports.

### Task 2: Make V1/V2 text presence and H3 maps use the index

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/geographic_text_presence.py`
- Modify: `src/osm_polygon_wikidata_only/hf/coverage_map.py`
- Modify: `src/osm_polygon_wikidata_only/hf/geographic_text_density.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_geographic/aggregation.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_geographic/parquet_inputs.py`
- Modify: `src/osm_polygon_wikidata_only/v2/maps.py`
- Test: `tests/hf/test_public_dataset_card_geography.py`
- Test: `tests/hf/test_coverage_map.py`
- Test: `tests/hf/test_geographic_text_coverage.py`
- Test: `tests/v2/test_public_artifacts.py`

- [ ] **Step 1: Write the overlapping regional RED regression**

  Extend the geographic fixture builder with canonical `osm_type`/`osm_id` columns and two regional polygon files. Link the same `(way, 7)` to a successful Wikipedia document in one file and a failed or whitespace-only document in the other; link a distinct relation to successful Wikivoyage text. Assert independently for V1 and V2 that `polygon_count`, text-covered identity count, rendered point count, and H3-cell total are the identity counts rather than regional rows. Assert that repeated calls produce the same record/point order and that failed/empty documents never qualify.

- [ ] **Step 2: Run the exact regression and verify RED**

  Run `uv run pytest tests/hf/test_public_dataset_card_geography.py -k overlapping -q tests/hf/test_geographic_text_coverage.py -k overlapping -q`. Expected result: the new assertions fail because the current code keys coverage by `polygon_id` and accepts non-empty text without consistently requiring successful extraction.

- [ ] **Step 3: Implement the shared text join**

  Have `load_text_presence` build one `PolygonIndex`, qualify Wikipedia and Wikivoyage documents only when `fetch_status == "ok"` and `full_text` is a trimmed non-empty string (retain the existing legacy-schema fallback only when the status column is absent), resolve link `polygon_id` values through `by_polygon_id`, and union text coverage by canonical identity. Add canonical identity fields to `CoveredPoint` and the snapshot while retaining existing string-ID fields as compatibility views. Use the index record for the deterministic coordinate and count all unique identities for the denominator. Keep legacy Wikivoyage-QID fallback constrained to the same identity index.

  Update `load_centroids_from_parquet` to read canonical identity columns when available, deduplicate before returning coordinate lists, and retain the current minimal-schema behavior for callers/tests that provide only `lon`/`lat`. Update the focused H3 input helpers so both numerator and denominator are based on canonical identity records; keep invalid-coordinate errors and low-sample semantics unchanged.

- [ ] **Step 4: Run focused GREEN verification**

  Run `uv run pytest tests/hf/test_public_dataset_card_geography.py tests/hf/test_coverage_map.py tests/hf/test_geographic_text_coverage.py tests/v2/test_public_artifacts.py -q`. Expected result: the overlap regressions and the existing geographic contracts pass.

- [ ] **Step 5: Refactor after green**

  Remove duplicate identity-set logic from the map readers, use the shared index at each boundary, and run `uv run ruff check src/osm_polygon_wikidata_only/hf/geographic_text_presence.py src/osm_polygon_wikidata_only/hf/coverage_map.py src/osm_polygon_wikidata_only/hf/geographic_text_density.py src/osm_polygon_wikidata_only/hf/_geographic/aggregation.py src/osm_polygon_wikidata_only/hf/_geographic/parquet_inputs.py src/osm_polygon_wikidata_only/v2/maps.py`.

### Task 3: Align V1 and V2 card/statistics reporting

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/_dataset_stats/aggregation.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_dataset_stats/rendering.py`
- Modify: `src/osm_polygon_wikidata_only/hf/continent_stats.py`
- Modify: `src/osm_polygon_wikidata_only/hf/dataset_card.py`
- Modify: `src/osm_polygon_wikidata_only/v2/card.py`
- Test: `tests/hf/test_dataset_stats.py`
- Test: `tests/hf/test_public_dataset_card_geography.py`
- Test: `tests/v2/test_card.py`

- [ ] **Step 1: Write RED reporting assertions**

  Add V1 and V2 card fixtures with overlapping regional identities, a successful document, a failed extraction with non-empty-looking text, and a whitespace-only document. Assert that the text-covered metric is the typed identity count, the denominator/caption says unique identities, and regional row metrics remain their original row totals. Assert that the continent text columns use one identity once. Preserve assertions for the existing geometry/area section and rich augmentation sections.

- [ ] **Step 2: Run the focused reporting tests and verify RED**

  Run `uv run pytest tests/hf/test_dataset_stats.py tests/hf/test_public_dataset_card_geography.py tests/v2/test_card.py -q`. Expected result: the new status/identity assertions fail against row-based or polygon-ID-based reporting.

- [ ] **Step 3: Implement minimal reporting alignment**

  Reuse the same identity index and successful-document join for the V1 text funnel and continent text counts. In V2, make the existing successful-document/link scan resolve through polygon records rather than trusting regional link rows; use canonical `(osm_type, osm_id)` keys for language funnels and all text-covered counts. Leave document rows, link rows, regional membership, storage totals, source/provenance comparisons, and `render_polygon_stats_section` untouched. Change only labels/definitions needed to distinguish regional row counts from globally unique text-covered polygon identities, preserving every existing rich card section.

- [ ] **Step 4: Run focused GREEN verification**

  Run `uv run pytest tests/hf/test_dataset_stats.py tests/hf/test_public_dataset_card_geography.py tests/v2/test_card.py -q`. Expected result: all focused reporting tests pass.

- [ ] **Step 5: Refactor after green**

  Keep row metrics named as rows and identity metrics named as identities, then run `uv run ruff check src/osm_polygon_wikidata_only/hf/_dataset_stats/aggregation.py src/osm_polygon_wikidata_only/hf/_dataset_stats/rendering.py src/osm_polygon_wikidata_only/hf/continent_stats.py src/osm_polygon_wikidata_only/hf/dataset_card.py src/osm_polygon_wikidata_only/v2/card.py`.

### Task 4: Document the metric contract and preserve release artifacts

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/api.md`
- Test: `tests/test_documentation.py`
- Test: `tests/contracts/test_golden_outputs.py`
- Test: `tests/v2/test_public_artifacts.py`

- [ ] **Step 1: Write documentation RED assertions**

  Assert that the public documentation names `(osm_type, osm_id)`, successful `fetch_status=ok` plus trimmed non-empty `full_text`, and deduplication across overlapping regional extracts. Assert that regional row counts and area-statistics release behavior remain explicitly separate.

- [ ] **Step 2: Run the exact documentation tests and verify RED**

  Run `uv run pytest tests/test_documentation.py tests/contracts/test_golden_outputs.py tests/v2/test_public_artifacts.py -q`. Expected result: new contract assertions fail until the wording is updated.

- [ ] **Step 3: Update only the metric definitions**

  Update map captions, README/card paragraphs, and architecture/API definitions without removing any existing dataset-card schema, attribution, comparison, sentence, storage, provenance, or polygon-area sections. Do not modify the local/Hugging Face publication targets.

- [ ] **Step 4: Run GREEN documentation verification**

  Run the same focused command and verify all existing golden-output and rich-section assertions remain green.

### Task 5: Full verification, review, and safe handoff

**Files:**
- Inspect all modified files and the preserved dirty release work with `git diff --check` and `git diff --stat`.

- [ ] **Step 1: Run the full required local quality gates**

  Run focused tests again, then `uv run ruff check .`, `uv run ty check src tests`, the repository architecture check from `justfile`, and the repository CRAP/mutation commands from the quality gate. Record each exact command, exit code, and any pre-existing or environment-only limitation. Do not claim remote CI, Docker, review, or publication verification from local commands.

- [ ] **Step 2: Run a fresh final diff audit**

  Confirm the target path, branch, preserved dirty files, no untracked accidental artifacts, unchanged area-statistics code, and retained rich card sections. Use `git diff --check` and inspect the staged diff before committing.

- [ ] **Step 3: Commit the bounded implementation**

  Stage only the intended implementation, regression/documentation changes, and the already-preserved release-work edits that belong to this current release branch. Commit with `git commit -m "fix: deduplicate text reporting by OSM identity"`.

- [ ] **Step 4: Push safely and link issue #6**

  Push the current branch without force. If the existing PR #5 is the target branch, update its body with a concise `Closes #6` reference and verification summary; otherwise create a normal PR from the current branch. Do not merge or publish to Hugging Face.

- [ ] **Step 5: Report exact evidence**

  Return the exact branch, commit, PR URL/number, issue linkage, focused and full test commands/results, Ruff/ty/architecture/CRAP results, preservation checks, and residual risks or blockers.
