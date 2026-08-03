# V2 Wikipedia-Tag Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Build an explicit V2 dataset that keeps all V1 outputs intact and adds polygons discovered through valid OSM Wikipedia tags in every language.

**Architecture:** V1 remains the default pipeline, schema, data root, manifests, and Hugging Face repository. V2 is an explicit opt-in package and command with its own `processed_v2` output, `cache/v2` state, versioned Parquet contracts, manifest, and Hugging Face repository `NoeFlandre/osm-polygon-wikidata-and-wikipedia`. V2 reads V1 artifacts as immutable inputs, reuses matching documents and sections, and fetches only direct Wikipedia references that are not already represented.

**Tech Stack:** Python 3.12, `uv`, PyArrow/Parquet, existing Wikipedia transport/cache/scheduler, atomic filesystem writes, pytest, Ruff, ty, and the existing HF upload queue.

---

### Task 1: Isolate V2 configuration

**Files:** create `src/osm_polygon_wikidata_only/v2/{__init__,config}.py`; modify `config/paths.py`; tests `tests/v2/test_config.py` and `tests/v2/test_v1_boundary.py`.

- [ ] Add RED tests proving V1 repository ID, V1 paths, and V1 schemas are unchanged, while V2 has `V2_CONTRACT_VERSION="wikipedia-tags-v2"`, `V2_REPO_ID="NoeFlandre/osm-polygon-wikidata-and-wikipedia"`, `processed_v2`, and `cache/v2`.
- [ ] Implement `DatasetVersion` and `DataRoot.processed_v2` / `DataRoot.v2_cache` without changing V1 defaults.
- [ ] Run `uv run pytest tests/v2/test_config.py tests/v2/test_v1_boundary.py -q` and commit `feat(v2): add isolated dataset configuration`.

### Task 2: Parse every OSM Wikipedia reference

**Files:** create `src/osm_polygon_wikidata_only/v2/wikipedia_tags.py`; test `tests/v2/test_wikipedia_tags.py`.

- [ ] Add RED tests for `wikipedia=en:Title`, `wikipedia:fr=Title`, arbitrary codes such as `ja`, `zh-yue`, and `sr-Latn`, URL values, Unicode, underscores, percent encoding, semicolon-separated values, duplicates, malformed values, and unrelated tags.
- [ ] Implement pure frozen `WikipediaTagRef` and `WikipediaTagRejection` models. Parse dynamic language codes without an allow-list, normalize titles deterministically, and return sorted refs plus structured rejections. Never perform I/O.
- [ ] Run `uv run pytest tests/v2/test_wikipedia_tags.py -q` and commit `feat(v2): parse multilingual OSM Wikipedia tags`.

### Task 3: Define V2 schemas and IDs

**Files:** create `src/osm_polygon_wikidata_only/v2/{models,schema}.py`; test `tests/v2/test_schema.py`.

- [ ] Add RED tests freezing V2 metadata and nullable `wikidata` for direct-only pages. Assert that V1 schema objects are unchanged.
- [ ] Implement V2 polygon, document, and link schemas. Preserve V1 fields where useful, add deterministic `link_sources`, and use stable page-based IDs for pages without QIDs. Keep V1 `domain.schema` and `domain.polygon_document_links` untouched.
- [ ] Run V2 schema tests plus `tests/contracts/test_public_imports.py` and commit `feat(v2): define nullable Wikipedia-tag document links`.

### Task 4: Index V1 artifacts read-only

**Files:** create `src/osm_polygon_wikidata_only/v2/v1_index.py`; test `tests/v2/test_v1_index.py`.

- [ ] Add RED tests with small fixtures for lookup by project/language/title, project/language/page ID, and QID; reject duplicate or malformed V1 rows; hash V1 inputs before and after.
- [ ] Implement `V1ReuseIndex` that validates and indexes V1 Parquet files without writing to V1 paths or issuing network requests.
- [ ] Run tests and commit `feat(v2): index immutable V1 artifacts for reuse`.

### Task 5: Enrich only missing direct Wikipedia pages

**Files:** create `src/osm_polygon_wikidata_only/v2/direct_enrichment.py`; tests `tests/v2/test_direct_enrichment.py`.

- [ ] Add RED tests proving matching V1 pages cause zero fetches, missing pages are fetched once, no-QID pages are retained, dual-source links merge provenance, conflicting pages both remain, and a second run has zero new calls.
- [ ] Implement a request planner and adapter around the existing `WikipediaClient.fetch_article`, `JsonFileCache`, retry classification, and scheduler. Reuse V1 rows first. Resolve a QID opportunistically, but never discard a valid page when QID resolution fails. Persist malformed/not-found/retryable outcomes separately.
- [ ] Run focused tests and Ruff; commit `feat(v2): enrich missing Wikipedia-tag pages`.

### Task 6: Persist V2 atomically and resumably

**Files:** create `src/osm_polygon_wikidata_only/v2/{storage,manifest}.py`; test `tests/v2/test_storage.py`.

- [ ] Add RED tests for atomic Parquet writes, cleanup after failures, manifest-last ordering, interruption recovery, unchanged-stem skipping, partial-stem resume, deterministic hashes, and V1 hash preservation.
- [ ] Implement writes only below `processed_v2`, per-stem content fingerprints under `cache/v2`, atomic replacement, strict validation before manifest update, and durable rejection/reuse receipts.
- [ ] Run tests and commit `feat(v2): add atomic resumable storage`.

### Task 7: Add explicit V2 sync and publication

**Files:** create `src/osm_polygon_wikidata_only/v2/runner.py`; modify `cli/parser.py`, `cli/commands.py`, and `hf/repo_layout.py`; tests `tests/v2/test_runner.py` and `tests/v2/test_cli.py`.

- [ ] Add RED tests proving V1 `sync-dir` remains unchanged and V2 requires an explicit `--dataset-version v2`, selects the V2 output/cache and exact V2 HF repo, reuses V1 artifacts, processes only tag deltas, resumes without duplicate commits, and rejects malformed V1 inputs.
- [ ] Implement the explicit V2 mode. Keep V1 as the default. Use the existing upload queue and pure publication operations, but prevent V2 paths or repo ID from targeting V1 accidentally. Publish one atomic region commit and a final metadata/card commit.
- [ ] Run V2 and existing CLI contract tests; commit `feat(v2): add explicit Wikipedia-tag sync mode`.

### Task 8: Generate the V2 public card and audit tool

**Files:** create `src/osm_polygon_wikidata_only/v2/dataset_card.py`, `scripts/v2_audit.py`, and `tests/v2/test_dataset_card.py`; modify public docs only after local review.

- [ ] Add RED golden tests for the union filter, all-language parsing, V1 reuse, nullable QIDs, provenance, counts by discovery source, and absence of private paths or internal notes.
- [ ] Implement deterministic V2 README/card generation from the V2 manifest and a read-only local/remote hash audit targeting the V2 HF repository.
- [ ] Run focused tests and commit `feat(v2): publish a public Wikipedia-tag dataset card`.

### Task 9: Full safety and release verification

**Files:** create `tests/v2/test_end_to_end.py` and `docs/v2-wikipedia-tags.md`; modify `docs/architecture.md` and `docs/development.md`.

- [ ] Add an injected end-to-end test for Wikidata-only, Wikipedia-only, and dual-tag polygons, including all-language output, deduplication, resumability, exact publication order, and unchanged V1 hashes.
- [ ] Document the V1/V2 boundary, command, separate HF target, direct-tag limitations, provenance, and recovery behavior without exposing private storage paths in public card text.
- [ ] Run `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, and `git diff --check`.
- [ ] Run an offline V2 dry run with stub Hub and in-memory clients. Review all diffs, generated card, V1 hash comparison, and publication plan before any remote upload.
- [ ] Only after review, push the commits and upload V2 to `NoeFlandre/osm-polygon-wikidata-and-wikipedia`.
