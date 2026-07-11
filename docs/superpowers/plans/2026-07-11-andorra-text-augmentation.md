# Andorra Text Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and upload multilingual Wikipedia documents/sections, Wikivoyage documents/sections, and selected Wikidata facts for the existing Andorra region without changing existing artifacts.

**Architecture:** A new augmentation package reads existing core Parquet files, derives Wikipedia documents locally, fetches exact-revision section HTML plus Wikivoyage and selected Wikidata claims through the shared session, and atomically publishes five sidecars plus an augmentation manifest. A dedicated `augment-region` CLI command isolates the pilot from existing processing commands.

**Tech Stack:** Python 3.12, stdlib MediaWiki HTTP/HTML parsing, PyArrow, existing Wikimedia scheduler/cache, pytest, Ruff, strict mypy.

---

### Task 1: Sidecar schemas and deterministic models

**Files:**
- Create: `src/osm_polygon_wikidata_only/augmentation/models.py`
- Create: `src/osm_polygon_wikidata_only/augmentation/schema.py`
- Create: `tests/test_augmentation.py`

- [ ] Write failing tests asserting exact document, section, and fact columns; deterministic IDs; and typed empty Parquet output.
- [ ] Run `uv run pytest tests/test_augmentation.py -q`; expect import failure.
- [ ] Implement frozen `Document`, `Section`, and `WikidataFact` records and schema-driven writers using `write_table`.
- [ ] Run the focused tests; expect pass.

### Task 2: Wikipedia reuse and exact-revision sections

**Files:**
- Create: `src/osm_polygon_wikidata_only/augmentation/sections.py`
- Create: `src/osm_polygon_wikidata_only/augmentation/mediawiki.py`
- Modify: `tests/test_augmentation.py`

- [ ] Write failing tests converting every existing article row into a Wikipedia document without network access and parsing lead plus nested headings from exact-revision HTML.
- [ ] Implement `document_from_article_row(row)` preserving existing fields and `parse_sections(document, html)` using `HTMLParser`, deterministic hierarchy, clean text, metrics, and hashes.
- [ ] Implement a cached MediaWiki fetcher whose request uses `action=parse`, `oldid`, and `prop=text|sections`.
- [ ] Run focused conversion/parser/cache tests; expect pass.

### Task 3: Wikivoyage discovery and Wikidata facts

**Files:**
- Create: `src/osm_polygon_wikidata_only/augmentation/wikimedia.py`
- Modify: `tests/test_augmentation.py`

- [ ] Write failing tests discovering every `*wikivoyage` sitelink, retaining every language, always resolving English fact labels, retaining additional label languages, and normalizing the selected claim allow-list.
- [ ] Implement batched `wbgetentities` augmentation requests for sitelinks, claims, labels, and entity-valued label resolution.
- [ ] Implement Wikivoyage document fetching and exact-revision section fetching through the shared scheduler/session.
- [ ] Run focused multilingual and fact-normalization tests; expect pass.

### Task 4: Incremental Andorra publication and CLI

**Files:**
- Create: `src/osm_polygon_wikidata_only/augmentation/orchestrator.py`
- Create: `src/osm_polygon_wikidata_only/augmentation/__init__.py`
- Modify: `src/osm_polygon_wikidata_only/cli/parser.py`
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Modify: `src/osm_polygon_wikidata_only/config/paths.py`
- Modify: `tests/test_augmentation.py`
- Modify: `tests/test_cli.py`

- [ ] Write failing tests for `augment-region andorra-latest`, input fingerprints, skip/resume, atomic five-file publication, augmentation-manifest-last ordering, and unchanged core file hashes.
- [ ] Implement the orchestrator reading existing `polygons/andorra-latest.parquet` and `articles/andorra-latest.parquet`, writing under `wikipedia/`, `wikivoyage/`, and `wikidata/`, and storing a versioned manifest.
- [ ] Add `augment-region <stem> [--push] [--dry-run]` without changing existing command defaults.
- [ ] Run augmentation, CLI, pipeline, schema, and packaging tests; expect pass.

### Task 5: Verification and requested Andorra upload

**Files:**
- Modify: `README.md`
- Modify: generated external-data artifacts only.

- [ ] Run the full offline suite, coverage, Ruff, mypy, and package build.
- [ ] Hash current Andorra core artifacts before the live run.
- [ ] Run `uv run osm-polygon-wikidata-only augment-region andorra-latest --push` with the configured data root and all languages.
- [ ] Read back all five local sidecars, validate joins/counts/schemas, and confirm core hashes are unchanged.
- [ ] Verify the atomic remote commit contains only the five sidecars, augmentation manifest, and documentation snapshot.
- [ ] Commit and push source changes to `main` after all verification succeeds.
