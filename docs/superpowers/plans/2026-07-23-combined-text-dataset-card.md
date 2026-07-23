# Combined Text Dataset Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a factual, automatically regenerated dataset card with combined Wikipedia/Wikivoyage language statistics and one canonical H3 text-density map.

**Architecture:** Reuse `load_text_presence` as the single coverage join, add a focused H3 text-density renderer, and derive combined language metrics from finalized documents plus their canonical polygon associations. Keep publication assembly as the only generation/upload workflow and atomically delete superseded remote assets.

**Tech Stack:** Python 3.12, PyArrow, H3, Matplotlib, pytest, Hugging Face Hub.

---

### Task 1: Freeze the public card contract

**Files:**
- Modify: `tests/contracts/test_dataset_card_augmentation.py`
- Modify: `tests/hf/test_public_dataset_card_geography.py`

- [ ] Add failing assertions that every canonical table has a concise description, `polygon_articles` is explicitly Wikipedia-only, and Wikivoyage QID association is explained.
- [ ] Add failing assertions for exactly three public maps in order, an explanatory paragraph after all polygons, and no references to either superseded H3 asset.
- [ ] Run `uv run pytest tests/contracts/test_dataset_card_augmentation.py tests/hf/test_public_dataset_card_geography.py -q`; expect assertion failures against current Markdown.
- [ ] Update `hf/dataset_card.py` minimally to satisfy the wording/order contract.
- [ ] Re-run the focused tests; expect pass.

### Task 2: Implement raw combined-text H3 density

**Files:**
- Create: `src/osm_polygon_wikidata_only/hf/geographic_text_density.py`
- Modify: `src/osm_polygon_wikidata_only/hf/repo_layout.py`
- Modify: `tests/hf/test_public_dataset_card_geography.py`
- Modify: `tests/contracts/test_remote_paths.py`

- [ ] Add failing synthetic tests proving Wikipedia/Wikivoyage overlap counts a polygon once, H3 values are raw counts, sorting/output path are deterministic, and rendering uses logarithmic purple-yellow encoding.
- [ ] Run the focused tests; expect missing-module/path failures.
- [ ] Implement `aggregate_geographic_text_density` by assigning `load_text_presence(...).covered_points` to H3 cells and returning sorted `PolygonCountCell` values.
- [ ] Implement rendering by reusing the existing count-map primitives with text-specific title, caption and colourbar.
- [ ] Add `REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE = "assets/geographic_text_density.png"` plus explicitly named legacy constants for the two retired assets.
- [ ] Re-run focused tests; expect pass.

### Task 3: Compute combined language statistics

**Files:**
- Create: `src/osm_polygon_wikidata_only/hf/_dataset_stats/combined_languages.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_dataset_stats/rendering.py`
- Modify: `src/osm_polygon_wikidata_only/hf/publication.py`
- Modify: `tests/hf/test_dataset_stats.py`
- Modify: `tests/hf/test_augmentation_stats.py`

- [ ] Add failing fixtures with overlapping Wikipedia and Wikivoyage languages/QIDs and assert combined document counts, per-language unique polygon counts, concentration and long-tail values.
- [ ] Run focused tests; expect missing combined snapshot and Wikipedia-only wording failures.
- [ ] Implement a frozen `CombinedLanguageStats` and deterministic, column-pruned scanner. Cache per-stem summaries by fingerprints under the existing stats cache directory.
- [ ] Join Wikipedia documents through `polygon_articles`; join Wikivoyage documents through QID to polygons; deduplicate `(language, polygon_id)` and `(project, document_id)`.
- [ ] Make public language rendering consume the combined snapshot while retaining clearly labelled Wikipedia-only funnel metrics.
- [ ] Re-run focused tests; expect pass.

### Task 4: Wire automatic publication and atomic migration

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/publication.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_uploader/operations.py`
- Modify: `tests/contracts/test_publication.py`
- Modify: `tests/hf/test_publication_augmentation.py`
- Modify: `tests/pipeline/test_sync_reconciliation.py`

- [ ] Add failing tests requiring the new density asset add plus both legacy asset deletes in the same publication, README last, and zero submission on generation failure.
- [ ] Add failing tests proving augmentation-only changes regenerate combined metrics/map while unrelated recovery changes respect `refresh_maps=False`.
- [ ] Replace old H3 generators/ops with the combined density generator/op and guarded legacy deletes.
- [ ] Update uploader canonical-replacement safety mapping so either legacy deletion requires the new density add.
- [ ] Re-run publication and sync tests; expect pass.

### Task 5: Documentation, deterministic real generation and live publication

**Files:**
- Modify: `docs/architecture.md`
- Modify: `tests/fixtures/golden/dataset_card.md`
- Modify: `tests/fixtures/golden/publication_file_list.json`
- Generate: `assets/geographic_text_density.png`

- [ ] Update architecture text only where it describes public statistics/maps and publication ordering.
- [ ] Regenerate established goldens and run `uv run pytest -q`.
- [ ] Run `uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`, and `git diff --check`.
- [ ] Generate the real README/map from `OSM_POLYGON_DATA_ROOT`, visually inspect the PNG and verify the README contains factual combined values.
- [ ] Commit changes, fast-forward `main`, push GitHub, publish the README/new asset/legacy deletions to `NoeFlandre/osm-polygon-wikidata-only`, and verify live paths plus a second idempotent publication plan.
