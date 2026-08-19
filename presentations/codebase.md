---
title: "The codebase in one view"
author: "Noé Flandre"
date: "2026-08-02"
theme: default
aspect_ratio: "16:9"
footer:
  left: "OSM Polygon Wikidata"
  right: "{n} / {N}"
custom_css: |
  :root {
    --colloquium-bg: #f7f9fc;
    --colloquium-text: #243447;
    --colloquium-heading: #102a43;
    --colloquium-accent: #1769aa;
    --colloquium-link: #1769aa;
    --colloquium-code-bg: #eaf1f7;
    --colloquium-muted: #52606d;
    --colloquium-border: #d9e2ec;
    --colloquium-font-body: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    --colloquium-font-heading: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    --colloquium-font-mono: "SFMono-Regular", Consolas, monospace;
    --colloquium-slide-padding: 58px 72px 54px;
  }
  .slide { font-size: 22px; }
  .slide h1 { font-size: 2.05em; letter-spacing: -0.02em; margin-bottom: 0.32em; }
  .slide h2 { font-size: 1.45em; letter-spacing: -0.01em; margin-bottom: 0.42em; }
  .slide h3 { color: var(--colloquium-accent); }
  .slide p { max-width: 1040px; }
  .slide ul { max-width: 1080px; }
  .slide table { font-size: 0.74em; margin-top: 0.35em; }
  .slide th, .slide td { padding: 0.34em 0.55em; }
  .slide--title { background: #102a43; color: #f7f9fc; }
  .slide--title h1, .slide--title h2 { color: #f7f9fc; max-width: 980px; }
  .slide--title .slide-content p { color: #d9e2ec; }
  .slide--title .slide-content { max-width: 900px; }
  .slide--title .kicker { color: #7cc4fa; font-size: 0.78em; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
  .slide--title-left { padding: 64px 78px 54px; }
  .slide--title-left h1 { font-size: 2.2em; max-width: 1080px; }
  .slide--content > .slide-content { max-width: 1136px; }
  .slide .slide-content img { max-height: 540px; margin: 0 auto; }
  .slide .map { width: 100%; max-height: 520px; object-fit: contain; border: 1px solid var(--colloquium-border); border-radius: 12px; }
  .slide .map-caption { color: var(--colloquium-muted); font-size: 0.72em; margin-top: 0.35em; text-align: center; }
  .slide .small { font-size: 0.75em; color: var(--colloquium-muted); }
  .slide .lead { font-size: 1.18em; line-height: 1.35; max-width: 980px; color: var(--colloquium-heading); }
  .slide .rule { height: 4px; width: 70px; background: #39a9db; border-radius: 4px; margin: 0.2em auto 0.9em; }
  .slide .mono { font-family: var(--colloquium-font-mono); font-size: 0.86em; }
  .slide .tight li { margin-bottom: 0.12em; }
  .slide .muted { color: var(--colloquium-muted); }
  .slide .accent { color: var(--colloquium-accent); }
  .slide .flow { font-size: 1.04em; line-height: 1.75; color: var(--colloquium-heading); }
  .slide .flow strong { color: var(--colloquium-accent); }
  .slide .callout { background: #eaf1f7; border-left: 5px solid #1769aa; padding: 0.65em 0.9em; border-radius: 4px; max-width: 1080px; }
  .slide .callout p { margin: 0; }
  .slide .two-up img { max-height: 390px; }
  .slide .two-up p { font-size: 0.8em; }
  .slide .source-note { font-size: 0.62em; color: var(--colloquium-muted); }
  .slide code { font-size: 0.78em; }
  .slide pre { font-size: 0.7em; }
  .slide--title-left .slide-content { max-width: 1080px; }
---

# A modular, resumable enrichment pipeline

<p class="kicker">Codebase overview</p>

<div class="rule"></div>

<p>The project turns Geofabrik PBF files into linked OSM, Wikidata, Wikipedia, and Wikivoyage tables.</p>
<p>It is built for long runs, public publication, and safe resume.</p>

<!-- notes: [Sources] Project README, docs/architecture.md, pyproject.toml, and the current source tree. Colloquium workflow: https://pypi.org/project/colloquium/0.2.2/ -->

---

<!-- layout: title-left -->
## The system is layered so each change stays local

| Package | Responsibility |
| --- | --- |
| <code>config</code> | Settings and external data-root resolution |
| <code>domain</code> | Pure IDs, models, geometry, and schemas |
| <code>io</code> | PBF, Parquet, cache, manifests, and atomic writes |
| <code>enrichment</code> | Wikidata and Wikipedia clients, cache, linking |
| <code>augmentation</code> | Wikivoyage, sections, facts, checkpoints |
| <code>pipeline</code> | Extraction, row construction, persistence, sync |
| <code>hf</code> | Dataset card, maps, publication, Hub uploads |
| <code>cli</code> and <code>utils</code> | Adapters, retries, logging, scheduling |

<p class="small">Stable facade modules keep public imports readable while focused private packages hold implementation details.</p>

<!-- notes: [Sources] docs/architecture.md and the source package layout under src/osm_polygon_wikidata_only. -->

---

<!-- layout: content -->
## Core processing has four focused phases

<!-- columns: 2 -->
<h3>1. Extract</h3>
<p>Stream OSM PBF elements and retain tagged closed ways and multipolygon relations.</p>
<h3>2. Enrich</h3>
<p>Resolve unique QIDs, discover sitelinks, and fetch Wikipedia data in bounded batches.</p>

|||

<h3>3. Construct</h3>
<p>Build deterministic polygon, document, and unified link rows.</p>
<h3>4. Persist</h3>
<p>Write atomic Parquet files and update the processed manifest after the files are valid.</p>

<p class="callout">The same contracts are tested at row, table, integration, publication, and golden-output levels.</p>

<!-- notes: [Sources] pipeline/extractor.py, pipeline/enrichment_phase.py, pipeline/row_construction.py, pipeline/persistence.py, and pipeline/processor.py. -->

---

<!-- layout: content -->
## Wikimedia access is shared, paced, and cached

| Concern | Design |
| --- | --- |
| Transport | One JSON helper validates responses and reports throttles |
| Authentication | Bot-password status is tracked per host |
| Scheduling | Host pacing, in-flight limits, cooldowns, and systemic backoff |
| Batching | QIDs and titles are sent in bounded API requests |
| Cache | JSON request results live below the selected data root |
| Failure handling | Retry temporary failures, preserve permanent absence, never cache transport failure as truth |

<p class="lead">This keeps network policy in one place while clients focus on request shape and result conversion.</p>

<!-- notes: [Sources] enrichment/wikimedia/transport.py, enrichment/wikimedia_auth.py, utils/request_scheduler.py, utils/retry.py, and io/cache.py. -->

---

<!-- layout: content -->
## Augmentation turns documents into reusable analysis units

<p class="lead">After core processing, the augmentation stage adds Wikivoyage documents, Wikipedia and Wikivoyage sections, and Wikidata facts.</p>

<!-- columns: 2 -->
<h3>Checkpoints</h3>
<ul class="tight">
  <li>Resolved entities</li>
  <li>Wikivoyage documents</li>
  <li>Document sections</li>
  <li>Wikidata facts</li>
</ul>

|||

<h3>Safety</h3>
<ul class="tight">
  <li>Core hashes are checked before and after work.</li>
  <li>Sidecars are written atomically.</li>
  <li>Partial work can be reused after interruption.</li>
  <li>Join integrity is checked before publication.</li>
</ul>

<!-- notes: [Sources] augmentation/orchestrator.py, augmentation/checkpoints.py, augmentation/steps.py, and augmentation/integrity.py. -->

---

<!-- layout: title-left -->
## Recovery and migration protect the data you already have

<ul>
  <li><strong>Wikidata recovery</strong> audits finalized regions and repairs only affected QIDs.</li>
  <li><strong>Link migration</strong> creates the unified polygon-to-document artifact with journaled atomic writes.</li>
  <li><strong>Containment retirement</strong> removes fully contained extracts only after a safe proof.</li>
  <li><strong>Atomic I/O</strong> writes temporary files, validates them, then replaces the target.</li>
  <li><strong>Upload queue</strong> keeps immutable snapshots so a retry does not read changing local bytes.</li>
</ul>

<p class="callout">The normal response to interruption is resume, not rebuild.</p>

<!-- notes: [Sources] pipeline/_wikidata_recovery, pipeline/link_migration.py, pipeline/containment_migration.py, io/atomic.py, and hf/upload_queue.py. -->

---

<!-- layout: content -->
## Publication converges local truth with the Hugging Face Hub

<p class="flow"><strong>Local artifacts</strong> → validate → <strong>ordered operations</strong> → <strong>one atomic Hub commit</strong></p>

<ul class="tight">
  <li>Remote inventory identifies missing or stale region artifacts.</li>
  <li>Publish-only repairs run before new PBF processing.</li>
  <li>README, maps, manifests, and Parquet files are assembled from the same local snapshot.</li>
  <li>Legacy paths are retired only with a canonical replacement in the same commit.</li>
</ul>

<p class="small">The Hub is the public dataset. Local caches and intermediate files are operator state, not published artifacts.</p>

<!-- notes: [Sources] hf/remote_inventory.py, hf/reconciliation.py, hf/publication.py, hf/repo_layout.py, hf/upload_queue.py, and cli/run_sync.py. -->

---

<!-- layout: content -->
## quality tools

<!-- columns: 2 -->
<h3>Developer loop</h3>
<ul class="tight">
  <li><code>uv</code> owns environments and the lockfile.</li>
  <li><code>pytest</code> covers unit, integration, contract, migration, and golden behavior.</li>
  <li><code>ruff</code> handles lint and formatting.</li>
  <li><code>ty</code> checks production code and maintained scripts.</li>
</ul>

|||

<h3>Automation</h3>
<ul class="tight">
  <li>Pre-commit runs fast local checks.</li>
  <li>Justfile commands are shared by developers and CI.</li>
  <li>GitHub Actions runs tests, coverage, lint, type checks, builds, and docs.</li>
  <li>Docs are built with MkDocs Material.</li>
</ul>

<p class="callout">A new change should preserve schemas, public imports, deterministic outputs, resumability, and publication order.</p>

<!-- notes: [Sources] pyproject.toml, Justfile, .pre-commit-config.yaml, .github/workflows/ci.yml, .github/workflows/docs.yml, and docs/architecture.md. -->

---

<!-- layout: title-left -->
## Start with one command and follow the contracts

```bash
export OSM_POLYGON_DATA_ROOT=/path/to/data-root
uv sync
uv run osm-polygon-wikidata-only sync-dir   "$OSM_POLYGON_DATA_ROOT/raw"   --skip-existing --push
```

<ul class="tight">
  <li>Read <code>README.md</code> for the public data model and operating rules.</li>
  <li>Read <code>docs/architecture.md</code> for dependency direction and workflow ownership.</li>
  <li>Read <code>docs/api.md</code> before changing a documented import or signature.</li>
  <li>Run the contract tests before touching schemas, paths, or publication.</li>
</ul>

<p><a href="https://github.com/NoeFlandre/osm-polygon-wikidata-only">Explore the source code</a></p>

<!-- notes: [Sources] README.md, docs/architecture.md, docs/api.md, and pyproject.toml. Public source link: https://github.com/NoeFlandre/osm-polygon-wikidata-only. -->
