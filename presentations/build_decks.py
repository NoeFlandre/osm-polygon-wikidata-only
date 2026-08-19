"""Generate the local Colloquium dataset and codebase decks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .snapshot import DatasetSnapshot, read_snapshot

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

COMMON_CSS = r"""
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
.slide .table-lg { font-size: 1em; margin-top: 0.4em; }
.slide .table-lg th, .slide .table-lg td { padding: 0.5em 0.75em; }
.slide .table-lg + .small { font-size: 0.85em; margin-top: 0.7em; }
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
"""


def _frontmatter(title: str, author: str, date: str) -> str:
    return f'''---
title: "{title}"
author: "{author}"
date: "{date}"
theme: default
aspect_ratio: "16:9"
footer:
  left: "OSM Polygon Wikidata"
  right: "{{n}} / {{N}}"
custom_css: |
{_indent_css(COMMON_CSS)}
---
'''


def _indent_css(css: str) -> str:
    return "\n".join(f"  {line}" if line else "" for line in css.strip("\n").splitlines())


def _notes(source: str) -> str:
    return f"<!-- notes: [Sources] {source} -->"


def _dataset_deck(s: DatasetSnapshot) -> str:
    continent_rows = "\n".join(
        f"| {name} | {polygons} | {wiki} | {combined} | {coverage} |"
        for name, polygons, wiki, combined, coverage in s.continents
    )
    language_rows = "\n".join(
        f"| `{code}` | {docs} | {share} |" for code, docs, share in s.top_language_rows
    )
    return (
        _frontmatter("The dataset in one view", "Noé Flandre", s.generated_on)
        + f"""
# Wordlwide OSM polygons with linked knowledge

<p class="kicker">Dataset overview</p>

<div class="rule"></div>

<p>Wikidata connects real places in OpenStreetMap to multilingual Wikipedia and Wikivoyage text.</p>
<p>Snapshot: {s.generated_on}</p>

{_notes("Generated from the local processed Parquet snapshot, processed_pbfs.json, and the project README. Colloquium workflow: https://pypi.org/project/colloquium/0.2.2/")}

---

<!-- layout: title-left -->
## Every row starts with a real OSM polygon

<p class="lead">The pipeline reads Geofabrik extracts and keeps closed ways and multipolygon relations with a non-empty <code>wikidata=*</code> tag.</p>

- <strong>{s.polygons}</strong> polygon rows across <strong>{s.active_regions}</strong> active regions
- each polygon keeps its OSM identity, geometry, tags, centroid, and area
- the raw input set has {s.raw_inputs} PBFs; {s.retired_inputs} contained extracts are retired from the active set

<p class="callout">This is a place-first dataset. Text and facts are linked to the polygon through its Wikidata Q-id.</p>

{_notes("Project README, domain schemas, and processed_pbfs.json. Selection is implemented in src/osm_polygon_wikidata_only/domain/filters.py and io/pbf_reader.py.")}

---

<!-- layout: content -->
## Seven Parquet tables form one linked dataset

<table class="table-lg">
<thead>
<tr><th>Table</th><th>What it contains</th></tr>
</thead>
<tbody>
<tr><td><code>polygons</code></td><td>One row per selected OSM polygon</td></tr>
<tr><td><code>polygon_articles</code></td><td>One link row per polygon and Wikipedia or Wikivoyage document</td></tr>
<tr><td><code>wikipedia/documents</code></td><td>Full Wikipedia document rows</td></tr>
<tr><td><code>wikipedia/sections</code></td><td>Section-level Wikipedia text</td></tr>
<tr><td><code>wikivoyage/documents</code></td><td>Full Wikivoyage document rows</td></tr>
<tr><td><code>wikivoyage/sections</code></td><td>Section-level Wikivoyage text</td></tr>
<tr><td><code>wikidata/facts</code></td><td>Selected structured Wikidata claims</td></tr>
</tbody>
</table>

<p class="small">Every table is partitioned by the Geofabrik stem. The link table has a <code>project</code> column so one join works for both text sources.</p>

{_notes("Generated dataset card schema section and src/osm_polygon_wikidata_only/hf/repo_layout.py. The link contract is defined in domain/polygon_document_links.py.")}

---

<!-- layout: title-left -->
## Wikidata is the bridge between geometry and text

<p class="flow"><strong>OSM polygon</strong> &nbsp; → &nbsp; <strong>Wikidata Q-id</strong> &nbsp; → &nbsp; <strong>Wikipedia documents</strong></p>
<p class="flow"><strong>OSM polygon</strong> &nbsp; → &nbsp; <strong>Wikidata Q-id</strong> &nbsp; → &nbsp; <strong>Wikivoyage documents</strong></p>
<p class="flow"><strong>Wikidata Q-id</strong> &nbsp; → &nbsp; <strong>facts table</strong></p>

<p class="lead">A document can be linked to several polygons. A polygon can have documents in many languages and in both projects.</p>

<ul class="tight">
  <li><code>document_id</code> identifies a page revision.</li>
  <li><code>project</code> distinguishes Wikipedia from Wikivoyage.</li>
  <li>Sections reference their parent document, so full text and section text can be used together.</li>
</ul>

{_notes("Project README schema sections and the canonical link schema in domain/polygon_document_links.py.")}

---

<!-- layout: content -->
## The snapshot has broad coverage, with a clear text boundary

<!-- columns: 2 -->
<p class="callout"><strong>Core scale</strong><br>{s.unique_wikidata} unique Wikidata entities<br>{s.wikipedia_documents} Wikipedia documents<br>{s.wikivoyage_documents} Wikivoyage documents</p>

|||

<p class="callout"><strong>Derived text and facts</strong><br>{s.wikipedia_sections} Wikipedia sections<br>{s.wikivoyage_sections} Wikivoyage sections<br>{s.wikidata_facts} Wikidata facts</p>

<p class="lead">{s.text_polygons} polygons have at least one non-empty Wikipedia or Wikivoyage document, or {s.text_polygon_rate} of the polygon set.</p>
<p class="small">Document words count full document rows. Section rows are excluded because they repeat document text.</p>

{_notes("Generated dataset card snapshot table and coverage map inputs. Counts are read from the current local card generated by the pipeline.")}

---

<!-- layout: content -->
## Text coverage is strongest in Europe, but not uniform

| Continent | Polygons | Wikipedia text | Wikipedia + Voyage | Coverage |
| --- | ---: | ---: | ---: | ---: |
{continent_rows}

<p class="small">A polygon is assigned by its WGS84 centroid and Natural Earth country boundaries. Offshore or unmatched centroids stay <code>Unassigned</code>. Coverage is text-covered polygons divided by polygons in that row.</p>

{_notes("Generated geographic distribution table from the dataset card snapshot. Method and caveats are documented in src/osm_polygon_wikidata_only/hf/continent_stats.py and the README.")}

---

<!-- layout: content -->
## Languages show a long tail

<!-- columns: 2 -->
<h3>Top document languages</h3>

| Language | Documents | Share |
| --- | ---: | ---: |
{language_rows}

|||

<h3>What the distribution means</h3>
<ul class="tight">
  <li><strong>{s.languages}</strong> languages occur across both projects.</li>
  <li>The top five languages account for {s.top_five_language_share} of all documents.</li>
  <li>The top twenty account for {s.top_twenty_language_share}.</li>
  <li>Language counts describe document rows.</li>
</ul>

{_notes("Generated language table and concentration values from the dataset card snapshot. Language definitions are in hf/dataset_stats.py and hf/continent_stats.py.")}

---

<!-- layout: content -->
## Two maps answer two different geographic questions

<!-- columns: 2 -->
![All dataset polygons](assets/coverage_map.png)

<p><strong>All polygons.</strong> Each point is an OSM polygon carrying a Wikidata tag.</p>

|||

![Polygons with text](assets/text_presence.png)

<p><strong>Polygons with text.</strong> Each point has non-empty Wikipedia or Wikivoyage text.</p>

{_notes("Maps are generated by hf/coverage_map.py, hf/geographic_text_presence.py, and hf/geographic_text_density.py. Natural Earth is used for land context.")}

---

<!-- layout: content -->
## H3 density shows where text-covered places cluster

![Text density](assets/text_density.png)

<p class="map-caption">Each H3 cell contains the number of unique polygons with non-empty Wikipedia or Wikivoyage text. A polygon is counted once even if it has several documents.</p>

{_notes("Generated by hf/geographic_text_density.py from local Parquet tables. H3 resolution and low-sample rules are defined in hf/_geographic/h3_geometry.py.")}

---

<!-- layout: title-left -->
## A reproducible snapshot that is ready to use

<ul>
  <li>Parquet files are partitioned by region and can be loaded with Hugging Face Datasets or PyArrow.</li>
  <li>Text is plain text with page and revision identifiers, license, and attribution fields.</li>
  <li>Source licenses are explicit: ODbL for OpenStreetMap, CC0 for Wikidata, and CC BY-SA 4.0 for Wikipedia and Wikivoyage.</li>
</ul>

```python
from datasets import load_dataset
ds = load_dataset(
    "parquet",
    data_files={{"polygons": "hf://datasets/NoeFlandre/osm-polygon-wikidata-only/polygons/*.parquet"}},
)
```

<p><a href="https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only">Open the dataset on Hugging Face</a></p>

{_notes("Project README data-loading and license sections. Hugging Face dataset URL: https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only.")}

---

<!-- layout: title-left -->
## The dataset grows through a resumable pipeline

<p class="lead">Each run reads new or changed PBFs, enriches only the regions that need work, writes atomic Parquet artifacts, refreshes the generated card and maps, then publishes one coherent Hub update.</p>

<div class="callout"><p>Use the codebase deck to see how the pipeline keeps that snapshot deterministic and recoverable.</p></div>

<p><a href="https://github.com/NoeFlandre/osm-polygon-wikidata-only">Source code on GitHub</a></p>

{_notes("Project README, docs/architecture.md, and pipeline publication modules. Public links: https://github.com/NoeFlandre/osm-polygon-wikidata-only and https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only.")}
"""
    )


def _codebase_deck(s: DatasetSnapshot) -> str:
    return (
        _frontmatter("The codebase in one view", "Noé Flandre", s.generated_on)
        + f"""
# A modular, resumable enrichment pipeline

<p class="kicker">Codebase overview</p>

<div class="rule"></div>

<p>The project turns Geofabrik PBF files into linked OSM, Wikidata, Wikipedia, and Wikivoyage tables.</p>
<p>It is built for long runs, public publication, and safe resume.</p>

{_notes("Project README, docs/architecture.md, pyproject.toml, and the current source tree. Colloquium workflow: https://pypi.org/project/colloquium/0.2.2/")}

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

{_notes("docs/architecture.md and the source package layout under src/osm_polygon_wikidata_only.")}

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

{_notes("pipeline/extractor.py, pipeline/enrichment_phase.py, pipeline/row_construction.py, pipeline/persistence.py, and pipeline/processor.py.")}

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

{_notes("enrichment/wikimedia/transport.py, enrichment/wikimedia_auth.py, utils/request_scheduler.py, utils/retry.py, and io/cache.py.")}

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

{_notes("augmentation/orchestrator.py, augmentation/checkpoints.py, augmentation/steps.py, and augmentation/integrity.py.")}

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

{_notes("pipeline/_wikidata_recovery, pipeline/link_migration.py, pipeline/containment_migration.py, io/atomic.py, and hf/upload_queue.py.")}

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

{_notes("hf/remote_inventory.py, hf/reconciliation.py, hf/publication.py, hf/repo_layout.py, hf/upload_queue.py, and cli/run_sync.py.")}

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

{_notes("pyproject.toml, Justfile, .pre-commit-config.yaml, .github/workflows/ci.yml, .github/workflows/docs.yml, and docs/architecture.md.")}

---

<!-- layout: title-left -->
## Start with one command and follow the contracts

```bash
export OSM_POLYGON_DATA_ROOT=/path/to/data-root
uv sync
uv run osm-polygon-wikidata-only sync-dir \
  "$OSM_POLYGON_DATA_ROOT/raw" \
  --skip-existing --push
```

<ul class="tight">
  <li>Read <code>README.md</code> for the public data model and operating rules.</li>
  <li>Read <code>docs/architecture.md</code> for dependency direction and workflow ownership.</li>
  <li>Read <code>docs/api.md</code> before changing a documented import or signature.</li>
  <li>Run the contract tests before touching schemas, paths, or publication.</li>
</ul>

<p><a href="https://github.com/NoeFlandre/osm-polygon-wikidata-only">Explore the source code</a></p>

{_notes("README.md, docs/architecture.md, docs/api.md, and pyproject.toml. Public source link: https://github.com/NoeFlandre/osm-polygon-wikidata-only.")}
"""
    )


def _write_deck(name: str, content: str) -> None:
    (ROOT / f"{name}.md").write_text(content, encoding="utf-8")


def main() -> None:
    snapshot = read_snapshot()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "snapshot.json").write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_deck("dataset", _dataset_deck(snapshot))
    _write_deck("codebase", _codebase_deck(snapshot))
    for name in ("coverage_map.png", "text_presence.png", "text_density.png"):
        destination = ROOT / "assets" / name
        if not destination.is_file():
            raise RuntimeError(f"Missing presentation asset: {destination}")
        html_asset = OUTPUT / "assets" / name
        html_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, html_asset)
    print(f"Generated decks for the {snapshot.generated_on} dataset snapshot")


if __name__ == "__main__":
    main()
