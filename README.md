# osm-polygon-wikidata-only

Extract polygonal OpenStreetMap features carrying a `wikidata=*` tag
from Geofabrik `.osm.pbf` extracts, enrich them with Wikidata and
Wikipedia (articles per sitelink language, with revisions, license,
attribution), and publish the result as a clean, multi-table Hugging
Face dataset.

* **GitHub**: <https://github.com/NoeFlandre/osm-polygon-wikidata-only>
* **Hugging Face dataset**: <https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only>

---

## What this project does

1. Reads Geofabrik `.osm.pbf` files (country / region extracts).
2. Keeps only the polygonal elements:
   * **Closed ways** carrying a non-empty `wikidata=*` tag.
   * **Multipolygon relations** carrying a non-empty `wikidata=*` tag.
3. Computes geometry metadata per polygon (centroid via
   equirectangular projection, area in m² and km², bbox, area bucket,
   primary OSM tag).
4. Looks up the polygons' Wikidata QIDs (entity, sitelinks,
   descriptions) and then fetches each linked Wikipedia article
   (lead text, full plain text, page/revision ID, license,
   attribution).
5. Publishes the result as **three parquet files per PBF** on the
   Hugging Face Hub, plus a manifest:
   * `polygons/<stem>.parquet` — one row per polygon.
   * `articles/<stem>.parquet` — one row per unique Wikipedia article.
   * `polygon_articles/<stem>.parquet` — many-to-many polygon↔article links.
   * `manifests/processed_pbfs.json` — aggregate stats per source PBF.

The repository is **code only**: every data artifact (PBFs, parquet,
HF caches, request caches) lives on an external drive.

---

## Repository layout

```
.
├── src/osm_polygon_wikidata_only/
│   ├── __init__.py
│   ├── cli/             # CLI entry point and argument parsing
│   │   ├── app.py
│   │   └── commands.py
│   ├── config/          # Paths (DataRoot) and runtime Settings
│   │   ├── paths.py
│   │   └── settings.py
│   ├── domain/          # Pure domain types and helpers
│   │   ├── analysis.py  # area_bucket, osm_primary_tag, bbox
│   │   ├── geometry.py  # PolygonGeometry, area, centroid
│   │   ├── ids.py       # polygon_id, article_id, content_hash
│   │   ├── models.py    # Polygon, Article, PolygonArticleLink, ManifestStats
│   │   └── schema.py    # Column lists, descriptions, pyarrow schemas
│   ├── enrichment/      # Wikidata + Wikipedia clients + linker
│   │   ├── article_linker.py
│   │   ├── text_cleaning.py
│   │   ├── wikidata_client.py
│   │   └── wikipedia_client.py
│   ├── hf/              # Hugging Face Hub integration
│   │   ├── dataset_card.py
│   │   ├── repo_layout.py
│   │   └── uploader.py
│   ├── io/              # PBF reader, parquet, manifest, file cache
│   │   ├── cache.py
│   │   ├── manifest.py
│   │   ├── parquet.py
│   │   └── pbf_reader.py
│   ├── pipeline/        # Extract → enrich → write → manifest
│   │   ├── extractor.py
│   │   ├── orchestrator.py
│   │   ├── processor.py
│   │   └── stats.py
│   └── utils/           # JSON, time, logging, retry helpers
│       ├── json.py
│       ├── logging.py
│       ├── retry.py
│       └── time.py
├── tests/               # pytest suite (114+ unit + 1 end-to-end smoke)
├── pyproject.toml       # Build, dev deps, ruff/mypy/pytest config
└── README.md
```

Each top-level sub-package has its own `__init__.py` and a tightly
focused public API. Cross-package imports go through dotted paths.

---

## Installation

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NoeFlandre/osm-polygon-wikidata-only.git
cd osm-polygon-wikidata-only
uv sync
```

This installs all runtime and development dependencies into a managed
`.venv`:

| Runtime | Purpose |
|---|---|
| `osmium` | Streaming OSM PBF parser |
| `datasets` | Hugging Face dataset utilities |
| `huggingface-hub` | HF Hub client |
| `pyarrow` | Parquet serialization |

| Dev | Purpose |
|---|---|
| `pytest`, `pytest-cov` | Tests |
| `ruff` | Lint + format |
| `mypy` | Type-check |

---

## External data root

All PBF inputs, intermediate outputs, Hugging Face caches, and the
local parquet/manifest files live on an external drive under a single
**data root**. The recommended local path is
`/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/`.

Resolution order:

1. `--data-root <path>` CLI flag.
2. `OSM_POLYGON_DATA_ROOT` environment variable.
3. The recommended local path above, when it exists.

A data root that does not exist is rejected (no silent fallback).

Default sub-directories under the data root:

| Sub-directory | Purpose |
|---|---|
| `raw/` | Geofabrik `.osm.pbf` files (input) |
| `processed/polygons/` | Written `polygons/<stem>.parquet` files |
| `processed/articles/` | Written `articles/<stem>.parquet` files |
| `processed/polygon_articles/` | Written `polygon_articles/<stem>.parquet` files |
| `processed/manifests/` | `processed_pbfs.json` aggregate manifest |
| `logs/` | Reserved for pipeline logs |
| `hf_cache/` | Hugging Face client-side cache |
| `cache/wikidata/`, `cache/wikipedia/` | Per-call JSON cache |
| `cache/` | Shared cache root |

Set the data root for a session:

```bash
export OSM_POLYGON_DATA_ROOT=/Volumes/Seagate\ M3/projects/osm-polygon-wikidata-only
```

---

## Usage

After `uv sync`, two entry points are available:

```bash
uv run osm-polygon-wikidata-only process-pbf <input.pbf> [--options]
uv run osm-polygon-wikidata-only process-dir  <dir>     [--options]
```

### Common options

| Flag | Purpose |
|---|---|
| `--data-root <path>` | Override the resolved external data root |
| `--repo-id <org/name>` | Target Hugging Face repo (default `NoeFlandre/osm-polygon-wikidata-only`) |
| `--user-agent <ua>` | Override Wikimedia User-Agent (default identifies this project) |
| `--languages en,fr,...` | Comma-separated ISO 639-1 codes for sitelinks |
| `--all-languages` | Fetch every available sitelink (default: 5 languages) |
| `--no-full-text` | Fetch only the lead section, not the full article |
| `--max-articles-per-qid <n>` | Cap articles per Wikidata QID (default `5`) |
| `--limit <n>` | Process only the first N polygons per PBF |
| `--skip-existing` | Skip PBFs already listed in the manifest |
| `--force` | Re-process even when `--skip-existing` applies |
| `--push` | Upload produced artifacts to the Hub |
| `--upload-threads <n>` | Concurrent transfer workers in the atomic Hub commit (default `5`) |
| `--commit-message <msg>` | Custom git commit message for the push |
| `--dry-run` | Use a stub HF client (records calls without uploading) |
| `--log-level <level>` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Examples

Process one PBF and write 3 parquet files + manifest locally:

```bash
uv run osm-polygon-wikidata-only process-pbf ~/pbfs/monaco-latest.osm.pbf
```

Push the result to the Hub with a stub client (no network):

```bash
uv run osm-polygon-wikidata-only process-pbf monaco-latest.osm.pbf --push --dry-run
```

Process every PBF under `<data-root>/raw/`, fetch only English and
French Wikipedia, skip already-processed:

```bash
uv run osm-polygon-wikidata-only process-dir \
    ~/pbfs/ \
    --languages en,fr \
    --skip-existing
```

Programmatic usage:

```python
from pathlib import Path
from osm_polygon_wikidata_only.config.paths import DataRoot, resolve_data_root
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikipedia_client import HttpWikipediaClient
from osm_polygon_wikidata_only.enrichment.wikidata_client import HttpWikidataClient
from osm_polygon_wikidata_only.pipeline.processor import process_pbf

data_root = resolve_data_root(repo_root=Path.cwd())
data_root.ensure()

settings = Settings(languages=("en", "fr"))
wd = HttpWikidataClient(settings)
wiki = HttpWikipediaClient(settings)

result = process_pbf(
    Path("monaco-latest.osm.pbf"),
    data_root=data_root,
    wikidata_client=wd,
    wikipedia_client=wiki,
    settings=settings,
)
print(result.polygon_count, "polygons")
```

## Reliability and performance behavior

The pipeline is designed to preserve dataset completeness while keeping
Wikimedia traffic polite:

* Candidate order, selected sitelinks, and Parquet row ordering are
  deterministic.
* Identical Wikidata QIDs and Wikipedia titles are fetched once per run and
  reused for every matching polygon.
* HTTP clients use the on-disk cache by default. Failed requests are cached
  briefly to avoid repeatedly hammering a failing endpoint.
* Concrete HTTP clients batch compatible Wikidata and same-language Wikipedia
  requests. The pipeline falls back to the established per-item request path
  if a batch response is incomplete or invalid.
* Per-host pacing, retries with jitter, and a shared `429` cooldown remain in
  force when batch jobs run concurrently.
* `--push` publishes every produced Parquet artifact and the final manifest in
  one atomic Hugging Face commit. Transfers use concurrent workers; increase
  `--upload-threads` only when local bandwidth and Hub quotas allow it.

For a repeatable production run, use `--skip-existing`; it consults the
manifest and leaves previously completed PBFs untouched. Use `--force` only
when you intentionally want to rebuild a completed PBF.

## Development quality checks

Run the complete local gate before contributing:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

The test suite uses in-memory clients and stub PBF readers for unit coverage.
It does not require a real PBF, external data root, or Wikimedia request.

---

## Output schema

Each PBF produces three parquet files. The schema lives in
`osm_polygon_wikidata_only.domain.schema` so the dataset card, the
parquet writer, and the tests share a single source of truth.

### `polygons/<stem>.parquet`

One row per polygon. Includes geometry metadata, OSM tags, primary
OSM tag, area bucket, and Wikipedia coverage counters.

### `articles/<stem>.parquet`

One row per unique Wikipedia article
(`(wikidata, language, page_id, revision_id)`). Includes lead text,
plain-text full text, thumbnails, license, attribution, and a
deterministic SHA-256 `content_hash`.

### `polygon_articles/<stem>.parquet`

Many-to-many links joining polygons to articles, plus a boolean
`is_best_language` flag (true for the language chosen by
`LinkSummary.best_language()`).

### `manifests/processed_pbfs.json`

Aggregate stats per source PBF: polygon/article counts, language
coverage, area-bucket counts, top tag keys.

---

## Wikimedia etiquette

Wikimedia APIs require a User-Agent identifying the project and a
contact. The defaults are in `config.settings.DEFAULT_USER_AGENT`.
Set `--user-agent` in production deployments.

The HTTP clients honor:

* configurable `request_timeout_s`, `request_max_retries`,
  `request_base_delay_s`,
* exponential backoff with jitter (`utils.retry.with_retries`),
* a disk-backed `JsonFileCache` (`io.cache.JsonFileCache`) that lets
  repeated runs avoid hammering the same endpoint,
* localized language lists (`--languages`) so we never fetch
  unwanted sitelinks.

---

## Development

### Run the tests

```bash
uv run pytest
```

The suite is fast (< 2 s) because nothing actually hits the network;
HTTP clients come in three flavors (`Http…`, `InMemory…`,
`Cached…`) and the tests use the in-memory flavors.

### Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

### Type-check

```bash
uv run mypy src
```

---

## Repository / data separation policy

The repository is **code-only**. Everything user-generated (datasets,
HF caches, Arrow/Parquet files, downloaded PBFs) is git-ignored and
must live on the configured external data root. This keeps the repo
tiny, makes data updates cheap, and prevents accidental commits of
multi-GB artifacts.

---

## Licensing and attribution

* **OpenStreetMap polygons**: (c) OpenStreetMap contributors, licensed
  under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/).
* **Wikidata** entity data: (c) Wikimedia contributors under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
* **Wikipedia** article text: (c) respective Wikipedia authors,
  licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Attribution and license are stored inline in the
  `articles.parquet` `license` and `attribution` columns.

Any derivative dataset must preserve OSM attribution as described on
<https://www.openstreetmap.org/copyright>.
