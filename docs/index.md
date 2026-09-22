# OSM Polygon Wikidata Only

`osm-polygon-wikidata-only` turns polygonal OpenStreetMap features into
multi-table Parquet datasets enriched with Wikidata, Wikipedia, and Wikivoyage
information. The repository contains the extraction pipeline, its tests, and
the tools used to publish the datasets on the Hugging Face Hub.

## Dataset contracts

The project publishes two deliberately separate contracts:

- **V1** is the default. It selects closed OSM ways and multipolygon relations
  with a non-empty `wikidata=*` tag, then joins Wikidata entities to Wikipedia
  and Wikivoyage documents.
- **V2** is opt-in with `--dataset-version v2`. It keeps the V1 rows and also
  selects valid multilingual `wikipedia=*` tags. A direct Wikipedia reference
  may therefore produce a row **without a Wikidata QID**. V2 is written to and
  published as a separate dataset; it never rewrites V1 artifacts.

The current published datasets are [V1 on Hugging Face](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only)
and [V2 on Hugging Face](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia).
The [GitHub repository](https://github.com/NoeFlandre/osm-polygon-wikidata-only)
contains the source and release history.

## Quick start

Install Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/), then
install the locked environment:

```bash
git clone https://github.com/NoeFlandre/osm-polygon-wikidata-only.git
cd osm-polygon-wikidata-only
uv sync --frozen
```

Choose a data root outside the source checkout. It must contain a `raw/`
directory with Geofabrik `.osm.pbf` extracts. The resumable V1 workflow is:

```bash
export OSM_POLYGON_DATA_ROOT=/path/to/osm-polygon-data
uv run osm-polygon-wikidata-only sync-dir \
  "$OSM_POLYGON_DATA_ROOT/raw" \
  --skip-existing
```

Add `--push` when the local artifacts have been reviewed and should be uploaded
to Hugging Face. The [README](https://github.com/NoeFlandre/osm-polygon-wikidata-only#usage)
documents authentication, optional Wikimedia credentials, and the V2 command.
The same command can be interrupted and run again; completed regions are
skipped and resumable state is kept under the selected data root.

## Releasing the card and statistics report

`release-stats` publishes only the dataset card and the machine-readable
`stats.json` report. It recomputes both from every published polygon row of the
selected contract and uploads nothing else; Parquet tables, manifests, and maps
are untouched. Each released dataset needs its own exact `--confirm-repo`.

The card's text-covered counts and explanatory captions use one deterministic
global identity per `(osm_type, osm_id)` across overlapping regional extracts.
Only successfully extracted (`fetch_status=ok`) documents with trimmed,
non-empty `full_text` qualify. The `stats.json` area and geometry report remains
row-based: it scans every published polygon row and keeps the deterministic
per-source breakdown.

```bash
# Dry run for both published datasets. No network writes.
uv run osm-polygon-wikidata-only release-stats \
  --data-root "$OSM_POLYGON_DATA_ROOT" \
  --confirm-repo NoeFlandre/osm-polygon-wikidata-only \
  --confirm-repo NoeFlandre/osm-polygon-wikidata-and-wikipedia

# Publish and verify both remote revisions.
uv run osm-polygon-wikidata-only release-stats \
  --data-root "$OSM_POLYGON_DATA_ROOT" \
  --confirm-repo NoeFlandre/osm-polygon-wikidata-only \
  --confirm-repo NoeFlandre/osm-polygon-wikidata-and-wikipedia \
  --apply
```

Use `--dataset-version v1` or `--dataset-version v2` to release one contract
alone, with that dataset's confirmation only. Each run prints one JSON report
per dataset recording the target repository, the verified remote revision,
every published file with its SHA-256 and size, and the polygon file and row
counts the statistics were computed from. Staged files are rewritten only when
their bytes change, so a second run over unchanged artifacts is a no-op.

## V1 contract tables

The default V1 contract publishes these logical tables for each region:

| Table | Contents |
| --- | --- |
| `polygons/<stem>.parquet` | Polygon identity, geometry, OSM tags, and coverage counters. |
| `wikipedia/documents/<stem>.parquet` | One row per Wikipedia revision and language. |
| `polygon_articles/<stem>.parquet` | Many-to-many links to Wikipedia and Wikivoyage documents. |
| `wikipedia/sections/<stem>.parquet` | Section-level Wikipedia text when augmentation is enabled. |
| `wikivoyage/documents/<stem>.parquet` | Wikivoyage documents associated with places. |
| `wikivoyage/sections/<stem>.parquet` | Section-level Wikivoyage text. |
| `wikidata/facts/<stem>.parquet` | Structured claims for polygon entities. |
| `manifests/processed_pbfs.json` | Aggregate counts and provenance for source extracts. |

## Row-level language splits

The [language-split contract ADR](adr/0002-language-split-contract.md) defines
an additive, row-level Hugging Face surface for both published datasets. The
split key is the `language` column on each textual/document row. Polygon
`best_language` is not a split key, so a polygon's French and German rows are
retained in both relevant partitions without collapsing them by OSM identity.

| Dataset contract | Language-bearing tables | Language-neutral tables remain in the default contract |
| --- | --- | --- |
| V1 `NoeFlandre/osm-polygon-wikidata-only` | `polygon_articles`, Wikipedia documents/sections, Wikivoyage documents/sections | `polygons`, `wikidata/facts` |
| V2 `NoeFlandre/osm-polygon-wikidata-and-wikipedia` | `polygon_document_links`, Wikipedia documents/sections | `polygons` |

Each language-bearing table uses an additive `<table>_by_language`
configuration. V1 uses `lang-<language>` Viewer splits; V2 uses
`lang_<language>` with language-code dashes replaced by underscores, while its
storage directories retain `lang-<language>`. Missing, blank, malformed, and
legacy-unusable values go to the corresponding `unknown` split with reason
counts; no row is dropped. The two inventories are generated independently
from their schema-validated artifacts and manifests.

For example, load only French V1 Wikipedia documents with:

```python
from datasets import load_dataset

french_documents = load_dataset(
    "NoeFlandre/osm-polygon-wikidata-only",
    name="wikipedia_documents_by_language",
    split="lang-fr",
)
```

## V2 contract differences

V2 keeps the V1 document and sidecar tables but stores its isolated artifacts
under `processed_v2/`. Its relationship table is
`polygon_document_links/<stem>.parquet`; each row has `link_sources` provenance
for Wikidata-sitelink and/or direct-Wikipedia-tag discovery. The V2 polygon
table adds `wikipedia_tag_refs`, `wikipedia_tag_rejections`, and
`discovery_sources`. V2 document rows permit a null Wikidata QID for a direct
Wikipedia page, so V1 and V2 paths must not be mixed in one run.

The generated dataset card describes the columns and reports statistics derived
from the finalized tables. Attribution and source licenses remain part of the
published contract; see the [README](https://github.com/NoeFlandre/osm-polygon-wikidata-only#licensing-and-attribution)
for details.

V2 Viewer selections use names such as `lang_fr` (the stored path remains
`language_splits/<configuration>/lang-fr/`).

Optional V2 sentence sidecars use only `segment-any-text/sat-3l-sm` for the
exact supported language-code set. Other language codes stay as one unsplit
row with explicit `unsupported_language` provenance. See the [V2 sentence
splitting guide](sentence-splitting.md) for the full list, resumable command,
and output contract.

## Learn more

- [Architecture](architecture.md) — data flow, V1/V2 boundaries, storage, and
  publication behavior.
- [API reference](api.md) — supported Python modules and compatibility rules.
- [Development](development.md) — local checks, Docker, tests, and contribution
  workflow.
- [Contributing guide](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CONTRIBUTING.md)
  — review expectations and scope boundaries.

## Licensing

The dataset combines OpenStreetMap data under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/),
Wikidata under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/),
and Wikipedia/Wikivoyage text under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Use the generated dataset card and the repository's [software citation metadata](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CITATION.cff)
when redistributing results. Dataset-specific citation files are maintained at
[`docs/citations/osm-polygon-wikidata-only.cff`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/docs/citations/osm-polygon-wikidata-only.cff)
and [`docs/citations/osm-polygon-wikidata-and-wikipedia.cff`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/docs/citations/osm-polygon-wikidata-and-wikipedia.cff).
