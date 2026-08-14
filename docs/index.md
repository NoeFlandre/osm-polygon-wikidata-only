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
