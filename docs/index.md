# OSM Polygon Wikidata Only

`osm-polygon-wikidata-only` converts polygonal OpenStreetMap features to multi-table Parquet datasets. The datasets have Wikidata, Wikipedia, and Wikivoyage information. The repository has the extraction pipeline, its tests, and the tools that publish the datasets on the Hugging Face Hub. For the project terms, see the [glossary](glossary.md).

## Dataset contracts

The project publishes two separate contracts:

- **V1** is the default. It selects the closed OSM ways and the multipolygon relations that have a `wikidata=*` tag that is not empty. Then it joins the Wikidata entities to the Wikipedia and Wikivoyage documents.
- **V2** is opt-in with `--dataset-version v2`. It keeps the V1 rows. It also selects the valid multilingual `wikipedia=*` tags. A direct Wikipedia reference can thus make a row **with no Wikidata QID**. The pipeline writes and publishes V2 as a separate dataset. It never rewrites the V1 artifacts.

The current published datasets are [V1 on Hugging Face](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only) and [V2 on Hugging Face](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia). The [GitHub repository](https://github.com/NoeFlandre/osm-polygon-wikidata-only) has the source and the release history.

## Quick start

Install Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/). Then install the locked environment:

```bash
git clone https://github.com/NoeFlandre/osm-polygon-wikidata-only.git
cd osm-polygon-wikidata-only
uv sync --frozen
```

Choose a data root outside the source checkout. It must have a `raw/` directory with Geofabrik `.osm.pbf` extracts. This is the resumable V1 workflow:

```bash
export OSM_POLYGON_DATA_ROOT=/path/to/osm-polygon-data
uv run osm-polygon-wikidata-only sync-dir \
  "$OSM_POLYGON_DATA_ROOT/raw" \
  --skip-existing
```

Review the local artifacts. Then add `--push` to upload them to Hugging Face. The [README](https://github.com/NoeFlandre/osm-polygon-wikidata-only#usage) describes the authentication, the optional Wikimedia credentials, and the V2 command. You can interrupt the command and run it again. The command skips the completed regions. It keeps the restart state under the selected data root.

## Release of the card and the statistics report

`release-stats` publishes only the dataset card and the machine-readable report `stats.json`. It recomputes both from each published polygon row of the selected contract. It uploads nothing else. The Parquet tables, manifests, and maps do not change. Each released dataset needs its own exact `--confirm-repo`.

The text-covered counts and the captions of the card use one deterministic global identity for each `(osm_type, osm_id)`. This identity is global across the regional extracts that overlap. A document qualifies only if it has these properties:

- It has a successful extraction (`fetch_status=ok`).
- Its `full_text` is not empty after trimming.

The area and geometry report in `stats.json` stays row-based. It scans each published polygon row. It keeps the deterministic breakdown for each source.

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

To release one contract only, use `--dataset-version v1` or `--dataset-version v2`. Give only the confirmation of that dataset. Each run prints one JSON report for each dataset. The report records these items:

- The target repository.
- The verified remote revision.
- Each published file with its SHA-256 and size.
- The polygon file counts and row counts that the statistics use.

The command rewrites a staged file only when its bytes change. A second run over unchanged artifacts does nothing.

## V1 contract tables

The default V1 contract publishes these logical tables for each region:

| Table | Contents |
| --- | --- |
| `polygons/<stem>.parquet` | The polygon identity, geometry, OSM tags, and coverage counters. |
| `wikipedia/documents/<stem>.parquet` | One row for each Wikipedia revision and language. |
| `polygon_articles/<stem>.parquet` | The many-to-many links to the Wikipedia and Wikivoyage documents. |
| `wikipedia/sections/<stem>.parquet` | The Wikipedia text by section, when augmentation is on. |
| `wikivoyage/documents/<stem>.parquet` | The Wikivoyage documents that belong to places. |
| `wikivoyage/sections/<stem>.parquet` | The Wikivoyage text by section. |
| `wikidata/facts/<stem>.parquet` | The structured claims for the polygon entities. |
| `manifests/processed_pbfs.json` | The aggregate counts and provenance of the source extracts. |

## Row-level language splits

The [language-split contract ADR](adr/0002-language-split-contract.md) defines an additive row-level Hugging Face surface. It applies to the two published datasets. The `language` column of each text row or document row is the split key. The polygon `best_language` is not a split key. The French rows and the German rows of a polygon thus stay in the two applicable partitions. The split does not merge them by OSM identity.

| Dataset contract | Tables that have a language | Tables with no language stay in the default contract |
| --- | --- | --- |
| V1 `NoeFlandre/osm-polygon-wikidata-only` | `polygon_articles`, Wikipedia documents and sections, Wikivoyage documents and sections | `polygons`, `wikidata/facts` |
| V2 `NoeFlandre/osm-polygon-wikidata-and-wikipedia` | `polygon_document_links`, Wikipedia documents and sections | `polygons` |

Each table that has a language uses additive language metadata. V1 keeps one `<table>_by_language` configuration with the Viewer splits `lang-<language>`. V2 uses one Viewer configuration for each pair of table and language. The name is `<table>_by_language__lang_<language>`, with underscores in place of the dashes of the language code. Each configuration has one `train` split. The storage directories keep `lang-<language>`.

The corresponding `unknown` partition keeps the values that are missing, blank, malformed, and legacy and not usable. It records the reason counts. The pipeline drops no row. The pipeline generates the two inventories independently from their artifacts and manifests. Both pass schema validation.

For example, load only the French V1 Wikipedia documents with this code:

```python
from datasets import load_dataset

french_documents = load_dataset(
    "NoeFlandre/osm-polygon-wikidata-only",
    name="wikipedia_documents_by_language",
    split="lang-fr",
)
```

## V2 contract differences

V2 keeps the V1 document tables and sidecar tables. It stores its isolated artifacts under `processed_v2/`. Its relationship table is `polygon_document_links/<stem>.parquet`. Each row has `link_sources` provenance. The provenance shows discovery by a Wikidata sitelink, by a direct Wikipedia tag, or by both. The V2 polygon table adds `wikipedia_tag_refs`, `wikipedia_tag_rejections`, and `discovery_sources`. The V2 document rows permit a null Wikidata QID for a direct Wikipedia page. Do not mix V1 paths and V2 paths in one run.

The generated dataset card describes the columns. It reports statistics from the finalized tables. The attribution and the source licenses stay part of the published contract. For details, see the [README](https://github.com/NoeFlandre/osm-polygon-wikidata-only#licensing-and-attribution).

The V2 Viewer selections have names such as `wikipedia_documents_by_language__lang_fr`, with the split `train`. The stored path stays `language_splits/<configuration>/lang-fr/`. The unknown partition is `<table>_by_language__lang_unknown`.

The optional V2 sentence sidecars use only the model `segment-any-text/sat-3l-sm`. They use it for the exact set of supported language codes. Each other language code stays as one row that is not split. The row has the explicit provenance `unsupported_language`. The [V2 sentence splitting guide](sentence-splitting.md) gives the complete list, the command that you can resume, and the output contract.

## More information

- [Architecture](architecture.md): the data flow, the V1 and V2 boundaries, the storage, and the publication behavior.
- [API reference](api.md): the supported Python modules and the compatibility rules.
- [Development](development.md): the local checks, Docker, tests, and the contribution workflow.
- [Contributing guide](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CONTRIBUTING.md): the review rules and the scope boundaries.

## Licensing

The dataset combines these sources:

- OpenStreetMap data under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/).
- Wikidata under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
- Wikipedia and Wikivoyage text under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

When you redistribute results, use the generated dataset card and the [software citation metadata](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CITATION.cff) of the repository. The dataset-specific citation files are [`docs/citations/osm-polygon-wikidata-only.cff`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/docs/citations/osm-polygon-wikidata-only.cff) and [`docs/citations/osm-polygon-wikidata-and-wikipedia.cff`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/docs/citations/osm-polygon-wikidata-and-wikipedia.cff).
