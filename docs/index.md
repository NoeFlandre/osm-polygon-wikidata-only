# OSM Polygon Wikidata Only

An open, reproducible pipeline that extracts polygonal OpenStreetMap features
with `wikidata=*` tags, enriches them with Wikidata, Wikipedia, and Wikivoyage,
and publishes a multi-table Parquet dataset on the Hugging Face Hub.

## Explore the dataset

The current generated dataset card, files, schemas, and statistics are
available on [Hugging Face](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only).
The [source repository](https://github.com/NoeFlandre/osm-polygon-wikidata-only)
contains the pipeline, tests, and documentation.

The published dataset includes:

- polygon geometry and OSM metadata;
- unified polygon-to-document links identifying the `wikipedia` or
  `wikivoyage` source corpus;
- Wikipedia and Wikivoyage documents and their section-level text;
- structured Wikidata facts for the polygon entities;
- manifests, geographic visualizations, and generated dataset-card statistics.

## Run the pipeline

Install Python 3.12+, [uv](https://docs.astral.sh/uv/), and the project:

```bash
git clone https://github.com/NoeFlandre/osm-polygon-wikidata-only.git
cd osm-polygon-wikidata-only
uv sync
```

Choose a data root outside the source checkout, then run the resumable unified
workflow:

```bash
export OSM_POLYGON_DATA_ROOT=/path/to/osm-polygon-data
uv run osm-polygon-wikidata-only sync-dir \
  "$OSM_POLYGON_DATA_ROOT/raw" \
  --skip-existing \
  --push
```

The workflow keeps local artifacts and request caches under the configured data
root, publishes completed regions atomically, and can be interrupted with
`Ctrl-C` and resumed with the same command. Wikimedia credentials are optional;
the [README](https://github.com/NoeFlandre/osm-polygon-wikidata-only#wikimedia-bot-password-authentication)
contains the full Bot Password setup and operational guidance.

## Learn more

- [API reference](api.md) — supported Python entry points and compatibility
  boundaries.
- [Architecture](architecture.md) — data flow, storage contracts, scheduling,
  recovery, and publication behavior.
- [Development](development.md) — reproducible setup, testing, and quality
  checks.

## Licensing

The dataset combines OpenStreetMap data under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/),
Wikidata under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/),
and Wikipedia/Wikivoyage text under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Refer to the generated dataset card for per-table attribution details.
