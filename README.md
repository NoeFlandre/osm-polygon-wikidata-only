# OSM Polygon Wikidata Only

Extract polygonal OpenStreetMap features from Geofabrik extracts, enrich them
with Wikidata, Wikipedia, and Wikivoyage, and write regional Parquet datasets.

The source repository, Python package, and CLI retain the historical name
`osm-polygon-wikidata-only`. V1 is the default; V2 is opt-in and also includes
valid multilingual `wikipedia=*` tags, including features without a Wikidata
QID. V2 is published separately as `osm-polygon-wikidata-and-wikipedia`.

[V1 dataset card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only) ·
[V2 dataset card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) ·
[Documentation site](https://noeflandre.github.io/osm-polygon-wikidata-only/)

## Quick start

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/), then install the
locked environment:

```bash
git clone https://github.com/NoeFlandre/osm-polygon-wikidata-only.git
cd osm-polygon-wikidata-only
uv sync --frozen
```

Download the small [Geofabrik Monaco extract](https://download.geofabrik.de/europe/monaco.html)
and process at most 100 polygons. Keep the data root outside the source
checkout. This writes locally and does not upload to Hugging Face:

```bash
DATA_ROOT=../osm-polygon-data
mkdir -p "$DATA_ROOT/raw"
curl --fail --location \
  --output "$DATA_ROOT/raw/monaco-latest.osm.pbf" \
  "https://download.geofabrik.de/europe/monaco-latest.osm.pbf"
uv run osm-polygon-wikidata-only process-pbf \
  "$DATA_ROOT/raw/monaco-latest.osm.pbf" \
  --data-root "$DATA_ROOT" --limit 100 --no-full-text
```

The download and Wikidata/Wikipedia enrichment use network access. Uploads are
opt-in with `--push`.

## Outputs and schema

V1 files are written below `<data-root>/processed/`. Each region has:

| Path | Contents |
| --- | --- |
| `polygons/<stem>.parquet` | Polygon identity, OSM tags, geometry, area, and coverage fields. |
| `wikipedia/documents/<stem>.parquet` | Wikipedia documents and revision metadata. |
| `polygon_articles/<stem>.parquet` | Links from polygons to Wikipedia and Wikivoyage documents. |
| `manifests/processed_pbfs.json` | Source PBF inventory and aggregate counts. |

Augmentation adds Wikipedia and Wikivoyage section tables and Wikidata facts.
V2 uses the separate `<data-root>/processed_v2/` tree and
`polygon_document_links/<stem>.parquet` for links with source provenance.
Detailed schemas and V1/V2 contracts are in the
[architecture guide](docs/architecture.md) and each dataset card.

![OSM Polygon Wikidata dataset overview](assets/dataset_hero.png)

## Configuration and Docker

Set `OSM_POLYGON_DATA_ROOT` or pass `--data-root`; the directory must exist
and stay outside the source checkout. `HF_TOKEN` is needed only for an
explicit Hub upload. Keep credentials in an untracked `.env`, never in source
control.

The Compose runtime builds the non-root `runtime` target, mounts the data root
at `/data`, and defaults to `--help`:

```bash
cp .env.example .env
mkdir -p ../osm-polygon-data/raw
docker compose run --rm runtime
docker compose run --rm runtime sync-dir /data/raw --data-root /data --skip-existing
```

The second command writes locally. Add `--push` only when you intend to
publish. See the [Docker guide](docs/development.md#docker-compose-runtime)
for host UID/GID mapping and bind-mount permissions.

## Documentation

[Architecture](docs/architecture.md) · [Python API](docs/api.md) ·
[Development and Docker](docs/development.md) ·
[Dataset snapshots](docs/dataset-snapshot.md) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## Citation and licenses

Cite the software with [`CITATION.cff`](CITATION.cff); cite a dataset with its
versioned metadata file: [V1 citation](docs/citations/osm-polygon-wikidata-only.cff)
or [V2 citation](docs/citations/osm-polygon-wikidata-and-wikipedia.cff).

The software is licensed under [Apache-2.0](LICENSE). Dataset content has
source-specific terms: [OpenStreetMap data](https://www.openstreetmap.org/copyright)
is under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), and
[Wikidata structured data](https://www.wikidata.org/wiki/Wikidata:Licensing) is
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Wikipedia text
is generally available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
or GFDL ([Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Terms_of_Use));
Wikivoyage written contributions use CC BY-SA 4.0, a compatible license, or
the public domain ([Wikivoyage copyleft policy](https://en.wikivoyage.org/wiki/Wikivoyage:Copyleft)).
Published document rows retain source-specific license and attribution fields;
check these for article-level terms.
