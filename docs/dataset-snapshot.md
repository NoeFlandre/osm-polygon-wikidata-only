# Dataset snapshots

These figures record the published snapshots described by each dataset card.
For the current files, schemas, and release metadata, see the
[V1 Hugging Face dataset card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only)
and [V2 Hugging Face dataset card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia).

## Published dataset versions

| Version | Dataset and code | Scope | Snapshot |
| --- | --- | --- | --- |
| **V1** | [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only) · [GitHub `v1.0.0`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/tree/v1.0.0) | Wikidata-first polygons | 1,184,110 polygons · 2,288,170 Wikipedia + Wikivoyage documents · 351 languages · 375 regions |
| **V2** | [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) · [GitHub `v2.0.0`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/releases/tag/v2.0.0) | V1 plus valid multilingual `wikipedia=*` references, including polygons without Wikidata | 1,259,424 polygons · 2,332,127 Wikipedia + Wikivoyage documents · 353 languages · 386 regions |

## V1 Trackio snapshot

The finished V1 dataset is recorded in the single public
[Trackio run `final-dataset-snapshot`](https://huggingface.co/spaces/NoeFlandre/osm-polygon-wikidata-only-trackio).
It contains three static plots: a text-coverage funnel, the ten largest
Wikipedia languages plus `Other languages`, and dataset composition on a
logarithmic scale. It is a snapshot, not a pipeline timeline.

| Snapshot metric | Value |
| --- | ---: |
| Polygons | 1,184,110 |
| Wikipedia + Wikivoyage documents | 2,288,170 |
| Document words | 801,528,334 |
| Languages | 351 |
| Geographic regions | 375 |

| Small snapshot table | Value |
| --- | ---: |
| Wikipedia polygon-document links | 2,468,604 |
| Polygon/link-table storage | 9.9 GB |
| Total Parquet storage | 19.2 GB |

The funnel's language thresholds use the canonical Wikipedia polygon fields.
Wikivoyage is included in the combined document, word, and dataset-composition
totals.

See the [dataset presentation](https://noeflandre.github.io/osm-polygon-wikidata-only/presentations/dataset.html)
for the published tables and geographic coverage.
