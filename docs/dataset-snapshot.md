# Dataset snapshots

These figures record the published snapshots. Each dataset card describes its snapshot. For the current files, schemas, and release metadata, read the [V1 Hugging Face dataset card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only) and the [V2 Hugging Face dataset card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia).

## Published dataset versions

| Version | Dataset and code | Scope | Snapshot |
| --- | --- | --- | --- |
| **V1** | [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only) · [GitHub `v1.0.0`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/tree/v1.0.0) | Polygons that start from Wikidata | 1,184,110 polygons · 2,288,170 Wikipedia + Wikivoyage documents · 351 languages · 375 regions |
| **V2** | [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) · [GitHub `v2.0.0`](https://github.com/NoeFlandre/osm-polygon-wikidata-only/releases/tag/v2.0.0) | V1 and the valid multilingual `wikipedia=*` references, with the polygons that have no Wikidata | 1,259,424 polygons · 2,332,127 Wikipedia + Wikivoyage documents · 353 languages · 386 regions |

## V1 Trackio snapshot

The single public [Trackio run `final-dataset-snapshot`](https://huggingface.co/spaces/NoeFlandre/osm-polygon-wikidata-only-trackio) records the finished V1 dataset. It has three static plots:

- A text-coverage funnel.
- The ten largest Wikipedia languages and `Other languages`.
- The dataset composition on a logarithmic scale.

The run is a snapshot. It is not a timeline of the pipeline.

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

The language thresholds of the funnel use the canonical Wikipedia polygon fields. The totals for documents, words, and dataset composition include Wikivoyage.

For the published tables and the geographic coverage, see the [dataset presentation](https://noeflandre.github.io/osm-polygon-wikidata-only/presentations/dataset.html).
