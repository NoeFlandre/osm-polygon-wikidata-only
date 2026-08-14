# Supported Python API

The command line is the primary interface. The modules below are the supported
Python entry points; imports not listed here may change without notice.

## Configuration

- `osm_polygon_wikidata_only.config.paths.DataRoot` and
  `resolve_data_root` describe and validate the operator-selected data root.
- `osm_polygon_wikidata_only.config.settings.Settings` contains immutable
  runtime settings used by the processing commands.

The data root must be outside the source checkout. Callers should construct it
through `resolve_data_root` rather than duplicating path rules.

## Processing and enrichment

`osm_polygon_wikidata_only.pipeline.processor` exposes the single-PBF facade:

- `PbfStem`, `ExtractedPbf`, and `ProcessResult` are the result models.
- `extract_pbf` reads polygon candidates from one PBF.
- `process_extracted_pbf` enriches an extracted input and writes its tables.
- `process_pbf` performs both phases.
- `IncompleteEnrichmentError` signals that expected enrichment did not finish;
  callers should not treat incomplete output as a completed region.

`osm_polygon_wikidata_only.pipeline.orchestrator` provides `collect_pbfs`,
`already_processed`, and `orchestrate` for callers that need the directory
workflow used by the CLI.

For service boundaries, use the compatibility facades
`osm_polygon_wikidata_only.enrichment.wikidata_client` and
`osm_polygon_wikidata_only.enrichment.wikipedia_client`. They expose the
typed client protocols, HTTP clients, in-memory clients for tests, parsers, and
the `WikidataEntity`, `WikipediaArticle`, and `FetchResult` models. The CLI
constructs these clients with the project defaults.

## Dataset cards and publication

- `osm_polygon_wikidata_only.hf.dataset_card.render_dataset_card` renders the
  Hugging Face dataset card from supplied schema descriptions and statistics.
- `osm_polygon_wikidata_only.hf.uploader` exposes `upload_parquet`,
  `upload_manifest`, `upload_card`, and `upload_files`, plus the `HfHub` and
  `StubHfHub` in-memory stub and token/authorization helpers. These functions
  can write to the Hub; use the CLI for the normal atomic publication path.
- `osm_polygon_wikidata_only.hf.trackio_snapshot.publish_trackio_snapshot`
  publishes the single static `final-dataset-snapshot` run. The corresponding
  console script is `osm-polygon-wikidata-only-trackio`.

`osm_polygon_wikidata_only.hf.coverage_map` provides
`load_centroids_from_parquet`, `generate_coverage_map`, and
`ensure_world_land` for the all-polygon map. The
`osm_polygon_wikidata_only.hf.geographic_text_coverage` facade provides the
typed H3 aggregation and render helpers used for geographic coverage assets, including
`assign_h3_cell`, `aggregate_geographic_text_coverage`, and
`render_geographic_text_coverage`.

## Compatibility rules

The CLI, Parquet schemas, manifest names, deterministic ordering, and public
client classes are compatibility contracts. New implementation details should
be introduced behind these facades. Clients are synchronous and may perform
network or filesystem I/O; tests can use the in-memory client variants and the
Hub stub to avoid both.

The V2 dataset is selected through the CLI's `sync-dir --dataset-version v2`
option. It has a separate storage and publication contract, so Python callers
should not mix V1 and V2 output paths in one run.
