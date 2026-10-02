# Supported Python API

The command line is the primary interface. The modules below are the supported Python entry points. Imports that this page does not list can change without notice.

## Configuration

- `osm_polygon_wikidata_only.config.paths.DataRoot` and `resolve_data_root` describe and validate the data root that the operator selects.
- `osm_polygon_wikidata_only.config.settings.Settings` holds the immutable runtime settings. The processing commands use it.

The data root must be outside the source checkout. Callers must construct it with `resolve_data_root`. They must not copy the path rules.

## Processing and enrichment

`osm_polygon_wikidata_only.pipeline.processor` exposes the facade for a single PBF:

- `PbfStem`, `ExtractedPbf`, and `ProcessResult` are the result models.
- `extract_pbf` reads the polygon candidates from one PBF.
- `process_extracted_pbf` enriches an extracted input and writes its tables.
- `process_pbf` does both phases.
- `IncompleteEnrichmentError` signals that the expected enrichment did not finish. Callers must not treat an incomplete output as a completed region.

`osm_polygon_wikidata_only.pipeline.orchestrator` provides `collect_pbfs`, `already_processed`, and `orchestrate`. Use them when you need the directory workflow that the CLI uses.

For service boundaries, use the compatibility facades `osm_polygon_wikidata_only.enrichment.wikidata_client` and `osm_polygon_wikidata_only.enrichment.wikipedia_client`. They expose these items:

- The typed client protocols.
- The HTTP clients.
- The in-memory clients for tests.
- The parsers.
- The models `WikidataEntity`, `WikipediaArticle`, and `FetchResult`.

The CLI constructs these clients with the project defaults.

## Dataset cards and publication

- `osm_polygon_wikidata_only.hf.minimal_card.render_minimal_card` renders the two public dataset cards from a `MinimalCardSnapshot`. A card has one section skeleton, one snapshot table of eight rows, the coverage maps, and pointers to `stats.json`. The function refuses a card body of 8 KiB or more.
- `osm_polygon_wikidata_only.hf.dataset_card.render_front_matter` renders the YAML front matter of the V1 card. The front matter declares the Dataset Viewer configurations. `osm_polygon_wikidata_only.v2.card_front_matter.render_front_matter` is the V2 equivalent.
- `osm_polygon_wikidata_only.hf.uploader` exposes `upload_parquet`, `upload_manifest`, `upload_card`, and `upload_files`. It also exposes the in-memory stubs `HfHub` and `StubHfHub`, and the token and authorization helpers. These functions can write to the Hub. For the normal atomic publication path, use the CLI.
- `osm_polygon_wikidata_only.hf.stats_release` exposes `release_v1_polygon_stats` and `release_v2_polygon_stats`. They are the deterministic compute, publish, and verify path behind the `release-stats` command. Each function does these actions:
    1. It recomputes `stats.json` and the dataset card from each published polygon row.
    2. It publishes only those two files to the confirmed dataset.
    3. It verifies the resulting remote revision.

    The text-covered metrics of the card remove duplicates globally by `(osm_type, osm_id)`. They need successful text (`fetch_status=ok`) that is not empty after trimming. `stats.json` keeps the complete row-based area and geometry statistics.
- `osm_polygon_wikidata_only.hf.trackio_snapshot.publish_trackio_snapshot` publishes the single static run `final-dataset-snapshot`. The console script for it is `osm-polygon-wikidata-only-trackio`.

`osm_polygon_wikidata_only.hf.coverage_map` provides `load_centroids_from_parquet`, `generate_coverage_map`, and `ensure_world_land` for the map of all polygons. The facade `osm_polygon_wikidata_only.hf.geographic_text_coverage` provides the typed H3 aggregation helpers and render helpers for the geographic coverage assets. These helpers include `assign_h3_cell`, `aggregate_geographic_text_coverage`, and `render_geographic_text_coverage`.

## Language-split inventory

`osm_polygon_wikidata_only.hf.language_splits` exposes the shared row-level normalizer and the separate V1 and V2 inventory contracts:

- `normalize_language` and `language_split_name` apply the language rule of the repository. They send the values that are not usable to `lang-unknown` with an explicit reason.
- `language_table_specs` returns the exact mapping of the tables that have a language to their schemas. It accepts `DatasetContract.V1` or `DatasetContract.V2`.
- `build_language_inventory` validates the manifest and the Parquet schemas of the selected contract. Then it streams only the `language` column of each text table or document table. It uses the column to find the languages and the counts for each table. For V1, pass `data_root.processed`. For V2, pass `data_root.processed_v2`. Never combine these roots.

This stage only defines and inventories the contract. It does not generate or publish the language partitions.

## Compatibility rules

These items are compatibility contracts: the CLI, the Parquet schemas, the manifest names, the deterministic ordering, and the public client classes. Add new implementation details behind these facades. The clients are synchronous. They can do network I/O or filesystem I/O. In tests, use the in-memory client variants and the Hub stub. They avoid both kinds of I/O.

Use the CLI option `sync-dir --dataset-version v2` to select the V2 dataset. V2 has a separate storage contract and publication contract. Python callers must not mix V1 and V2 output paths in one run.
