# Supported Python API

The command line is the primary interface. The following Python entry points
are also supported:

- `config.paths.DataRoot` and `resolve_data_root`
- `config.settings.Settings`
- `enrichment.wikidata_client` public client/entity names
- `enrichment.wikipedia_client` public client/article/result names
- `pipeline.processor.PbfStem`, `ProcessResult`, `ExtractedPbf`,
  `IncompleteEnrichmentError`, `process_pbf`, `process_extracted_pbf`,
  and `extract_pbf`
- `pipeline.orchestrator.orchestrate`
- `hf.dataset_card.render_dataset_card`
- `hf.uploader` public upload helpers
- `hf.coverage_map.generate_coverage_map`, `ensure_world_land`,
  and `load_centroids_from_parquet`
- `hf.geographic_text_coverage` documented types, constants, and
  helpers (`CoverageCell`, `CoverageMapError`, `PolygonCountCell`,
  `RenderResult`, `DEFAULT_H3_RESOLUTION`,
  `DEFAULT_MIN_POLYGONS_PER_CELL`, the `LOCAL_*` / `REMOTE_*` asset
  path constants, `assign_h3_cell`, the `aggregate_*`,
  `generate_*`, and `render_*` helpers)

The remaining names exposed by the codebase -- including
`pipeline.sync_planner`, `pipeline.sync_runner`,
`hf.publication`, `hf.dataset_stats`, `hf.repo_layout`,
`augmentation.orchestrator`, `utils.request_scheduler`,
`utils.http_retry`, `enrichment.wikimedia`, and
`enrichment.wikimedia_auth` -- are implementation details behind
compatibility facades. They are not part of the supported Python
surface; importing them directly is at the caller's risk.

Compatibility facades preserve the supported imports above while
focused internal packages may change. Underscore-prefixed packages
(for example `hf._dataset_stats`, `hf._geographic`, `hf._uploader`)
are private. Other focused modules (for example
`enrichment.wikidata.transport`, `enrichment.wikipedia.transport`,
`augmentation.steps`, `pipeline.sync_runner`) may also be
implementation details even though they do not carry an underscore
prefix; they remain reachable through compatibility facades and may
change without notice. Names beginning with `_` at the attribute
level are always implementation details. Parquet schemas and
identifiers are data compatibility contracts documented in the README,
not merely Python implementation details.

Clients are synchronous and may perform network or filesystem I/O. Pipeline
publication is fail-closed: unresolved expected enrichment raises
`IncompleteEnrichmentError` before final artifacts or manifest completion.

Exception boundary policy:

- **Atomic writes** (`io.atomic.atomic_write_text`,
  `hf._geographic.rendering.atomic_save_png`) catch `BaseException` so
  the temporary-file cleanup branch fires even on `KeyboardInterrupt`
  and `SystemExit`. Narrowing to `Exception` would leak temp files on
  Ctrl-C.
- **Upload backend translation** in `hf._uploader.operations`
  (`_ensure_repo_exists`, `upload_parquet`, `upload_files`,
  `upload_card`) catches `Exception` so the unstable exception types
  raised by `huggingface_hub` are translated uniformly into
  `UploadError`. The same module's `_translate_hf_error` handles
  401/403/404/auth-marker translation. Token resolution and
  verification in `hf._uploader.token` catches `Exception` for the
  same reason. `hf.upload_queue` is different: it does **not**
  translate every exception into `UploadError`; it records failures
  (with the underlying exception detail appended to the message) into
  its `failures` list and lets the daemon worker survive to process
  the next queued job.
- **Heartbeat isolation** in
  `pipeline.heartbeat.EnrichmentHeartbeat.run` catches `Exception` so
  observational heartbeat failures are contained, logged at debug,
  and the daemon thread exits without propagating uncaught
  exceptions into the calling pipeline.
- **World-land basemap fallback** in
  `hf.publication.refresh_coverage_assets` and
  `hf.publication.snapshot_upload_manifests` catches `Exception`
  around `ensure_world_land` because the helper performs network
  I/O via `urllib.request.urlretrieve` (raises `URLError`,
  `HTTPError`, `ContentTooShortError`, `socket.timeout`, `OSError`);
  the documented fallback is to render the map without continents.
- **PyArrow schema-introspection fallback** in
  `hf._geographic.parquet_inputs.read_required_columns` catches
  `Exception` around `pq.read_metadata`. When the metadata read
  fails, the implementation falls through with an empty `actual`
  column-name set and lets the subsequent column-pruned
  `pq.read_table` call determine the outcome: a valid parquet with
  the requested columns still loads successfully; missing columns
  are translated into `CoverageMapError`.
