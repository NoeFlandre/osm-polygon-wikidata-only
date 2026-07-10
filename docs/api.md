# Supported Python API

The command line is the primary interface. The following Python entry points
are also supported:

- `config.paths.DataRoot` and `resolve_data_root`
- `config.settings.Settings`
- `enrichment.wikidata_client` public client/entity names
- `enrichment.wikipedia_client` public client/article/result names
- `pipeline.processor.PbfStem`, `ProcessResult`, and `process_pbf`
- `pipeline.orchestrator.orchestrate`
- `hf.dataset_card.render_dataset_card`
- `hf.uploader` public upload helpers

Compatibility facades preserve these imports while focused internal packages
may change. Names beginning with `_` are implementation details. Parquet
schemas and identifiers are data compatibility contracts documented in the
README, not merely Python implementation details.

Clients are synchronous and may perform network or filesystem I/O. Pipeline
publication is fail-closed: unresolved expected enrichment raises
`IncompleteEnrichmentError` before final artifacts or manifest completion.
