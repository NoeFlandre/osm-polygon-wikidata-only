# V2 Row-Level Language Split Generation

> **Historical, archived.** This document records a completed plan or design and
> may not match the current code (paths, names, unchecked boxes). Do not execute
> it; see `docs/adr/` and the code for the current record.

## Goal

Generate deterministic, local language partitions for the complete validated
V2 text/document tables without changing V1 artifacts or publishing to the
Hugging Face Hub.

## Design

`osm_polygon_wikidata_only.v2.language_splits` is the V2-specific generator
and command boundary. It accepts a `processed_v2/` path, calls the accepted
`hf.language_splits.build_language_inventory` with `DatasetContract.V2`, and
uses that inventory as the only source of table, schema, manifest, and
language-bucket truth. The generator never opens the V2 polygon table for
routing and never reads `best_language`.

For each validated V2 language-bearing table, the generator streams complete
Parquet record batches in the contract's sorted source-file order. Every row
is normalized exactly once through `normalize_language`; rows are written to
the corresponding `<table>_by_language/lang-<language>/` output while keeping
their original column order, schema metadata, physical row order, provenance,
and identity. A source region remains one output shard per table/language, so
V2 region boundaries remain visible and memory is bounded by one row group.

Outputs live below `processed_v2/language_splits/` and are addressed by the
accepted configuration/split names. The V2-specific manifest at
`processed_v2/manifests/language_splits.json` records the V2 contract marker,
source manifest digest, validated artifact fingerprint, table identities,
bucket counts, output paths, and output SHA-256 values. JSON is serialized by
the repository's deterministic helper and Parquet files are staged and
atomically replaced. The manifest is authoritative for the generated tree;
no Hugging Face upload operation is called.

The local command is intentionally independent of the shared CLI and package
entry-point files:

```bash
uv run python -m osm_polygon_wikidata_only.v2.language_splits \
  /path/to/processed_v2
```

## Verification

Focused tests use real temporary Parquet artifacts and assert multilingual
row preservation, row/document identity, explicit unknown routing, complete
conservation, stable ordering and bytes, exact schema preservation, V1/V2
isolation, and standard `datasets.load_dataset("parquet", ...)` loading of a
single language split. Source Ruff, ty, architecture, and focused CRAP
checks are run with their caches under `/private/tmp`.
