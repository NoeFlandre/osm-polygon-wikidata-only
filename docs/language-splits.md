# Language-split release command

The existing V1 and V2 language-split generators can be run through one
deterministic local command:

```console
uv run osm-polygon-wikidata-only language-splits \
  --data-root /path/to/data \
  --dataset-version both \
  --batch-size 65536
```

`--dataset-version` accepts `v1`, `v2`, or `both` and defaults to `both`.
The command validates every selected processed-data inventory before it
writes any output. `--dry-run` performs that validation and prints the same
deterministic JSON plan without creating language-split files:

```console
uv run osm-polygon-wikidata-only language-splits \
  --data-root /path/to/data \
  --dataset-version both \
  --dry-run
```

Outputs remain isolated by dataset contract:

- V1: `processed/language_splits/`
- V2: `processed_v2/language_splits/`

The JSON result reports the selected contract, validated source-manifest
fingerprints, expected files, row counts, and generated manifest paths. The
command streams source Parquet batches, preserves row-level language
semantics, maps unusable language values to the accepted `unknown` bucket,
and uses Hugging Face-compatible names such as `lang-be-tarask`.

This is a local generation path only. It does not read raw PBF files,
sample or truncate rows, upload to Hugging Face, or modify dataset cards and
website metadata. The existing undifferentiated V1/V2 commands and the
`release-stats` card/statistics path remain separate.
