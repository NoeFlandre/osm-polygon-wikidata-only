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
and uses Hugging Face-compatible names such as `lang-be-tarask`. Both
published dataset cards declare additive Hugging Face Dataset Viewer language
selections. V1 uses one configuration per language-bearing table with
`lang-<language>` split names. V2 uses one configuration per table/language
pair, named `<configuration>__lang_<language>` with language-code dashes
replaced by underscores, and one `train` split; its storage directories retain
`lang-<language>`.

The plain `language-splits` command is local generation only. It does not read
raw PBF files, sample or truncate rows, upload to Hugging Face, or modify
dataset cards and website metadata. The existing undifferentiated V1/V2
commands and the `release-stats` card/statistics path remain separate.

## Exact-target publication

After reviewing the dry-run plan, publish with the separate exact-target
command. It creates one atomic Hub commit per selected dataset, updates only
the managed language metadata and section of the existing card, verifies the
uploaded files at the returned revision, and makes an unchanged second run a
no-op:

```console
uv run osm-polygon-wikidata-only publish-language-splits \
  --data-root /path/to/data \
  --dataset-version v1 \
  --confirm-repo NoeFlandre/osm-polygon-wikidata-only \
  --apply
```

Use the V2 repository confirmation for `--dataset-version v2`, or repeat both
exact confirmations for `both`. V1 publishes
`data/<configuration>/lang-<language>-00000-of-00001.parquet` files; V2
publishes deterministic bounded
`language_splits/<configuration>/lang-<language>/part-*.parquet` shards. The
V2 Viewer configuration names use the form described above and expose the
single `train` split. The generated
manifests remain under `manifests/` and are the ownership record
used to remove only obsolete generated shards on later releases.

V2 shard planning uses the validated row inventory, keeps each shard at most
100,000 rows, and records the contributing source files. The command still
refuses an apply before local generation or Hub mutation when any selected
release exceeds Hugging Face's 25,000-file atomic-commit limit.
