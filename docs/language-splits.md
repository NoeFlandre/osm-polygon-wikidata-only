# Language-split release command

One deterministic local command runs the existing V1 and V2 language-split generators:

```console
uv run osm-polygon-wikidata-only language-splits \
  --data-root /path/to/data \
  --dataset-version both \
  --batch-size 65536
```

`--dataset-version` accepts `v1`, `v2`, or `both`. The default is `both`. The command validates the inventory of each selected processed dataset before it writes any output. With `--dry-run`, the command does this validation and prints the same deterministic JSON plan. It does not create language-split files:

```console
uv run osm-polygon-wikidata-only language-splits \
  --data-root /path/to/data \
  --dataset-version both \
  --dry-run
```

The outputs stay separate for each dataset contract:

- V1: `processed/language_splits/`
- V2: `processed_v2/language_splits/`

The JSON result reports these items:

- The selected contract.
- The fingerprints of the validated source manifests.
- The expected files.
- The row counts.
- The paths of the generated manifests.

The command streams the source Parquet batches. It keeps the row-level language meaning. It maps the language values that are not usable to the accepted `unknown` bucket. It uses names that Hugging Face accepts, for example `lang-be-tarask`.

The two published dataset cards declare additive language selections for the Hugging Face Dataset Viewer:

- V1 uses one configuration for each table that has a language. The splits have the names `lang-<language>`.
- V2 uses one configuration for each pair of table and language. The name is `<configuration>__lang_<language>`, with underscores in place of the dashes of the language code. Each configuration has one `train` split. The storage directories keep `lang-<language>`.

The plain `language-splits` command only generates files on the local disk. It does not do these actions:

- Read the raw PBF files.
- Sample or truncate rows.
- Upload to Hugging Face.
- Change the dataset cards or the website metadata.

The existing V1 and V2 commands that do not use language splits stay separate. The `release-stats` path for the card and the statistics stays separate also.

## Exact-target publication

First, review the dry-run plan. Then publish with the separate exact-target command. For each selected dataset, the command does these actions:

1. It creates one atomic Hub commit.
2. It updates only the managed language metadata and the managed section of the existing card.
3. It verifies the uploaded files at the returned revision.

If you run the command a second time with no change, the command does nothing.

```console
uv run osm-polygon-wikidata-only publish-language-splits \
  --data-root /path/to/data \
  --dataset-version v1 \
  --confirm-repo NoeFlandre/osm-polygon-wikidata-only \
  --apply
```

For `--dataset-version v2`, use the V2 repository confirmation. For `both`, give the two exact confirmations. V1 publishes the files `data/<configuration>/lang-<language>-00000-of-00001.parquet`. V2 publishes deterministic bounded shards `language_splits/<configuration>/lang-<language>/part-*.parquet`. The names of the V2 Viewer configurations use the form above. They show the single `train` split. The generated manifests stay under `manifests/`. They are the ownership record. A later release uses them to remove only the old generated shards.

The V2 shard plan uses the validated row inventory. Each shard has 100,000 rows at most. The plan records the source files that contribute to the shard. The command refuses an apply when a selected release has more than 25,000 files. This is the atomic-commit limit of Hugging Face. The command refuses before local generation and before it changes the Hub.
