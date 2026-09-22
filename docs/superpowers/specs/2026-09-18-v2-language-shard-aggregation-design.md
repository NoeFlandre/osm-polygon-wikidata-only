# V2 language-shard aggregation design

## Goal

Publish row-level language partitions for both V1 and V2 as selectable
Hugging Face Dataset Viewer configurations/splits. The current V2 layout
creates 179,847 data files, which exceeds the Hub `create_commit` limit of
25,000 files. V1 data files remain unchanged; only its dataset-card metadata
needs to expose the already-published language files to the Viewer.

## Chosen design

V2 will write deterministic aggregated Parquet shards under the existing
language namespace:

```text
language_splits/<configuration>/lang-<language>/part-00000-of-000NN.parquet
```

Rows are streamed in validated source-file order, grouped by the row's
normalized language, and written with the original schema (including the
language column). A shard contains at most 100,000 rows; a language with more
rows receives sequential shards. The shard count is calculated from the
validated inventory before generation, so the filename total is deterministic
and can be checked against the Hub limit before any output or remote mutation.

Each manifest file record will contain its row count, hash, configuration,
language, split, and the ordered source-file list that contributed rows to the
shard. The V2 language-split contract version will be incremented to identify
the aggregated layout. Row order within each language remains the source
inventory order; no rows are deduplicated or dropped.

## Dataset Viewer contract

The dataset-card YAML front matter is the Viewer configuration contract. For
each language-bearing table, the release will add a distinct
`<configuration>_by_language` config and one `data_files` entry per non-empty
language split, including `lang-unknown` when present:

- V1: Viewer split `lang-<language>`, stored at
  `data/<configuration>/lang-<language>-00000-of-00001.parquet`
- V2: Viewer split `lang_<language>` with dashes in the language code replaced
  by underscores, stored at
  `language_splits/<configuration>/lang-<language>/*.parquet`

The existing default configurations and all unrelated front matter remain
unchanged. The managed language-config block is replaced deterministically on
reruns, so both cards expose language values as Viewer-selectable splits and
the update is idempotent. The release acceptance check calls the Dataset
Viewer `/splits` and `/first-rows` APIs at the final revision for representative
language splits in both datasets.

## Atomicity and publication

The existing local staging/rollback transaction remains in place. The
publication planner will predict the aggregated V2 file set from inventory
counts, include the manifest and card, and retain the fail-closed 25,000-file
guard. For the current V2 inventory the predicted set must fit below the
limit; otherwise generation stops before writing. Remote publication remains
one commit, with revision-bound file, manifest, card, schema, count, and hash
verification.

The managed card section will describe the aggregated `part-*` layout. Old
manifest-owned V2 shards are deleted only when they are absent from the new
manifest. Unmanaged files are preserved.

## Alternatives rejected

1. Multiple remote commits would expose partial V2 releases and make card and
   manifest updates non-atomic.
2. Archives would avoid the file limit but would remove standard Parquet/
   Dataset Viewer discoverability and require custom extraction.
3. Raising the Hub limit is not available to the repository and would not
   solve the underlying excessive-shard layout.

## Tests and acceptance

- RED tests cover aggregation across source files, deterministic shard names,
  rotation at the row limit, source provenance, schema preservation, unknown
  language routing, and conservation.
- Publication tests cover the new V2 plan count, the under-limit preflight,
  stale aggregated-shard cleanup, card wording, and exact atomic operations.
- Existing V1 tests and publication behavior remain green.
- The release run will update and verify both cards' Viewer config metadata,
  verify the V2 Hub revision, complete remote inventory, manifest/card
  contents, representative shard hashes, Dataset Viewer/API visibility, and a
  second no-op for each target.
