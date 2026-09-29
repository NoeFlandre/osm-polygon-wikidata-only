# Command-line reference

The main command provides the following subcommands. Every command has a
`--help` page with its options and defaults. Run commands from the repository
root with `uv run`:

| Command | Purpose |
| --- | --- |
| `uv run osm-polygon-wikidata-only process-pbf` | Process one `.osm.pbf` file. |
| `uv run osm-polygon-wikidata-only process-dir` | Process every PBF in a directory. |
| `uv run osm-polygon-wikidata-only sync-dir` | Converge core and augmentation for every PBF, using V1 by default or an explicit dataset version. |
| `uv run osm-polygon-wikidata-only augment-region` | Augment one completed region without reading its PBF again. |
| `uv run osm-polygon-wikidata-only augment-dir` | Augment every completed core region. |
| `uv run osm-polygon-wikidata-only split-v2-sentences` | Materialize resumable V2 sentence sidecars. |
| `uv run osm-polygon-wikidata-only language-splits` | Generate deterministic V1/V2 language partitions locally. |
| `uv run osm-polygon-wikidata-only publish-language-splits` | Generate and publish exact-target V1/V2 language partitions; use `--apply` to publish. |
| `uv run osm-polygon-wikidata-only release-stats` | Publish a dataset card and polygon statistics report; use `--apply` to publish. |
| `uv run osm-polygon-wikidata-only enforce-integrity` | Remove join-integrity violations from processed tables and write an audit. |
| `uv run osm-polygon-wikidata-only audit-remote` | Read-only audit of local files against the configured Hugging Face dataset. |
| `uv run osm-polygon-wikidata-only trackio-snapshot` | Publish the frozen dataset metrics snapshot to Trackio. |
| `uv run osm-polygon-wikidata-only grid5000` | Run the Grid5000 sentence-splitting controller or a reserved-node job. |
| `uv run osm-polygon-wikidata-only audit-containment` | Produce a read-only audit of whole-file containment retirements. |

For example, inspect a command before running it:

```bash
uv run osm-polygon-wikidata-only sync-dir --help
```

Commands that publish to Hugging Face or Trackio are opt-in; inspect their
help and target repository before using their apply/publish options. Keep
credentials in environment variables or an untracked `.env` file, never in
source code or a container image.
