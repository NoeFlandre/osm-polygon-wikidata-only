# Command-line reference

The main command has the subcommands in the table. Each subcommand has a `--help` page. The page shows the options and the default values. Run the commands from the repository root with `uv run`.

| Command | Purpose |
| --- | --- |
| `uv run osm-polygon-wikidata-only process-pbf` | Process one `.osm.pbf` file. |
| `uv run osm-polygon-wikidata-only process-dir` | Process each PBF in a directory. |
| `uv run osm-polygon-wikidata-only sync-dir` | Complete the core step and the augmentation for each PBF. It uses V1 by default. You can select a dataset version. |
| `uv run osm-polygon-wikidata-only augment-region` | Augment one completed region. It does not read the PBF again. |
| `uv run osm-polygon-wikidata-only augment-dir` | Augment each completed core region. |
| `uv run osm-polygon-wikidata-only split-v2-sentences` | Create the V2 sentence sidecars. You can resume the command. |
| `uv run osm-polygon-wikidata-only language-splits` | Generate the deterministic V1 and V2 language splits locally. |
| `uv run osm-polygon-wikidata-only publish-language-splits` | Generate the V1 and V2 language splits for the exact target and publish them. Use `--apply` to publish. |
| `uv run osm-polygon-wikidata-only release-stats` | Publish a dataset card and a polygon statistics report. Use `--apply` to publish. |
| `uv run osm-polygon-wikidata-only enforce-integrity` | Remove the join-integrity violations from the processed tables and write an audit. |
| `uv run osm-polygon-wikidata-only audit-remote` | Compare the local files with the configured Hugging Face dataset. The command only reads. |
| `uv run osm-polygon-wikidata-only trackio-snapshot` | Publish the frozen dataset metrics snapshot to Trackio. |
| `uv run osm-polygon-wikidata-only grid5000` | Run the Grid5000 sentence-splitting controller or a job on a reserved node. |
| `uv run osm-polygon-wikidata-only audit-containment` | Create an audit of the whole-file containment retirements. The command only reads. |

## Legacy commands

These older entry points stay installed. Each one prints a deprecation notice on stderr. Its stdout and exit codes do not change. Use the replacement in new scripts and jobs. If a script merges stderr into stdout (for example with `2>&1`), the notice appears in that output.

| Legacy command | Replacement | Notes |
| --- | --- | --- |
| `uv run osm-polygon-wikidata-only-enforce-integrity` | `uv run osm-polygon-wikidata-only enforce-integrity` | Same options. |
| `uv run osm-polygon-wikidata-only-audit-remote` | `uv run osm-polygon-wikidata-only audit-remote` | Same options. |
| `uv run osm-polygon-wikidata-only-trackio` | `uv run osm-polygon-wikidata-only trackio-snapshot` | Publishes the V1 run, which is the default of `--dataset-version`. |
| `uv run osm-polygon-wikidata-and-wikipedia-trackio` | `uv run osm-polygon-wikidata-only trackio-snapshot --dataset-version v2` | Publishes the V2 run. |
| `python scripts/audit_containment.py` | `uv run osm-polygon-wikidata-only audit-containment` | Shim with the same arguments, output and exit codes. |

## Exit codes

| Code | Meaning | Commands |
| --- | --- | --- |
| `0` | Success. | All commands. |
| `1` | Expected operator failure, such as an unusable data root, a missing or unreadable file, or a data-contract violation. The message goes to stderr. | `enforce-integrity`, `audit-containment`, `audit-remote`, `process-pbf`, `process-dir` (a background upload failed). |
| `2` | `audit-containment` found at least one blocked parent. The report is still printed. Argparse also uses `2` for a usage error, so check the output to tell the two apart. | `audit-containment`, and any command for a usage error. |

To see the options of a command before you run it, use `--help`. For example:

```bash
uv run osm-polygon-wikidata-only sync-dir --help
```

A command that publishes to Hugging Face or Trackio is opt-in. Before you use its apply or publish option, read its help and check the target repository. Keep the credentials in environment variables or in an untracked `.env` file. Never put them in source code or in a container image.
