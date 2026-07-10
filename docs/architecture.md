# Architecture

The project is intentionally layered so each concern can be tested in
isolation.

| Layer | Responsibility |
| --- | --- |
| `config` | Immutable runtime settings and external data-root resolution. |
| `domain` | Stable IDs, geometry/analysis helpers, flat dataset records, schemas. |
| `io` | PBF streaming, cache files, manifests, and Parquet persistence. |
| `enrichment` | Wikidata/Wikipedia clients, cache wrappers, batching, and linking. |
| `pipeline` | Extract, enrich, construct rows, write artifacts, and update manifests. |
| `hf` | Remote paths, dataset card rendering, and atomic Hub uploads. |
| `cli` | Argument parsing and dependency wiring only. |

## Completeness and publication

Normal production runs fetch full text for every valid language-Wikipedia
sitelink with no per-QID article cap. Wikimedia requests share a global
unauthenticated scheduler capped at three in-flight requests. Successful
responses are cached atomically; transient failures never satisfy completion.

A PBF is published locally only after every expected article succeeds. Its
three Parquet files, manifest snapshot, and generated Hugging Face dataset card
are then queued in one background upload commit while the next PBF begins.
Failed upload jobs persist under the external data root and resume on the next
invocation. The dataset and pipeline are maintained by Noé Flandre.

## Compatibility contract

The CLI, Parquet schemas, manifest paths, deterministic ordering, and public
client classes are stable. New internals must be introduced behind existing
public functions or explicit capability protocols.

## Verification

Run `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`,
and `uv run mypy src` before merging a change.
