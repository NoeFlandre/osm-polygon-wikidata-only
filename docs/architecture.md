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

## Dependency direction

Dependencies point inward: CLI and pipeline orchestration compose I/O and
enrichment; enrichment depends on configuration, cache interfaces, and small
utilities; domain code is pure and does not import infrastructure. Stable
facade modules preserve documented imports while focused subpackages contain
models and implementation details.

The largest workflows are split by responsibility:

- `cli.parser` owns argparse and immutable settings conversion;
- `pipeline.rows` owns deterministic domain-row construction;
- `pipeline.processor` sequences extraction, enrichment, publication, and
  metrics;
- `enrichment.wikipedia.models` and `enrichment.wikidata.models` define the
  typed contracts used across clients and linkers.

Private implementation modules may evolve, but the supported imports in
[`docs/api.md`](api.md) are compatibility boundaries.

## Completeness and publication

Normal production runs fetch full text for every valid language-Wikipedia
sitelink with no per-QID article cap. Wikimedia requests share one process-wide
scheduler capped at three in-flight requests. With a configured Bot Password,
one transport owns a cookie-preserving session per API host and lazily performs
the MediaWiki token/login handshake once for Wikidata and each language-specific
Wikipedia host. Without credentials, the same transport remains anonymous.

Anonymous pacing stays fixed at 180 requests per minute. Authenticated unified
runs use their configured ceiling (1,200 requests per minute by default) while
retaining the global three-request concurrency cap. A 429 response applies
`Retry-After` globally and halves the active rate before later successful windows
restore it. The session, rather than either domain client, owns HTTP
cookies and scheduled response reads. Successful
responses are cached atomically; transient failures never satisfy completion.
When TextExtracts is empty for a valid page, enrichment uses the Action API's
exact-revision parse output as a deterministic plain-text fallback.

Long enrichment is observable without request-level noise. A thread-safe tracker
records completed and total QIDs, completed and total Wikipedia sites, and
articles attempted. The processor reads an immutable snapshot in a two-minute
heartbeat that also names the active Wikidata or Wikipedia phase, and stops its
daemon immediately when enrichment returns or raises. This is a liveness signal,
not an ETA, and it does not alter request pacing, ordering, retries, or output
construction.

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
