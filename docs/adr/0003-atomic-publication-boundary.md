# ADR 0003: Atomic publication boundary

- Status: Accepted
- Date: 2026-09-26

## Context

Regional extraction and enrichment can be interrupted, while the published
dataset is consumed as a complete set of tables and manifests. Exposing a
partially written region, or reporting success before a remote upload is
verified, would make resumability unsafe and public counts unreliable.

## Decision

- Keep raw inputs, resumable state, and generated artifacts in the operator's
  data root, outside the source checkout.
- Finalize a region only after its Parquet files pass schema and join checks;
  install the completed files atomically and write the manifest entry last.
- Publish related release files in one atomic Hugging Face commit; do not split
  a release's manifest and data across commits.
- Follow each publication workflow's explicit success contract. Workflows with
  revision-bound inventory checks verify the expected remote files and metadata
  at the uploaded revision before marking the release complete.
- On failure, retain complete local artifacts and enough state to retry. An
  incomplete candidate must never be represented as a published region.

## Consequences

Local outputs are the recovery boundary: operators can resume without
re-extracting completed regions, and a failed upload does not discard valid
local work. Each workflow must distinguish a local candidate, a committed Hub
revision, and any stronger remote-verification condition it promises. The
operational details are documented in the [architecture guide](../architecture.md)
and [language-split release guide](../language-splits.md).
