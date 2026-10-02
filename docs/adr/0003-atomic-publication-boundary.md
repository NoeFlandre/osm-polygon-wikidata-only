# ADR 0003: Atomic publication boundary

- Status: Accepted
- Date: 2026-09-26

## Context

An interruption can stop the regional extraction and enrichment. Users consume the published dataset as a complete set of tables and manifests. Two events make the resume unsafe and the public counts not reliable:

- A region is visible when it is only partially written.
- The pipeline reports success before it verifies the remote upload.

## Decision

- Keep the raw inputs, the restart state, and the generated artifacts in the data root of the operator. The data root is outside the source checkout.
- Finalize a region only after its Parquet files pass the schema checks and join checks. Install the completed files atomically. Write the manifest entry last.
- Publish the related release files in one atomic Hugging Face commit. Do not put the manifest and the data of a release in different commits.
- Follow the explicit success contract of each publication workflow. A workflow with revision-bound inventory checks verifies the expected remote files and metadata at the uploaded revision. Only then does it mark the release as complete.
- If a failure occurs, keep the complete local artifacts and enough state to retry. Never represent an incomplete candidate as a published region.

## Consequences

The local outputs are the recovery boundary. The operator can resume without a new extraction of the completed regions. A failed upload does not delete valid local work. Each workflow must distinguish these states: a local candidate, a committed Hub revision, and any stronger remote verification that the workflow promises. The [architecture guide](../architecture.md) and the [language-split release guide](../language-splits.md) give the operational details.
