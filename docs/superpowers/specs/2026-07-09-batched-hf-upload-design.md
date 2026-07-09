# Batched Hugging Face Upload Design

## Goal

Publish the exact current artifacts faster by replacing repeated sequential
Hub commits with atomic, concurrent commits.

## Design

Create a public `upload_files` helper that validates local paths, converts
them to Hugging Face `CommitOperationAdd` operations, and calls
`HfApi.create_commit` once with a configurable worker count. The CLI collects
all three Parquet files for every completed PBF and the final manifest, then
submits them together.

The existing single-file upload helpers remain unchanged for compatibility.
The stub client records one commit containing all operations, so tests can
prove exact remote paths and atomic grouping. The default worker count uses
the Hub client's supported concurrency and remains configurable through the
CLI.

## Invariants

Remote paths, files, content, manifest timing, dry-run behavior, and failed
upload reporting remain unchanged. The only observable change is fewer,
atomic commits and faster concurrent transfer.
