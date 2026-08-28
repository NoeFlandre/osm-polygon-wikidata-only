# Fast Sentence Publication Verification

## Goal

Reduce the local Hugging Face publication latency for Grid5000 sentence batches without changing sentence artifacts, publication paths, dataset-card protection, or resume behavior.

## Root cause

After each atomic upload, the controller currently lists the complete dataset repository and downloads every expected artifact again before marking the batch published. The sentence Parquet files are large LFS objects, so this repeats the dominant network transfer solely to verify bytes that the Hub already exposes as LFS SHA-256 metadata.

## Design

`RemoteInventory` gains an exact-path metadata query backed by `HfApi.get_paths_info`. The query returns the requested remote file paths, sizes, and LFS SHA-256 values in one request; it does not enumerate the whole repository.

`HfHubSentencePublisher.verify_sentence_batch` will:

1. Build the existing deterministic publication list and protected comparison-map list.
2. Query metadata for exactly those remote paths.
3. Fail with the existing missing-file error when a requested path is absent.
4. For files with LFS SHA-256 metadata, compare remote size and the local file SHA-256 without downloading the remote object.
5. Fall back to the existing exact-byte download for files without a usable digest, preserving verification for regular Git blobs and older Hub responses.

The sentence publisher will also use a bounded upload pool of at most four workers (or one per publication operation when fewer are present). This changes transfer concurrency only; the atomic commit contents and ordering remain unchanged.

The README and comparison-map baseline checks remain unchanged. No sentence splitting, batch planning, HF paths, commit contents, or ledger state transitions change.

## Testing and quality gates

Tests will cover metadata parsing, exact-path lookup, LFS verification without a download, and the regular-blob fallback. The focused tests must run RED before implementation and GREEN after it. The project quality gauntlet, CRAP gate (<6), and mutation gate must pass before the controller is resumed with the optimized code.
