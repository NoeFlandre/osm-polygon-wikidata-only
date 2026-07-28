# Resumable Augmentation Checkpoints

## Goal

Preserve almost all completed augmentation work across `Ctrl-C`, process
failure, or machine restart without changing dataset rows, schemas, ordering,
publication behavior, or the currently running process.

## Storage boundary

All checkpoint files live below:

`<OSM_POLYGON_DATA_ROOT>/cache/augmentation_checkpoints/<stem>/<plan-key>/`

For the production data root this is on the Seagate volume. No checkpoint is
written into the source repository, `/tmp`, or the Hugging Face cache.
Malformed stems and paths escaping the configured checkpoint root are rejected.

## Validity

The plan key includes a checkpoint contract version, the exact core artifact
hashes, ordered QIDs, and ordered Wikipedia document identities and revisions.
Every checkpoint also records the expected input identity for its phase.
Schema, metadata, identity, or contract mismatches make a checkpoint
non-reusable. Corrupt or incomplete checkpoints are ignored.

## Checkpoint boundaries

1. Wikidata entity payload: saved atomically after the complete entity response.
2. Wikivoyage documents: saved as an exact-schema Parquet after the phase.
3. Article sections: documents are processed in deterministic batches of 50.
   Each completed batch is saved atomically as exact-schema Parquet with its
   ordered document identities. On restart, completed batches are reused and
   only the interrupted or remaining batches run.
4. Wikidata facts: saved as exact-schema Parquet after the phase.

Loaded checkpoint rows pass through the same deterministic final sorting as
fresh rows. Request-level caches remain unchanged and provide a second layer of
resumability inside the currently active batch.

## Completion and failure behavior

Canonical sidecars are written only after all phases are available. Existing
core drift checks run before and after the sidecar write. Checkpoints are
cleared only after integrity enforcement and the augmentation manifest update
succeed. `BaseException`, including `KeyboardInterrupt`, preserves completed
checkpoints. No code attempts to alter or signal an already-running process.

## Testing

Tests must first fail on the current implementation, then prove:

- checkpoint storage is confined to the supplied data-root cache;
- complete phases and section batches survive a fresh-process restart;
- an interruption loses at most the active section batch;
- corrupt, stale, mismatched, or partial checkpoints are not reused;
- reused and uninterrupted runs produce identical rows and ordering;
- successful completion clears checkpoints, while failure preserves them;
- existing augmentation, manifest, integrity, and full-suite contracts remain
  green.
