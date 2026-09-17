# Language Release Hardening Design

## Goal

Make the existing V1/V2 language-split release command safe and reproducible
enough to generate and later publish the two datasets without losing rows,
mixing dataset versions, deleting unrelated files, or leaving a partial V2
release after a failure.

This design covers the implementation issues #24–#28. It does not publish to
the Hugging Face Hub, recompute polygon statistics, or change the source
Parquet artifacts.

## Current boundary

The repository already contains:

- a shared row-level language contract and inventory validator;
- a V1 generator and a V2 generator;
- the `language-splits` CLI facade with dry-run support;
- separate V1 and V2 output roots;
- the public `docs/language-splits.md` page.

The V1 generator remains the compatibility boundary. The hardening work is
primarily in the V2 writer and in regression coverage for the facade,
documentation, and mutation scope.

## Architecture

The release flow has four phases:

1. **Preflight.** Resolve the processed root and output root, validate the
   selected language inventory, and reject an output path that equals or
   contains any source artifact or source directory. No output directory is
   created and no stale file is inspected before this phase succeeds.
2. **Stage.** Stream each validated V2 source Parquet file into a temporary
   sibling tree on the same filesystem as the destination. Preserve the
   source schema, row order, source-file identity, normalized language bucket,
   row counts, and deterministic file ordering. Validate every staged file
   before installation and write the staged manifest last.
3. **Install.** Determine obsolete generated shards only from the previous
   valid language-split manifest. Preserve files not owned by that manifest.
   Move current owned files and the previous manifest to transaction-local
   backups, replace staged files, and install the manifest last.
4. **Rollback or cleanup.** If any backup or replacement fails, remove newly
   installed files and restore every backup, leaving the previous release
   byte-for-byte unchanged. On success, remove transaction files and empty
   directories that are owned by the generated release.

The V2 transaction is implemented as a sequence of same-filesystem renames
with explicit rollback. A whole-directory rename is not used because the
output tree may contain operator-owned files and the manifest lives outside
the output directory.

V1 keeps its existing output layout and generation behavior. The shared facade
continues to preflight all selected versions before generation, so selecting
`both` cannot start V1 or V2 generation after the other version has already
mutated its output due to an inventory failure.

## Defect-specific changes

### Documentation and mutation scope

Keep `docs/language-splits.md` in MkDocs navigation and retain a test that
asserts the navigation target exists. Add a mutation-scope regression test
that requires both the release facade and its integration tests in the
configured mutation source/test selections.

### Stale V2 shards

The previous manifest is the ownership ledger. Only `.parquet` paths listed in
its generated file records and resolved below the current output root may be
removed. A stale-looking file not listed by the manifest remains untouched.
Malformed, missing, or foreign-output manifests do not authorize deletion.

### Source/output overlap

The resolved output root must remain inside the processed root while remaining
disjoint from every validated source file and source directory, including the
processed manifest. The check runs before staging, directory creation, stale
discovery, or cleanup.

### Atomic V2 installation

Generation errors leave the existing release untouched because all new files
are staged outside it. Installation errors are injected in tests at multiple
points, including during backup, and must restore the exact prior file
snapshot with no `.tmp`, `.backup`, or staging directories left behind.

## Testing strategy

Use RED → GREEN → REFACTOR for each defect. Permanent tests cover:

- MkDocs navigation and mutation-scope contracts;
- V1 output compatibility and V1/V2 path isolation;
- stale generated-shard removal while preserving operator-owned files;
- rejection of equal, nested, and source-overlapping output roots before any
  mutation;
- schema, row-count, language-bucket, and source-row conservation checks;
- deterministic manifest and output bytes across repeated runs;
- failure injection during staging, backup, and replacement with full rollback;
- cleanup of transaction artifacts and safe handling of nested output parents.

Focused tests run before the repository-wide quality gauntlet. The mutation
configuration must measure the facade and V2 writer, and meaningful mutants
must be killed by the focused tests rather than excluded.

## Acceptance boundary

This sub-project is complete only when the code and tests are merged through a
reviewed GitHub PR, the full configured quality checks pass, and both V1 and
V2 dry-run/generation plans can be reproduced from the clean latest-main
worktree. Hugging Face publication and remote verification are separate later
steps and must not be claimed by this design alone.
