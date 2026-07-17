# Contained Region Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove whole-file Geofabrik containment duplicates locally and remotely without losing polygons, links, text, sections, facts, provenance, or resumability.

**Architecture:** A private containment policy identifies retained parent stems and retired child stems. A fail-closed migration module audits source-independent identities, stages lossless parent-sidecar unions, assembles one atomic Hugging Face publication containing parent replacements plus child deletions and refreshed metadata, and quarantines local child artifacts only after upload success. `sync-dir` consults the durable retirement manifest so retired raw PBFs never re-enter planning.

**Tech Stack:** Python 3.12, PyArrow/Parquet, existing `PublicationOp`/Hugging Face upload queue, pytest, Ruff, mypy.

---

### Task 1: Characterize policy and containment safety

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/containment_policy.py`
- Create: `tests/migration/test_containment_policy.py`

- [ ] Write failing tests pinning the seven retained parents, eleven children, exact remote/local table paths, source-independent primary keys, deterministic ordering, and path-traversal rejection.
- [ ] Run the focused test and confirm import/contract failure.
- [ ] Implement frozen `ContainmentRule` values and private path/key helpers with no I/O.
- [ ] Run the focused test green and refactor names/docstrings.

### Task 2: Implement read-only losslessness audit

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/containment_migration.py`
- Create: `tests/migration/test_containment_migration.py`

- [ ] Write failing tests for exact polygon containment, missing parent rows, malformed Parquet, missing files, duplicate identities, sidecar deltas, deterministic reports, and fail-closed `safe_to_stage`.
- [ ] Confirm RED failures.
- [ ] Implement schema-preserving batched identity scans and frozen audit result models. Polygon containment is mandatory; missing parent polygons block the rule. Missing non-core rows become an explicit union plan rather than data loss.
- [ ] Run focused tests green and refactor shared scanners.

### Task 3: Stage lossless canonical parent artifacts

**Files:**
- Modify: `src/osm_polygon_wikidata_only/pipeline/containment_migration.py`
- Modify: `tests/migration/test_containment_migration.py`

- [ ] Write failing tests proving parent Parquet schemas/metadata/order remain exact, missing sidecar rows are unioned once by stable identity, duplicate rows choose the existing parent deterministically, polygon links are remapped to the retained parent polygon ID/provenance, original files remain untouched during staging, and a second staging run is byte/logically idempotent.
- [ ] Confirm RED failures.
- [ ] Implement staging under `cache/containment_retirement/<parent>/`, atomic writes, manifest snapshots with child entries removed and parent counts/hashes updated, and a deterministic retirement manifest.
- [ ] Run focused tests green and refactor.

### Task 4: Assemble atomic remote migration and post-upload quarantine

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/publication.py`
- Modify: `src/osm_polygon_wikidata_only/hf/repo_layout.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_uploader/operations.py`
- Modify: `src/osm_polygon_wikidata_only/cli/run_sync.py`
- Modify: `tests/contracts/test_publication.py`
- Modify: `tests/migration/test_containment_migration.py`

- [ ] Write failing tests proving a migration commit adds every staged parent artifact, both canonical manifests, retirement manifest, README/maps, and deletes all seven child table paths atomically; unsafe unpaired child deletion is rejected.
- [ ] Write failing crash-safety tests proving failed/dry-run uploads never move local files, successful uploads quarantine children and install staged parents/manifests atomically, retries are idempotent, and unrelated files are untouched.
- [ ] Confirm RED failures.
- [ ] Implement the pure assembler, uploader structural safety check, and post-success local finalizer using `quarantine/containment-v1/<child>/...`.
- [ ] Run focused tests green and refactor.

### Task 5: Wire durable exclusion and migration into `sync-dir`

**Files:**
- Modify: `src/osm_polygon_wikidata_only/cli/run_sync.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/orchestrator.py`
- Modify: `tests/pipeline/test_sync_reconciliation.py`
- Modify: `tests/contracts/test_resumability.py`

- [ ] Write failing tests proving only successfully retired children are excluded, unretired/blocked children remain eligible, no Wikimedia calls occur for migration-only work, upload failure remains retryable, dry-run is mutation-free, and the second run emits no containment commit.
- [ ] Confirm RED failures.
- [ ] Integrate audit/staging before normal planning, enqueue migration publications before PROCESS work, finalize after confirmed upload, and filter PBFs using the durable local retirement manifest.
- [ ] Run focused tests green and refactor.

### Task 6: Public documentation and real-data guarded execution

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Create: `scripts/audit_containment.py`
- Modify: relevant documentation tests

- [ ] Add tests that README/architecture describe canonical parent retention, quarantine, atomic remote deletion, and the fail-closed blocked state without changing unrelated passages.
- [ ] Implement the minimal documentation and read-only audit script.
- [ ] Run the complete quality matrix.
- [ ] Run the audit against the external dataset twice and compare deterministic reports.
- [ ] Stage and publish only safe rules; update/reprocess Italy before retiring its children; verify the remote listing, regenerated README/maps, local quarantine, and a second no-op sync.

