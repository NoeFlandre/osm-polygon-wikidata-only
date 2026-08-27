# Grid5000 Sentence-Splitting Design

**Status:** Approved 2026-08-27

## Goal

Finish sentence splitting for the remaining V2 PBF stems on reserved
Grid5000 GPU nodes, while keeping the local Seagate data root authoritative,
publishing one verified atomic Hugging Face update after every successful GPU
job, and preserving restartability across job, network, and controller
failures.

The remaining work is currently 338 sentence-incomplete stems. Three stems
already have incomplete local checkpoint state and must be resumed rather than
treated as fresh work.

## Scope and non-goals

In scope:

- execute all remaining SaT inference on Grid5000 GPUs;
- use only `segment-any-text/sat-3l-sm`, revision `137da05`;
- preserve the existing exact 85-language routing policy: supported codes are
  split, every other code remains one explicitly marked unsplit row;
- stage bounded batches, retrieve verified outputs/checkpoints, and publish
  after each successful job;
- retain the existing dataset card and comparison map byte-for-byte during
  sentence publication;
- clean only this run's confirmed project-owned Grid5000 temporary data;
- add tests and quality gates for the new orchestration and GPU boundary.

Out of scope:

- changing the sentence model, language policy, source section tables, or
  sentence row schema;
- running inference on the Mac or on a Grid5000 frontend;
- publishing HF credentials or private local data to Grid5000;
- broad deletion of existing Grid5000 files, home directories, or shared
  storage;
- changing V1 artifacts or rebuilding the V2 card/map for each sentence batch.

## Interpretation of the attached tutorial

The attached `grid5000-agent-tutorial.md` supplies operating constraints. It
does not expand the requested scope. Its requirements are adopted here:

- Grid5000 is used only for approved research/education work;
- `usagepolicycheck -t` is run before and after job submission and at final
  shutdown;
- frontends perform only checkout, file management, submission, and
  monitoring; inference runs only on a reserved node;
- requests use the smallest suitable resource and shortest realistic
  walltime, with no duplicate or speculative reservations;
- checkpoints and provenance are recorded frequently;
- important artifacts are copied to the Seagate data root before Grid5000
  cleanup;
- credentials are supplied through environment/credential mechanisms and are
  never printed or committed;
- logs, artifacts, remote publication, cancellation, and cleanup are verified
  before completion.

The authoritative policy references are the [Grid5000 Usage
Policy](https://www.grid5000.fr/w/Grid5000:UsagePolicy), [Grid5000
Storage](https://www.grid5000.fr/w/Storage), and [Getting
Started](https://www.grid5000.fr/w/Getting_Started) pages.

## Alternatives considered

### A. Serial, size-bounded GPU jobs with a local publisher — selected

A lightweight Mac controller selects the next missing stems, transfers only
that batch to a project-owned Grid5000 staging directory, submits one OAR
job, retrieves and validates the result, and publishes it locally. Inference
and model execution happen only inside the reserved GPU job. The HF token
never enters Grid5000.

This gives the strongest credential boundary and the simplest recovery rule:
the Seagate state is authoritative, and a job is not considered complete
until its outputs and its HF commit are both verified.

### B. Persistent dataset staging on Grid5000

Keeping all section inputs on a persistent site disk would reduce repeated
transfers, but would require a disk reservation, larger site-local storage,
and more complex ownership and cleanup handling. It is unnecessary because
the remaining section inputs are about 6.72 GiB in total and can be staged in
bounded batches.

### C. Direct HF publication from GPU jobs

Giving jobs the HF write token would reduce result transfer, but would put a
credential on Grid5000 and make publication recovery depend on remote state.
It is rejected in favor of local publication after verified retrieval.

## Architecture

### Local controller

The controller is a lightweight process that performs no sentence inference.
It owns a run ledger under the Seagate data root and processes one batch at a
time. It:

1. loads the finalized V2 manifest and sentence manifest;
2. skips stems whose required Wikipedia output and optional Wikivoyage output
   are already complete and hash-valid;
3. selects a deterministic batch of at most four stems and at most 256 MiB of
   section input;
4. records source hashes, code commit, model revision, batch parameters, and
   a unique run/job staging path;
5. transfers the batch and relevant partial checkpoints to the site frontend;
6. runs policy and duplicate-job checks, then submits one OAR GPU job;
7. retrieves the job receipt, outputs, and verified checkpoint state;
8. atomically installs the local result, merges the sentence manifest, and
   validates supported/unsupported language accounting;
9. publishes the batch's sentence files, merged manifest, and unchanged
   README in one HF commit;
10. verifies the remote commit and exact hashes, then removes only the
    confirmed project-owned remote staging directory.

If the controller stops, the ledger and local checkpoints identify the exact
batch and job state. Restart first reconciles any submitted job and remote
publication before submitting another job, so it cannot duplicate work.

### Grid5000 job

The job entry point runs on a reserved node only. It receives a pinned source
tree, selected section Parquets, the selected checkpoint directories, a
minimal V2 manifest context, and a shared per-run model cache. It:

- verifies the staged input hashes and source contract;
- verifies `nvidia-smi` and the ONNX Runtime CUDA provider;
- constructs `SaT3lSegmenter` with GPU required and no silent CPU fallback;
- runs the existing resumable sentence runner for the selected stems;
- writes a receipt containing provenance, provider, timing, row counts, and
  output/checkpoint hashes;
- leaves its project-owned result directory available for retrieval even if
  the job exits unsuccessfully, so completed checkpoint batches can be
  salvaged and retried.

No job code receives `HF_TOKEN`.

### Resource policy

The initial request is one GPU on one host with a 30-minute walltime in the
explicit `besteffort` queue, expressed as
`-q besteffort -l host=1/gpu=1,walltime=0:30`. The controller will
not assume that a particular GPU model is available; it will inspect the
selected site immediately before submission and fail closed if a CUDA-capable
GPU cannot be reserved. Batch size may be reduced only based on recorded
runtime or memory evidence.

Only one project job may be active at a time. This avoids speculative
reservations, prevents duplicate batch publication, and keeps the remote
model cache single-writer. Frontend commands remain limited to lightweight
SSH, OAR, transfer, and monitoring operations.

## GPU execution contract

The existing local behavior remains unchanged: the Mac may continue to use
CoreML or CPU through the default `SaT3lSegmenter` path. A separate pinned
GPU extra installs `wtpsplit[onnx-gpu]==2.2.1`.

The adapter gains an explicit GPU-required mode. In that mode it must:

- require `CUDAExecutionProvider` in the installed ONNX Runtime providers;
- pass `CUDAExecutionProvider` before any permitted fallback provider;
- record the effective provider set in the job receipt;
- raise before processing if CUDA is absent.

This prevents a successful-looking CPU run from being counted as Grid5000 GPU
work. The model's official ONNX documentation describes CUDA provider usage
and the `onnx-gpu` extra: [wtpsplit ONNX
documentation](https://github.com/segment-any-text/wtpsplit).

## State and transfer contract

The Seagate data root remains the source of truth. The local run ledger is an
atomic JSON document containing:

- contract version and immutable source Git commit;
- model ID, model revision, package/runtime versions, and requested provider;
- Grid5000 site, batch plan, OAR job ID, job state, and retry count;
- source file hashes and byte sizes;
- result hashes, checkpoint progress, row counts, language routing counts,
  timestamps, and the HF commit SHA.

Every transfer has a file manifest with SHA-256 hashes. A result is accepted
only when the receipt, source hashes, Parquet schemas, checkpoint identities,
and output hashes agree. Local replacement uses temporary files and atomic
renames. Existing complete outputs are never overwritten with a different
hash.

Partial checkpoint batches are valid resumable state; an incomplete metadata
record or an invalid/corrupt batch is never imported. A failed job therefore
causes the same batch to be retried from the last verified local checkpoint,
without changing the source sections.

## Hugging Face publication invariant

After every successful GPU job, one atomic commit contains:

- the newly completed Wikipedia and available Wikivoyage sentence Parquets;
- the merged `manifests/sentence_splitting.json`;
- the existing `README.md` snapshot.

The controller must not call `write_v2_card` during this loop. Before every
publication it verifies that the local README and
`assets/v2_added_wikipedia_tag_documents.png` match the known-good baseline;
after publication it downloads/inspects the remote files and verifies exact
hash equality. A publication failure leaves the verified local batch and
ledger state intact as `ready_to_publish`; restart retries publication before
starting another GPU job.

## Cleanup and failure handling

All remote temporary data lives below an explicit run namespace such as
`$HOME/osm-polygon-wikidata-only-grid5000/<run-id>/`. The controller validates
that every cleanup target resolves below that namespace before removal.

After result retrieval and HF verification, it copies receipts and logs to the
Seagate data root, then deletes only the completed job directory. The shared
model cache and any unresolved current job directory remain until the
controller has safely reconciled them or the full run has completed. At full
completion, the controller removes the remaining run-owned site cache and
verifies that no project-owned files remain. It never removes unrelated home,
shared, or site files.

Interrupts are handled at job boundaries: the controller stops submitting,
collects any available verified checkpoint state, releases its local lock,
and leaves the ledger ready to resume. OAR jobs are cancelled when they are
known to be stale or after final completion; active jobs are never duplicated
just because the controller was disconnected.

## Testing and quality gates

Implementation follows strict RED→GREEN→REFACTOR cycles. Tests are written
and observed failing before production changes for:

- GPU-provider requirement and fail-closed behavior;
- deterministic size-bounded batch planning;
- input/result manifest and receipt validation;
- partial checkpoint import and restart transitions;
- duplicate-job prevention and ledger recovery;
- safe cleanup path validation;
- unchanged README/map publication and atomic HF operation construction.

The existing sentence behavior tests remain the regression suite. The quality
configuration will include new pure helpers in the relevant CRAP scope and
mutation selection. Completion requires:

- all targeted and full regression tests passing;
- Ruff, formatting, and type checks passing;
- CRAP strictly below 6 for the changed pure-helper scope;
- every reported mutant killed, with no survivor, timeout, or untested mutant;
- a fresh HF verification of every published batch;
- a final Grid5000 policy check, job cancellation check, and scoped cleanup
  audit.

## Acceptance criteria

The work is complete only when:

1. every one of the 386 finalized V2 stems has valid sentence outputs for its
   Wikipedia table and every available Wikivoyage table;
2. every new sentence output has a receipt proving the pinned model revision,
   CUDA provider, source hash, row accounting, and Grid5000 OAR job ID;
3. each completed job has its own verified HF commit and the final remote
   manifest contains all local completed regions;
4. the README and comparison map remain byte-identical to the approved
   baseline unless separately approved;
5. unsupported languages remain explicitly unsplit and documented;
6. the local ledger/checkpoints are restartable and no duplicate job was
   submitted;
7. the requested TDD, CRAP, mutation, policy, publication, and cleanup gates
   have fresh passing evidence; and
8. only confirmed project-owned Grid5000 temporary files have been removed.
