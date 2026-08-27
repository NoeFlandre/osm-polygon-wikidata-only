# Grid5000 sentence splitting

This guide describes the resumable production path for the remaining V2
sentence sidecars. The external data root remains authoritative; the source
checkout is only staged into a short-lived Grid5000 job directory. The
controller runs locally, while model loading and inference run only inside a
reserved GPU job.

## Fixed execution contract

Every job uses `SaT-3l-sm` at revision `137da05` through
`wtpsplit[onnx-gpu]==2.2.1`, plus the pinned ONNX Runtime CUDA/cuDNN runtime
wheels. The compute entry point requires an active
`CUDAExecutionProvider` and fails closed when CUDA is not available; it does
not accept a session that silently falls back to CPU. It also
records the GPU name, UUID, memory, ONNX Runtime providers, source commit,
model revision, selected stems, row counts, and artifact hashes in a receipt.

The existing V2 language boundary remains unchanged. Only the exact codes in
the [sentence-splitting language list](sentence-splitting.md) are sent to SaT.
An unsupported language remains one unsplit row with
`segmentation_status=unsupported_language`; it is never sent to SaT. The
sentence manifest records the observed unsupported codes, so a published
result makes this boundary explicit.

## Local controller and GPU job

The Mac-side controller owns the data root, the durable resume ledger
`grid5000_sentence_run.json`, Hugging Face authentication, and publication.
The HF token stays local and is never placed in the staged tree, remote
command, job arguments, or receipt. The reserved job has no publication
responsibility and only writes its result tree and receipt.

The default run is intentionally bounded and serial:

- at most four region stems or 256 MiB of section input per job;
- one `host=1/gpu=1` reservation at a time;
- a default `0:30` walltime;
- one short job submitted, retrieved, verified, and published before the next
  job is submitted.

From the repository checkout, start or resume the operation with:

```bash
UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run python scripts/grid5000_sentence_controller.py \
    --data-root "$OSM_POLYGON_DATA_ROOT" \
    --site grenoble \
    --queue besteffort \
    --repo-id NoeFlandre/osm-polygon-wikidata-and-wikipedia \
    --max-stems 4 \
    --max-input-bytes 268435456 \
    --batch-size 256 \
    --inference-batch-size 16 \
    --walltime 0:30
```

The compute-node entry point is separate and is intended to be invoked by
the controller inside the reservation:

```bash
uv run --no-sync python scripts/grid5000_sentence_job.py \
    --data-root /path/to/result/data \
    --stems REGION_STEM \
    --model-cache /path/to/run/model-cache \
    --source-commit GIT_COMMIT \
    --job-id "$OAR_JOB_ID" \
    --batch-size 256 \
    --inference-batch-size 16 \
    --receipt /path/to/result/receipt.json
```

The controller stages only the selected section tables, manifests, selected
checkpoint trees, pinned source files, package-forced assets, and job metadata. It does not stage
the full dataset or raw PBF collection. The model and package download are
performed on the reserved node, with a run-owned model and uv cache reused by
later jobs. Because the compute-node image does not guarantee a system `uv`,
the first job creates a run-owned bootstrap environment and installs the
the first job creates a run-owned bootstrap environment and installs the
pinned `uv==0.11.16` package into it; later jobs reuse that bootstrap. After
the locked environment is synced, the job adds the installed NVIDIA library
directories to `LD_LIBRARY_PATH` before constructing the SaT session.

## Policy, resume, and publication sequence

The controller acquires the local lock before reading or writing the ledger.
It runs `usagepolicycheck -t` before submission and immediately after
submission. OAR requests use the explicitly recorded `besteffort` queue and
exactly `oarsub -q besteffort -l host=1/gpu=1,walltime=0:30`; monitoring polls
only the recorded OAR job ID. A different queue may be supplied only after a
site qualification probe. The frontend performs only policy, OAR, SSH/rsync,
monitoring, and scoped file-management operations. `uv sync`, `nvidia-smi`,
model loading, and sentence inference happen inside the reservation.

For every successful GPU job, the controller:

1. downloads the receipt, sentence Parquets, merged sentence manifest, and
   checkpoint state before removing anything remotely;
2. validates receipt identity, schemas, SHA-256 hashes, manifest invariants,
   and checkpoint identity locally;
3. atomically installs verified output and checkpoint files;
4. publishes the selected sentence sidecars, merged manifest, and unchanged
   README in one Hugging Face commit;
5. verifies remote presence and exact hashes, then records the HF commit in
   the ledger;
6. removes only that batch's directory below the run-owned namespace.

The controller never regenerates or uploads the dataset card during sentence
publication, so the protected README and comparison-map bytes cannot be
silently replaced by an older card. Their baseline hashes are recorded when
the run is created and any change blocks publication for operator review.

If a job fails, valid partial checkpoints are imported and the batch remains
retryable without publishing incomplete sentence outputs. A terminal job with
no valid receipt is recorded as a retryable failure and its run-owned
directory is cleaned, rather than leaving the ledger falsely `running`. If
publication fails, the batch remains `ready_to_publish`; a restart retries the
local HF publication without submitting another GPU job. `Ctrl-C` records the
current state, cancels only a known active OAR job, and releases the local
lock.

The ledger is the source of resume truth. An omitted `--run-id` resumes the
existing ledger, while immutable fields such as source commit, model
revision, site, queue, limits, and protected asset hashes must match after the
first successful publication. An unpublished run may adopt a newer source
commit only when its batches are planned or have a recorded terminal receipt
failure, with the change recorded in `source_commit_updates`. A successful run
performs one final policy check and removes the entire run namespace only
after all planned batches are published. Cleanup rejects paths outside that
namespace.

## Verification after completion

The controller exits successfully only when every finalized V2 stem is
sentence-complete, every batch has a verified HF commit, and the final policy
and cleanup checks have succeeded. Keep the ledger and receipts with the
external data root for provenance. The following checks are local and do not
submit another job:

```bash
just test
just ruff
just ty
just crap-all
just mutation
```

The `sentence_protocol.py` helpers are included in the pure CRAP and mutation
scopes. SSH/OAR transitions remain covered by focused fake-boundary tests;
they are intentionally not mutated as if they were pure functions.
