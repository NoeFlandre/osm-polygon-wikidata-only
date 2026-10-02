# Grid5000 sentence splitting

This guide describes the production path that you can resume. It creates the remaining V2 sentence sidecars. The external data root stays authoritative. The source checkout goes only into a short-lived Grid5000 job directory. The controller runs locally. The model loading and the inference run only inside a reserved GPU job.

## Fixed execution contract

Each job uses `SaT-3l-sm` at revision `137da05` through `wtpsplit[onnx-gpu]==2.2.1`. It also uses the pinned ONNX Runtime CUDA and cuDNN runtime wheels. The compute entry point needs an active `CUDAExecutionProvider`. It fails closed when CUDA is not available. It does not accept a session that changes silently to the CPU. It records these items in a receipt:

- The GPU name, UUID, and memory.
- The ONNX Runtime providers.
- The source commit and the model revision.
- The selected stems and the row counts.
- The artifact hashes.

The V2 language boundary does not change. The command sends to SaT only the exact codes in the [sentence-splitting language list](sentence-splitting.md). An unsupported language stays as one unsplit row. The row has `segmentation_status=unsupported_language`. The command never sends it to SaT. The sentence manifest records the unsupported codes that the command observed. A published result thus shows this boundary.

## Local controller and GPU job

The Mac-side controller owns these items: the data root, the durable resume ledger `grid5000_sentence_run.json`, the Hugging Face authentication, and the publication. The HF token stays local. Never put it in the staged tree, in the remote command, in the job arguments, or in the receipt. The reserved job does not publish. It only writes its result tree and its receipt.

The default run is bounded and serial on purpose:

- Each job has four region stems or 256 MiB of section input at most.
- The controller makes one `host=1/gpu=1` reservation at a time.
- The default walltime is `0:30`.
- The controller submits, retrieves, verifies, and publishes one short job. Then it submits the next job.

To start or resume the operation, run this command from the repository checkout:

```bash
UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run osm-polygon-wikidata-only grid5000 controller \
    --data-root "$OSM_POLYGON_DATA_ROOT" \
    --site rennes \
    --queue besteffort \
    --gpu-model A40 \
    --repo-id NoeFlandre/osm-polygon-wikidata-and-wikipedia \
    --max-stems 4 \
    --max-input-bytes 268435456 \
    --batch-size 256 \
    --inference-batch-size 16 \
    --walltime 0:30
```

The compute-node entry point is separate. The controller starts it inside the reservation. The controller stages and runs the equivalent compatibility shim `scripts/grid5000_sentence_job.py`:

```bash
uv run --no-sync osm-polygon-wikidata-only grid5000 job \
    --data-root /path/to/result/data \
    --stems REGION_STEM \
    --model-cache /path/to/run/model-cache \
    --source-commit GIT_COMMIT \
    --job-id "$OAR_JOB_ID" \
    --batch-size 256 \
    --inference-batch-size 16 \
    --receipt /path/to/result/receipt.json
```

The controller stages only these items:

- The selected section tables and manifests.
- The selected checkpoint trees.
- The pinned source files.
- The package-forced assets.
- The job metadata.

It does not stage the full dataset. It does not stage the raw PBF collection.

The reserved node downloads the model and the packages. The jobs reuse a model cache and a uv cache that belong to the run. The compute-node image does not guarantee a system `uv`. The first job thus creates a bootstrap environment that belongs to the run. It installs the pinned `uv==0.11.16` package into this environment. Later jobs reuse the bootstrap. After the locked environment syncs, the job adds the installed NVIDIA library directories to `LD_LIBRARY_PATH`. It does this before it constructs the SaT session.

## Policy, resume, and publication sequence

The controller gets the local lock before it reads or writes the ledger. It runs `usagepolicycheck -t` before the submission and immediately after the submission. The invocation records the selected GPU model. It limits each short job to this model. For example, use the qualified Rennes A40 resources: `oarsub -q besteffort -p gpu_model='A40' -l host=1/gpu=1,walltime=0:30`. This prevents mixed GPU allocations that cannot run the locked CUDA stack.

The monitoring polls only the recorded OAR job ID. Use a different GPU model, queue, or site only after a site qualification probe. The frontend does only these operations: policy, OAR, SSH/rsync, monitoring, and scoped file management. These operations happen inside the reservation: `uv sync`, `nvidia-smi`, model loading, and sentence inference.

For each successful GPU job, the controller does these steps:

1. It downloads the receipt, the sentence Parquets, the merged sentence manifest, and the checkpoint state. It does this before it removes anything remotely.
2. It validates these items locally: the receipt identity, the schemas, the SHA-256 hashes, the manifest invariants, and the checkpoint identity.
3. It installs the verified output files and checkpoint files atomically.
4. It publishes the selected sentence sidecars, the merged manifest, and the unchanged README in one Hugging Face commit.
5. It verifies that the remote files exist and have the exact hashes. Then it records the HF commit in the ledger.
6. It removes only the directory of that batch below the namespace of the run.

The controller never regenerates or uploads the dataset card during the sentence publication. An older card thus cannot replace the protected README and the comparison-map bytes silently. The controller records their baseline hashes when it creates the run. Any change blocks the publication until the operator reviews it.

If a job fails, the controller imports the valid partial checkpoints. The batch stays retryable. The controller does not publish incomplete sentence outputs. If a job ends with no valid receipt, the controller records a retryable failure. It cleans the directory of the run. The ledger then does not show the false state `running`. If the publication fails, the batch stays `ready_to_publish`. After a restart, the controller retries the local HF publication. It does not submit another GPU job. `Ctrl-C` records the current state. It cancels only a known active OAR job. It releases the local lock.

The ledger is the source of truth for a resume. If you omit `--run-id`, the controller resumes the existing ledger. After the first successful publication, the immutable fields must match. These fields are the source commit, the model revision, the site, the queue, the limits, and the protected asset hashes. An unpublished run can adopt a newer source commit in only two cases. In the first case, its batches are planned. In the second case, its batches have a terminal receipt failure on record. The controller records the change in `source_commit_updates`. A successful run does one final policy check. It removes the entire namespace of the run only after it publishes all the planned batches. The cleanup refuses paths outside this namespace.

## Verification after completion

The controller exits with success only when all these conditions are true:

- Each finalized V2 stem is sentence-complete.
- Each batch has a verified HF commit.
- The final policy check and the cleanup checks succeed.

Keep the ledger and the receipts with the external data root for provenance. The checks below are local. They do not submit another job:

```bash
just test
just ruff
just ty
just crap-all
just mutation
```

The pure CRAP scope and the pure mutation scope include the `sentence_protocol.py` helpers. Focused fake-boundary tests cover the SSH and OAR transitions. The mutation tests do not mutate them, because they are not pure functions.
