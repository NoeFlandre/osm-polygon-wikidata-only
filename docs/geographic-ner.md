# Geographic name extraction

This optional, additive stage extracts only named geographic proper names from
already split Wikipedia and Wikivoyage sentences. It does not classify
landuse, judge sentence relevance, geocode a mention, or replace the original
source text.

The pilot uses `whoisjones/otter-cross-mmbert`, pinned at
`8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d`, with the single label
`named geographic location` and an uncalibrated score threshold of `0.5`.
Long sentences use overlapping token windows; character offsets always refer
to the unchanged original sentence. A place mention is not evidence that the
place is the polygon linked to its document.

Each output row keeps the original join keys: `sentence_id`, `document_id`,
`project`, and `language`. The `entities` sidecar field contains validated
character offsets and surfaces for the original sentence, together with the
pinned model label and score. Join sidecars back to the source by those keys;
do not join on a row number or on the extracted text alone.

## Statuses and evidence

Row statuses are explicit:

- `ok` — a split, supported-language, non-empty sentence was inferred;
- `empty_text` — the sentence is split and supported but contains no text;
- `skipped_language` — its language is outside the contract; and
- `skipped_unsplit` — sentence segmentation is not marked `split`.

The receipt status is `paused` or `completed`. Failures during shard execution
are recorded as `failed` when a receipt exists; earlier startup failures are
reported by the worker's exit status. `validation_status` is
`pilot_unvalidated` by default and is not a calibration or quality claim.
The pilot must not report precision, recall, or validated-language coverage
until a human-labelled geographic-name gold set exists.

## Automatic silver validation

When a human-labelled set is unavailable, an additive silver check compares the
Otter output with a second NER model and deterministic references. It counts
primary spans matching successful, non-empty document titles, labels, or
aliases and linked OSM polygon names; separately counts obvious artifacts such
as QIDs, URLs, and numeric strings; and reports second-model agreement,
per-language and per-project counts, unique linked documents, and
unique/entity-bearing OSM polygons. These are quality proxies, not precision or
recall.

The cross-check uses the WikiNEuRal model
`Babelscape/wikineural-multilingual-ner` at pinned revision
`89ab4613336445bde46866ddc825561fe69e6e6c`. Its contract covers the six
languages overlapping this pilot (`de`, `en`, `es`, `fr`, `pt`, `ru`); the other
pilot languages remain explicitly outside the cross-check contract.

Prepare a cross-check from the selected primary pilot, stage it, and run it
through the same resumable short-job controller:

```sh
export NER_CROSSCHECK_DIR="$NER_DATA_ROOT/geographic-ner/crosscheck-20260908"
export NER_CROSSCHECK_STAGING="$NER_DATA_ROOT/geographic-ner/staging/geographic-ner-crosscheck-20260908"
export NER_CROSSCHECK_RUN_DIR="$NER_DATA_ROOT/geographic-ner/runs/geographic-ner-crosscheck-20260908"

.venv/bin/python scripts/prepare_geographic_ner_crosscheck.py \
  --pilot-dir "$NER_PILOT_DIR" \
  --output-dir "$NER_CROSSCHECK_DIR"
.venv/bin/python scripts/grid5000_geographic_ner.py \
  --staging-dir "$NER_CROSSCHECK_STAGING" \
  --source "$NER_CROSSCHECK_DIR/input.parquet" \
  --contract "$NER_CROSSCHECK_DIR/contract.json" \
  --source-root "$PWD" \
  --requirements-lock "$PWD/requirements/geographic-ner-gpu.txt" \
  --prepare-only
.venv/bin/python scripts/grid5000_geographic_ner.py \
  --staging-dir "$NER_CROSSCHECK_STAGING" \
  --run-dir "$NER_CROSSCHECK_RUN_DIR" \
  --run-id geographic-ner-crosscheck-20260908 \
  --site rennes --queue besteffort --gpu-model A40 --period day \
  --repo-id "$NER_REPO_ID"
```

Evaluate the completed primary and cross-check outputs locally; this writes no
Hub files:

```sh
.venv/bin/python scripts/evaluate_geographic_ner_pilot.py \
  --data-root "$NER_DATA_ROOT" \
  --pilot-dir "$NER_PILOT_DIR" \
  --primary-output "$NER_RUN_DIR/output" \
  --secondary-output "$NER_CROSSCHECK_RUN_DIR/output" \
  --output "$NER_CROSSCHECK_RUN_DIR/silver-validation.json"
```

Every run binds its contract, model revision, source SHA-256, and batch size in
`receipt.json`. Completed batches are separate `batch-XXXXXX.parquet` sidecars,
each recorded with a SHA-256 and row count. `predictions.sqlite3` is a local
deduplication cache; it does not replace the source joins or the receipt.

## Local readiness and dependency preflight

Run these commands from the repository root. They use the existing `.venv` and
do not resync dependencies, contact Grid5000, download a model, or publish to
the Hub:

```sh
.venv/bin/python - <<'PY'
import huggingface_hub
import osmium
import pyarrow

from osm_polygon_wikidata_only.io import atomic, hashing, pbf_reader
from osm_polygon_wikidata_only.ner import pipeline

print("runtime imports: ok")
PY
.venv/bin/python scripts/grid5000_geographic_ner.py --help
.venv/bin/pytest -q tests/ner tests/grid5000/test_ner_controller.py
.venv/bin/ruff check src/osm_polygon_wikidata_only/ner src/osm_polygon_wikidata_only/grid5000/ner_controller.py scripts tests/ner tests/grid5000/test_ner_controller.py
.venv/bin/ruff format --check src/osm_polygon_wikidata_only/ner src/osm_polygon_wikidata_only/grid5000/ner_controller.py scripts tests/ner tests/grid5000/test_ner_controller.py
.venv/bin/ty check src scripts
```

`pyarrow`, `huggingface-hub`, and `osmium` are normal runtime dependencies in
`pyproject.toml` and `uv.lock`. CUDA-only `torch` and `transformers` are kept in
the hashed `requirements/geographic-ner-gpu.txt` lock for the reserved worker;
they are not required for the offline contract tests.

### Mutation review policy

`just mutation` requires zero **unreviewed** survivors and rejects timeouts,
untested mutants, and empty reports. Equivalent mutations remain visible in
the results and are reported separately from kills. Each exception in
`quality/mutation-equivalents.json` records its exact mutant name, source-file
and generated-function SHA-256 hashes, and a written equivalence argument.
Changed code, changed mutants, or stale exceptions fail the gate and require
a new review; there are no wildcard or whole-function exclusions.

Examples include codec-name aliases, runtime-erased `typing.cast` arguments,
and redundant length checks. Default-encoding changes for Unicode metadata
and altered ledger keys are **not** exempt: regression tests cover them.
Passing offline checks establishes implementation readiness, not measured
model accuracy or a successful live GPU run. Review the unvalidated pilot
before authorizing full-corpus inference.

## Preparation

The deterministic sampler reads the external `processed_v2` tree and writes a
link-backed pilot. Keep all data and run state on the external data root, not in
the repository:

```sh
export NER_DATA_ROOT="/path/to/external/ner-data"
export NER_PILOT_DIR="$NER_DATA_ROOT/geographic-ner/pilot-20260907"
export NER_RUN_ID="geographic-ner-pilot-YYYYMMDD"
export NER_STAGING_DIR="$NER_DATA_ROOT/geographic-ner/staging/$NER_RUN_ID"
export NER_RUN_DIR="$NER_DATA_ROOT/geographic-ner/runs/$NER_RUN_ID"
export NER_REPO_ID="owner/dataset"

.venv/bin/python scripts/prepare_geographic_ner_pilot.py \
  --data-root "$NER_DATA_ROOT" \
  --output-dir "$NER_PILOT_DIR" \
  --sample-size 1000 \
  --seed geographic-ner-pilot-v1
.venv/bin/python scripts/grid5000_geographic_ner.py \
  --staging-dir "$NER_STAGING_DIR" \
  --source "$NER_PILOT_DIR/input.parquet" \
  --contract "$NER_PILOT_DIR/contract.json" \
  --source-root "$PWD" \
  --requirements-lock "$PWD/requirements/geographic-ner-gpu.txt" \
  --prepare-only
```

The sampler writes deterministic rows, the unvalidated contract, and its
selection manifest. The following `--prepare-only` staging command then
validates required input columns, non-empty Parquet, the contract, the complete
source tree, and hashes on every GPU-lock package. It refuses to overwrite an
existing staging directory.

Replace the example data root, date, and repository ID before preparation.
The sampler requests ten pilot languages (`ar`, `de`, `en`, `es`, `fr`, `hy`,
`ja`, `pt`, `ru`, `zh`); this is a sampling choice, not validated model coverage.

Before any future launch, verify the immutable staging result:

```sh
test -s "$NER_STAGING_DIR/input.parquet"
test -s "$NER_STAGING_DIR/contract.json"
test -d "$NER_STAGING_DIR/code"
.venv/bin/python - "$NER_STAGING_DIR/contract.json" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if contract.get("validation_status") != "pilot_unvalidated":
    raise SystemExit("pilot contract must remain pilot_unvalidated")
print("pilot contract: unvalidated by design")
PY
```

## Future short-job invocation

The controller requests one serial, bounded twenty-minute allocation: `host=1`,
`gpu=1`, `walltime=0:20`, batch size `128`, and inference batch size `16`.
Rennes, the `besteffort` queue, and A40 are candidate settings only. Recheck
account entitlement and current usage-policy availability immediately before
launch; this repository does not establish that the site is eligible.

After preparation and preflight pass, the future launch command is:

```sh
.venv/bin/python scripts/grid5000_geographic_ner.py \
  --staging-dir "$NER_STAGING_DIR" \
  --run-dir "$NER_RUN_DIR" \
  --run-id "$NER_RUN_ID" \
  --site rennes \
  --queue besteffort \
  --gpu-model A40 \
  --period day \
  --repo-id "$NER_REPO_ID"
```

This submits a live job and is intentionally not run in the current readiness
pass. The node-local worker contract is explicitly bounded as follows, but it
must only be run inside the reserved CUDA job:

```sh
python -m osm_polygon_wikidata_only.ner.job \
  --source input.parquet \
  --output-dir output \
  --contract contract.json \
  --model-cache shared-cache \
  --seconds 60 \
  --batch-size 128 \
  --inference-batch-size 16
```

It rejects a non-OAR or non-CUDA environment before model download.
The dependency environment is reused by lock-file hash, but marked ready only
after a successful installation; an interrupted installation is retried.

## Safe resume

Resume by rerunning the same controller command, without `--publish`, with the
same staging directory, run directory, run ID, and repository ID:

```sh
.venv/bin/python scripts/grid5000_geographic_ner.py \
  --staging-dir "$NER_STAGING_DIR" \
  --run-dir "$NER_RUN_DIR" \
  --run-id "$NER_RUN_ID" \
  --site rennes \
  --queue besteffort \
  --gpu-model A40 \
  --period day \
  --repo-id "$NER_REPO_ID"
```

`paused` resubmits without re-uploading the immutable staging tree. Output is
retained after a failed or incomplete job. `submitting` and `failed` states
stop with an operator-recovery error; do not blindly submit a second job.

## Per-job verified publication

Publication is a separate, explicit action. Use it only after the ledger is
`completed` or `ready_to_publish` and the local receipt and artifact hashes have
been reviewed:

```sh
.venv/bin/python - "$NER_RUN_DIR/ledger.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["state"]
if state not in {"completed", "ready_to_publish"}:
    raise SystemExit(f"refusing publication from ledger state: {state}")
print(f"verified local publication boundary: {state}")
PY
.venv/bin/python scripts/grid5000_geographic_ner.py \
  --staging-dir "$NER_STAGING_DIR" \
  --run-dir "$NER_RUN_DIR" \
  --run-id "$NER_RUN_ID" \
  --site rennes \
  --queue besteffort \
  --gpu-model A40 \
  --period day \
  --repo-id "$NER_REPO_ID" \
  --publish
```

The controller validates the receipt, contiguous batch names, source and
contract identities, row counts, and hashes again immediately before calling
the receipt-bound publisher. The publisher verifies the remote namespace and
SHA-256 values after the commit. This command contacts the Hub and is not run
under the current no-publication instruction.

## Full-corpus guard

The pilot contract is deliberately `pilot_unvalidated`. Do not launch a full
corpus from that contract or report precision, recall, or language coverage.
The exact guard for any future full-corpus wrapper is:

```sh
.venv/bin/python - "$NER_STAGING_DIR/contract.json" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("validation_status")
if status != "validated":
    raise SystemExit("refusing full-corpus execution: validation_status must be validated")
print("validated contract: full-corpus execution may be considered")
PY
```

Changing the field without a human-labelled geographic-name gold set does not
constitute validation.

## Contract and quality gates

The offline contract tests exercise the resumable pipeline, publication
boundary, sampler, and Grid5000 controller without downloading a model or
requiring CUDA. The repository quality gate includes all NER implementation,
controller, CLI, and sampler paths in mutation selection and in a dedicated
CRAP score below 6:

```sh
.venv/bin/pytest -q tests/ner tests/grid5000/test_ner_controller.py
just crap-ner
just mutation
```

These checks do not establish model precision, recall, validated-language
coverage, Grid5000 entitlement, live job success, or remote publication.
