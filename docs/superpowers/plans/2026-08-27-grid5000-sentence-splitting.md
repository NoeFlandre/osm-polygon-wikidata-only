# Grid5000 Sentence Splitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run all remaining V2 sentence splitting on reserved Grid5000 GPUs, publish one verified atomic Hugging Face commit after every successful GPU job, and preserve restartability and policy-compliant cleanup.

**Architecture:** Keep the Seagate data root authoritative and add a lightweight local controller that plans bounded batches, submits at most one OAR GPU job, retrieves verified artifacts, publishes locally, and records every transition in an atomic ledger. A separate compute-node entry point runs the existing sentence runner with a CUDA-required SaT adapter, emits a provenance receipt, and leaves checkpoint state available for retrieval. Remote staging is limited to a run-owned site namespace and is removed only after local/HF verification.

**Tech Stack:** Python 3.12, `uv`, `wtpsplit[onnx-gpu]==2.2.1`, ONNX Runtime CUDA, PyArrow, pytest/pytest-cov, Ruff, ty, crap4py, mutmut, SSH/rsync, Grid5000 OAR, Hugging Face Hub uploader.

---

## Implementation boundaries

| Boundary | Responsibility | Must not do |
| --- | --- | --- |
| `v2/sat.py` | Select and enforce the requested ONNX provider | Change default Mac CoreML/CPU behavior |
| `grid5000/sentence_protocol.py` | Pure batch, receipt, manifest, and path contracts | Open SSH, submit jobs, or upload to HF |
| `grid5000/sentence_job.py` | Compute-node preflight and sentence execution | Read the HF token or publish remotely |
| `grid5000/sentence_controller.py` | Local ledger, transfer, OAR lifecycle, import, publish, cleanup | Run sentence inference locally |
| `scripts/grid5000_sentence_job.py` | Thin CLI for the reserved node | Contain orchestration policy |
| `scripts/grid5000_sentence_controller.py` | Thin CLI for the Mac controller | Implement duplicate logic hidden from tests |

The existing `run_v2_sentence_split` remains the only sentence-row materializer. The Grid5000 job invokes it with selected stems and a GPU-required `SaT3lSegmenter`; the controller never calls it.

### Task 1: Lock the GPU runtime contract with RED→GREEN tests

**Files:**
- Modify: `tests/v2/test_sat.py`
- Modify: `src/osm_polygon_wikidata_only/v2/sat.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add failing provider-enforcement tests.**

  Add tests to `tests/v2/test_sat.py` using the existing fake `wtpsplit` and `onnxruntime` modules:

  ```python
  def test_gpu_mode_requires_cuda_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
      monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())
      ort = ModuleType("onnxruntime")
      ort.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
      monkeypatch.setitem(sys.modules, "onnxruntime", ort)

      with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
          SaT3lSegmenter(cache_dir=tmp_path, revision="model-revision", require_gpu=True)


  def test_gpu_mode_passes_cuda_before_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
      monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())
      ort = ModuleType("onnxruntime")
      ort.get_available_providers = lambda: [  # type: ignore[attr-defined]
          "CUDAExecutionProvider",
          "CPUExecutionProvider",
      ]
      monkeypatch.setitem(sys.modules, "onnxruntime", ort)

      segmenter = SaT3lSegmenter(cache_dir=tmp_path, revision="model-revision", require_gpu=True)

      assert segmenter.ort_providers == (
          "CUDAExecutionProvider",
          "CPUExecutionProvider",
      )
      assert _FakeSaT.init_kwargs["ort_providers"] == [
          "CUDAExecutionProvider",
          "CPUExecutionProvider",
      ]
  ```

- [ ] **Step 2: Run the focused tests and verify the correct RED failure.**

  Run:

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/v2/test_sat.py
  ```

  Expected: the new tests fail because `SaT3lSegmenter` has no `require_gpu` mode or `ort_providers` property; existing tests continue to collect.

- [ ] **Step 3: Implement the minimal GPU-required adapter.**

  Extend `SaT3lSegmenter.__init__` with `require_gpu: bool = False`, resolve providers through `_resolve_ort_providers(override, require_gpu)`, store `self.ort_providers` as a tuple, and fail with an actionable `RuntimeError` when `require_gpu` is true and `CUDAExecutionProvider` is absent. Preserve the existing default resolution exactly: CoreML then CPU on Apple Silicon, CPU elsewhere. Keep the `wtpsplit` optional-dependency error unchanged except for allowing the GPU extra in its message.

  Use this provider contract:

  ```python
  def _resolve_ort_providers(
      override: Sequence[str] | None,
      *,
      require_gpu: bool = False,
  ) -> list[str]:
      if override is not None:
          providers = list(override)
          if not providers:
              raise ValueError("ort_providers must not be empty")
      else:
          providers = _default_ort_providers()
      if require_gpu and _CUDA_PROVIDER not in providers:
          raise RuntimeError("Grid5000 sentence splitting requires CUDAExecutionProvider")
      return providers
  ```

  The GPU job will pass `("CUDAExecutionProvider", "CPUExecutionProvider")`; this makes provider order explicit while the preflight rejects a machine without CUDA.

- [ ] **Step 4: Run the focused tests and verify GREEN.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/v2/test_sat.py
  ```

  Expected: all tests in `tests/v2/test_sat.py` pass, including both new GPU tests and the existing CoreML/CPU contracts.

- [ ] **Step 5: Add the pinned GPU optional dependency and refresh the lock.**

  Add this separate optional extra to `pyproject.toml` without changing the existing CPU extra:

  ```toml
  sentence-splitting-gpu = ["wtpsplit[onnx-gpu]==2.2.1"]
  ```

  Run:

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv lock
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv lock --check
  ```

  Expected: the lock resolves `onnxruntime-gpu` for the new extra and the lock check exits 0. Do not install the GPU extra on the Mac.

- [ ] **Step 6: Commit the isolated runtime boundary.**

  ```bash
  git add src/osm_polygon_wikidata_only/v2/sat.py tests/v2/test_sat.py pyproject.toml uv.lock
  git diff --cached --check
  git commit -m "feat: enforce CUDA sentence splitting mode"
  ```

### Task 2: Add deterministic batch and artifact contracts

**Files:**
- Create: `src/osm_polygon_wikidata_only/grid5000/__init__.py`
- Create: `src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py`
- Create: `tests/grid5000/__init__.py`
- Create: `tests/grid5000/test_sentence_protocol.py`

- [ ] **Step 1: Write failing tests for batch planning and safe namespaces.**

  Create a test fixture with four finalized stems, section Parquets with known byte sizes, one optional Wikivoyage table, and a sentence manifest marking one stem complete. Add tests asserting:

  - completed stems are excluded;
  - stems are sorted deterministically;
  - each batch has at most four stems;
  - each batch stays at or below the byte cap unless a single stem itself exceeds the cap;
  - a Wikivoyage section file contributes to the batch size;
  - a run ID accepts only lowercase letters, digits, `_`, and `-`;
  - a cleanup path is accepted only when it resolves below the explicit remote run root.

  Use the wished-for API:

  ```python
  batches = plan_sentence_batches(
      processed_v2,
      sentence_manifest,
      max_stems=4,
      max_input_bytes=256 * 1024 * 1024,
  )
  assert batches[0].stems == ("alpha-latest", "beta-latest")
  assert batches[0].input_bytes == expected_bytes
  assert is_safe_run_id("run-20260827-01")
  assert not is_safe_run_id("../outside")
  assert validate_cleanup_target(run_root, run_root / "jobs" / "123")
  assert not validate_cleanup_target(run_root, run_root.parent / "other")
  ```

- [ ] **Step 2: Run the protocol tests and verify RED.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_protocol.py
  ```

  Expected: collection fails because the `grid5000` package and protocol functions do not exist.

- [ ] **Step 3: Implement the pure protocol module.**

  Define:

  ```python
  GRID5000_SENTENCE_CONTRACT_VERSION = "grid5000-sentence-v1"
  DEFAULT_MAX_STEMS = 4
  DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024
  DEFAULT_WALLTIME = "0:30"

  @dataclass(frozen=True, slots=True)
  class SentenceBatch:
      index: int
      stems: tuple[str, ...]
      input_bytes: int

  @dataclass(frozen=True, slots=True)
  class FileDigest:
      relative_path: str
      size: int
      sha256: str

  def plan_sentence_batches(
      processed_v2: Path,
      sentence_manifest: Mapping[str, object] | None,
      *,
      max_stems: int = DEFAULT_MAX_STEMS,
      max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
  ) -> tuple[SentenceBatch, ...]: ...

  def sentence_source_paths(processed_v2: Path, stem: str) -> tuple[Path, ...]: ...
  def is_safe_run_id(value: str) -> bool: ...
  def validate_cleanup_target(run_root: Path, target: Path) -> bool: ...
  def sha256_manifest(paths: Sequence[Path], *, root: Path) -> tuple[FileDigest, ...]: ...
  ```

  `plan_sentence_batches` must require positive limits, use only finalized stems from `processed_v2/manifests/processed_pbfs.json`, require Wikipedia sentence completion, require Wikivoyage completion only when its source exists, and allow an oversized single stem as one batch. It must not read Parquet rows.

- [ ] **Step 4: Run protocol tests and verify GREEN.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_protocol.py
  ```

  Expected: all protocol tests pass.

- [ ] **Step 5: Run pure-helper CRAP coverage before adding more orchestration.**

  Add the protocol module to the dedicated quality scope only after its tests exist. Run the focused test/coverage command and inspect the function-level scores. Keep each changed pure helper below 6 by extracting only genuinely independent decisions and covering their error branches.

- [ ] **Step 6: Commit the deterministic contracts.**

  ```bash
  git add src/osm_polygon_wikidata_only/grid5000 tests/grid5000
  git diff --cached --check
  git commit -m "feat: add Grid5000 sentence batch contracts"
  ```

### Task 3: Add verified manifest and checkpoint import helpers

**Files:**
- Modify: `src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py`
- Create: `tests/grid5000/test_sentence_artifacts.py`

- [ ] **Step 1: Write failing artifact-validation tests.**

  Add tests for:

  - accepting a remote sentence manifest that preserves every local region and adds exactly the selected completed regions;
  - rejecting a manifest that removes a local region or changes the model/revision/policy;
  - accepting a complete sentence output only when its schema and SHA-256 match the receipt;
  - importing valid checkpoint metadata and contiguous `batch-*.parquet` files into a temporary local checkpoint directory;
  - rejecting a checkpoint with the wrong source fingerprint, wrong model revision, invalid schema, or path traversal;
  - installing files through temporary files and leaving the prior target unchanged when validation fails.

  Use these APIs:

  ```python
  def validate_manifest_extension(
      local_payload: Mapping[str, object],
      incoming_payload: Mapping[str, object],
      *,
      selected_stems: Sequence[str],
  ) -> None: ...

  def validate_sentence_output(path: Path, *, expected_sha256: str) -> None: ...

  def import_checkpoint_tree(
      incoming_root: Path,
      local_root: Path,
      *,
      expected_identity: Mapping[str, object],
  ) -> tuple[int, ...]: ...
  ```

- [ ] **Step 2: Run the artifact tests and verify RED.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_artifacts.py
  ```

  Expected: collection fails because the new validation/import functions do not exist.

- [ ] **Step 3: Implement validation and atomic installation.**

  Reuse `sentence_schema()`, `SentenceCheckpoint` identity fields, `sha256_file`, and `atomic_write_text`. Validate all incoming paths relative to the incoming root before copying. Copy each file to a temporary sibling and replace it atomically; write checkpoint metadata last. Never delete an existing local checkpoint batch during import. `validate_manifest_extension` must compare all existing local region dictionaries exactly and require the incoming selected stems to be present.

- [ ] **Step 4: Run the artifact tests and verify GREEN.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_artifacts.py
  ```

  Expected: all artifact tests pass.

- [ ] **Step 5: Commit the import boundary.**

  ```bash
  git add src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py tests/grid5000/test_sentence_artifacts.py
  git diff --cached --check
  git commit -m "feat: validate Grid5000 sentence artifacts"
  ```

### Task 4: Add the compute-node job entry point

**Files:**
- Create: `src/osm_polygon_wikidata_only/grid5000/sentence_job.py`
- Create: `scripts/grid5000_sentence_job.py`
- Create: `tests/grid5000/test_sentence_job.py`

- [ ] **Step 1: Write failing job-contract tests.**

  Test with temporary staged data and injected fakes that:

  - reject a missing `nvidia-smi` result;
  - reject an ONNX Runtime provider set without CUDA;
  - construct `SaT3lSegmenter(require_gpu=True, ort_providers=("CUDAExecutionProvider", "CPUExecutionProvider"))`;
  - pass the selected stems and configured batch sizes to `run_v2_sentence_split`;
  - write a success receipt containing source commit, model ID/revision, providers, GPU identity, selected stems, output hashes, and row counts;
  - write a failed receipt without exposing arbitrary exception payloads or environment variables;
  - never call an HF uploader.

  Use the wished-for boundary:

  ```python
  def run_sentence_job(
      data_root: DataRoot,
      *,
      stems: Sequence[str],
      model_cache: Path,
      source_commit: str,
      job_id: str,
      batch_size: int,
      inference_batch_size: int,
      receipt_path: Path,
      command_runner: CommandRunner | None = None,
  ) -> JobReceipt: ...
  ```

- [ ] **Step 2: Run the job tests and verify RED.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_job.py
  ```

  Expected: collection fails because the job module does not exist.

- [ ] **Step 3: Implement the compute-node job.**

  `run_sentence_job` must:

  1. validate the stems and source manifest;
  2. run `nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader` through an injected command boundary;
  3. import ONNX Runtime and require `CUDAExecutionProvider` in `get_available_providers()`;
  4. create the GPU-required segmenter with the pinned revision;
  5. invoke `run_v2_sentence_split` exactly once for the selected stems;
  6. hash completed output Parquets and checkpoint metadata/batches;
  7. write the receipt atomically on success or failure and re-raise the failure after recording a sanitized error class/message.

  The script CLI must accept `--data-root`, `--stems`, `--model-cache`, `--source-commit`, `--job-id`, `--batch-size`, `--inference-batch-size`, and `--receipt`. It must not define `HF_TOKEN` or call Hub APIs. Keep all inference inside the function called from the OAR job.

- [ ] **Step 4: Run the job tests and verify GREEN.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_job.py
  ```

  Expected: all job-contract tests pass.

- [ ] **Step 5: Add a no-network compute smoke command.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run python scripts/grid5000_sentence_job.py --help
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run ty check src scripts
  ```

  Expected: help exits 0 and type checking reports no new errors.

- [ ] **Step 6: Commit the compute-node entry point.**

  ```bash
  git add src/osm_polygon_wikidata_only/grid5000/sentence_job.py scripts/grid5000_sentence_job.py tests/grid5000/test_sentence_job.py
  git diff --cached --check
  git commit -m "feat: add Grid5000 GPU sentence job"
  ```

### Task 5: Add the resumable local controller and ledger

**Files:**
- Create: `src/osm_polygon_wikidata_only/grid5000/sentence_controller.py`
- Create: `scripts/grid5000_sentence_controller.py`
- Create: `tests/grid5000/test_sentence_controller.py`

- [ ] **Step 1: Write failing controller tests with fake SSH/OAR/transfer/HF services.**

  Test these behaviors without network or real subprocesses:

  - a new run creates an atomic ledger containing source commit, model/revision, site, baseline README/map hashes, limits, and the planned first batch;
  - an existing `submitted` or `running` ledger is reconciled before any new submission;
  - an active OAR job is never duplicated;
  - a completed job retrieves its receipt and artifacts, imports them, publishes exactly once, verifies the HF callback, and marks the batch published;
  - a failed job imports valid partial checkpoints, leaves the batch retryable, and does not call the publisher;
  - a publisher failure leaves the local batch `ready_to_publish` and restart retries publication before new GPU work;
  - a changed README or comparison-map hash blocks publication;
  - cleanup rejects paths outside the run namespace and runs only after successful publication;
  - Ctrl-C releases the local lock and persists the current ledger state.

  Model the fake external boundary as:

  ```python
  class Grid5000Transport(Protocol):
      def run_frontend(self, args: Sequence[str]) -> CompletedProcess[str]: ...
      def upload_tree(self, local_root: Path, remote_root: str) -> None: ...
      def download_tree(self, remote_root: str, local_root: Path) -> None: ...
      def remove_tree(self, remote_root: str) -> None: ...

  class HubPublisher(Protocol):
      def publish_sentence_batch(self, processed_v2: Path, stems: Sequence[str], message: str) -> str: ...
      def verify_sentence_batch(self, processed_v2: Path, stems: Sequence[str]) -> None: ...
  ```

- [ ] **Step 2: Run the controller tests and verify RED.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_controller.py
  ```

  Expected: collection fails because the controller and ledger do not exist.

- [ ] **Step 3: Implement the atomic ledger.**

  Store the ledger at `data_root.cache / "grid5000_sentence_run.json"` using `atomic_write_text`. Validate immutable fields on resume: contract version, source commit, model ID/revision, segmenter version, site, baseline README/map hashes, and batch limits. Store one record per batch with `planned`, `submitted`, `running`, `retrieved`, `ready_to_publish`, `published`, or `failed` state, OAR job ID, attempt count, remote path, hashes, timings, and HF commit URL/SHA. Store receipt/log copies below `data_root.cache / "grid5000_sentence_runs" / <run-id>`.

- [ ] **Step 4: Implement bounded staging and OAR lifecycle.**

  The controller must:

  - select the next deterministic batch with `plan_sentence_batches`;
  - create a local temporary staging tree containing only selected source sections, the two V2/sentence manifests, selected checkpoint trees, the pinned Git archive, and job metadata;
  - use the explicit remote root `$HOME/osm-polygon-wikidata-only-grid5000/<run-id>/`;
  - run `usagepolicycheck -t` on the site frontend before submission and immediately after submission;
  - submit exactly `oarsub -q besteffort -l host=1/gpu=1,walltime=0:30 <job-script>`;
  - poll only the recorded OAR job ID;
  - download receipt, outputs, manifest, and checkpoint state before cleanup;
  - import and validate locally before calling the HF publisher;
  - remove only the finished job directory after publication verification.

  The frontend must only execute SSH, rsync, file management, policy, OAR, and monitoring commands. `uv sync`, model loading, `nvidia-smi`, and sentence inference must run inside the reserved job script.

- [ ] **Step 5: Implement local HF publishing and post-commit verification.**

  The controller's concrete publisher must call existing `sentence_publication_ops` and `upload_files` with the local token. It must upload the selected sentence Parquets, merged sentence manifest, and unchanged README in one commit. It must never call `write_v2_card`. Before upload, compare local README/map hashes to the ledger baseline. After upload, verify remote file presence and exact hashes for the selected sidecars, manifest, README, and comparison map using the existing bounded HF transport. Record the returned commit URL/SHA only after verification succeeds.

- [ ] **Step 6: Implement the controller CLI and resume semantics.**

  The CLI must expose:

  ```text
  --data-root PATH
  --site NAME (default: grenoble)
  --repo-id REPO_ID
  --max-stems INTEGER (default: 4)
  --max-input-bytes INTEGER (default: 268435456)
  --batch-size INTEGER (default: 256)
  --inference-batch-size INTEGER (default: 16)
  --walltime TEXT (default: 0:30)
  --run-id TEXT (optional; resume ledger when omitted)
  --hf-token TEXT (optional; otherwise local HF token resolution)
  ```

  On startup it acquires `exclusive_run_lock(data_root.cache / "grid5000-sentence-splitting.lock")`, loads/reconciles the ledger, and exits 0 only when every finalized V2 stem is complete and the final policy/cleanup checks pass. It must persist state before and after every external transition and handle `KeyboardInterrupt` by saving state, cancelling only a known stale job, and releasing the lock.

- [ ] **Step 7: Run controller tests and verify GREEN.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/grid5000/test_sentence_controller.py
  ```

  Expected: all controller tests pass, including failure/restart and no-duplicate cases.

- [ ] **Step 8: Commit the controller.**

  ```bash
  git add src/osm_polygon_wikidata_only/grid5000/sentence_controller.py scripts/grid5000_sentence_controller.py tests/grid5000/test_sentence_controller.py
  git diff --cached --check
  git commit -m "feat: add resumable Grid5000 sentence controller"
  ```

### Task 6: Document the operational contract and quality scopes

**Files:**
- Create: `docs/grid5000-sentence-splitting.md`
- Modify: `README.md`
- Modify: `docs/development.md`
- Modify: `Justfile`
- Modify: `pyproject.toml`
- Modify: `tests/test_documentation.py`
- Modify: `tests/v2/test_sentence_docs.py`

- [ ] **Step 1: Add failing documentation and quality-contract tests.**

  Require the documentation to mention the controller command, `host=1/gpu=1`, 30-minute default walltime, local HF token boundary, CUDA fail-closed behavior, exact supported-language policy, per-job atomic publication, `usagepolicycheck`, Seagate authority, scoped cleanup, and resume ledger. Require `Justfile`/`pyproject.toml` to include the new pure helper files in CRAP and mutation scopes.

- [ ] **Step 2: Run the documentation tests and verify RED.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/test_documentation.py tests/v2/test_sentence_docs.py
  ```

  Expected: the new assertions fail because the Grid5000 documentation and quality scope do not exist.

- [ ] **Step 3: Write the operational documentation.**

  Document this exact starting command, with the repository's actual data root substituted by the operator:

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

  Explain that unsupported language codes remain one unsplit row, that all model inference runs on the reserved GPU node, and that the local controller publishes only verified artifacts.

- [ ] **Step 4: Extend quality configuration without broadening unrelated scopes.**

  Add `src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py` to the pure CRAP coverage/source lists and add `tests/grid5000/test_sentence_protocol.py` to the mutation test selection. Keep subprocess/SSH/controller code outside mutation mutation targets; cover its transitions with tests instead. Ensure the CRAP command reports a maximum strictly below `5.99` and the mutation gate rejects survivors, timeouts, and untested mutants.

- [ ] **Step 5: Run documentation tests and verify GREEN.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/test_documentation.py tests/v2/test_sentence_docs.py
  ```

- [ ] **Step 6: Commit documentation and quality wiring.**

  ```bash
  git add docs/grid5000-sentence-splitting.md README.md docs/development.md Justfile pyproject.toml tests/test_documentation.py tests/v2/test_sentence_docs.py
  git diff --cached --check
  git commit -m "docs: document Grid5000 sentence execution"
  ```

### Task 7: Run local no-regression and quality gates

**Files:**
- No additional files; inspect only the scoped changes and pre-existing worktree state.

- [ ] **Step 1: Run all focused sentence and Grid5000 tests.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --no-cov -q tests/v2 tests/grid5000
  ```

  Expected: zero failures.

- [ ] **Step 2: Run the full regression suite with coverage.**

  ```bash
  COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-coverage-grid5000 UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q
  ```

  Expected: zero failures and the configured coverage threshold is met.

- [ ] **Step 3: Run static checks.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run ruff check src tests scripts
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run ruff format --check src tests scripts
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run ty check src scripts
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv build
  ```

  Expected: all commands exit 0. If a repository-wide hook reports a pre-existing unrelated untracked file, do not edit it; record the exact failure and run targeted checks on the changed files before committing.

- [ ] **Step 4: Run CRAP and mutation gates.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv just crap-all
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv just mutation
  ```

  Expected: every reported changed pure helper has CRAP `< 6.00`, and the mutation gate reports only killed mutants with no survivor, timeout, or untested mutant.

- [ ] **Step 5: Inspect the complete diff and preserve unrelated work.**

  ```bash
  git diff --check
  git status --short --branch
  git diff --stat origin/main...HEAD
  ```

  Stage only the implementation paths named in this plan. Preserve `:memory:.ses`, the existing superpowers plan/spec files, `output/`, `tests/unit/`, and `tmp/` unless a later explicit request changes that scope.

- [ ] **Step 6: Push the implementation commits.**

  ```bash
  git push origin main
  git rev-parse HEAD
  git rev-parse origin/main
  ```

  Expected: the two revisions match. Record the pushed source commit for every Grid5000 receipt.

### Task 8: Preflight Grid5000 and execute the remaining batches

**Files:**
- Modify only external Grid5000 run-owned staging and the Seagate data-root artifacts created by the controller.

- [ ] **Step 1: Run the read-only site preflight.**

  From the configured `grenoble` frontend, run:

  ```bash
  hostname
  usagepolicycheck -t
  quota
  oarnodes --sql "gpu > 0 AND state = 'Alive'" -J
  oarstat -u
  ```

  Select only an alive CUDA-capable GPU resource. Do not submit if the policy check errors, quota is insufficient for the bounded staging namespace, or the site has no suitable GPU. Do not use the access frontend for heavy work.

- [ ] **Step 2: Verify the local baseline before the first job.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run python -c 'from pathlib import Path; from osm_polygon_wikidata_only.io.hashing import sha256_file; root=Path("/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/processed_v2"); print(sha256_file(root/"README.md")); print(sha256_file(root/"assets/v2_added_wikipedia_tag_documents.png"))'
  ```

  Confirm the sentence manifest has 48 complete stems, 91 project entries, and 338 remaining stems before starting. The controller must record these values in its ledger.

- [ ] **Step 3: Start the controller in a resumable session.**

  Run the documented controller command from Task 6. Keep the controller on the Mac; only the submitted job script runs inference on the reserved Grid5000 GPU. Monitor the controller output and `oarstat -u`; do not submit another controller while this ledger is active.

- [ ] **Step 4: Verify every job boundary before allowing the next batch.**

  For each job, confirm in the controller ledger and logs:

  - exactly one OAR job ID and no duplicate active job;
  - policy check before and after submission;
  - `nvidia-smi` GPU identity and CUDA provider in the receipt;
  - source/model/revision/input hashes match;
  - sentence schemas, row accounting, supported/unsupported routing, and checkpoint state validate;
  - one HF commit contains the batch sidecars, merged sentence manifest, and unchanged README;
  - remote README/map hashes match local baselines;
  - the finished job directory is removed only after those checks.

- [ ] **Step 5: Continue until the ledger reports all 386 finalized stems complete.**

  The controller must reuse the three existing partial checkpoint states and must not rerun the 48 already complete stems. If a job fails, retrieve valid partial checkpoint batches, retry the same batch, and do not publish until every selected project completes.

- [ ] **Step 6: Verify final publication and cleanup.**

  ```bash
  UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run python -m osm_polygon_wikidata_only.grid5000.sentence_controller --data-root "$OSM_POLYGON_DATA_ROOT" --site grenoble --repo-id NoeFlandre/osm-polygon-wikidata-and-wikipedia --verify-only
  ssh grenoble 'usagepolicycheck -t; oarstat -u'
  ```

  Expected: local and HF manifests contain all 386 stems, all expected sentence outputs are hash-valid, README/map remain byte-identical, no project OAR jobs remain, the final policy check reports no flagged jobs, and the remote run namespace contains no project-owned files.

- [ ] **Step 7: Record the final evidence and report.**

  Preserve the local run ledger, receipts, HF commit URLs, source commit, job IDs, runtimes, GPU models, row counts, and final hashes under the Seagate data root. Report the exact number of finalized PBFs, sentence-complete stems, HF commits, Grid5000 jobs, failed/retried jobs, and any policy or environmental limitation.

## Plan self-review

- Spec coverage: GPU-only execution, local token boundary, bounded OAR jobs, receipts, checkpoint import, manifest preservation, atomic HF publication, policy checks, scoped cleanup, TDD, CRAP, mutation, documentation, and final verification are covered by Tasks 1–8.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation step is required; all commands, paths, contracts, and expected outcomes are named.
- Type consistency: `SentenceBatch`, `FileDigest`, `Grid5000Transport`, `HubPublisher`, `run_sentence_job`, `plan_sentence_batches`, `validate_manifest_extension`, `validate_sentence_output`, and `import_checkpoint_tree` are defined before their consumers.
- Scope: subprocess/controller code is intentionally tested with fakes and excluded from mutation targets; pure deterministic protocol code is included in CRAP and mutation scopes.
