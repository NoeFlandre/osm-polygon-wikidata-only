from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from subprocess import CompletedProcess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000 import sentence_controller
from osm_polygon_wikidata_only.grid5000.sentence_job import GpuIdentity, JobReceipt
from osm_polygon_wikidata_only.grid5000.sentence_protocol import (
    FileDigest,
    sha256_manifest,
)
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteFileInfo, RemoteInventory
from osm_polygon_wikidata_only.v2.sentence_logic import sentence_schema


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        ({"state": "planned", "oar_job_id": None, "hf_commit": None}, True),
        ({"state": "planned", "oar_job_id": "123", "hf_commit": None}, False),
        ({"state": "failed", "oar_job_id": "123", "error": "missing_receipt"}, True),
        ({"state": "failed", "oar_job_id": "123", "error": "unknown"}, False),
        ({"state": "published", "oar_job_id": None, "hf_commit": None}, False),
        (None, False),
    ],
)
def test_source_commit_batch_safety_is_explicit(batch: object, expected: bool) -> None:
    assert sentence_controller._source_commit_batch_is_safe(batch) is expected


def test_receipt_artifact_decoder_preserves_fields() -> None:
    assert sentence_controller._receipt_artifact(
        {"relative_path": "processed_v2/file.parquet", "size": 7, "sha256": "digest"}
    ) == FileDigest("processed_v2/file.parquet", 7, "digest")


def test_receipt_value_normalizes_stem_lists() -> None:
    assert sentence_controller._normalized_receipt_value("stems", ["alpha-latest"]) == (
        "alpha-latest",
    )
    assert sentence_controller._normalized_receipt_value("job_id", "123") == "123"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("baseline_readme_sha256", None, "README baseline"),
        ("baseline_map_sha256", None, "comparison-map baseline"),
        ("batches", {}, "batches must be a list"),
    ],
)
def test_ledger_baselines_reject_invalid_values(
    key: str,
    value: object,
    message: str,
) -> None:
    ledger: dict[str, object] = {
        "baseline_readme_sha256": "readme-digest",
        "baseline_map_sha256": "map-digest",
        "batches": [],
    }
    ledger[key] = value

    with pytest.raises(sentence_controller.ControllerRunError, match=message):
        sentence_controller._validate_ledger_baselines(ledger)


def _write_table(path: Path, schema: pa.Schema, text: str = "First.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field.name: None for field in schema}
    if "text" in schema.names:
        row["text"] = text
    pq.write_table(pa.Table.from_pylist([row], schema=schema), path)


def _sentence_manifest(regions: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "contract_version": "v2-sentence-splitting-v1",
        "segmenter": "sat-3l-sm",
        "model_id": "segment-any-text/sat-3l-sm",
        "model_revision": "137da05",
        "segmenter_version": "2.2.1",
        "supported_languages": ["en", "fr"],
        "unsupported_languages": [],
        "unsupported_language_policy": "one unsplit row; never passed to SaT",
        "regions": regions or [],
    }


def _data_root(tmp_path: Path) -> DataRoot:
    root = DataRoot(tmp_path)
    root.ensure()
    source = root.processed_v2 / "wikipedia/sections/alpha-latest.parquet"
    _write_table(source, section_schema(), "Alpha source.")
    manifest = root.processed_v2 / "manifests/processed_pbfs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"regions": {"alpha-latest": {"sections_path": str(source)}}}),
        encoding="utf-8",
    )
    (root.processed_v2 / "README.md").write_text("stable card\n", encoding="utf-8")
    map_path = root.processed_v2 / "assets/v2_added_wikipedia_tag_documents.png"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_bytes(b"stable map")
    (root.processed_v2 / "manifests/sentence_splitting.json").write_text(
        json.dumps(_sentence_manifest()) + "\n", encoding="utf-8"
    )
    return root


def test_rsync_resolves_remote_home_placeholder(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(args, **_kwargs):
        calls.append(tuple(args))
        if args[0] == "ssh":
            return CompletedProcess(args, 0, stdout="/home/test-user\n", stderr="")
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(sentence_controller.subprocess, "run", fake_run)
    monkeypatch.setattr(sentence_controller, "_required_executable", lambda name: name)
    staging = tmp_path / "staging"
    staging.mkdir()

    sentence_controller.SubprocessGrid5000Transport("grenoble").upload_tree(
        staging, "$HOME/project"
    )

    assert calls == [
        ("ssh", "grenoble", "printf", "%s", "$HOME"),
        ("rsync", "-a", f"{staging}/", "grenoble:/home/test-user/project/"),
    ]


def test_oarsub_job_command_is_quoted_for_ssh(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []
    job_command = 'cd "$HOME/run/code" && uv sync --frozen'
    property_filter = "gpu_model='H100 NVL'"

    def fake_run(args, **_kwargs):
        calls.append(tuple(args))
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(sentence_controller.subprocess, "run", fake_run)
    monkeypatch.setattr(sentence_controller, "_required_executable", lambda name: name)

    sentence_controller.SubprocessGrid5000Transport("grenoble").run_frontend(
        (
            "oarsub",
            "-q",
            "besteffort",
            "-p",
            property_filter,
            "-l",
            "host=1/gpu=1,walltime=0:30",
            job_command,
        )
    )

    assert calls == [
        (
            "ssh",
            "grenoble",
            " ".join(
                sentence_controller.shlex.quote(argument)
                for argument in (
                    "oarsub",
                    "-q",
                    "besteffort",
                    "-p",
                    property_filter,
                    "-l",
                    "host=1/gpu=1,walltime=0:30",
                    job_command,
                )
            ),
        )
    ]


def test_remote_job_bootstraps_pinned_uv_on_compute_node(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)

    command = controller._remote_job_command(
        {
            "remote_job_root": f"{controller.remote_run_root}/jobs/batch-00000000-attempt-01",
            "stems": ["alpha-latest"],
        }
    )

    assert (
        'python3 -m venv "$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/uv-bootstrap"'
        in command
    )
    assert (
        '"$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/uv-bootstrap/bin/python" '
        '-m pip install --disable-pip-version-check --no-input "uv==0.11.16"'
    ) in command
    assert (
        '"$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/uv-bootstrap/bin/uv" sync --frozen'
        in command
    )
    assert (
        '"$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/uv-bootstrap/bin/uv" run --no-sync'
        in command
    )
    assert 'find "$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/jobs' in command
    assert "LD_LIBRARY_PATH=" in command


def test_batch_staging_includes_package_forced_assets(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    controller = _controller(data_root, _FakeTransport(tmp_path), _FakePublisher())
    ledger = controller.initialize()
    staging = tmp_path / "staging"
    staging.mkdir()

    controller._stage_batch(staging, ledger["batches"][0])

    assert (staging / "code/assets/dataset_hero.png").is_file()
    assert (staging / "code/assets/dataset_hero_v2.png").is_file()


class _FakeTransport:
    def __init__(
        self,
        tmp_path: Path,
        *,
        success: bool = True,
        interrupt_on_poll: bool = False,
        terminal_state: str = "Terminated",
    ) -> None:
        self.tmp_path = tmp_path
        self.success = success
        self.interrupt_on_poll = interrupt_on_poll
        self.terminal_state = terminal_state
        self.frontend_calls: list[tuple[str, ...]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.removals: list[str] = []
        self.staged: Path | None = None
        self.polls = 0

    def run_frontend(self, args: Sequence[str]) -> CompletedProcess[str]:
        args = tuple(args)
        self.frontend_calls.append(args)
        if args[0] == "usagepolicycheck":
            return CompletedProcess(args, 0, stdout="No jobs flagged\n", stderr="")
        if args[0] == "mkdir":
            return CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "oarsub":
            return CompletedProcess(args, 0, stdout="OAR job id : 12345\n", stderr="")
        if args[0] == "oarstat":
            self.polls += 1
            if self.interrupt_on_poll:
                raise KeyboardInterrupt
            state = self.terminal_state
            exit_code = "0" if self.success else "1"
            return CompletedProcess(
                args,
                0,
                stdout=f"state = {state}\nexit_code = {exit_code}\n",
                stderr="",
            )
        if args[0] == "oardel":
            return CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected frontend command: {args}")

    def upload_tree(self, local_root: Path, remote_root: str) -> None:
        self.uploads.append((local_root, remote_root))
        self.staged = self.tmp_path / f"staged-{len(self.uploads)}"
        shutil.copytree(local_root, self.staged)

    def download_tree(self, remote_root: str, local_root: Path) -> None:
        self.downloads.append((remote_root, local_root))
        if self.staged is None:
            (local_root / "data").mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                self.tmp_path / "processed_v2",
                local_root / "data/processed_v2",
                dirs_exist_ok=True,
            )
            shutil.copytree(
                self.tmp_path / "cache/v2",
                local_root / "data/cache/v2",
                dirs_exist_ok=True,
            )
        else:
            result = self.staged / "result"
            shutil.copytree(result, local_root, dirs_exist_ok=True)
        data_root = DataRoot(local_root / "data")
        if self.success:
            self._write_success_result(data_root, local_root)
        else:
            self._write_failure_result(data_root, local_root)

    def remove_tree(self, remote_root: str) -> None:
        self.removals.append(remote_root)

    def _write_success_result(self, data_root: DataRoot, result_root: Path) -> None:
        output = data_root.processed_v2 / "wikipedia/sentences/alpha-latest.parquet"
        _write_table(output, sentence_schema())
        manifest_path = data_root.processed_v2 / "manifests/sentence_splitting.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["regions"] = [
            {
                "stem": "alpha-latest",
                "project": "wikipedia",
                "sections": 1,
                "split_sections": 1,
                "unsplit_sections": 0,
                "sentence_rows": 1,
                "supported_languages": ["en"],
                "unsupported_languages": [],
            }
        ]
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        self._write_receipt(
            data_root, result_root, status="succeeded", paths=(output, manifest_path)
        )

    def _write_failure_result(self, data_root: DataRoot, result_root: Path) -> None:
        source = data_root.processed_v2 / "wikipedia/sections/alpha-latest.parquet"
        checkpoint = data_root.v2_cache / "sentence-checkpoints/alpha-latest/wikipedia"
        checkpoint.mkdir(parents=True, exist_ok=True)
        identity = {
            "contract_version": "v2-sentence-checkpoints-v1",
            "stem": "alpha-latest",
            "project": "wikipedia",
            "input_fingerprint": sentence_controller.sha256_file(source),
            "model_id": "segment-any-text/sat-3l-sm",
            "model_revision": "137da05",
            "batch_size": 256,
        }
        (checkpoint / "metadata.json").write_text(
            json.dumps({"identity": identity, "complete": False, "batch_count": 0, "row_count": 0}),
            encoding="utf-8",
        )
        _write_table(checkpoint / "batch-00000000.parquet", sentence_schema())
        self._write_receipt(data_root, result_root, status="failed", paths=())

    def _write_receipt(
        self,
        data_root: DataRoot,
        result_root: Path,
        *,
        status: str,
        paths: tuple[Path, ...],
    ) -> None:
        artifacts: tuple[FileDigest, ...] = sha256_manifest(paths, root=data_root.path)
        receipt = JobReceipt(
            status=status,
            job_id="12345",
            source_commit="abc123",
            model_id="segment-any-text/sat-3l-sm",
            model_revision="137da05",
            segmenter_version="2.2.1",
            ort_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            gpu=(GpuIdentity("NVIDIA A100", "GPU-uuid-1", "40960 MiB"),),
            stems=("alpha-latest",),
            batch_size=256,
            inference_batch_size=16,
            regions=(),
            artifacts=artifacts,
            started_at="2026-08-27T00:00:00+00:00",
            finished_at="2026-08-27T00:01:00+00:00",
            error_type="RuntimeError" if status == "failed" else None,
            error_message="redacted; inspect the reserved-job log" if status == "failed" else None,
        )
        (result_root / "receipt.json").write_text(
            json.dumps(receipt.to_payload()) + "\n", encoding="utf-8"
        )


class _FakePublisher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.publish_calls: list[tuple[Path, tuple[str, ...], str]] = []
        self.verify_calls: list[tuple[Path, tuple[str, ...]]] = []

    def publish_sentence_batch(self, processed_v2: Path, stems: Sequence[str], message: str) -> str:
        stems = tuple(stems)
        self.publish_calls.append((processed_v2, stems, message))
        if self.fail:
            raise RuntimeError("publisher unavailable")
        return "https://huggingface.co/datasets/example/commit/abc"

    def verify_sentence_batch(self, processed_v2: Path, stems: Sequence[str]) -> None:
        stems = tuple(stems)
        self.verify_calls.append((processed_v2, stems))


def _controller(
    data_root: DataRoot,
    transport: _FakeTransport,
    publisher: _FakePublisher,
    *,
    run_id: str = "run-20260827-01",
    queue: str = "besteffort",
    gpu_model: str = "A40",
    source_commit: str = "abc123",
) -> sentence_controller.Grid5000SentenceController:
    return sentence_controller.Grid5000SentenceController(
        data_root,
        site="grenoble",
        queue=queue,
        gpu_model=gpu_model,
        repo_id="example/v2",
        transport=transport,
        publisher=publisher,
        run_id=run_id,
        source_commit=source_commit,
        repo_root=Path.cwd(),
        sleep=lambda _seconds: None,
    )


def test_unsafe_queue_is_rejected_before_a_run_can_start(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()

    with pytest.raises(sentence_controller.ControllerRunError, match="Unsafe Grid5000 queue"):
        _controller(data_root, transport, publisher, queue="best effort")


def test_initialize_persists_immutable_ledger_and_first_planned_batch(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)

    ledger = controller.initialize()

    assert ledger["contract_version"] == "grid5000-sentence-v1"
    assert ledger["source_commit"] == "abc123"
    assert ledger["model_id"] == "segment-any-text/sat-3l-sm"
    assert ledger["model_revision"] == "137da05"
    assert ledger["site"] == "grenoble"
    assert ledger["queue"] == "besteffort"
    assert ledger["baseline_readme_sha256"]
    assert ledger["baseline_map_sha256"]
    assert ledger["limits"] == {
        "max_stems": 4,
        "max_input_bytes": 268435456,
        "batch_size": 256,
        "inference_batch_size": 16,
        "walltime": "0:30",
    }
    assert ledger["batches"][0]["state"] == "planned"
    assert ledger["batches"][0]["stems"] == ["alpha-latest"]
    assert (
        json.loads(controller.ledger_path.read_text(encoding="utf-8"))["run_id"] == ledger["run_id"]
    )


def test_process_batch_publishes_a_ready_batch_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root(tmp_path)
    controller = _controller(data_root, _FakeTransport(tmp_path), _FakePublisher())
    batch = {"index": 0, "state": "ready_to_publish", "stems": ["alpha-latest"]}
    published: list[dict[str, object]] = []
    monkeypatch.setattr(controller, "_publish_batch", published.append)

    controller._process_batch(batch)

    assert published == [batch]


def test_custom_gpu_model_is_persisted_and_requested(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher, gpu_model="L40S")

    ledger = controller.run()

    assert ledger["gpu_model"] == "L40S"
    submit_command = next(command for command in transport.frontend_calls if command[0] == "oarsub")
    assert submit_command[4] == "gpu_model='L40S'"


def test_exotic_gpu_model_requests_exotic_job_type(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher, gpu_model="A100-PCIE-40GB")

    controller.run()

    submit_command = next(command for command in transport.frontend_calls if command[0] == "oarsub")
    job_type_position = submit_command.index("-t")
    assert submit_command[job_type_position : job_type_position + 2] == ("-t", "exotic")


def test_pre_submission_resume_records_a_new_source_commit(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    first = _controller(data_root, transport, publisher)
    ledger = first.initialize()
    ledger["source_commit"] = "old123"
    ledger["batches"][0].update(state="failed", attempt=1, error="ControllerRunError")
    first._write_ledger(ledger)

    resumed = _controller(data_root, transport, publisher, source_commit="new123")

    updated = resumed.initialize()

    assert updated["source_commit"] == "new123"
    assert updated["source_commit_updates"][-1]["from"] == "old123"
    assert updated["source_commit_updates"][-1]["to"] == "new123"


def test_failed_receipt_resume_can_adopt_a_new_source_commit(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    first = _controller(data_root, transport, publisher)
    ledger = first.initialize()
    ledger["batches"][0].update(
        state="failed",
        attempt=1,
        oar_job_id="12345",
        remote_job_root="$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/jobs/batch-00000000-attempt-01",
        error="missing_receipt",
    )
    first._write_ledger(ledger)

    resumed = _controller(data_root, transport, publisher, source_commit="new123")

    updated = resumed.initialize()

    assert updated["source_commit"] == "new123"
    assert updated["source_commit_updates"][-1]["reason"] == "resumable_unpublished_run"


def test_completed_batch_is_retrieved_published_verified_and_cleaned(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)

    ledger = controller.run()

    assert ledger["batches"][0]["state"] == "published"
    assert ledger["batches"][0]["hf_commit"]
    assert publisher.publish_calls
    assert publisher.verify_calls
    assert any(command[0] == "oarsub" for command in transport.frontend_calls)
    mkdir_command = next(command for command in transport.frontend_calls if command[0] == "mkdir")
    assert f"{controller.remote_run_root}/jobs" in mkdir_command
    submit_command = next(command for command in transport.frontend_calls if command[0] == "oarsub")
    assert submit_command[:7] == (
        "oarsub",
        "-q",
        "besteffort",
        "-p",
        "gpu_model='A40'",
        "-l",
        "host=1/gpu=1,walltime=0:30",
    )
    assert '--job-id "$OAR_JOB_ID"' in submit_command[-1]
    assert "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN" in submit_command[-1]
    assert sum(command[0] == "usagepolicycheck" for command in transport.frontend_calls) >= 3
    assert transport.removals[-1].endswith("run-20260827-01")


def test_finishing_success_receipt_is_published(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path, terminal_state="Finishing")
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)

    ledger = controller.run()

    assert ledger["batches"][0]["state"] == "published"
    assert publisher.publish_calls


def test_running_ledger_is_reconciled_without_duplicate_submission(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    first_transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    first = _controller(data_root, first_transport, publisher)
    ledger = first.initialize()
    ledger["batches"][0].update(
        state="running",
        oar_job_id="12345",
        remote_job_root="$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/jobs/batch-00000000-attempt-01",
    )
    first._write_ledger(ledger)

    transport = _FakeTransport(tmp_path)
    resumed = _controller(data_root, transport, publisher)
    resumed.run()

    assert not any(command[0] == "oarsub" for command in transport.frontend_calls)
    assert any(command[0] == "oarstat" for command in transport.frontend_calls)


def test_failed_job_imports_partial_checkpoint_and_does_not_publish(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path, success=False)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)

    with pytest.raises(sentence_controller.ControllerRunError, match="failed"):
        controller.run()

    checkpoint = data_root.v2_cache / "sentence-checkpoints/alpha-latest/wikipedia"
    assert (checkpoint / "batch-00000000.parquet").is_file()
    assert publisher.publish_calls == []
    assert json.loads(controller.ledger_path.read_text())["batches"][0]["state"] == "failed"


def test_retry_resets_remote_cleanup_state_for_the_new_attempt(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    first_transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    first = _controller(data_root, first_transport, publisher)
    ledger = first.initialize()
    ledger["batches"][0].update(
        state="failed",
        attempt=1,
        remote_job_root="$HOME/osm-polygon-wikidata-only-grid5000/run-20260827-01/jobs/batch-00000000-attempt-01",
        remote_cleaned=True,
        error="missing_receipt",
    )
    first._write_ledger(ledger)

    transport = _FakeTransport(tmp_path)
    resumed = _controller(data_root, transport, publisher)

    resumed.run()

    assert any(removal.endswith("batch-00000000-attempt-02") for removal in transport.removals)


def test_missing_receipt_marks_terminal_batch_failed_and_cleans_it(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)

    original_download = transport.download_tree

    def download_without_receipt(remote_root: str, local_root: Path) -> None:
        original_download(remote_root, local_root)
        (local_root / "receipt.json").unlink()

    transport.download_tree = download_without_receipt  # type: ignore[method-assign]

    with pytest.raises(sentence_controller.ControllerRunError, match="receipt"):
        controller.run()

    batch = json.loads(controller.ledger_path.read_text())["batches"][0]
    assert batch["state"] == "failed"
    assert batch["error"] == "missing_receipt"
    assert transport.removals[-1].endswith("batch-00000000-attempt-01")


def test_publisher_failure_leaves_ready_state_and_resume_does_not_submit_again(
    tmp_path: Path,
) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher(fail=True)
    controller = _controller(data_root, transport, publisher)

    with pytest.raises(RuntimeError, match="publisher unavailable"):
        controller.run()
    first_submission_count = sum(command[0] == "oarsub" for command in transport.frontend_calls)
    assert (
        json.loads(controller.ledger_path.read_text())["batches"][0]["state"] == "ready_to_publish"
    )

    publisher.fail = False
    controller.run()

    assert (
        sum(command[0] == "oarsub" for command in transport.frontend_calls)
        == first_submission_count
    )
    assert len(publisher.publish_calls) == 2


def test_changed_card_or_map_blocks_publication(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)
    controller.initialize()
    (data_root.processed_v2 / "README.md").write_text("changed card\n", encoding="utf-8")

    with pytest.raises(sentence_controller.ControllerRunError, match="baseline"):
        controller.run()

    assert publisher.publish_calls == []


def test_hf_sentence_verification_uses_lfs_metadata_without_download(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = _data_root(tmp_path)
    sentence_path = data_root.processed_v2 / "wikipedia/sentences/alpha-latest.parquet"
    _write_table(sentence_path, sentence_schema())
    publisher = sentence_controller.HfHubSentencePublisher(
        "example/v2", token="token", cache_dir=tmp_path / "verify"
    )
    expected_paths = [
        operation.path_in_repo
        for operation in sentence_controller.sentence_publication_ops(
            data_root.processed_v2, ("alpha-latest",)
        )
    ]
    expected_paths.append("assets/v2_added_wikipedia_tag_documents.png")
    metadata = {
        path: RemoteFileInfo(
            path=path,
            size=(data_root.processed_v2 / path).stat().st_size,
            sha256=sentence_controller.sha256_file(data_root.processed_v2 / path),
        )
        for path in expected_paths
    }
    inventory = RemoteInventory(set(metadata), metadata)
    fetch_calls: list[tuple[str, tuple[str, ...]]] = []

    def fetch_paths(repo_id: str, *, paths: Sequence[str], **_kwargs) -> RemoteInventory:
        fetch_calls.append((repo_id, tuple(paths)))
        return inventory

    monkeypatch.setattr(RemoteInventory, "fetch", lambda *_args, **_kwargs: inventory)
    monkeypatch.setattr(RemoteInventory, "fetch_paths", fetch_paths)
    monkeypatch.setattr(
        sentence_controller,
        "_download_hf_file",
        lambda *_args, **_kwargs: pytest.fail("LFS verification should not download files"),
    )

    publisher.verify_sentence_batch(data_root.processed_v2, ("alpha-latest",))

    assert fetch_calls == [("example/v2", tuple(expected_paths))]


def test_sentence_verification_plan_includes_the_comparison_map(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_table(
        data_root.processed_v2 / "wikipedia/sentences/alpha-latest.parquet",
        sentence_schema(),
    )

    expected = sentence_controller._expected_sentence_files(
        data_root.processed_v2,
        ("alpha-latest",),
    )

    assert expected[-1][1] == "assets/v2_added_wikipedia_tag_documents.png"


def test_hf_sentence_verification_downloads_files_without_lfs_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = _data_root(tmp_path)
    sentence_path = data_root.processed_v2 / "wikipedia/sentences/alpha-latest.parquet"
    _write_table(sentence_path, sentence_schema())
    publisher = sentence_controller.HfHubSentencePublisher(
        "example/v2", token="token", cache_dir=tmp_path / "verify"
    )
    expected_paths = [
        operation.path_in_repo
        for operation in sentence_controller.sentence_publication_ops(
            data_root.processed_v2, ("alpha-latest",)
        )
    ]
    expected_paths.append("assets/v2_added_wikipedia_tag_documents.png")
    metadata = {
        path: RemoteFileInfo(
            path=path,
            size=(data_root.processed_v2 / path).stat().st_size,
            sha256=None
            if path == "README.md"
            else sentence_controller.sha256_file(data_root.processed_v2 / path),
        )
        for path in expected_paths
    }
    inventory = RemoteInventory(set(metadata), metadata)
    downloaded: list[str] = []

    monkeypatch.setattr(
        RemoteInventory,
        "fetch_paths",
        lambda *_args, **_kwargs: inventory,
    )

    def download(_repo_id: str, filename: str, *, local_dir: Path, **_kwargs) -> Path:
        downloaded.append(filename)
        target = local_dir / "downloaded-file"
        target.write_bytes((data_root.processed_v2 / filename).read_bytes())
        return target

    monkeypatch.setattr(sentence_controller, "_download_hf_file", download)

    publisher.verify_sentence_batch(data_root.processed_v2, ("alpha-latest",))

    assert downloaded == ["README.md"]


def test_hf_sentence_verification_rejects_lfs_digest_mismatch(tmp_path: Path, monkeypatch) -> None:
    data_root = _data_root(tmp_path)
    sentence_path = data_root.processed_v2 / "wikipedia/sentences/alpha-latest.parquet"
    _write_table(sentence_path, sentence_schema())
    publisher = sentence_controller.HfHubSentencePublisher(
        "example/v2", token="token", cache_dir=tmp_path / "verify"
    )
    expected_paths = [
        operation.path_in_repo
        for operation in sentence_controller.sentence_publication_ops(
            data_root.processed_v2, ("alpha-latest",)
        )
    ]
    expected_paths.append("assets/v2_added_wikipedia_tag_documents.png")
    metadata = {
        path: RemoteFileInfo(
            path=path,
            size=(data_root.processed_v2 / path).stat().st_size,
            sha256="0" * 64
            if path == "README.md"
            else sentence_controller.sha256_file(data_root.processed_v2 / path),
        )
        for path in expected_paths
    }
    monkeypatch.setattr(
        RemoteInventory,
        "fetch_paths",
        lambda *_args, **_kwargs: RemoteInventory(set(metadata), metadata),
    )
    monkeypatch.setattr(
        sentence_controller,
        "_download_hf_file",
        lambda *_args, **_kwargs: pytest.fail("digest mismatches must fail before download"),
    )

    with pytest.raises(sentence_controller.ControllerRunError, match=r"hash mismatch: README\.md"):
        publisher.verify_sentence_batch(data_root.processed_v2, ("alpha-latest",))


def test_hf_sentence_publisher_uses_bounded_upload_concurrency(tmp_path: Path, monkeypatch) -> None:
    data_root = _data_root(tmp_path)
    for stem in ("alpha-latest", "beta-latest"):
        _write_table(
            data_root.processed_v2 / f"wikipedia/sentences/{stem}.parquet",
            sentence_schema(),
        )
    publisher = sentence_controller.HfHubSentencePublisher(
        "example/v2", token="token", cache_dir=tmp_path / "verify"
    )
    observed: list[int] = []

    def upload_files(*_args, num_threads: int, **_kwargs) -> str:
        observed.append(num_threads)
        return "commit"

    monkeypatch.setattr(sentence_controller, "upload_files", upload_files)

    publisher.publish_sentence_batch(
        data_root.processed_v2,
        ("alpha-latest", "beta-latest"),
        "publish",
    )

    assert observed == [4]


def test_keyboard_interrupt_cancels_only_recorded_job_and_saves_ledger(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    transport = _FakeTransport(tmp_path, interrupt_on_poll=True)
    publisher = _FakePublisher()
    controller = _controller(data_root, transport, publisher)

    with pytest.raises(KeyboardInterrupt):
        controller.run()

    assert any(
        command[0] == "oardel" and command[1] == "12345" for command in transport.frontend_calls
    )
    assert json.loads(controller.ledger_path.read_text())["batches"][0]["state"] == "cancelled"
