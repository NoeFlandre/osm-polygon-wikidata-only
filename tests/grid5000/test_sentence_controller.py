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
from osm_polygon_wikidata_only.v2.sentence_logic import sentence_schema


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


class _FakeTransport:
    def __init__(
        self, tmp_path: Path, *, success: bool = True, interrupt_on_poll: bool = False
    ) -> None:
        self.tmp_path = tmp_path
        self.success = success
        self.interrupt_on_poll = interrupt_on_poll
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
            state = "Terminated"
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
) -> sentence_controller.Grid5000SentenceController:
    return sentence_controller.Grid5000SentenceController(
        data_root,
        site="grenoble",
        repo_id="example/v2",
        transport=transport,
        publisher=publisher,
        run_id=run_id,
        source_commit="abc123",
        repo_root=Path.cwd(),
        sleep=lambda _seconds: None,
    )


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
    submit_command = next(command for command in transport.frontend_calls if command[0] == "oarsub")
    assert '--job-id "$OAR_JOB_ID"' in submit_command[3]
    assert "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN" in submit_command[3]
    assert sum(command[0] == "usagepolicycheck" for command in transport.frontend_calls) >= 3
    assert transport.removals[-1].endswith("run-20260827-01")


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
