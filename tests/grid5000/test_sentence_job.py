from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000 import sentence_job
from osm_polygon_wikidata_only.v2.sentence_logic import (
    SAT_MODEL_ID,
    sentence_schema,
)
from osm_polygon_wikidata_only.v2.sentence_runner import (
    SentenceRegionSummary,
    SentenceRunResult,
)


def _write_table(path: Path, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field.name: None for field in schema}
    if "text" in schema.names:
        row["text"] = "First."
    pq.write_table(pa.Table.from_pylist([row], schema=schema), path)


def _data_root(tmp_path: Path) -> DataRoot:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    section_path = data_root.processed_v2 / "wikipedia/sections/alpha-latest.parquet"
    _write_table(section_path, section_schema())
    manifest_path = data_root.processed_v2 / "manifests/processed_pbfs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"regions": {"alpha-latest": {"sections_path": str(section_path)}}}),
        encoding="utf-8",
    )
    return data_root


def _fake_onnxruntime(providers: list[str]) -> ModuleType:
    module = ModuleType("onnxruntime")
    module.get_available_providers = lambda: providers  # type: ignore[attr-defined]
    return module


class _FakeSegmenter:
    model_id = SAT_MODEL_ID
    version = "2.2.1"
    revision = "137da05"
    ort_providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    init_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs = kwargs


def _gpu_runner(_args: object) -> CompletedProcess[str]:
    return CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="NVIDIA A100, GPU-uuid-1, 40960 MiB\n",
        stderr="",
    )


def test_sentence_job_requires_nvidia_smi_and_writes_sanitized_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _data_root(tmp_path)
    receipt_path = tmp_path / "receipt.json"

    def failed_runner(_args: object) -> CompletedProcess[str]:
        return CompletedProcess(
            args=["nvidia-smi"],
            returncode=1,
            stdout="",
            stderr="secret-token should not be copied",
        )

    with pytest.raises(RuntimeError, match="nvidia-smi"):
        sentence_job.run_sentence_job(
            data_root,
            stems=("alpha-latest",),
            model_cache=tmp_path / "model-cache",
            source_commit="abc123",
            job_id="job-1",
            batch_size=256,
            inference_batch_size=16,
            receipt_path=receipt_path,
            command_runner=failed_runner,
        )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert "secret-token" not in receipt_path.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in receipt_path.read_text(encoding="utf-8")


def test_sentence_job_rejects_onnxruntime_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _data_root(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_onnxruntime(["CPUExecutionProvider"]))

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        sentence_job.run_sentence_job(
            data_root,
            stems=("alpha-latest",),
            model_cache=tmp_path / "model-cache",
            source_commit="abc123",
            job_id="job-1",
            batch_size=256,
            inference_batch_size=16,
            receipt_path=receipt_path,
            command_runner=_gpu_runner,
        )

    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_sentence_job_redacts_inference_exception_from_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _data_root(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _fake_onnxruntime(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    monkeypatch.setattr(sentence_job, "SaT3lSegmenter", _FakeSegmenter)

    def failed_run(
        _data_root: DataRoot,
        *,
        segmenter: object,
        batch_size: int,
        stems: tuple[str, ...],
    ) -> SentenceRunResult:
        del segmenter, batch_size, stems
        raise RuntimeError("secret-token from an arbitrary model failure")

    monkeypatch.setattr(sentence_job, "run_v2_sentence_split", failed_run)

    with pytest.raises(RuntimeError, match="secret-token"):
        sentence_job.run_sentence_job(
            data_root,
            stems=("alpha-latest",),
            model_cache=tmp_path / "model-cache",
            source_commit="abc123",
            job_id="job-1",
            batch_size=256,
            inference_batch_size=16,
            receipt_path=receipt_path,
            command_runner=_gpu_runner,
        )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["error_message"] == "redacted; inspect the reserved-job log"
    assert "secret-token" not in receipt_path.read_text(encoding="utf-8")


def test_sentence_job_runs_selected_stems_and_writes_provenance_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _data_root(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _fake_onnxruntime(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    monkeypatch.setattr(sentence_job, "SaT3lSegmenter", _FakeSegmenter)
    calls: list[tuple[tuple[str, ...], int, int]] = []

    def fake_run(
        _data_root: DataRoot,
        *,
        segmenter: object,
        batch_size: int,
        stems: tuple[str, ...],
    ) -> SentenceRunResult:
        calls.append((stems, batch_size, id(segmenter)))
        output_path = _data_root.processed_v2 / "wikipedia/sentences/alpha-latest.parquet"
        _write_table(output_path, sentence_schema())
        manifest_path = _data_root.processed_v2 / "manifests/sentence_splitting.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"regions": []}\n', encoding="utf-8")
        checkpoint = _data_root.v2_cache / "sentence-checkpoints/alpha-latest/wikipedia"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
        return SentenceRunResult(
            manifest_path=manifest_path,
            regions=(
                SentenceRegionSummary(
                    stem="alpha-latest",
                    project="wikipedia",
                    sections=1,
                    split_sections=1,
                    unsplit_sections=0,
                    sentence_rows=1,
                    supported_languages=("en",),
                    unsupported_languages=(),
                ),
            ),
        )

    monkeypatch.setattr(sentence_job, "run_v2_sentence_split", fake_run)

    receipt = sentence_job.run_sentence_job(
        data_root,
        stems=("alpha-latest",),
        model_cache=tmp_path / "model-cache",
        source_commit="abc123",
        job_id="job-1",
        batch_size=32,
        inference_batch_size=7,
        receipt_path=receipt_path,
        command_runner=_gpu_runner,
    )

    assert receipt.status == "succeeded"
    assert receipt.source_commit == "abc123"
    assert receipt.stems == ("alpha-latest",)
    assert calls and calls[0][0] == ("alpha-latest",)
    assert calls[0][1] == 32
    assert _FakeSegmenter.init_kwargs == {
        "cache_dir": tmp_path / "model-cache",
        "revision": "137da05",
        "inference_batch_size": 7,
        "ort_providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "require_gpu": True,
    }
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["model_id"] == SAT_MODEL_ID
    assert payload["model_revision"] == "137da05"
    assert payload["ort_providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert payload["gpu"][0]["uuid"] == "GPU-uuid-1"
    assert payload["artifacts"]
    assert payload["regions"][0]["sentence_rows"] == 1
