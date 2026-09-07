"""Reserved-node execution cannot silently use a frontend or CPU."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_reservation_required_before_downloading(tmp_path, monkeypatch):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    monkeypatch.delenv("OAR_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match=r"^Geographic NER requires an OAR reservation$"):
        job.main(
            [
                "--source",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path),
                "--contract",
                str(tmp_path / "contract.json"),
                "--model-cache",
                str(tmp_path),
            ]
        )


def test_node_identity_and_cuda_are_required(monkeypatch):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(job, "import_module", lambda name: torch)
    with pytest.raises(
        RuntimeError, match=r"^Geographic NER requires CUDA; CPU fallback is disabled$"
    ):
        job.gpu_identity()


def test_gpu_identity_reports_the_reserved_cuda_runtime(monkeypatch):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    calls = []

    def device_name(index):
        calls.append(index)
        return "A40"

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=device_name),
        __version__="2.8.0",
        version=SimpleNamespace(cuda="12.8"),
    )

    def import_runtime(name):
        assert name == "torch"
        return torch

    monkeypatch.setattr(job, "import_module", import_runtime)
    assert job.gpu_identity() == {"name": "A40", "torch": "2.8.0", "cuda": "12.8"}
    assert calls == [0]


def test_parser_keeps_typed_inputs_and_runtime_defaults(tmp_path):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    args = job._parser().parse_args(
        [
            "--source",
            str(tmp_path / "input.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--contract",
            str(tmp_path / "contract.json"),
            "--model-cache",
            str(tmp_path / "cache"),
            "--seconds",
            "60",
            "--batch-size",
            "3",
            "--inference-batch-size",
            "2",
        ]
    )
    assert args.source == Path(tmp_path / "input.parquet")
    assert args.output_dir == Path(tmp_path / "out")
    assert args.contract == Path(tmp_path / "contract.json")
    assert args.model_cache == Path(tmp_path / "cache")
    assert (args.seconds, args.batch_size, args.inference_batch_size) == (60, 3, 2)

    defaults = job._parser().parse_args(
        [
            "--source",
            str(tmp_path / "input.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--contract",
            str(tmp_path / "contract.json"),
            "--model-cache",
            str(tmp_path / "cache"),
        ]
    )
    assert (defaults.seconds, defaults.batch_size, defaults.inference_batch_size) == (900, 128, 16)


def test_parser_help_describes_the_gpu_job():
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")

    assert job._parser().description == job.__doc__


@pytest.mark.parametrize("missing", ["--source", "--output-dir", "--contract", "--model-cache"])
def test_parser_requires_each_input_path(tmp_path, missing):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    values = {
        "--source": str(tmp_path / "input.parquet"),
        "--output-dir": str(tmp_path / "out"),
        "--contract": str(tmp_path / "contract.json"),
        "--model-cache": str(tmp_path / "cache"),
    }
    argv = [item for option in values for item in (option, values[option]) if option != missing]
    with pytest.raises(SystemExit):
        job._parser().parse_args(argv)


@pytest.mark.parametrize("seconds", [0, 1021])
def test_invalid_budget_is_rejected_before_gpu_checks(tmp_path, monkeypatch, seconds):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")

    def fail_gpu_check():
        raise AssertionError("GPU checks must not run for invalid CLI input")

    monkeypatch.setattr(job, "gpu_identity", fail_gpu_check)
    with pytest.raises(ValueError, match=r"^Job budget must be between 1 and 1020 seconds$"):
        job.main(
            [
                "--source",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path),
                "--contract",
                str(tmp_path / "contract.json"),
                "--model-cache",
                str(tmp_path),
                "--seconds",
                str(seconds),
            ]
        )


def test_job_rejects_string_languages_before_model_download(tmp_path, monkeypatch):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"languages": "en"}))
    monkeypatch.setattr(
        job, "gpu_identity", lambda: {"name": "A40", "torch": "2.8.0", "cuda": "12.8"}
    )

    def fail_download(**kwargs):
        raise AssertionError("model download must not run for malformed contract input")

    monkeypatch.setattr(job, "snapshot_download", fail_download)
    with pytest.raises(ValueError, match=r"^Contract languages must be a list or tuple$"):
        job.main(
            [
                "--source",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path),
                "--contract",
                str(contract_path),
                "--model-cache",
                str(tmp_path),
            ]
        )


@pytest.mark.parametrize("seconds", [1, 1020])
def test_job_pins_snapshot_and_records_execution(tmp_path, monkeypatch, seconds, capsys):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    from dataclasses import asdict

    from osm_polygon_wikidata_only.ner.pipeline import MODEL_ID, MODEL_REVISION, Contract

    monkeypatch.setenv("OAR_JOB_ID", "12345")
    clock = iter([100.0, 101.25])
    monkeypatch.setattr(job, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(asdict(Contract(languages=("en",)))))
    monkeypatch.setattr(
        job, "gpu_identity", lambda: {"name": "A40", "torch": "2.8.0", "cuda": "12.8"}
    )
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path / "model")

    monkeypatch.setattr(job, "snapshot_download", download)
    atomic_paths = []
    real_atomic_write_json = job.atomic_write_json

    def record_atomic_path(path, value):
        atomic_paths.append(path)
        real_atomic_write_json(path, value)

    monkeypatch.setattr(job, "atomic_write_json", record_atomic_path)
    extractors = []

    def make_extractor(*args, **kwargs):
        extractors.append((args, kwargs))
        return "extractor"

    monkeypatch.setattr(job, "OtterLocationExtractor", make_extractor)
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        receipt = {"processed_rows": 7, "status": "completed"}
        (tmp_path / "receipt.json").write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(job, "run_shard", run)
    assert (
        job.main(
            [
                "--source",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path),
                "--contract",
                str(contract_path),
                "--model-cache",
                str(tmp_path),
                "--seconds",
                str(seconds),
            ]
        )
        == 0
    )
    assert downloads[0]["repo_id"] == MODEL_ID
    assert downloads[0]["revision"] == MODEL_REVISION
    assert downloads[0]["cache_dir"] == tmp_path
    assert downloads[0]["max_workers"] == 2
    assert extractors == [((Path(tmp_path / "model"),), {"batch_size": 16, "threshold": 0.5})]
    assert calls[0][0][:2] == (Path(tmp_path / "input.parquet"), tmp_path)
    assert calls[0][0][3] == Contract(languages=("en",))
    assert calls[0][1]["batch_size"] == 128
    assert calls[0][1]["deadline"] == 100 + seconds
    assert calls[0][0][2] == "extractor"
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    execution = receipt["executions"][0]
    assert set(execution) == {"job_id", "gpu", "status", "elapsed_seconds"}
    assert execution["job_id"] == "12345"
    assert execution["gpu"]["name"] == "A40"
    assert execution["status"] == "completed"
    assert execution["elapsed_seconds"] == pytest.approx(1.25)
    assert atomic_paths == [tmp_path / "receipt.json"]
    assert json.loads(capsys.readouterr().out) == {
        "status": "completed",
        "processed_rows": 7,
    }


def test_job_records_failed_execution_when_shard_raises(tmp_path, monkeypatch):
    job = importlib.import_module("osm_polygon_wikidata_only.ner.job")
    from dataclasses import asdict

    from osm_polygon_wikidata_only.ner.pipeline import Contract

    monkeypatch.setenv("OAR_JOB_ID", "12345")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(asdict(Contract(languages=("en",)))))
    monkeypatch.setattr(
        job, "gpu_identity", lambda: {"name": "A40", "torch": "2.8.0", "cuda": "12.8"}
    )
    monkeypatch.setattr(job, "snapshot_download", lambda **kwargs: str(tmp_path / "model"))
    monkeypatch.setattr(job, "OtterLocationExtractor", lambda *args, **kwargs: "extractor")

    def fail_run(*args, **kwargs):
        (tmp_path / "receipt.json").write_text(json.dumps({"status": "failed"}))
        raise RuntimeError("inference failed")

    monkeypatch.setattr(job, "run_shard", fail_run)
    with pytest.raises(RuntimeError, match="inference failed"):
        job.main(
            [
                "--source",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path),
                "--contract",
                str(contract_path),
                "--model-cache",
                str(tmp_path),
                "--seconds",
                "1",
            ]
        )

    receipt = json.loads((tmp_path / "receipt.json").read_text())
    execution = receipt["executions"][0]
    assert set(execution) == {"job_id", "gpu", "status", "elapsed_seconds"}
    assert execution["status"] == "failed"
