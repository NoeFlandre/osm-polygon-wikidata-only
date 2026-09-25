from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli import commands
from osm_polygon_wikidata_only.cli import grid5000 as grid5000_sentence_controller
from scripts import grid5000_sentence_controller as controller_shim
from scripts import grid5000_sentence_job as job_shim


def test_controller_cli_forwards_all_resumable_run_options(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_run(data_root, **kwargs):
        captured["data_root"] = data_root
        captured.update(kwargs)
        return {"run_id": "run-test", "batches": [{"state": "published"}]}

    monkeypatch.setattr(
        grid5000_sentence_controller,
        "run_grid5000_sentence_controller",
        fake_run,
    )

    assert (
        commands.main(
            [
                "grid5000",
                "controller",
                "--data-root",
                str(tmp_path / "data-root"),
                "--site",
                "lyon",
                "--queue",
                "besteffort",
                "--repo-id",
                "example/dataset",
                "--max-stems",
                "3",
                "--max-input-bytes",
                "1234",
                "--batch-size",
                "64",
                "--inference-batch-size",
                "8",
                "--walltime",
                "0:20",
                "--run-id",
                "run-test",
                "--hf-token",
                "test-token",
            ]
        )
        == 0
    )

    assert captured["data_root"].path == tmp_path / "data-root"
    assert captured["site"] == "lyon"
    assert captured["queue"] == "besteffort"
    assert captured["repo_id"] == "example/dataset"
    assert captured["max_stems"] == 3
    assert captured["max_input_bytes"] == 1234
    assert captured["batch_size"] == 64
    assert captured["inference_batch_size"] == 8
    assert captured["walltime"] == "0:20"
    assert captured["run_id"] == "run-test"
    assert captured["hf_token"] == "test-token"
    assert "run-test" in capsys.readouterr().out


def test_job_subcommand_forwards_reserved_node_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    class Receipt:
        job_id = "42"
        status = "succeeded"

    def fake_job(data_root, **kwargs):
        captured["data_root"] = data_root
        captured.update(kwargs)
        return Receipt()

    monkeypatch.setattr(grid5000_sentence_controller, "run_sentence_job", fake_job)
    argv = [
        "--data-root",
        str(tmp_path / "data"),
        "--stems",
        "a-latest",
        "b-latest",
        "--model-cache",
        str(tmp_path / "models"),
        "--source-commit",
        "abc",
        "--job-id",
        "42",
        "--receipt",
        str(tmp_path / "receipt.json"),
    ]

    assert commands.main(["grid5000", "job", *argv]) == 0
    assert captured["data_root"].path == tmp_path / "data"
    assert captured["stems"] == ["a-latest", "b-latest"]
    assert captured["source_commit"] == "abc"
    assert captured["batch_size"] == 256
    assert captured["receipt_path"] == tmp_path / "receipt.json"
    assert "Grid5000 sentence job 42: succeeded" in capsys.readouterr().out

    assert job_shim.main(argv) == 0


def test_scripts_are_shims_over_the_packaged_commands() -> None:
    assert controller_shim.main is grid5000_sentence_controller.controller_main
    assert job_shim.main is grid5000_sentence_controller.job_main
