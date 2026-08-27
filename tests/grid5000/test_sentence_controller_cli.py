from __future__ import annotations

from pathlib import Path

from scripts import grid5000_sentence_controller


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
        grid5000_sentence_controller.main(
            [
                "--data-root",
                str(tmp_path / "data-root"),
                "--site",
                "lyon",
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
    assert captured["repo_id"] == "example/dataset"
    assert captured["max_stems"] == 3
    assert captured["max_input_bytes"] == 1234
    assert captured["batch_size"] == 64
    assert captured["inference_batch_size"] == 8
    assert captured["walltime"] == "0:20"
    assert captured["run_id"] == "run-test"
    assert captured["hf_token"] == "test-token"
    assert "run-test" in capsys.readouterr().out
