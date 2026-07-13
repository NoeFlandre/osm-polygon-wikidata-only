"""Tests for the CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import osm_polygon_wikidata_only.cli.commands as commands
from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.cli.commands import _build_settings, build_parser, main
from osm_polygon_wikidata_only.config.paths import DataRoot


def test_parser_has_two_subcommands() -> None:
    parser = build_parser()
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub_action.choices) == {
        "augment-dir",
        "augment-region",
        "process-pbf",
        "process-dir",
        "sync-dir",
    }


def test_parser_accepts_canonical_sync_dir() -> None:
    args = build_parser().parse_args(["sync-dir", "/tmp/raw", "--skip-existing", "--push"])
    assert args.command == "sync-dir"
    assert args.input == Path("/tmp/raw")
    assert args.skip_existing is True


def test_sync_dir_handles_empty_directory_without_network(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    assert main(["sync-dir", str(raw), "--data-root", str(tmp_path), "--skip-existing"]) == 0


def test_parser_accepts_additive_region_augmentation() -> None:
    args = build_parser().parse_args(["augment-region", "andorra-latest", "--push"])
    assert args.command == "augment-region"
    assert args.stem == "andorra-latest"
    assert args.push is True


def test_parser_process_pbf_accepts_input(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["process-pbf", str(tmp_path / "x.osm.pbf"), "--all-languages"])
    assert args.command == "process-pbf"
    assert args.all_languages is True
    assert args.no_full_text is False


def test_parser_process_pbf_no_full_text_disables_field(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "process-pbf",
            str(tmp_path / "x.osm.pbf"),
            "--no-full-text",
            "--languages",
            "en,fr",
        ]
    )
    assert args.no_full_text is True
    assert args.languages == "en,fr"


def test_parser_process_dir_default_skip() -> None:
    parser = build_parser()
    args = parser.parse_args(["process-dir", "/tmp/abc"])
    assert args.skip_existing is False
    assert args.force is False


def test_parser_push_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "process-pbf",
            "/tmp/x.osm.pbf",
            "--push",
            "--repo-id",
            "foo/bar",
        ]
    )
    assert args.push is True
    assert args.repo_id == "foo/bar"


def test_parser_accepts_upload_worker_count() -> None:
    parser = build_parser()
    args = parser.parse_args(["process-pbf", "/tmp/x.osm.pbf", "--upload-threads", "8"])
    assert args.upload_threads == 8


def test_parser_accepts_explicit_hf_token() -> None:
    parser = build_parser()
    args = parser.parse_args(["process-pbf", "/tmp/x.osm.pbf", "--push", "--hf-token", "hf_secret"])
    assert args.hf_token == "hf_secret"
    settings = _build_settings(args)
    assert settings.hf_token == "hf_secret"


def test_main_push_without_token_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: None)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        main(["process-dir", str(raw), "--data-root", str(tmp_path), "--push"])
    assert excinfo.value.code == 2


def test_main_push_rejects_token_rejected_by_whoami(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osm_polygon_wikidata_only.hf.uploader import UploadError

    def _fake_verify(token: str | None) -> str | None:
        raise UploadError("Hugging Face rejected HF_TOKEN: invalid.")

    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "present")
    monkeypatch.setattr(commands, "verify_hf_token", _fake_verify)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        main(["process-dir", str(raw), "--data-root", str(tmp_path), "--push"])
    assert excinfo.value.code == 2


def test_main_push_logs_authenticated_username(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "present")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "noeflandre")
    monkeypatch.setattr(commands, "verify_repo_authorization", lambda token, repo_id: "noeflandre")
    raw = tmp_path / "raw"
    raw.mkdir()
    caplog.set_level("INFO", logger="osm_polygon_wikidata_only.cli")
    assert main(["process-dir", str(raw), "--data-root", str(tmp_path), "--push", "--dry-run"]) == 0
    assert any("noeflandre" in record.getMessage() for record in caplog.records)


def test_main_push_aborts_when_namespace_does_not_match_token_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osm_polygon_wikidata_only.hf.uploader import UploadError

    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "present")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "someoneelse")

    def _fake_authorize(token: str | None, repo_id: str) -> str:
        raise UploadError(
            "HF_TOKEN authenticates as 'someoneelse', but --repo-id "
            "'NoeFlandre/osm-polygon-wikidata-only' lives in the 'NoeFlandre' namespace."
        )

    monkeypatch.setattr(commands, "verify_repo_authorization", _fake_authorize)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        main(["process-dir", str(raw), "--data-root", str(tmp_path), "--push"])
    assert excinfo.value.code == 2


def test_main_push_distinguishes_missing_token_from_invalid_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxxxxxxxxxxxxxxxxxxx")
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: None)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        main(["process-dir", str(raw), "--data-root", str(tmp_path), "--push"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "rejected" in err or "invalid" in err
    assert "https://huggingface.co/settings/tokens" in err


def test_main_push_reports_invalid_explicit_hf_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: None)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "process-dir",
                str(raw),
                "--data-root",
                str(tmp_path),
                "--push",
                "--hf-token",
                "definitely-not-a-real-token",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--hf-token" in err
    assert "rejected" in err or "invalid" in err


def test_normal_command_defaults_to_every_language_without_article_cap() -> None:
    args = build_parser().parse_args(["process-dir", "/tmp/pbfs"])
    settings = _build_settings(args)
    assert settings.languages is None
    assert settings.fetch_full_text is True
    assert settings.max_articles_per_qid is None
    assert settings.enrichment_site_workers == 8


def test_main_handles_empty_directory_without_network(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    assert main(["process-dir", str(raw), "--data-root", str(tmp_path)]) == 0


def test_main_drains_dry_run_upload_queue_for_empty_directory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    assert (
        main(
            [
                "process-dir",
                str(raw),
                "--data-root",
                str(tmp_path),
                "--push",
                "--dry-run",
            ]
        )
        == 0
    )


def test_write_readme_snapshot_uses_canonical_current_dataset_state(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    data_root.processed_manifests.joinpath("processed_pbfs.json").write_text(
        json.dumps(
            {
                "andorra-latest.osm.pbf": {
                    "polygon_count": 2,
                    "article_count": 3,
                    "unique_wikidata_count": 1,
                }
            }
        )
    )
    destination = tmp_path / "README.md"

    commands._write_readme_snapshot(data_root, "org/dataset", destination)

    markdown = destination.read_text()
    assert "# org/dataset" in markdown
    assert "polygon_count: 2" in markdown
    assert "wikivoyage/documents/<stem>.parquet" in markdown


def test_augmentation_upload_files_include_fresh_readme(tmp_path: Path) -> None:
    paths = [tmp_path / f"artifact-{index}" for index in range(6)]
    readme = tmp_path / "README.md"
    result = AugmentationResult(*paths[:5], paths[5], {})

    files = commands._augmentation_upload_files(result, tmp_path, readme)

    assert files[-1] == (readme, "README.md")
