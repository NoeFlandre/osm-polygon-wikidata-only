"""Tests for the CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest

import osm_polygon_wikidata_only.cli.commands as commands
from osm_polygon_wikidata_only.cli.commands import _build_settings, build_parser, main
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult


def test_parser_has_documented_subcommands() -> None:
    parser = build_parser()
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub_action.choices) == {
        "augment-dir",
        "augment-region",
        "process-pbf",
        "process-dir",
        "sync-dir",
        "split-v2-sentences",
    }


def test_parser_accepts_canonical_sync_dir() -> None:
    args = build_parser().parse_args(["sync-dir", "/tmp/raw", "--skip-existing", "--push"])
    assert args.command == "sync-dir"
    assert args.input == Path("/tmp/raw")
    assert args.skip_existing is True


@pytest.mark.parametrize("command", ["process-pbf", "process-dir"])
def test_processing_inputs_preserve_the_cli_path(command: str, tmp_path: Path) -> None:
    input_path = tmp_path / ("region.osm.pbf" if command == "process-pbf" else "raw")

    assert commands._processing_inputs(command, input_path) == [input_path]


@pytest.mark.parametrize("push", [False, True])
def test_processing_command_forwards_completion_to_optional_upload_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, push: bool
) -> None:
    root = DataRoot(tmp_path)
    config = Settings(repo_id="example/repo")
    input_path = tmp_path / "region.osm.pbf"
    args = argparse.Namespace(
        command="process-pbf",
        input=input_path,
        push=push,
        dry_run=True,
        upload_threads=1,
        commit_message=None,
    )
    result = ProcessResult(
        polygons_path=tmp_path / "polygons.parquet",
        articles_path=tmp_path / "articles.parquet",
        polygon_articles_path=tmp_path / "polygon-articles.parquet",
        manifest_path=tmp_path / "manifest.json",
        polygon_count=1,
        article_count=1,
        link_count=1,
        manifest_entry={"source_pbf": "region.osm.pbf"},
        stage_timings_s={},
    )
    queue_closed = False
    enqueue_calls: list[tuple[object, DataRoot, str, str, ProcessResult]] = []

    class FakeUploadQueue:
        def close_and_wait(self) -> list[str]:
            nonlocal queue_closed
            queue_closed = True
            return []

    queue = FakeUploadQueue()

    def fake_build_clients(
        received_settings: Settings, *, data_root: DataRoot
    ) -> tuple[str, str, str]:
        assert received_settings is config
        assert data_root is root
        return "wikidata", "wikipedia", "cache"

    def fake_build_upload_queue(
        _args: argparse.Namespace,
        _settings: Settings,
        *,
        data_root: DataRoot,
    ) -> FakeUploadQueue | None:
        assert data_root is root
        return queue if push else None

    def fake_orchestrate(
        inputs: list[Path],
        *,
        data_root: DataRoot,
        settings: Settings,
        wikidata_client: str,
        wikipedia_client: str,
        cache: str,
        on_complete: Callable[[ProcessResult], None],
    ) -> list[ProcessResult]:
        assert inputs == [input_path]
        assert data_root is root
        assert settings is config
        assert (wikidata_client, wikipedia_client, cache) == (
            "wikidata",
            "wikipedia",
            "cache",
        )
        on_complete(result)
        return [result]

    def fake_enqueue_core_upload(
        received_queue: object,
        *,
        data_root: DataRoot,
        repo_id: str,
        commit_message: str,
        result: ProcessResult,
    ) -> None:
        enqueue_calls.append((received_queue, data_root, repo_id, commit_message, result))

    monkeypatch.setattr(commands, "_build_clients", fake_build_clients)
    monkeypatch.setattr(commands, "_build_upload_queue", fake_build_upload_queue)
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_enqueue_core_upload", fake_enqueue_core_upload)

    assert commands._run_processing_command(args, data_root=root, settings=config) == 0
    assert queue_closed is push
    if push:
        assert enqueue_calls == [
            (
                queue,
                root,
                "example/repo",
                "Update PBF region.osm.pbf",
                result,
            )
        ]
    else:
        assert enqueue_calls == []


def test_augmentation_stems_selects_one_region_or_completed_regions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    completed = ["b-latest", "a-latest"]
    monkeypatch.setattr(commands, "completed_region_stems", lambda _root: completed)

    assert commands._augmentation_stems("augment-region", "one-latest", data_root) == ["one-latest"]
    assert commands._augmentation_stems("augment-dir", None, data_root) == completed


def test_load_augmentation_result_skips_a_current_canonical_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_wikidata_only.pipeline.link_migration as migration

    data_root = DataRoot(tmp_path)
    args = argparse.Namespace(skip_existing=True)
    monkeypatch.setattr(commands, "augmentation_is_current", lambda *_args: True)

    class EmptyPlan:
        stems: tuple[object, ...] = ()

    monkeypatch.setattr(migration, "plan_link_migration", lambda *_args, **_kwargs: EmptyPlan())

    assert (
        commands._load_augmentation_result(
            args,
            data_root=data_root,
            stem="andorra-latest",
            augmentation_client=object(),  # type: ignore[arg-type]
        )
        is None
    )


def test_load_augmentation_result_augments_and_loads_when_not_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_wikidata_only.augmentation.orchestrator as orchestrator
    import osm_polygon_wikidata_only.pipeline.link_migration as migration

    data_root = DataRoot(tmp_path)
    args = argparse.Namespace(skip_existing=False)
    calls: list[str] = []
    marker = object()
    monkeypatch.setattr(
        commands, "augment_region", lambda *_args, **_kwargs: calls.append("augment")
    )
    monkeypatch.setattr(
        migration, "apply_link_migration", lambda *_args, **_kwargs: calls.append("migrate")
    )
    monkeypatch.setattr(
        orchestrator,
        "load_existing_augmentation_result",
        lambda *_args, **_kwargs: marker,
    )

    result = commands._load_augmentation_result(
        args,
        data_root=data_root,
        stem="andorra-latest",
        augmentation_client=object(),  # type: ignore[arg-type]
    )

    assert result is marker
    assert calls == ["augment", "migrate"]


def test_publish_augmentation_submits_the_assembled_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_wikidata_only.hf.publication as publication

    data_root = DataRoot(tmp_path)
    args = argparse.Namespace(
        push=True,
        dry_run=True,
        commit_message=None,
        upload_threads=2,
    )
    settings = Settings(repo_id="example/repo")
    submitted: list[tuple[object, str]] = []
    monkeypatch.setattr(publication, "assemble_augmentation_upload", lambda **_kwargs: ["op"])
    monkeypatch.setattr(
        commands,
        "upload_files",
        lambda repo_id, **kwargs: submitted.append((kwargs["ops"], kwargs["commit_message"])),
    )

    commands._publish_augmentation(
        args,
        settings,
        data_root=data_root,
        stem="andorra-latest",
        result=object(),  # type: ignore[arg-type]
    )

    assert submitted == [(["op"], "Add text augmentation for andorra-latest")]


def test_run_augmentation_command_processes_selected_stems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    args = argparse.Namespace(
        command="augment-dir",
        stem=None,
        skip_existing=False,
        push=False,
    )
    result = argparse.Namespace(counts={"wikipedia_documents": 1})
    monkeypatch.setattr(commands, "AugmentationWikimediaClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(commands, "_augmentation_stems", lambda *_args: ["andorra-latest"])
    monkeypatch.setattr(commands, "_load_augmentation_result", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(commands, "_publish_augmentation", lambda *_args, **_kwargs: None)

    assert commands._run_augmentation_command(args, data_root=data_root, settings=Settings()) == 0


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.hf.uploader import UploadError

    messages: list[str] = []

    def _fake_verify(token: str | None) -> str | None:
        raise UploadError("Hugging Face rejected HF_TOKEN: invalid.")

    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "present")
    monkeypatch.setattr(commands, "verify_hf_token", _fake_verify)
    monkeypatch.setattr(commands.LOGGER, "info", messages.append)
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        main(["process-dir", str(raw), "--data-root", str(tmp_path), "--push"])
    assert excinfo.value.code == 2
    assert any("Connecting to Hugging Face" in message for message in messages)


def test_main_push_logs_authenticated_username(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "present")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "noeflandre")
    monkeypatch.setattr(commands, "verify_repo_authorization", lambda token, repo_id: "noeflandre")
    raw = tmp_path / "raw"
    raw.mkdir()
    caplog.set_level("INFO", logger="osm_polygon_wikidata_only.cli")
    # Authentication is intentionally exercised on the real push path. A
    # dry-run skips Hugging Face verification, so it cannot prove this log.
    assert main(["process-dir", str(raw), "--data-root", str(tmp_path), "--push"]) == 0
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


def test_commands_main_signature() -> None:
    import inspect

    sig = inspect.signature(main)
    assert len(sig.parameters) == 1
    assert "argv" in sig.parameters
    assert sig.parameters["argv"].default is None
