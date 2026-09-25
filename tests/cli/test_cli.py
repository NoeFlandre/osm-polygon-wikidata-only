"""Tests for the CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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
        "language-splits",
        "publish-language-splits",
        "release-stats",
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
        defer_metadata_assets: bool = False,
    ) -> None:
        del defer_metadata_assets
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


def test_load_augmentation_result_skip_existing_plans_link_migration_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_wikidata_only.augmentation.orchestrator as orchestrator
    import osm_polygon_wikidata_only.pipeline.link_migration as migration

    data_root = DataRoot(tmp_path)
    args = argparse.Namespace(skip_existing=True)
    marker = object()
    legacy_plan = SimpleNamespace(
        stems=(SimpleNamespace(classification=migration.StemClassification.MIGRATABLE),)
    )
    plan_calls: list[object] = []
    applied: list[object] = []
    monkeypatch.setattr(commands, "augmentation_is_current", lambda *_args: True)
    monkeypatch.setattr(
        migration,
        "plan_link_migration",
        lambda *_args, **kwargs: plan_calls.append(kwargs) or legacy_plan,
    )
    monkeypatch.setattr(
        migration,
        "apply_link_migration",
        lambda *_args, **kwargs: applied.append(kwargs.get("plan")),
    )
    monkeypatch.setattr(
        orchestrator, "load_existing_augmentation_result", lambda *_args, **_kwargs: marker
    )

    result = commands._load_augmentation_result(
        args,
        data_root=data_root,
        stem="andorra-latest",
        augmentation_client=object(),  # type: ignore[arg-type]
    )

    assert result is marker
    assert plan_calls == [{"stems": {"andorra-latest"}}]
    assert applied == [legacy_plan]


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


def test_languages_option_is_trimmed_deduplicated_and_sorted(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["process-pbf", str(tmp_path / "x.osm.pbf"), "--languages", " fr, en,fr ,, de "]
    )

    assert _build_settings(args).languages == ("de", "en", "fr")


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


def test_process_dir_defers_metadata_assets_until_the_queue_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory run publishes each region, then refreshes metadata once."""
    submissions: list[tuple[str, object]] = []
    deferrals: list[bool] = []

    class _StubQueue:
        def submit(self, ops: object, message: str) -> None:
            submissions.append((message, ops))

        def close_and_wait(self) -> list[str]:
            return []

    queue = _StubQueue()
    root = DataRoot(tmp_path)
    config = Settings(repo_id="example/repo")
    args = argparse.Namespace(
        command="process-dir",
        input=tmp_path / "pbfs",
        commit_message=None,
        push=True,
    )
    result = SimpleNamespace(manifest_entry={"source_pbf": "region.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    def fake_enqueue_core_upload(
        received_queue: object,
        *,
        data_root: DataRoot,
        repo_id: str,
        commit_message: str,
        result: object,
        defer_metadata_assets: bool = False,
    ) -> None:
        deferrals.append(defer_metadata_assets)
        submissions.append(("region", received_queue))

    def fake_metadata_refresh(
        received_queue: object,
        *,
        data_root: DataRoot,
        repo_id: str,
    ) -> None:
        submissions.append(("metadata", received_queue))

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: queue)
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_enqueue_core_upload", fake_enqueue_core_upload)
    monkeypatch.setattr(commands, "_upload_metadata_refresh", fake_metadata_refresh)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)

    assert commands._run_processing_command(args, data_root=root, settings=config) == 0

    assert deferrals == [True]
    # Exactly one metadata refresh, after the region commit.
    assert [name for name, _ in submissions] == ["region", "metadata"]


def test_process_pbf_publishes_metadata_assets_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-PBF run keeps the historical inline metadata publication."""
    deferrals: list[bool] = []
    refreshes: list[object] = []

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    queue = _StubQueue()
    args = argparse.Namespace(
        command="process-pbf",
        input=tmp_path / "region.osm.pbf",
        commit_message=None,
        push=True,
    )
    result = SimpleNamespace(manifest_entry={"source_pbf": "region.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: queue)
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(
        commands,
        "_enqueue_core_upload",
        lambda *a, **kw: deferrals.append(bool(kw["defer_metadata_assets"])),
    )
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: refreshes.append(kw))
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)

    assert (
        commands._run_processing_command(
            args, data_root=DataRoot(tmp_path), settings=Settings(repo_id="example/repo")
        )
        == 0
    )

    assert deferrals == [False]
    assert refreshes == []


def test_process_dir_persists_and_clears_the_metadata_refresh_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred refresh leaves a durable marker until it succeeds."""
    markers: list[tuple[list[str], dict[str, str]]] = []
    cleared: list[DataRoot] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "region-latest.parquet").write_bytes(b"polygons")

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    args = argparse.Namespace(
        command="process-dir",
        input=tmp_path / "pbfs",
        commit_message=None,
        push=True,
    )
    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(
        commands,
        "set_metadata_refresh_marker",
        lambda data_root, stems, hashes: markers.append((stems, hashes)),
    )
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            args, data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 0
    )

    assert len(markers) == 1
    stems, hashes = markers[0]
    assert stems == ["region-latest"]
    assert set(hashes) == {"region-latest"}
    assert len(hashes["region-latest"]) == 64
    assert cleared == [root]


def test_process_dir_keeps_the_marker_when_an_upload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed upload must leave the refresh intent on disk for the next run."""
    cleared: list[DataRoot] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "region-latest.parquet").write_bytes(b"polygons")

    class _FailingQueue:
        def close_and_wait(self) -> list[str]:
            return ["upload failed"]

    args = argparse.Namespace(
        command="process-dir",
        input=tmp_path / "pbfs",
        commit_message=None,
        push=True,
    )
    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _FailingQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "set_metadata_refresh_marker", lambda *a, **kw: None)
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            args, data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 1
    )
    assert cleared == []


def _marker(stems: list[str]) -> dict[str, object]:
    """Return a metadata-refresh marker payload shaped like the real one."""
    return {
        "stems": sorted(stems),
        "fingerprint_hashes": {stem: "a" * 64 for stem in sorted(stems)},
    }


def _deferring_args(tmp_path: Path, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        command="process-dir",
        input=tmp_path / "pbfs",
        commit_message=None,
        push=True,
        dry_run=dry_run,
    )


def test_a_failed_regional_upload_skips_the_metadata_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata must never describe a region whose upload failed."""
    refreshes: list[object] = []
    cleared: list[DataRoot] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "region-latest.parquet").write_bytes(b"polygons")

    class _FailingQueue:
        def close_and_wait(self) -> list[str]:
            return ["region upload failed"]

    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _FailingQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: refreshes.append(kw))
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "set_metadata_refresh_marker", lambda *a, **kw: None)
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 1
    )
    assert refreshes == []
    assert cleared == []


def test_a_resumed_run_never_refreshes_from_a_marker_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marked region this run did not publish may not be described remotely.

    The marker can name a region whose upload never reached the Hub, so a
    rerun that skips it must leave the marker for the sync command, which
    reconciles against the remote before refreshing.
    """
    refreshes: list[object] = []
    cleared: list[DataRoot] = []
    root = DataRoot(tmp_path)
    root.ensure()

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", lambda inputs, **kwargs: [])
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(
        commands, "load_metadata_refresh_marker", lambda data_root: _marker(["region-latest"])
    )
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: refreshes.append(kw))
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 0
    )
    assert refreshes == []
    assert cleared == []


def test_a_marker_naming_an_unpublished_region_blocks_the_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing one region does not license metadata for another."""
    refreshes: list[object] = []
    cleared: list[DataRoot] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "published-latest.parquet").write_bytes(b"polygons")

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    result = SimpleNamespace(manifest_entry={"source_pbf": "published-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "set_metadata_refresh_marker", lambda *a, **kw: None)
    monkeypatch.setattr(
        commands,
        "load_metadata_refresh_marker",
        lambda data_root: _marker(["published-latest", "stranded-latest"]),
    )
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: refreshes.append(kw))
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 0
    )
    assert refreshes == []
    assert cleared == []


def test_a_failed_metadata_refresh_closes_the_queue_and_keeps_its_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An assembly failure is reported as an upload failure, never a hang."""
    cleared: list[DataRoot] = []
    closed: list[bool] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "region-latest.parquet").write_bytes(b"polygons")

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            closed.append(True)
            return []

    def failing_refresh(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("manifest drift")

    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "set_metadata_refresh_marker", lambda *a, **kw: None)
    monkeypatch.setattr(
        commands, "load_metadata_refresh_marker", lambda data_root: _marker(["region-latest"])
    )
    monkeypatch.setattr(commands, "_upload_metadata_refresh", failing_refresh)
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 1
    )
    assert closed == [True]
    assert cleared == []


def test_the_refresh_marker_is_written_before_the_regional_job_is_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between the two must not leave an uploadable region unrecorded."""
    order: list[str] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "region-latest.parquet").write_bytes(b"polygons")

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: order.append("submit"))
    monkeypatch.setattr(
        commands, "set_metadata_refresh_marker", lambda *a, **kw: order.append("marker")
    )
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "clear_metadata_refresh_marker", lambda data_root: None)

    assert (
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 0
    )
    assert order == ["marker", "submit"]


def test_a_malformed_marker_cannot_leave_the_upload_queue_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue closes before the marker is read, so a bad marker never hangs."""
    closed: list[bool] = []
    root = DataRoot(tmp_path)
    root.ensure()

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            closed.append(True)
            return []

    def malformed_marker(_data_root: DataRoot) -> object:
        raise ValueError("malformed metadata refresh marker")

    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "set_metadata_refresh_marker", lambda *a, **kw: None)
    monkeypatch.setattr(commands, "load_metadata_refresh_marker", malformed_marker)

    with pytest.raises(ValueError, match="malformed metadata refresh marker"):
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )

    # The validation error still surfaces, but only after the worker was
    # told to stop, so the CLI can exit.
    assert closed == [True]


def test_an_aborted_run_drains_without_publishing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A region that failed to submit must not get repository-wide assets."""
    refreshes: list[object] = []
    cleared: list[DataRoot] = []
    closed: list[bool] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "region-latest.parquet").write_bytes(b"polygons")

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            closed.append(True)
            return []

    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    def failing_enqueue(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("canonical document snapshot failed")

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "_enqueue_core_upload", failing_enqueue)
    monkeypatch.setattr(commands, "set_metadata_refresh_marker", lambda *a, **kw: None)
    monkeypatch.setattr(
        commands, "load_metadata_refresh_marker", lambda data_root: _marker(["region-latest"])
    )
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: refreshes.append(kw))
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    with pytest.raises(RuntimeError, match="canonical document snapshot failed"):
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )

    # The queue still drains, but the marker survives for the next run and
    # no repository-wide asset describes the region that never uploaded.
    assert closed == [True]
    assert refreshes == []
    assert cleared == []


def test_recording_a_region_preserves_a_surviving_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processing a new region must not discard an earlier stranded one."""
    recorded: list[tuple[list[str], dict[str, str]]] = []
    refreshes: list[object] = []
    cleared: list[DataRoot] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "fresh-latest.parquet").write_bytes(b"polygons")

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    result = SimpleNamespace(manifest_entry={"source_pbf": "fresh-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    def record(data_root: DataRoot, stems: list[str], hashes: dict[str, str]) -> None:
        recorded.append((stems, hashes))

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(
        commands, "load_metadata_refresh_marker", lambda data_root: _marker(["stranded-latest"])
    )
    monkeypatch.setattr(commands, "set_metadata_refresh_marker", record)
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: refreshes.append(kw))
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
        )
        == 0
    )

    # The stranded region stays in the marker, and keeps the refresh shut.
    assert len(recorded) == 1
    stems, hashes = recorded[0]
    assert stems == ["fresh-latest", "stranded-latest"]
    assert set(hashes) == {"fresh-latest", "stranded-latest"}
    assert refreshes == []
    assert cleared == []


def test_an_unverifiable_marker_is_reported_to_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Staleness this command cannot repair must not be silent."""
    root = DataRoot(tmp_path)
    root.ensure()

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", lambda inputs, **kwargs: [])
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(
        commands, "load_metadata_refresh_marker", lambda data_root: _marker(["stranded-latest"])
    )

    with caplog.at_level("WARNING"):
        assert (
            commands._run_processing_command(
                _deferring_args(tmp_path), data_root=root, settings=Settings(repo_id="example/repo")
            )
            == 0
        )

    assert "stranded-latest" in caplog.text
    assert "sync-dir" in caplog.text


def test_a_dry_run_never_touches_the_refresh_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulated push must not retire durable publication intent."""
    recorded: list[object] = []
    cleared: list[DataRoot] = []
    refreshes: list[object] = []
    root = DataRoot(tmp_path)
    root.ensure()
    (root.processed_polygons / "region-latest.parquet").write_bytes(b"polygons")

    class _StubQueue:
        def close_and_wait(self) -> list[str]:
            return []

    result = SimpleNamespace(manifest_entry={"source_pbf": "region-latest.osm.pbf"})

    def fake_orchestrate(inputs: object, **kwargs: object) -> list[object]:
        on_complete = kwargs["on_complete"]
        assert callable(on_complete)
        on_complete(result)
        return [result]

    monkeypatch.setattr(
        commands, "_build_clients", lambda *a, **kw: ("wikidata", "wikipedia", "cache")
    )
    monkeypatch.setattr(commands, "_build_upload_queue", lambda *a, **kw: _StubQueue())
    monkeypatch.setattr(commands, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(commands, "_log_process_results", lambda results: None)
    monkeypatch.setattr(commands, "_enqueue_core_upload", lambda *a, **kw: None)
    monkeypatch.setattr(
        commands, "set_metadata_refresh_marker", lambda *a, **kw: recorded.append(kw)
    )
    monkeypatch.setattr(
        commands, "load_metadata_refresh_marker", lambda data_root: _marker(["region-latest"])
    )
    monkeypatch.setattr(commands, "_upload_metadata_refresh", lambda *a, **kw: refreshes.append(kw))
    monkeypatch.setattr(
        commands, "clear_metadata_refresh_marker", lambda data_root: cleared.append(data_root)
    )

    assert (
        commands._run_processing_command(
            _deferring_args(tmp_path, dry_run=True),
            data_root=root,
            settings=Settings(repo_id="example/repo"),
        )
        == 0
    )

    # The refresh is still exercised against the stub hub, but the durable
    # marker is neither written nor retired by a simulated push.
    assert len(refreshes) == 1
    assert recorded == []
    assert cleared == []


def test_release_stats_requires_one_confirmation_per_released_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from osm_polygon_wikidata_only.v2.storage import write_v2_region

    write_v2_region(
        tmp_path / "processed_v2",
        "region-latest",
        polygons=[{"polygon_id": "p1", "osm_type": "way", "osm_id": 1, "has_wikidata": False}],
        documents=[],
        links=[],
    )

    with pytest.raises(SystemExit):
        main(
            [
                "release-stats",
                "--data-root",
                str(tmp_path),
                "--dataset-version",
                "v2",
                "--confirm-repo",
                "NoeFlandre/osm-polygon-wikidata-only",
            ]
        )

    assert "one --confirm-repo per released dataset" in capsys.readouterr().err


def test_release_stats_dry_run_reports_the_v2_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from osm_polygon_wikidata_only.v2.storage import write_v2_region

    write_v2_region(
        tmp_path / "processed_v2",
        "region-latest",
        polygons=[{"polygon_id": "p1", "osm_type": "way", "osm_id": 1, "has_wikidata": False}],
        documents=[],
        links=[],
    )

    assert (
        main(
            [
                "release-stats",
                "--data-root",
                str(tmp_path),
                "--dataset-version",
                "v2",
                "--confirm-repo",
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["repo_id"] == "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
    assert payload["published"] is False
    # The coverage assets are released with the card so their captions can
    # never drift from the statistics the card reports.
    assert [item["path_in_repo"] for item in payload["files"]] == [
        "README.md",
        "stats.json",
        "assets/coverage_map.png",
        "assets/geographic_text_presence.png",
        "assets/geographic_text_density.png",
    ]


def test_release_stats_rejects_apply_and_dry_run_together() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["release-stats", "--apply", "--dry-run"])


def test_release_apply_runs_credential_preflight_even_without_push_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        commands,
        "_require_push_token",
        lambda parser, settings: calls.append(f"token:{settings.hf_token}"),
    )
    monkeypatch.setattr(
        commands,
        "_verify_push_access",
        lambda parser, settings: calls.append(f"access:{settings.hf_token}"),
    )

    commands._authenticate_for_push(
        argparse.ArgumentParser(),
        argparse.Namespace(
            command="release-stats",
            dataset_version="v2",
            push=False,
            apply=True,
            dry_run=False,
        ),
        Settings(hf_token="explicit-token"),
    )

    assert calls == ["token:explicit-token", "access:explicit-token"]


def test_release_forwards_explicit_token_to_the_release_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from osm_polygon_wikidata_only.hf import stats_release

    captured: dict[str, object] = {}

    def fake_release(data_root: DataRoot, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            to_payload=lambda: {"repo_id": kwargs["confirm_repo"], "published": False}
        )

    monkeypatch.setattr(stats_release, "release_v2_polygon_stats", fake_release)
    args = argparse.Namespace(
        dataset_version="v2",
        confirm_repo=["NoeFlandre/osm-polygon-wikidata-and-wikipedia"],
        dry_run=True,
        apply=False,
        hf_token="explicit-token",
        source_revision=None,
        data_revision=None,
        generated_on=None,
    )

    assert (
        commands._run_release_stats(argparse.ArgumentParser(), args, data_root=DataRoot(tmp_path))
        == 0
    )
    capsys.readouterr()
    assert captured["token"] == "explicit-token"
