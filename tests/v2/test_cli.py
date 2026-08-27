import argparse
from pathlib import Path
from types import SimpleNamespace

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory


def test_sync_dir_has_explicit_v2_selector() -> None:
    from osm_polygon_wikidata_only.cli.parser import build_parser

    args = build_parser().parse_args(["sync-dir", "/tmp/raw", "--dataset-version", "v2"])
    assert args.dataset_version == "v2"


def test_v2_selector_defaults_to_v1() -> None:
    from osm_polygon_wikidata_only.cli.parser import build_parser

    args = build_parser().parse_args(["sync-dir", str(Path("/tmp/raw"))])
    assert args.dataset_version == "v1"


def test_v2_selector_routes_to_isolated_executor(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli
    from osm_polygon_wikidata_only.cli.commands import main

    calls: list[str] = []
    monkeypatch.setattr(v2_cli, "execute_v2", lambda *args, **kwargs: calls.append("v2") or 0)
    raw = tmp_path / "raw"
    raw.mkdir()
    assert (
        main(["sync-dir", str(raw), "--data-root", str(tmp_path), "--dataset-version", "v2"]) == 0
    )
    assert calls == ["v2"]


def test_v2_executor_honors_an_explicit_repository_override(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli

    repos: list[str] = []
    monkeypatch.setattr(
        v2_cli,
        "build_wikimedia_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(
            wikipedia=object(), scheduler=object(), session=object()
        ),
    )
    monkeypatch.setattr(
        v2_cli.RemoteInventory,
        "fetch",
        staticmethod(lambda repo_id, **_kwargs: repos.append(repo_id) or RemoteInventory(set())),
    )
    monkeypatch.setattr(
        v2_cli,
        "upload_files",
        lambda repo_id, **_kwargs: repos.append(repo_id),
    )

    def run(_input, *, upload, **_kwargs):
        upload([], "test")
        return 0

    monkeypatch.setattr(v2_cli, "run_v2_sync", run)
    args = argparse.Namespace(
        dry_run=True,
        push=True,
        hf_token=None,
        commit_message=None,
        upload_threads=1,
        input=str(tmp_path / "raw"),
    )
    assert (
        v2_cli.execute_v2(
            args,
            data_root=DataRoot(tmp_path),
            settings=Settings(repo_id="example/v2"),
        )
        == 0
    )
    assert repos == ["example/v2", "example/v2"]


def test_sentence_executor_constructs_pinned_cpu_segmenter(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli
    from osm_polygon_wikidata_only.v2.sentence_runner import SentenceRunResult

    calls: dict[str, object] = {}

    class FakeSegmenter:
        model_id = "segment-any-text/sat-3l-sm"
        revision = "model-revision"
        version = "test"

        def __init__(self, **kwargs: object) -> None:
            calls["segmenter"] = kwargs

    def fake_run(data_root, *, segmenter, batch_size):
        calls["run"] = (data_root, segmenter, batch_size)
        return SentenceRunResult(
            manifest_path=data_root.processed_v2 / "manifests/sentence_splitting.json",
            regions=(),
        )

    monkeypatch.setattr(v2_cli, "SaT3lSegmenter", FakeSegmenter)
    monkeypatch.setattr(v2_cli, "run_v2_sentence_split", fake_run)
    monkeypatch.setattr(v2_cli, "write_v2_card", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(
        batch_size=32,
        inference_batch_size=7,
        push=False,
        dry_run=False,
        commit_message=None,
        upload_threads=1,
    )

    assert (
        v2_cli.execute_v2_sentence_split(args, data_root=DataRoot(tmp_path), settings=Settings())
        == 0
    )
    assert calls["segmenter"] == {
        "cache_dir": tmp_path / "hf_cache" / "models" / "sat-3l-sm",
        "revision": "137da05",
        "inference_batch_size": 7,
    }
    assert calls["run"][2] == 32  # type: ignore[index]


def test_sentence_executor_renders_card_against_v1_artifacts(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli
    from osm_polygon_wikidata_only.v2.sentence_runner import SentenceRunResult

    class FakeSegmenter:
        model_id = "segment-any-text/sat-3l-sm"
        revision = "model-revision"
        version = "test"

        def __init__(self, **_kwargs: object) -> None:
            pass

    def fake_run(_data_root, *, segmenter, batch_size):
        del segmenter, batch_size
        return SentenceRunResult(manifest_path=tmp_path / "manifest.json", regions=())

    card_calls: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(v2_cli, "SaT3lSegmenter", FakeSegmenter)
    monkeypatch.setattr(v2_cli, "run_v2_sentence_split", fake_run)
    monkeypatch.setattr(
        v2_cli,
        "write_v2_card",
        lambda processed_v2, **kwargs: card_calls.append((processed_v2, kwargs)),
    )
    args = argparse.Namespace(
        batch_size=32,
        inference_batch_size=7,
        push=False,
        dry_run=False,
        commit_message=None,
        upload_threads=1,
    )
    data_root = DataRoot(tmp_path)

    assert v2_cli.execute_v2_sentence_split(args, data_root=data_root, settings=Settings()) == 0

    assert card_calls == [(data_root.processed_v2, {"v1_processed": data_root.processed})]


def test_sentence_executor_publishes_only_completed_sentence_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli
    import osm_polygon_wikidata_only.v2.publication as publication
    from osm_polygon_wikidata_only.v2.sentence_runner import (
        SentenceRegionSummary,
        SentenceRunResult,
    )

    class FakeSegmenter:
        model_id = "segment-any-text/sat-3l-sm"
        revision = "model-revision"
        version = "test"

        def __init__(self, **_kwargs: object) -> None:
            pass

    result = SentenceRunResult(
        manifest_path=tmp_path / "manifest.json",
        regions=(
            SentenceRegionSummary(
                stem="region-latest",
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
    plan_calls: list[tuple[str, ...]] = []
    upload: dict[str, object] = {}
    monkeypatch.setattr(v2_cli, "SaT3lSegmenter", FakeSegmenter)
    monkeypatch.setattr(v2_cli, "run_v2_sentence_split", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(v2_cli, "write_v2_card", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        publication,
        "sentence_publication_ops",
        lambda _root, stems: plan_calls.append(tuple(stems)) or ["op"],
    )
    monkeypatch.setattr(
        v2_cli,
        "upload_files",
        lambda repo_id, **kwargs: upload.update(repo_id=repo_id, **kwargs),
    )
    args = argparse.Namespace(
        batch_size=32,
        inference_batch_size=7,
        push=True,
        dry_run=True,
        commit_message=None,
        upload_threads=3,
    )

    assert (
        v2_cli.execute_v2_sentence_split(
            args,
            data_root=DataRoot(tmp_path),
            settings=Settings(repo_id="example/v2"),
        )
        == 0
    )
    assert plan_calls == [("region-latest",)]
    assert upload["repo_id"] == "example/v2"
    assert upload["ops"] == ["op"]
    assert upload["commit_message"] == "Add V2 sentence sidecars"
    assert upload["num_threads"] == 3
