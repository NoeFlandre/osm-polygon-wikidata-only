"""Contract tests for exact-target language publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import osm_polygon_wikidata_only.cli.commands as commands
from osm_polygon_wikidata_only.cli.parser import build_parser
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.language_split_publication import (
    MAX_ATOMIC_PUBLICATION_FILES,
    LanguagePublicationError,
    LanguagePublicationPlan,
    LanguagePublicationReport,
    LanguagePublicationResult,
    LanguagePublishedFile,
    _git_blob_sha1,
    _merge_language_card,
    _plan_from_version_plan,
    _remote_matches,
    _remote_path_for_local,
    _verify_remote_release,
    plan_language_split_publication,
    run_language_split_publication,
)
from osm_polygon_wikidata_only.hf.language_split_release import LanguageSplitVersion
from osm_polygon_wikidata_only.hf.language_splits import DatasetContract

V1_REPO = "NoeFlandre/osm-polygon-wikidata-only"
V2_REPO = "NoeFlandre/osm-polygon-wikidata-and-wikipedia"


def test_publication_cli_has_explicit_apply_and_target_confirmation() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "publish-language-splits",
            "--dataset-version",
            "v2",
            "--confirm-repo",
            V2_REPO,
            "--apply",
        ]
    )

    assert args.command == "publish-language-splits"
    assert args.apply is True
    assert args.confirm_repo == [V2_REPO]


def test_remote_paths_keep_v1_and_v2_contracts_separate(tmp_path: Path) -> None:
    v1_root = tmp_path / "processed"
    v2_root = tmp_path / "processed_v2"

    assert (
        _remote_path_for_local(
            DatasetContract.V1,
            v1_root
            / "language_splits/data/polygon_articles_by_language/lang-fr-00000-of-00001.parquet",
            processed_root=v1_root,
            output_root=v1_root / "language_splits",
        )
        == "data/polygon_articles_by_language/lang-fr-00000-of-00001.parquet"
    )
    assert (
        _remote_path_for_local(
            DatasetContract.V2,
            v2_root / "language_splits/wikipedia_documents_by_language/lang-fr/a.parquet",
            processed_root=v2_root,
            output_root=v2_root / "language_splits",
        )
        == "language_splits/wikipedia_documents_by_language/lang-fr/a.parquet"
    )


def test_language_card_merge_preserves_unmanaged_content() -> None:
    existing = (
        "# Existing card\n\n"
        "## Language partitions\n\nOld release.\n\n"
        "## Citation\n\nKeep this citation.\n"
    )

    merged = _merge_language_card(
        existing,
        version=LanguageSplitVersion.V1,
        configurations=("polygon_articles_by_language", "wikipedia_documents_by_language"),
        languages=("en", "fr", "unknown"),
    )

    assert "Old release." not in merged
    assert "## Language partitions" in merged
    assert "row-level" in merged
    assert "lang-unknown" in merged
    assert "Keep this citation." in merged


def test_publication_requires_exact_target_confirmation(tmp_path: Path) -> None:
    with pytest.raises(LanguagePublicationError, match="confirm-repo"):
        run_language_split_publication(
            tmp_path,
            dataset_version="v1",
            confirm_repos=(V2_REPO,),
            dry_run=True,
        )


def test_plan_and_evidence_serialization_are_stable(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    output_root = processed_root / "language_splits"
    manifest = output_root / "manifests/language_splits_v1.json"
    version_plan = SimpleNamespace(
        version=LanguageSplitVersion.V1,
        processed_root=processed_root,
        output_root=output_root,
        manifest_path=manifest,
        inventory=SimpleNamespace(
            languages=("en", "unknown"),
            tables=(SimpleNamespace(configuration="polygon_articles_by_language"),),
        ),
    )

    def as_dict(_root: Path) -> dict[str, object]:
        return {
            "expected_files": [
                {
                    "path": "processed/language_splits/data/polygon_articles_by_language/"
                    "lang-en-00000-of-00001.parquet"
                }
            ]
        }

    version_plan.to_dict = as_dict
    plan = _plan_from_version_plan(version_plan)
    assert plan.manifest_remote_path == "manifests/language_splits_v1.json"
    assert plan.to_dict()["files"][0]["path_in_repo"].startswith("data/")

    report = LanguagePublicationReport(
        version=LanguageSplitVersion.V1,
        repo_id=V1_REPO,
        dry_run=True,
        published=False,
        committed=False,
        no_op=False,
        revision=None,
        files=plan.files,
    )
    result = LanguagePublicationResult(reports=(report,))
    assert result.to_payload()["command"] == "publish-language-splits"
    assert json.loads(result.to_json())["reports"][0]["dry_run"] is True


def test_publication_plan_facade_maps_validated_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processed_root = tmp_path / "processed"
    output_root = processed_root / "language_splits"
    version_plan = SimpleNamespace(
        version=LanguageSplitVersion.V1,
        processed_root=processed_root,
        output_root=output_root,
        manifest_path=output_root / "manifests/language_splits_v1.json",
        inventory=SimpleNamespace(
            languages=("en", "unknown"),
            tables=(SimpleNamespace(configuration="polygon_articles_by_language"),),
        ),
    )
    version_plan.to_dict = lambda _root: {
        "expected_files": [
            {"path": "processed/language_splits/data/polygon_articles_by_language/lang-en.parquet"}
        ]
    }
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.language_split_publication.plan_language_split_release",
        lambda *args, **kwargs: SimpleNamespace(releases=(version_plan,)),
    )

    result = plan_language_split_publication(
        tmp_path,
        dataset_version="v1",
        confirm_repos=(V1_REPO,),
    )
    assert result[0].repo_id == V1_REPO
    assert result[0].files[0].path_in_repo == "data/polygon_articles_by_language/lang-en.parquet"


def test_dry_run_returns_only_the_planned_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = LanguagePublicationPlan(
        version=LanguageSplitVersion.V1,
        repo_id=V1_REPO,
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "processed/language_splits",
        manifest_path=tmp_path / "manifest.json",
        manifest_remote_path="manifests/language_splits_v1.json",
        languages=("unknown",),
        configurations=(),
        files=(),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.language_split_publication.plan_language_split_publication",
        lambda *args, **kwargs: (plan,),
    )

    result = run_language_split_publication(
        tmp_path,
        dataset_version="v1",
        confirm_repos=(V1_REPO,),
        dry_run=True,
    )
    assert result.reports[0].dry_run is True
    assert result.reports[0].published is False


def test_apply_rejects_oversized_plan_before_generation_or_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = LanguagePublicationPlan(
        version=LanguageSplitVersion.V1,
        repo_id=V1_REPO,
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "processed/language_splits",
        manifest_path=tmp_path / "manifest.json",
        manifest_remote_path="manifests/language_splits_v1.json",
        languages=("en",),
        configurations=("polygon_articles_by_language",),
        files=tuple(
            LanguagePublishedFile(
                local_path=tmp_path / f"part-{index}.parquet",
                path_in_repo=f"part-{index}.parquet",
                size_bytes=None,
                sha256=None,
            )
            for index in range(MAX_ATOMIC_PUBLICATION_FILES - 1)
        ),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.language_split_publication.plan_language_split_publication",
        lambda *args, **kwargs: (plan,),
    )
    generation_called = False

    def unexpected_generation(*args: object, **kwargs: object) -> None:
        nonlocal generation_called
        generation_called = True

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.language_split_publication.run_language_split_release",
        unexpected_generation,
    )

    with pytest.raises(LanguagePublicationError, match="no remote mutation was attempted"):
        run_language_split_publication(
            tmp_path,
            dataset_version="v1",
            confirm_repos=(V1_REPO,),
            apply=True,
            hub=StubHfHub(),
        )

    assert generation_called is False


def test_cli_publication_handler_forwards_release_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    received: dict[str, object] = {}
    data_root = DataRoot(tmp_path)

    def fake_publish(*args: object, **kwargs: object) -> SimpleNamespace:
        received["data_root"] = args[0]
        received.update(kwargs)
        return SimpleNamespace(to_json=lambda: "{}")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.language_split_publication.run_language_split_publication",
        fake_publish,
    )
    args = argparse.Namespace(
        dataset_version="v1",
        batch_size=17,
        confirm_repo=[V1_REPO],
        apply=False,
        dry_run=True,
        hf_token="test-token",
    )

    assert (
        commands._run_publish_language_splits(argparse.ArgumentParser(), args, data_root=data_root)
        == 0
    )
    assert received["data_root"] is data_root
    assert received["dataset_version"] == "v1"
    assert received["batch_size"] == 17
    assert received["confirm_repos"] == (V1_REPO,)
    assert received["apply"] is False
    assert received["dry_run"] is True
    assert received["token"] == "test-token"
    assert capsys.readouterr().out == "{}\n"


def test_remote_matches_lfs_blob_and_size_paths(tmp_path: Path) -> None:
    local_path = tmp_path / "part.parquet"
    local_path.write_bytes(b"fixture")
    local = LanguagePublishedFile(
        local_path=local_path,
        path_in_repo="part.parquet",
        size_bytes=local_path.stat().st_size,
        sha256=hashlib.sha256(b"fixture").hexdigest(),
    )
    lfs_remote = SimpleNamespace(
        size=local.size_bytes,
        lfs=SimpleNamespace(sha256=local.sha256),
    )
    blob_remote = SimpleNamespace(
        size=local.size_bytes,
        lfs=None,
        blob_id=_git_blob_sha1(local_path),
    )
    wrong_size = SimpleNamespace(size=local.size_bytes + 1, lfs=None, blob_id=None)

    assert _remote_matches(local, lfs_remote, StubHfHub(), V1_REPO, "rev", tmp_path)
    assert _remote_matches(local, blob_remote, StubHfHub(), V1_REPO, "rev", tmp_path)
    assert not _remote_matches(local, wrong_size, StubHfHub(), V1_REPO, "rev", tmp_path)


def test_remote_verification_rejects_missing_and_stale_files(tmp_path: Path) -> None:
    missing_plan = LanguagePublicationPlan(
        version=LanguageSplitVersion.V1,
        repo_id=V1_REPO,
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "processed/language_splits",
        manifest_path=tmp_path / "manifest.json",
        manifest_remote_path="manifests/language_splits_v1.json",
        languages=("en",),
        configurations=("polygon_articles_by_language",),
        files=(
            LanguagePublishedFile(
                local_path=tmp_path / "missing.parquet",
                path_in_repo="missing.parquet",
                size_bytes=1,
                sha256="0" * 64,
            ),
        ),
    )
    with pytest.raises(LanguagePublicationError, match="remote verification failed"):
        _verify_remote_release(
            StubHfHub(remote_files=set()),
            missing_plan,
            revision="rev",
            stale_files=(),
            data_root=tmp_path,
        )

    stale_plan = LanguagePublicationPlan(
        version=LanguageSplitVersion.V1,
        repo_id=V1_REPO,
        processed_root=tmp_path / "processed",
        output_root=tmp_path / "processed/language_splits",
        manifest_path=tmp_path / "manifest.json",
        manifest_remote_path="manifests/language_splits_v1.json",
        languages=("en",),
        configurations=(),
        files=(),
    )
    with pytest.raises(LanguagePublicationError, match="stale language file remains"):
        _verify_remote_release(
            StubHfHub(remote_files={"old.parquet"}),
            stale_plan,
            revision="rev",
            stale_files=("old.parquet",),
            data_root=tmp_path,
        )


def test_publication_uses_one_commit_and_second_run_is_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_root = tmp_path / "processed"
    output_root = processed_root / "language_splits"
    output_root.mkdir(parents=True)
    shard = output_root / "data/polygon_articles_by_language/lang-en-00000-of-00001.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"parquet-fixture")
    manifest = output_root / "manifests/language_splits_v1.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "data/polygon_articles_by_language/lang-en-00000-of-00001.parquet"}
                ]
            }
        ),
        encoding="utf-8",
    )

    version_plan = SimpleNamespace(
        version=LanguageSplitVersion.V1,
        processed_root=processed_root,
        output_root=output_root,
        manifest_path=manifest,
        inventory=SimpleNamespace(
            contract=DatasetContract.V1,
            languages=("en", "unknown"),
            tables=(
                SimpleNamespace(configuration="polygon_articles_by_language"),
                SimpleNamespace(configuration="wikipedia_documents_by_language"),
            ),
        ),
    )
    generated = SimpleNamespace(
        output_root=output_root,
        manifest_path=manifest,
        files=(
            SimpleNamespace(
                path=shard,
                configuration="polygon_articles_by_language",
                language="en",
                split="lang-en",
            ),
        ),
    )
    result = SimpleNamespace(plan=SimpleNamespace(releases=(version_plan,)), generated=(generated,))
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.language_split_publication.run_language_split_release",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.language_split_publication.plan_language_split_publication",
        lambda *args, **kwargs: (
            LanguagePublicationPlan(
                version=LanguageSplitVersion.V1,
                repo_id=V1_REPO,
                processed_root=processed_root,
                output_root=output_root,
                manifest_path=manifest,
                manifest_remote_path="manifests/language_splits_v1.json",
                languages=("en", "unknown"),
                configurations=("polygon_articles_by_language",),
                files=(),
            ),
        ),
    )

    hub = StubHfHub(
        remote_files={"README.md"},
        remote_content={"README.md": b"# Card\n\n## Citation\n\nKeep it.\n"},
    )
    first = run_language_split_publication(
        tmp_path,
        dataset_version="v1",
        confirm_repos=(V1_REPO,),
        apply=True,
        hub=hub,
    )
    second = run_language_split_publication(
        tmp_path,
        dataset_version="v1",
        confirm_repos=(V1_REPO,),
        apply=True,
        hub=hub,
    )

    assert first.reports[0].committed is True
    assert second.reports[0].no_op is True
    assert second.reports[0].committed is False
    assert len(hub.commits) == 1
    assert "README.md" in hub.commits[0]["paths"]
    assert (
        "data/polygon_articles_by_language/lang-en-00000-of-00001.parquet"
        in hub.commits[0]["paths"]
    )
    assert "## Language partitions" in hub.remote_content["README.md"].decode()
