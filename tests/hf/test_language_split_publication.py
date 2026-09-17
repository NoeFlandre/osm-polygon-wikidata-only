"""Contract tests for exact-target language publication."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_wikidata_only.cli.parser import build_parser
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.language_split_publication import (
    LanguagePublicationError,
    _merge_language_card,
    _remote_path_for_local,
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
