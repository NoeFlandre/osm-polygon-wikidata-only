"""Validation contracts for planned sentence-publication files."""

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.grid5000.sentence_publication import (
    ControllerRunError,
    validate_expected_sentence_files,
    verify_expected_sentence_files,
)
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory


def test_expected_sentence_files_rejects_incomplete_local_plan() -> None:
    with pytest.raises(ControllerRunError, match="incomplete publication plan"):
        validate_expected_sentence_files([(None, "sentences/region.parquet")])


def test_expected_sentence_files_accepts_complete_local_plan(tmp_path: Path) -> None:
    validate_expected_sentence_files([(tmp_path / "region.parquet", "sentences/region.parquet")])


def test_verification_refuses_an_incomplete_plan_before_downloading(tmp_path: Path) -> None:
    downloads: list[tuple[object, ...]] = []

    def record_download(*args: object, **_kwargs: object) -> Path:
        downloads.append(args)
        return tmp_path / "unexpected"

    with pytest.raises(ControllerRunError, match="incomplete publication plan"):
        verify_expected_sentence_files(
            "owner/repo",
            [(None, "sentences/a.parquet")],
            inventory=RemoteInventory({"sentences/a.parquet"}),
            token=None,
            cache_dir=tmp_path,
            download_function=record_download,
        )

    assert downloads == []
