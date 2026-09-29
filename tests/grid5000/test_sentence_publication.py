"""Validation contracts for planned sentence-publication files."""

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.grid5000.sentence_publication import (
    ControllerRunError,
    validate_expected_sentence_files,
)


def test_expected_sentence_files_rejects_incomplete_local_plan() -> None:
    with pytest.raises(ControllerRunError, match="incomplete publication plan"):
        validate_expected_sentence_files([(None, "sentences/region.parquet")])


def test_expected_sentence_files_accepts_complete_local_plan(tmp_path: Path) -> None:
    validate_expected_sentence_files([(tmp_path / "region.parquet", "sentences/region.parquet")])
