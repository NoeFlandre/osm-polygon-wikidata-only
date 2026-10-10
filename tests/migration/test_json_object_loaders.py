"""Characterisation of the link-migration JSON object loaders' error policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.pipeline._link_migration import artifacts


def test_load_json_object_returns_the_parsed_object(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"city": "München"}', encoding="utf-8")

    assert artifacts._load_json_object(path, "manifest.json") == {"city": "München"}


def test_load_json_object_labels_malformed_json_and_chains_the_parser_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        artifacts._load_json_object(path, "manifest.json")

    assert str(error.value).startswith("manifest.json: ")
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_load_json_object_rejects_a_non_object_document(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"^manifest\.json must be a JSON object$"):
        artifacts._load_json_object(path, "manifest.json")


def test_load_json_object_propagates_unreadable_paths_unchanged(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        artifacts._load_json_object(tmp_path, "manifest.json")


def test_missing_processed_manifest_loads_as_no_entries(tmp_path: Path) -> None:
    assert artifacts._load_processed_entries(tmp_path / "processed_pbfs.json") == {}


def test_missing_pending_envelope_loads_as_the_empty_contract(tmp_path: Path) -> None:
    assert artifacts._load_pending_publications(tmp_path / "pending.json") == {
        "contract_version": "pending-publications-v1",
        "stems": [],
    }


def test_malformed_pending_envelope_is_labelled_with_its_filename(tmp_path: Path) -> None:
    path = tmp_path / "pending.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match=r"^pending_migration_publications\.json: "):
        artifacts._load_pending_publications(path)


def test_pending_envelope_rejects_an_unknown_contract(tmp_path: Path) -> None:
    path = tmp_path / "pending.json"
    path.write_text('{"contract_version": "other"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported pending-publications contract"):
        artifacts._load_pending_publications(path)
