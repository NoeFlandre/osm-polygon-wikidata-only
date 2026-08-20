"""Contracts for loading an already-published augmentation result."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation import orchestrator
from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot

STEM = "demo-latest"
COUNTS = {
    "wikipedia_documents": 0,
    "wikipedia_sections": 0,
    "wikivoyage_documents": 0,
    "wikivoyage_sections": 0,
    "wikidata_facts": 0,
}


def _write_valid_result(tmp_path: Path) -> DataRoot:
    data_root = DataRoot(tmp_path / "data")
    data_root.ensure()
    schemas = (
        wikipedia_document_schema(),
        section_schema(),
        document_schema(),
        section_schema(),
        fact_schema(),
    )
    for path, schema in zip(orchestrator.sidecar_paths(data_root, STEM), schemas, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([], schema=schema), path)
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                STEM: {
                    "contract_version": orchestrator.CONTRACT_VERSION,
                    "counts": COUNTS,
                }
            }
        ),
        encoding="utf-8",
    )
    return data_root


def test_load_existing_result_validates_and_returns_sidecars(tmp_path: Path) -> None:
    data_root = _write_valid_result(tmp_path)
    result = orchestrator.load_existing_augmentation_result(data_root, STEM)
    assert result.counts == COUNTS
    assert result.manifest_path.is_file()
    assert result.wikipedia_documents_path.name == f"{STEM}.parquet"


def test_manifest_reader_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        orchestrator._read_augmentation_manifest(missing)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        orchestrator._read_augmentation_manifest(malformed)


def test_manifest_entry_and_count_validation_reject_bad_shapes() -> None:
    with pytest.raises(KeyError):
        orchestrator._augmentation_manifest_entry({}, STEM)
    with pytest.raises(TypeError):
        orchestrator._augmentation_manifest_entry({STEM: []}, STEM)
    with pytest.raises(ValueError, match="contract version"):
        orchestrator._validate_augmentation_entry(
            {"contract_version": "wrong", "counts": COUNTS}, STEM
        )
    with pytest.raises(TypeError, match="counts"):
        orchestrator._validate_augmentation_entry(
            {"contract_version": orchestrator.CONTRACT_VERSION, "counts": []}, STEM
        )
    with pytest.raises(ValueError, match="missing required fields"):
        orchestrator._validate_augmentation_entry(
            {"contract_version": orchestrator.CONTRACT_VERSION, "counts": {}}, STEM
        )
    with pytest.raises(TypeError, match="non-negative"):
        orchestrator._validate_augmentation_entry(
            {
                "contract_version": orchestrator.CONTRACT_VERSION,
                "counts": {**COUNTS, "wikidata_facts": -1},
            },
            STEM,
        )
    with pytest.raises(TypeError, match="non-negative"):
        orchestrator._validate_augmentation_entry(
            {
                "contract_version": orchestrator.CONTRACT_VERSION,
                "counts": {**COUNTS, "wikidata_facts": "0"},
            },
            STEM,
        )


def test_sidecar_file_validation_covers_missing_unreadable_and_mismatch(tmp_path: Path) -> None:
    valid = _write_valid_result(tmp_path)
    valid_path = orchestrator.sidecar_paths(valid, STEM)[0]
    orchestrator._validate_sidecar_file(valid_path, wikipedia_document_schema())

    with pytest.raises(FileNotFoundError):
        orchestrator._validate_sidecar_file(
            tmp_path / "missing.parquet", wikipedia_document_schema()
        )
    unreadable = tmp_path / "unreadable.parquet"
    unreadable.write_text("not parquet", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        orchestrator._validate_sidecar_file(unreadable, wikipedia_document_schema())
    mismatch = tmp_path / "mismatch.parquet"
    pq.write_table(pa.table({"value": [1]}), mismatch)
    with pytest.raises(ValueError, match="schema mismatch"):
        orchestrator._validate_sidecar_file(mismatch, wikipedia_document_schema())


def test_reused_section_batch_logging_is_silent_when_none_reused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    orchestrator._log_reused_section_batches(STEM, 0, 2)
    assert not caplog.records
    orchestrator._log_reused_section_batches(STEM, 1, 2)
    assert "reused 1/2" in caplog.records[-1].message
