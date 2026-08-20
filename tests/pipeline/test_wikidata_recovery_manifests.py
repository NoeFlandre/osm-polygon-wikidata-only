"""Unit contracts for recovery manifest staging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import RecoveryRepairError
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.repair import (
    _stage_augmentation_manifest,
    _stage_processed_manifest,
)


def test_stage_processed_manifest_updates_region_statistics(tmp_path: Path) -> None:
    stem = "sample-latest"
    processed_manifest = tmp_path / "processed.json"
    staged_manifest = tmp_path / "staged-processed.json"
    processed_manifest.write_text(
        json.dumps({f"{stem}.osm.pbf": {"source": "fixture"}}),
        encoding="utf-8",
    )

    _stage_processed_manifest(
        stem,
        paths={"processed_manifest": processed_manifest},
        staged=staged_manifest,
        polygons=[
            {
                "wikidata": "Q1",
                "has_wikipedia": True,
                "text_available": True,
            }
        ],
        documents=[{"language": "en", "full_text": "text"}],
        affected_qids=("Q1",),
        affected_polygon_count=1,
    )

    entry = json.loads(staged_manifest.read_text(encoding="utf-8"))[f"{stem}.osm.pbf"]
    assert entry["source"] == "fixture"
    assert entry["polygon_count"] == 1
    assert entry["unique_wikidata_count"] == 1
    assert entry["article_count"] == 1
    assert entry["languages"] == ["en"]
    assert entry["rows_with_wikipedia"] == 1
    assert entry["rows_with_full_text"] == 1
    assert entry["total_full_text_chars"] == 4
    assert entry["wikidata_recovery"]["affected_qids"] == ["Q1"]


def test_stage_augmentation_manifest_updates_counts_and_hashes(tmp_path: Path) -> None:
    stem = "sample-latest"
    augmentation_manifest = tmp_path / "augmentation.json"
    staged_manifest = tmp_path / "staged-augmentation.json"
    augmentation_manifest.write_text(
        json.dumps({stem: {"counts": {"existing": 1}, "source": "fixture"}}),
        encoding="utf-8",
    )
    polygons = tmp_path / "polygons.parquet"
    documents = tmp_path / "documents.parquet"
    staged_polygons = tmp_path / "staged-polygons.parquet"
    staged_documents = tmp_path / "staged-documents.parquet"
    polygons.write_bytes(b"old polygons")
    documents.write_bytes(b"old documents")
    staged_polygons.write_bytes(b"new polygons")
    staged_documents.write_bytes(b"new documents")

    _stage_augmentation_manifest(
        stem,
        paths={
            "augmentation_manifest": augmentation_manifest,
            "polygons": polygons,
            "documents": documents,
        },
        staged=staged_manifest,
        staged_polygons=staged_polygons,
        staged_documents=staged_documents,
        documents=[{}] * 2,
        sections=[{}] * 3,
        facts=[{}] * 4,
    )

    entry = json.loads(staged_manifest.read_text(encoding="utf-8"))[stem]
    assert entry["source"] == "fixture"
    assert entry["counts"] == {
        "existing": 1,
        "wikipedia_documents": 2,
        "wikipedia_sections": 3,
        "wikidata_facts": 4,
    }
    assert entry["core_hashes"] == {
        str(polygons): hashlib.sha256(b"new polygons").hexdigest(),
        str(documents): hashlib.sha256(b"new documents").hexdigest(),
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"{not-json", "Augmentation manifest is unreadable"),
        (b"{}", "Augmentation manifest is missing region"),
        (b'{"sample-latest": {"counts": []}}', "Augmentation manifest counts are invalid"),
    ],
)
def test_stage_augmentation_manifest_rejects_invalid_input(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    stem = "sample-latest"
    source = tmp_path / "augmentation.json"
    source.write_bytes(payload)
    staged = tmp_path / "staged.json"

    with pytest.raises(RecoveryRepairError, match=message):
        _stage_augmentation_manifest(
            stem,
            paths={"augmentation_manifest": source},
            staged=staged,
            staged_polygons=tmp_path / "polygons.parquet",
            staged_documents=tmp_path / "documents.parquet",
            documents=[],
            sections=[],
            facts=[],
        )


def test_stage_augmentation_manifest_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(RecoveryRepairError, match="Augmentation manifest is unreadable"):
        _stage_augmentation_manifest(
            "sample-latest",
            paths={"augmentation_manifest": tmp_path / "missing.json"},
            staged=tmp_path / "staged.json",
            staged_polygons=tmp_path / "polygons.parquet",
            staged_documents=tmp_path / "documents.parquet",
            documents=[],
            sections=[],
            facts=[],
        )


def test_stage_processed_manifest_rejects_missing_region(tmp_path: Path) -> None:
    source = tmp_path / "processed.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(RecoveryRepairError, match="Processed manifest is missing"):
        _stage_processed_manifest(
            "sample-latest",
            paths={"processed_manifest": source},
            staged=tmp_path / "staged.json",
            polygons=[],
            documents=[],
            affected_qids=(),
            affected_polygon_count=0,
        )
