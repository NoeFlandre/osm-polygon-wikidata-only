from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_wikidata_only.augmentation import orchestrator
from osm_polygon_wikidata_only.augmentation.checkpoints import (
    AugmentationCheckpointStore,
    _documents_from_rows,
    _validated_section_batch,
)
from osm_polygon_wikidata_only.augmentation.integrity import (
    RejectionRecord,
    WikivoyageIntegrityResult,
)
from osm_polygon_wikidata_only.augmentation.models import Section
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    _integrity_rejections_payload,
    _is_valid_core_hash_entry,
    _is_valid_core_hash_path,
    _processed_link_manifest_is_current,
)
from osm_polygon_wikidata_only.augmentation.schema import document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot


def _section(*, document_id: str, section_id: str, section_index: int) -> Section:
    return Section(
        section_id=section_id,
        document_id=document_id,
        article_id="article-1",
        wikidata="Q1",
        project="wikipedia",
        language="en",
        site="enwiki",
        page_id=1,
        revision_id=1,
        section_index=section_index,
        heading="",
        anchor="",
        level=0,
        parent_section_id="",
        section_path="[]",
        text="section text",
        text_length_chars=12,
        text_length_words=2,
        text_length_tokens_estimate=3,
        content_hash="hash",
        license="CC BY-SA 4.0",
        attribution="Wikipedia",
    )


def test_core_hash_path_and_entry_reject_invalid_membership(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    path = str(root / "region.parquet")
    allowed = {path}

    assert not _is_valid_core_hash_path(path, set(), root)
    assert _is_valid_core_hash_entry(42, "a" * 64, allowed, root) is False
    assert _is_valid_core_hash_entry(path, "g" * 64, allowed, root) is False
    assert _is_valid_core_hash_entry(path, "a" * 64, allowed, root)


def test_core_hash_path_fails_closed_when_resolution_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.resolve() / "region.parquet"
    original_resolve = Path.resolve

    def fail_target(self: Path, strict: bool = False) -> Path:
        if self == path:
            raise OSError("unresolvable path")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_target)
    assert not _is_valid_core_hash_path(str(path), {str(path)}, tmp_path.resolve())


def test_processed_link_manifest_returns_false_for_missing_or_malformed_data(
    tmp_path: Path,
) -> None:
    root = DataRoot(tmp_path)
    links = tmp_path / "links.parquet"
    manifest = root.processed_manifests / "processed_pbfs.json"

    assert not _processed_link_manifest_is_current(root, "region-latest", links)
    for content in ("{broken", "[]", "{}"):
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(content, encoding="utf-8")
        assert not _processed_link_manifest_is_current(root, "region-latest", links)


def test_integrity_rejection_payload_serializes_rejected_documents() -> None:
    rejection = RejectionRecord(
        shard="region",
        source_table="wikivoyage_documents",
        identifier="doc-2",
        wikidata="Q2",
        expected=None,
        reason="wikidata_absent_from_polygons",
    )
    integrity = WikivoyageIntegrityResult(
        shard="region",
        original_document_count=2,
        retained_document_count=1,
        rejected_document_count=1,
        original_section_count=3,
        retained_section_count=2,
        cascaded_section_count=1,
        rewritten_documents=True,
        rewritten_sections=True,
        rejections=(rejection,),
    )

    payload = _integrity_rejections_payload(integrity)

    assert payload is not None
    assert payload["rejected_document_count"] == 1
    assert payload["rejections"] == [rejection.to_dict()]


@pytest.mark.parametrize(
    ("schema_version", "link_count", "expected"),
    [
        ("polygon-document-links-v1", 3, True),
        ("legacy", 3, False),
        ("polygon-document-links-v1", 2, False),
    ],
)
def test_processed_link_manifest_requires_matching_schema_and_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: str,
    link_count: int,
    expected: bool,
) -> None:
    stem = "region-latest"
    data_root = DataRoot(tmp_path)
    manifest_path = data_root.processed_manifests / "processed_pbfs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                f"{stem}.osm.pbf": {
                    "link_schema_version": schema_version,
                    "link_count": link_count,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator.pq, "read_metadata", lambda _path: SimpleNamespace(num_rows=3))

    result = _processed_link_manifest_is_current(data_root, stem, tmp_path / "links.parquet")

    assert result is expected


def test_checkpoint_row_helpers_reject_malformed_documents_and_sections() -> None:
    assert _documents_from_rows([{}]) is None

    expected = (("doc-1", 1, "hash"),)
    unknown = [_section(document_id="other", section_id="s1", section_index=0)]
    assert _validated_section_batch(unknown, expected) is None

    duplicates = [
        _section(document_id="doc-1", section_id="same", section_index=0),
        _section(document_id="doc-1", section_id="same", section_index=1),
    ]
    assert _validated_section_batch(duplicates, expected) is None


def test_checkpoint_directory_replacement_and_failed_write_cleanup(tmp_path: Path) -> None:
    store = AugmentationCheckpointStore(tmp_path, "region-latest", "a" * 64)

    def write_payload(value: str):
        def write(directory: Path) -> None:
            (directory / "payload.txt").write_text(value, encoding="utf-8")

        return write

    target = store._save_directory("snapshot", write_payload("old"))
    store._save_directory("snapshot", write_payload("new"))
    assert (target / "payload.txt").read_text(encoding="utf-8") == "new"

    def fail(directory: Path) -> None:
        (directory / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated write failure")

    with pytest.raises(RuntimeError, match="simulated write failure"):
        store._save_directory("snapshot", fail)

    assert (target / "payload.txt").read_text(encoding="utf-8") == "new"
    assert list(store.plan_root.glob(".snapshot-*")) == []


def test_checkpoint_table_reader_rejects_a_missing_file(tmp_path: Path) -> None:
    assert (
        AugmentationCheckpointStore._read_table(tmp_path / "missing.parquet", document_schema())
        is None
    )
