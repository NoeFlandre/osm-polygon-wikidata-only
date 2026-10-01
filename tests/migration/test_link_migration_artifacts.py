from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
    LEDGER_CONTRACT_VERSION,
    IntegrityPlan,
    RejectionRecord,
    save_ledger,
)
from osm_polygon_wikidata_only.augmentation.steps import CONTRACT_VERSION
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import LINK_CONTRACT_VERSION
from osm_polygon_wikidata_only.pipeline._link_migration import artifacts
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    StemApplyContext,
    StemApplyInputs,
    StemClassification,
    StemPlan,
)


def test_pending_publication_update_repairs_metadata_marker() -> None:
    payload = {"metadata_refresh": "legacy-shape"}

    updated = artifacts._updated_pending_publications(payload, "alpha", "link-hash")

    assert updated == {
        "stems": ["alpha"],
        "metadata_refresh": {
            "stems": ["alpha"],
            "fingerprint_hashes": {"alpha": "link-hash"},
        },
    }


def test_updated_processed_entry_preserves_source_pbf_for_reconstructed_entry() -> None:
    inputs = cast(
        StemApplyInputs,
        SimpleNamespace(
            polygons_table=pa.Table.from_pylist([{"region": "north"}]),
            stem_plan=SimpleNamespace(stem="north-latest"),
        ),
    )
    entries: dict[str, dict[str, object]] = {}

    updated = artifacts._updated_processed_entry(
        entries,
        inputs,
        "north-latest.osm.pbf",
        3,
    )

    assert updated is entries
    assert entries["north-latest.osm.pbf"]["source_pbf"] == "north-latest.osm.pbf"


def test_rejection_record_conversion_preserves_expected_qid_and_cascade_count() -> None:
    records = artifacts._rejection_records(
        [
            {
                "shard": "alpha",
                "source_table": "polygon_articles",
                "identifier": "Q1:en:1:1",
                "wikidata": "Q1",
                "expected": "Q2",
                "reason": "unexpected-link",
                "cascaded_sections": 3,
            }
        ],
        None,
    )

    assert records == [
        RejectionRecord(
            shard="alpha",
            source_table="polygon_articles",
            identifier="Q1:en:1:1",
            wikidata="Q1",
            expected="Q2",
            reason="unexpected-link",
            cascaded_sections=3,
        )
    ]


def test_rejection_ledger_staging_preserves_history(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    ledger_path = processed / "integrity" / "rejection_ledger.json"
    existing = RejectionRecord(
        shard="prior",
        source_table="polygon_articles",
        identifier="Q1:en:1:1",
        wikidata="Q1",
        expected=None,
        reason="prior-rejection",
    )
    save_ledger(ledger_path, [existing])

    staged = artifacts._stage_rejection_ledger(processed, processed / "staging", [], None)

    expected_path = processed / "staging" / "integrity" / "rejection_ledger.json"
    assert staged == expected_path
    assert json.loads(staged.read_text(encoding="utf-8")) == {
        "contract_version": LEDGER_CONTRACT_VERSION,
        "records": [existing.to_dict()],
    }


def test_retained_voyage_table_keeps_its_original_schema_and_metadata(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    target = processed / "wikivoyage" / "documents" / "alpha.parquet"
    target.parent.mkdir(parents=True)
    schema = pa.schema(
        [("document_id", pa.string()), ("revision_id", pa.int64()), ("label", pa.string())],
        metadata={b"source-contract": b"wikivoyage-v1"},
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"document_id": "doc-1", "revision_id": 1, "label": None}], schema=schema
        ),
        target,
    )

    staged = artifacts._stage_retained_voyage_table(
        processed,
        processed / "staging",
        target,
        [{"document_id": "doc-2", "revision_id": 2, "label": None}],
    )

    assert pq.read_schema(staged).equals(schema, check_metadata=True)


def test_stem_replacement_stages_keep_full_manifest_integrity_and_schema_contracts(
    tmp_path: Path,
) -> None:
    stem = "alpha-latest"
    processed = tmp_path / "processed"
    data_root = DataRoot(tmp_path)
    polygons_path = processed / "polygons" / f"{stem}.parquet"
    wikipedia_documents_path = processed / "wikipedia" / "documents" / f"{stem}.parquet"
    links_path = processed / "polygon_articles" / f"{stem}.parquet"
    voyage_documents_path = processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    voyage_sections_path = processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    for path in (polygons_path, wikipedia_documents_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("utf-8"))

    voyage_schema = pa.schema(
        [("document_id", pa.string()), ("revision_id", pa.int64())],
        metadata={b"source-contract": b"wikivoyage-v1"},
    )
    voyage_documents_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"document_id": "old-1", "revision_id": 1},
                {"document_id": "old-2", "revision_id": 2},
            ],
            schema=voyage_schema,
        ),
        voyage_documents_path,
    )

    existing_augmentation_entry = {
        "completed_at": "2026-01-02T03:04:05+00:00",
        "counts": {"legacy_rows": 8, "polygon_articles": 99},
        "custom_metadata": {"kept": True},
    }
    augmentation_path = processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    augmentation_path.parent.mkdir(parents=True)
    augmentation_path.write_text(
        json.dumps(
            {
                stem: existing_augmentation_entry,
                "other-stem": {"untouched": True},
            }
        ),
        encoding="utf-8",
    )

    pending_path = processed / "manifests" / "pending_migration_publications.json"
    pending_path.parent.mkdir(parents=True)
    pending_path.write_text(
        json.dumps(
            {
                "contract_version": "pending-publications-v1",
                "stems": ["zeta-latest"],
                "metadata_refresh": {
                    "stems": ["zeta-latest"],
                    "fingerprint_hashes": {"zeta-latest": "old-hash"},
                },
            }
        ),
        encoding="utf-8",
    )

    ledger_record = RejectionRecord(
        shard=stem,
        source_table="wikivoyage_documents",
        identifier="voyage-doc-1",
        wikidata="Q1",
        expected="Q2",
        reason="unexpected-link",
    )
    integrity_plan = cast(
        IntegrityPlan,
        SimpleNamespace(
            retained_documents=[{"document_id": "retained", "revision_id": 2}],
            retained_sections=[],
            rejections=[ledger_record],
        ),
    )
    stem_plan = StemPlan(
        stem=stem,
        classification=StemClassification.MIGRATABLE,
        reason="legacy schema",
        polygons_fingerprint="polygons-hash",
        links_fingerprint="links-hash",
        documents_fingerprint="documents-hash",
        row_count=1,
        canonical_digest=None,
    )
    inputs = StemApplyInputs(
        stem_plan=stem_plan,
        links_path=links_path,
        polygons_path=polygons_path,
        docs_path=wikipedia_documents_path,
        voyage_documents_path=voyage_documents_path,
        voyage_sections_path=voyage_sections_path,
        legacy_table=pa.table({"placeholder": []}),
        polygons_table=pa.table({"source_pbf": [f"{stem}.osm.pbf"], "region": ["north"]}),
        docs_table=pa.table({"placeholder": []}),
        data_root=data_root,
    )
    context = StemApplyContext(
        inputs=inputs,
        integrity_plan=integrity_plan,
        canonical_table=pa.table({"canonical": ["link"]}),
    )

    replacements = artifacts.stage_stem_replacements(
        processed,
        context,
        [
            {
                "shard": stem,
                "source_table": "polygon_articles",
                "identifier": "legacy-link-1",
                "wikidata": "Q1",
                "expected": "Q2",
                "reason": "unexpected-link",
                "cascaded_sections": 0,
            }
        ],
    )
    staged_by_target = dict(replacements)

    assert staged_by_target[voyage_documents_path] == (
        processed
        / ".link_migration_staging"
        / stem
        / "wikivoyage"
        / "documents"
        / f"{stem}.parquet"
    )
    assert pq.read_table(staged_by_target[voyage_documents_path]).to_pylist() == [
        {"document_id": "retained", "revision_id": 2}
    ]
    assert pq.read_schema(staged_by_target[voyage_documents_path]).equals(
        voyage_schema, check_metadata=True
    )

    staged_augmentation = staged_by_target[
        processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    ]
    augmentation = json.loads(staged_augmentation.read_text(encoding="utf-8"))
    entry = augmentation[stem]
    assert augmentation["other-stem"] == {"untouched": True}
    assert entry["custom_metadata"] == {"kept": True}
    assert entry["contract_version"] == CONTRACT_VERSION
    assert entry["link_schema_version"] == LINK_CONTRACT_VERSION
    assert entry["completed_at"] == existing_augmentation_entry["completed_at"]
    assert entry["counts"] == {
        "legacy_rows": 8,
        "polygon_articles": 1,
        "wikivoyage_documents": 1,
        "wikivoyage_sections": 0,
    }
    assert entry["paths"] == [
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
    ]
    assert entry["core_hashes"] == {
        str(polygons_path): hashlib.sha256(polygons_path.read_bytes()).hexdigest(),
        str(wikipedia_documents_path): hashlib.sha256(
            wikipedia_documents_path.read_bytes()
        ).hexdigest(),
    }
    staged_links = staged_by_target[links_path]
    assert entry["link_artifact_sha256"] == hashlib.sha256(staged_links.read_bytes()).hexdigest()

    staged_pending = staged_by_target[pending_path]
    assert json.loads(staged_pending.read_text(encoding="utf-8")) == {
        "contract_version": "pending-publications-v1",
        "stems": ["alpha-latest", "zeta-latest"],
        "metadata_refresh": {
            "stems": ["alpha-latest", "zeta-latest"],
            "fingerprint_hashes": {
                "alpha-latest": entry["link_artifact_sha256"],
                "zeta-latest": "old-hash",
            },
        },
    }

    staged_ledger = staged_by_target[processed / "integrity" / "rejection_ledger.json"]
    ledger = json.loads(staged_ledger.read_text(encoding="utf-8"))
    assert {record["identifier"] for record in ledger["records"]} == {
        "legacy-link-1",
        "voyage-doc-1",
    }

    new_entry = artifacts._augmentation_entry(
        data_root,
        stem,
        {"counts": {}},
        context,
        staged_links,
        processed,
    )
    timestamp = new_entry["completed_at"]
    assert isinstance(timestamp, str) and timestamp
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@pytest.mark.parametrize(
    ("loader", "filename"),
    [
        (artifacts._load_processed_entries, "processed_pbfs.json"),
        (artifacts._load_augmentation_manifest, "augmentation_manifest.json"),
        (artifacts._load_pending_publications, "pending_migration_publications.json"),
    ],
)
def test_corrupt_manifest_loaders_name_the_offending_artifact(
    tmp_path: Path,
    loader: Callable[[Path], object],
    filename: str,
) -> None:
    path = tmp_path / filename
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        loader(path)

    assert str(error.value).startswith(f"{filename}:")


def test_pending_publication_loader_initializes_versioned_envelope_contract(tmp_path: Path) -> None:
    assert artifacts._load_pending_publications(tmp_path / "missing.json") == {
        "contract_version": "pending-publications-v1",
        "stems": [],
    }


def test_json_object_loader_decodes_utf8_independent_of_default_encoding_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unicode.json"
    path.write_text('{"place":"Montréal"}', encoding="utf-8")
    original_read_text = Path.read_text

    def read_with_ascii_default(
        candidate: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        return original_read_text(candidate, encoding=encoding or "ascii", errors=errors)

    monkeypatch.setattr(Path, "read_text", read_with_ascii_default)

    assert artifacts._load_json_object(path, "unicode.json") == {"place": "Montréal"}
