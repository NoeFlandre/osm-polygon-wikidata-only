from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.rejection_ledger import (
    LEDGER_CONTRACT_VERSION,
    RejectionRecord,
    save_ledger,
)
from osm_polygon_wikidata_only.pipeline._link_migration import artifacts
from osm_polygon_wikidata_only.pipeline._link_migration import artifacts as link_artifacts
from osm_polygon_wikidata_only.pipeline._link_migration.models import StemApplyInputs


def test_pending_publication_update_initializes_missing_stems_and_replaces_bad_marker() -> None:
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


def test_rejection_ledger_staging_keeps_history_at_the_contract_path(tmp_path: Path) -> None:
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


def test_source_pbf_helper_rejects_zero_or_multiple_values() -> None:
    class Column:
        def __init__(self, values: list[str]) -> None:
            self.values = values

        def to_pylist(self) -> list[str]:
            return self.values

    class Table:
        def __init__(self, values: list[str]) -> None:
            self.values = values

        def column(self, name: str) -> Column:
            assert name == "source_pbf"
            return Column(self.values)

    for values, count in [([], 0), (["a.pbf", "b.pbf"], 2)]:
        inputs = SimpleNamespace(
            stem_plan=SimpleNamespace(stem="region-latest"),
            polygons_table=Table(values),
        )
        with pytest.raises(RuntimeError, match=f"{count} distinct source_pbf"):
            link_artifacts.source_pbf_for_stem(cast(Any, inputs))


def test_processed_manifest_that_is_not_an_object_is_refused_with_its_filename(
    tmp_path: Path,
) -> None:
    path = tmp_path / "processed_pbfs.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"^processed_pbfs\.json must be a JSON object$"):
        artifacts._load_processed_entries(path)
