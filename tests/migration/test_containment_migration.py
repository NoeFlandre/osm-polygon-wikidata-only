"""Losslessness contracts for contained-region migration."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.pipeline.containment_migration import (
    audit_rule,
    load_retired_children,
    prepare_local_rule,
    stage_rule,
)
from osm_polygon_wikidata_only.pipeline.containment_policy import (
    TABLE_CONTRACTS,
    ContainmentRule,
)

PARENT = "parent-latest"
CHILD = "child-latest"


def _row(columns: tuple[str, ...], token: int) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in columns:
        values[column] = token if column == "osm_id" else f"{column}-{token}"
    return values


def _seed(processed: Path, *, child_polygon_token: int = 1, sidecar_extra: bool = False) -> None:
    for contract in TABLE_CONTRACTS:
        parent_rows = [_row(contract.identity_columns, 1)]
        child_rows = [_row(contract.identity_columns, child_polygon_token)]
        if sidecar_extra and contract.subdir == "wikidata/facts":
            child_rows.append(_row(contract.identity_columns, 2))
        schema = pa.schema(
            [
                pa.field(column, pa.int64() if column == "osm_id" else pa.string())
                for column in contract.identity_columns
            ],
            metadata={b"contract": b"fixture"},
        )
        for stem, rows in ((PARENT, parent_rows), (CHILD, child_rows)):
            path = processed / contract.subdir / f"{stem}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_exact_polygon_containment_is_safe_and_sidecar_delta_is_explicit(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, sidecar_extra=True)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    assert audit.safe_to_stage
    facts = next(table for table in audit.children[0].tables if table.subdir == "wikidata/facts")
    assert facts.child_rows == 2
    assert facts.missing_from_parent == 1


def test_missing_parent_polygon_blocks_staging(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, child_polygon_token=2)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    assert not audit.safe_to_stage
    assert audit.blockers == ("child-latest: polygons missing 1 identity from parent",)


def test_duplicate_identity_blocks_staging(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    path = processed / "polygons" / f"{PARENT}.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table]), path)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    assert not audit.safe_to_stage
    assert audit.children[0].tables[0].parent_duplicate_identities == 1


def test_missing_required_file_blocks_staging(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    (processed / "wikipedia" / "sections" / f"{CHILD}.parquet").unlink()
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    assert not audit.safe_to_stage
    assert any("missing file" in blocker for blocker in audit.blockers)


def test_report_order_is_deterministic(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    rule = ContainmentRule(PARENT, (CHILD,))
    assert audit_rule(processed, rule) == audit_rule(processed, rule)


def test_stage_unions_missing_sidecar_rows_without_touching_originals(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, sidecar_extra=True)
    parent_facts = processed / "wikidata" / "facts" / f"{PARENT}.parquet"
    original = parent_facts.read_bytes()
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    staged = stage_rule(processed, tmp_path / "cache", audit)
    staged_facts = staged.artifact("wikidata/facts")
    assert parent_facts.read_bytes() == original
    assert pq.read_table(staged_facts).num_rows == 2
    assert pq.read_schema(staged_facts).equals(pq.read_schema(parent_facts), check_metadata=True)


def test_stage_keeps_newest_polygon_values_with_parent_provenance(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    schema = pa.schema(
        [
            pa.field("osm_type", pa.string()),
            pa.field("osm_id", pa.int64()),
            pa.field("polygon_id", pa.string()),
            pa.field("region", pa.string()),
            pa.field("source_pbf", pa.string()),
            pa.field("extracted_at", pa.string()),
            pa.field("tags", pa.string()),
        ],
        metadata={b"contract": b"fixture"},
    )
    parent = {
        "osm_type": "way",
        "osm_id": 1,
        "polygon_id": "parent:way:1",
        "region": "parent",
        "source_pbf": "parent-latest.osm.pbf",
        "extracted_at": "2026-01-01T00:00:00Z",
        "tags": '{"old":"value"}',
    }
    child = {
        **parent,
        "polygon_id": "child:way:1",
        "region": "child",
        "source_pbf": "child-latest.osm.pbf",
        "extracted_at": "2026-02-01T00:00:00Z",
        "tags": '{"new":"value"}',
    }
    for stem, row in ((PARENT, parent), (CHILD, child)):
        pq.write_table(
            pa.Table.from_pylist([row], schema=schema),
            processed / "polygons" / f"{stem}.parquet",
        )
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    staged = stage_rule(processed, tmp_path / "cache", audit)
    row = pq.read_table(staged.artifact("polygons")).to_pylist()[0]
    assert row["tags"] == '{"new":"value"}'
    assert row["extracted_at"] == "2026-02-01T00:00:00Z"
    assert row["polygon_id"] == "parent:way:1"
    assert row["region"] == "parent"
    assert row["source_pbf"] == "parent-latest.osm.pbf"


def test_stage_is_logically_idempotent(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, sidecar_extra=True)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    first = stage_rule(processed, tmp_path / "cache", audit)
    second = stage_rule(processed, tmp_path / "cache", audit)
    for contract in TABLE_CONTRACTS:
        assert pq.read_table(first.artifact(contract.subdir)).equals(
            pq.read_table(second.artifact(contract.subdir)), check_metadata=True
        )


def test_unsafe_audit_cannot_be_staged(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, child_polygon_token=2)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    try:
        stage_rule(processed, tmp_path / "cache", audit)
    except ValueError as error:
        assert "not safe" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unsafe audit staged")


def test_prepare_quarantines_then_installs_parent_and_retires_child(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, sidecar_extra=True)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    prepare_local_rule(tmp_path, audit)
    assert load_retired_children(processed) == frozenset({CHILD})
    assert pq.read_table(processed / "wikidata/facts" / f"{PARENT}.parquet").num_rows == 2
    for contract in TABLE_CONTRACTS:
        assert not (processed / contract.subdir / f"{CHILD}.parquet").exists()
        assert (
            tmp_path
            / "quarantine"
            / "containment-v1"
            / CHILD
            / contract.subdir
            / f"{CHILD}.parquet"
        ).is_file()


def test_prepare_is_idempotent_after_success(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, sidecar_extra=True)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    first = prepare_local_rule(tmp_path, audit)
    second = prepare_local_rule(tmp_path, audit)
    assert first == second
    assert load_retired_children(processed) == frozenset({CHILD})
