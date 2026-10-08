"""Losslessness contracts for contained-region migration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_wikidata_only.pipeline.containment_migration as containment_migration
from osm_polygon_wikidata_only.pipeline import containment_audit
from osm_polygon_wikidata_only.pipeline.containment_migration import (
    ChildAudit,
    RuleAudit,
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
SECOND_CHILD = "second-child-latest"


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


def test_newer_child_polygon_is_losslessly_added_to_parent(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed, child_polygon_token=2)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    assert audit.safe_to_stage
    staged = stage_rule(processed, tmp_path / "cache", audit)
    table = pq.read_table(staged.artifact("polygons"))
    assert table.num_rows == 2
    assert set(table.column("osm_id").to_pylist()) == {1, 2}
    assert pq.read_schema(staged.artifact("polygons")).equals(
        pq.read_schema(processed / "polygons" / f"{PARENT}.parquet"), check_metadata=True
    )


def test_stage_reads_each_polygon_table_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    schema = pa.schema(
        [pa.field("osm_type", pa.string()), pa.field("osm_id", pa.int64())],
        metadata={b"contract": b"fixture"},
    )
    children = (CHILD, SECOND_CHILD)
    for stem, osm_id in ((PARENT, 1), (CHILD, 2), (SECOND_CHILD, 3)):
        path = processed / "polygons" / f"{stem}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([{"osm_type": "way", "osm_id": osm_id}], schema=schema),
            path,
        )
    monkeypatch.setattr(containment_migration, "TABLE_CONTRACTS", (TABLE_CONTRACTS[0],))
    audit = RuleAudit(PARENT, tuple(ChildAudit(child, ()) for child in children), ())
    original_read_table = pq.read_table
    polygon_reads: dict[str, int] = {}

    def read_table(path: str | Path, *args: object, **kwargs: object) -> pa.Table:
        source = Path(path)
        if source.parent == processed / "polygons":
            polygon_reads[source.stem] = polygon_reads.get(source.stem, 0) + 1
        return original_read_table(path, *args, **kwargs)

    monkeypatch.setattr(containment_migration.pq, "read_table", read_table)

    stage_rule(processed, tmp_path / "cache", audit)

    assert polygon_reads == {PARENT: 1, CHILD: 1, SECOND_CHILD: 1}


def test_two_child_polygon_union_matches_golden_rows_and_manifest_stats(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    for contract in TABLE_CONTRACTS:
        child_path = processed / contract.subdir / f"{CHILD}.parquet"
        second_child_path = processed / contract.subdir / f"{SECOND_CHILD}.parquet"
        second_child_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pq.read_table(child_path), second_child_path)

    polygon_schema = pa.schema(
        [
            pa.field("osm_type", pa.string()),
            pa.field("osm_id", pa.int64()),
            pa.field("polygon_id", pa.string()),
            pa.field("region", pa.string()),
            pa.field("source_pbf", pa.string()),
            pa.field("extracted_at", pa.string()),
            pa.field("wikidata", pa.string()),
            pa.field("has_wikipedia", pa.bool_()),
            pa.field("text_available", pa.bool_()),
            pa.field("area_bucket", pa.string()),
            pa.field("tag_keys", pa.string()),
        ],
        metadata={b"contract": b"fixture"},
    )

    def polygon_row(
        osm_id: int,
        *,
        extracted_at: str,
        wikidata: str,
        has_wikipedia: bool,
        text_available: bool,
        area_bucket: str,
        tag_keys: str,
        source: str,
    ) -> dict[str, object]:
        stem = source.removesuffix(".osm.pbf")
        return {
            "osm_type": "way",
            "osm_id": osm_id,
            "polygon_id": f"{stem}:way:{osm_id}",
            "region": stem.removesuffix("-latest"),
            "source_pbf": source,
            "extracted_at": extracted_at,
            "wikidata": wikidata,
            "has_wikipedia": has_wikipedia,
            "text_available": text_available,
            "area_bucket": area_bucket,
            "tag_keys": tag_keys,
        }

    polygon_tables = {
        PARENT: [
            polygon_row(
                1,
                extracted_at="2026-01-01T00:00:00Z",
                wikidata="Q1",
                has_wikipedia=True,
                text_available=False,
                area_bucket="small",
                tag_keys='["parent", "name"]',
                source="parent-latest.osm.pbf",
            ),
            polygon_row(
                2,
                extracted_at="2026-01-01T00:00:00Z",
                wikidata="Q2",
                has_wikipedia=False,
                text_available=True,
                area_bucket="small",
                tag_keys='["name"]',
                source="parent-latest.osm.pbf",
            ),
        ],
        CHILD: [
            polygon_row(
                1,
                extracted_at="2026-02-01T00:00:00Z",
                wikidata="Q8",
                has_wikipedia=False,
                text_available=True,
                area_bucket="small",
                tag_keys='["intermediate"]',
                source="child-latest.osm.pbf",
            ),
            polygon_row(
                3,
                extracted_at="2026-02-01T00:00:00Z",
                wikidata="Q3",
                has_wikipedia=True,
                text_available=True,
                area_bucket="large",
                tag_keys='["name", "shop"]',
                source="child-latest.osm.pbf",
            ),
        ],
        SECOND_CHILD: [
            polygon_row(
                1,
                extracted_at="2026-03-01T00:00:00Z",
                wikidata="Q9",
                has_wikipedia=True,
                text_available=False,
                area_bucket="small",
                tag_keys='["latest", "name"]',
                source="second-child-latest.osm.pbf",
            ),
            polygon_row(
                4,
                extracted_at="2026-03-01T00:00:00Z",
                wikidata="Q4",
                has_wikipedia=False,
                text_available=True,
                area_bucket="large",
                tag_keys='["name", "amenity"]',
                source="second-child-latest.osm.pbf",
            ),
        ],
    }
    for stem, rows in polygon_tables.items():
        pq.write_table(
            pa.Table.from_pylist(rows, schema=polygon_schema),
            processed / "polygons" / f"{stem}.parquet",
        )

    document_schema = pa.schema(
        [
            pa.field("document_id", pa.string()),
            pa.field("language", pa.string()),
            pa.field("article_length_chars", pa.int64()),
        ],
        metadata={b"contract": b"fixture"},
    )
    for stem in (PARENT, CHILD, SECOND_CHILD):
        pq.write_table(
            pa.Table.from_pylist(
                [{"document_id": "article-1", "language": "en", "article_length_chars": 42}],
                schema=document_schema,
            ),
            processed / "wikipedia" / "documents" / f"{stem}.parquet",
        )

    rule = ContainmentRule(PARENT, (CHILD, SECOND_CHILD))
    audit = audit_rule(processed, rule)
    assert audit.safe_to_stage
    staged = stage_rule(processed, tmp_path / "cache", audit)

    polygon_path = staged.artifact("polygons")
    staged_table = pq.read_table(polygon_path)
    assert staged_table.schema.equals(polygon_schema, check_metadata=True)
    assert staged_table.to_pylist() == [
        polygon_row(
            1,
            extracted_at="2026-03-01T00:00:00Z",
            wikidata="Q9",
            has_wikipedia=True,
            text_available=False,
            area_bucket="small",
            tag_keys='["latest", "name"]',
            source="parent-latest.osm.pbf",
        ),
        polygon_row(
            2,
            extracted_at="2026-01-01T00:00:00Z",
            wikidata="Q2",
            has_wikipedia=False,
            text_available=True,
            area_bucket="small",
            tag_keys='["name"]',
            source="parent-latest.osm.pbf",
        ),
        polygon_row(
            3,
            extracted_at="2026-02-01T00:00:00Z",
            wikidata="Q3",
            has_wikipedia=True,
            text_available=True,
            area_bucket="large",
            tag_keys='["name", "shop"]',
            source="parent-latest.osm.pbf",
        ),
        polygon_row(
            4,
            extracted_at="2026-03-01T00:00:00Z",
            wikidata="Q4",
            has_wikipedia=False,
            text_available=True,
            area_bucket="large",
            tag_keys='["name", "amenity"]',
            source="parent-latest.osm.pbf",
        ),
    ]
    assert containment_migration._canonical_manifest_stats(staged) == {
        "polygon_count": 4,
        "unique_wikidata_count": 4,
        "rows_with_wikipedia": 2,
        "rows_with_full_text": 3,
        "area_bucket_counts": {"small": 2, "large": 2},
        "top_tag_keys": {"latest": 1, "name": 4, "shop": 1, "amenity": 1},
        "article_count": 1,
        "language_count": 1,
        "languages": ["en"],
        "total_full_text_chars": 42,
    }


def test_duplicate_identity_blocks_staging(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    path = processed / "polygons" / f"{PARENT}.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table]), path)
    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    assert not audit.safe_to_stage
    assert audit.children[0].tables[0].parent_duplicate_identities == 1


def test_duplicate_child_polygon_identity_blocks_staging(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    child_path = processed / "polygons" / f"{CHILD}.parquet"
    child_table = pq.read_table(child_path)
    pq.write_table(pa.concat_tables([child_table, child_table]), child_path)

    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))

    assert not audit.safe_to_stage
    assert audit.children[0].tables[0].child_duplicate_identities == 1
    with pytest.raises(ValueError, match="not safe to stage"):
        stage_rule(processed, tmp_path / "cache", audit)


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
    _seed(processed)
    (processed / "wikipedia" / "sections" / f"{CHILD}.parquet").unlink()
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


def test_stage_remaps_child_polygon_articles_to_parent_provenance(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed(processed)
    polygon_schema = pa.schema(
        [
            pa.field("osm_type", pa.string()),
            pa.field("osm_id", pa.int64()),
            pa.field("polygon_id", pa.string()),
            pa.field("region", pa.string()),
            pa.field("source_pbf", pa.string()),
        ]
    )
    link_schema = polygon_schema.append(pa.field("article_id", pa.string()))
    parent_polygon = {
        "osm_type": "way",
        "osm_id": 1,
        "polygon_id": "parent:way:1",
        "region": "parent",
        "source_pbf": "parent-latest.osm.pbf",
    }
    child_polygon = {
        **parent_polygon,
        "polygon_id": "child:way:1",
        "region": "child",
        "source_pbf": "child-latest.osm.pbf",
    }
    for stem, polygon, article_id in (
        (PARENT, parent_polygon, "article-parent"),
        (CHILD, child_polygon, "article-child"),
    ):
        pq.write_table(
            pa.Table.from_pylist([polygon], schema=polygon_schema),
            processed / "polygons" / f"{stem}.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([{**polygon, "article_id": article_id}], schema=link_schema),
            processed / "polygon_articles" / f"{stem}.parquet",
        )

    audit = audit_rule(processed, ContainmentRule(PARENT, (CHILD,)))
    staged = stage_rule(processed, tmp_path / "cache", audit)
    links = pq.read_table(staged.artifact("polygon_articles")).to_pylist()

    assert {row["article_id"]: row["polygon_id"] for row in links} == {
        "article-parent": "parent:way:1",
        "article-child": "parent:way:1",
    }
    assert {(row["region"], row["source_pbf"]) for row in links} == {
        ("parent", "parent-latest.osm.pbf")
    }


def test_containment_retirement_payload_and_staged_artifact_contract(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    assert containment_migration._load_retirement_payload(processed) == {
        "contract_version": containment_migration.RETIREMENT_CONTRACT_VERSION,
        "retired": {},
    }

    path = processed / "manifests" / containment_migration.RETIREMENT_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text('{"contract_version":"wrong","retired":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported containment retirement"):
        containment_migration._load_retirement_payload(processed)
    path.write_text(
        json.dumps(
            {
                "contract_version": containment_migration.RETIREMENT_CONTRACT_VERSION,
                "retired": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed containment retirement"):
        containment_migration._load_retirement_payload(processed)
    path.write_text(
        json.dumps(
            {
                "contract_version": containment_migration.RETIREMENT_CONTRACT_VERSION,
                "retired": {"region": ["child"]},
            }
        ),
        encoding="utf-8",
    )
    assert containment_migration._load_retirement_payload(processed)["retired"] == {
        "region": ["child"]
    }

    staged_path = tmp_path / "polygons.parquet"
    staged = containment_migration.StagedRule("parent", ("child",), (("polygons", staged_path),))
    assert staged.artifact("polygons") == staged_path
    with pytest.raises(KeyError, match="wikipedia/documents"):
        staged.artifact("wikipedia/documents")


def test_present_containment_contract_reports_schema_duplicates_and_read_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = SimpleNamespace(subdir="polygons", identity_columns=("identity",))
    parent, child = Path("parent.parquet"), Path("child.parquet")
    schema = pa.schema([("identity", pa.string())])
    monkeypatch.setattr(
        containment_audit.pq,
        "read_schema",
        lambda path: schema if path == parent else pa.schema([("other", pa.string())]),
    )
    audit, blockers = cast(Any, containment_audit._audit_present_contract)(
        Path("processed"), contract, "parent", "child", parent, child
    )
    assert audit.child_rows == 0
    assert blockers == ["child: schema mismatch for polygons"]

    monkeypatch.setattr(containment_audit.pq, "read_schema", lambda _path: schema)
    identities = iter((({("p",)}, 1), ({("c",)}, 2)))
    monkeypatch.setattr(containment_audit, "_identity_set", lambda *_args: next(identities))
    audit, blockers = cast(Any, containment_audit._audit_present_contract)(
        Path("processed"), contract, "parent", "child", parent, child
    )
    assert audit.child_rows == 1
    assert audit.missing_from_parent == 1
    assert blockers == [
        "child: polygons parent has 1 duplicate identities",
        "child: polygons child has 2 duplicate identities",
    ]

    def fail_read(_path: Path) -> pa.Schema:
        raise OSError("unreadable parquet")

    monkeypatch.setattr(containment_audit.pq, "read_schema", fail_read)
    audit, blockers = cast(Any, containment_audit._audit_present_contract)(
        Path("processed"), contract, "parent", "child", parent, child
    )
    assert audit.child_rows == 0
    assert blockers == ["child: unreadable polygons: OSError"]
