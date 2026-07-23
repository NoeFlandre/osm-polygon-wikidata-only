from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.models import Section, stable_id
from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.steps import CONTRACT_VERSION, sha256_file
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema
from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult, WikipediaArticle
from osm_polygon_wikidata_only.enrichment.wikipedia.transport import InMemoryWikipediaClient
from osm_polygon_wikidata_only.io.manifest import save_manifest
from osm_polygon_wikidata_only.pipeline._wikidata_recovery import repair as repair_module
from osm_polygon_wikidata_only.pipeline.wikidata_recovery import (
    RecoveryRepairError,
    audit_wikidata_integrity,
    repair_wikidata_region,
)
from osm_polygon_wikidata_only.utils.json import dumps

from .test_wikidata_recovery_audit import (
    _data_root,
    _entity,
    _RecordingWikidataClient,
    _write_region,
)


class _AugmentationClient:
    def __init__(self, affected_qids: set[str]) -> None:
        self.affected_qids = affected_qids
        self.entity_calls: list[tuple[list[str], str]] = []
        self.parse_calls: list[tuple[str, str, int]] = []

    def entities(self, qids: list[str] | set[str], *, props: str) -> dict[str, dict[str, Any]]:
        requested = sorted(qids)
        self.entity_calls.append((requested, props))
        if props == "sitelinks|claims":
            return {
                qid: {
                    "id": qid,
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datatype": "wikibase-item",
                                    "datavalue": {"value": {"id": "Q5"}},
                                },
                                "rank": "normal",
                            }
                        ]
                    },
                }
                for qid in requested
            }
        return {qid: {"id": qid, "labels": {"en": {"value": f"Label {qid}"}}} for qid in requested}

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        self.parse_calls.append((project, language, revision_id))
        return "<p>Recovered article text.</p>"


def _wikipedia_article(qid: str, index: int = 9) -> WikipediaArticle:
    return WikipediaArticle(
        language="en",
        site="enwiki",
        title=f"Title {qid}",
        page_id=3000 + index,
        revision_id=4000 + index,
        revision_timestamp="2026-01-01T00:00:00Z",
        url=f"https://en.wikipedia.org/wiki/Title_{qid}",
        lead_text="Recovered article text.",
        extract="Recovered article text.",
        full_text="Recovered article text.",
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="CC BY-SA",
        attribution="test",
        source_api="test",
        retrieved_at="2026-01-01T00:00:00Z",
    )


def _empty_row(schema: pa.Schema) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in schema:
        row[field.name] = (
            0
            if pa.types.is_integer(field.type)
            else None
            if pa.types.is_floating(field.type)
            else ""
        )
    return row


def _finish_region(data_root: DataRoot, stem: str, *, healthy_qid: str = "Q1") -> None:
    wikipedia_documents = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    wikipedia_sections = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
    documents = pq.read_table(wikipedia_documents).to_pylist()  # type: ignore[no-untyped-call]
    if documents:
        document = documents[0]
        section = Section(
            section_id=stable_id(document["document_id"], 0, ""),
            document_id=str(document["document_id"]),
            article_id=str(document["article_id"]),
            wikidata=str(document["wikidata"]),
            project="wikipedia",
            language="en",
            site="enwiki",
            page_id=int(document["page_id"]),
            revision_id=int(document["revision_id"]),
            section_index=0,
            heading="",
            anchor="",
            level=0,
            parent_section_id="",
            section_path="[]",
            text="Healthy text.",
            text_length_chars=13,
            text_length_words=2,
            text_length_tokens_estimate=3,
            content_hash="healthy",
            license="CC BY-SA",
            attribution="test",
        )
        pq.write_table(
            pa.Table.from_pylist([section.to_dict()], schema=section_schema()),
            wikipedia_sections,
        )

    old_fact = _empty_row(fact_schema())
    old_fact.update(
        {
            "fact_id": "healthy-fact",
            "wikidata": healthy_qid,
            "property_id": "P31",
            "value_type": "wikibase-item",
            "value_entity_id": "Q5",
            "value_text": "human",
            "rank": "normal",
        }
    )
    facts_path = data_root.processed / "wikidata" / "facts" / f"{stem}.parquet"
    pq.write_table(pa.Table.from_pylist([old_fact], schema=fact_schema()), facts_path)

    for project in ("wikivoyage",):
        documents_path = data_root.processed / project / "documents" / f"{stem}.parquet"
        sections_path = data_root.processed / project / "sections" / f"{stem}.parquet"
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        sections_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([], schema=document_schema()), documents_path)
        pq.write_table(pa.Table.from_pylist([], schema=section_schema()), sections_path)

    polygon_rows = pq.read_table(data_root.processed_polygons / f"{stem}.parquet").to_pylist()  # type: ignore[no-untyped-call]
    link_rows = pq.read_table(data_root.processed_links / f"{stem}.parquet").to_pylist()  # type: ignore[no-untyped-call]
    entry = {
        "source_pbf": f"{stem}.osm.pbf",
        "region": stem,
        "polygons_path": f"polygons/{stem}.parquet",
        "articles_path": f"articles/{stem}.parquet",
        "wikipedia_documents_path": f"wikipedia/documents/{stem}.parquet",
        "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        "extraction_version": "test",
        "processed_at": "2026-01-01T00:00:00Z",
        "polygon_count": len(polygon_rows),
        "unique_wikidata_count": len({row["wikidata"] for row in polygon_rows}),
        "article_count": len(documents),
        "language_count": 1 if documents else 0,
        "languages": ["en"] if documents else [],
        "rows_with_wikipedia": len({row["polygon_id"] for row in link_rows}),
        "rows_with_full_text": len({row["polygon_id"] for row in link_rows}),
        "total_full_text_chars": sum(len(str(row["full_text"])) for row in documents),
        "area_bucket_counts": {},
        "top_tag_keys": {},
    }
    save_manifest(data_root.processed_manifests / "processed_pbfs.json", {f"{stem}.osm.pbf": entry})

    sidecars = (
        wikipedia_documents,
        wikipedia_sections,
        data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
        data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet",
        facts_path,
    )
    augmentation_manifest = {
        stem: {
            "contract_version": CONTRACT_VERSION,
            "core_hashes": {
                str(data_root.processed_polygons / f"{stem}.parquet"): sha256_file(
                    data_root.processed_polygons / f"{stem}.parquet"
                ),
                str(wikipedia_documents): sha256_file(wikipedia_documents),
            },
            "paths": [str(path.relative_to(data_root.processed)) for path in sidecars],
            "counts": {
                "wikipedia_documents": len(documents),
                "wikipedia_sections": pq.read_table(wikipedia_sections).num_rows,  # type: ignore[no-untyped-call]
                "wikivoyage_documents": 0,
                "wikivoyage_sections": 0,
                "wikidata_facts": 1,
            },
            "completed_at": "2026-01-01T00:00:00Z",
        }
    }
    path = data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(augmentation_manifest) + "\n", encoding="utf-8")


def _settings() -> Settings:
    return Settings(
        languages=None,
        fetch_full_text=True,
        max_articles_per_qid=None,
        enrichment_batch_size=50,
    )


def _audit_plan(data_root: DataRoot, stem: str, affected_qid: str):
    return audit_wikidata_integrity(
        data_root,
        [stem],
        _RecordingWikidataClient({affected_qid: _entity(affected_qid)}),
    ).region(stem)


def _repair(
    data_root: DataRoot,
    stem: str,
    affected_qid: str,
    *,
    before_commit: Any = None,
):
    plan = _audit_plan(data_root, stem, affected_qid)
    return repair_wikidata_region(
        data_root,
        plan,
        wikidata_client=_RecordingWikidataClient({affected_qid: _entity(affected_qid)}),
        wikipedia_client=InMemoryWikipediaClient(
            {
                ("enwiki", f"Title {affected_qid}"): FetchResult(
                    "ok", _wikipedia_article(affected_qid)
                )
            }
        ),
        augmentation_client=_AugmentationClient({affected_qid}),
        settings=_settings(),
        before_commit=before_commit,
    )


def test_repair_changes_only_affected_qid_and_preserves_existing_rows(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "repair"
    _write_region(data_root, stem, ["Q1", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    paths = {
        "polygons": data_root.processed_polygons / f"{stem}.parquet",
        "links": data_root.processed_links / f"{stem}.parquet",
        "documents": data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
        "sections": data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet",
        "facts": data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
    }
    before = {name: pq.read_table(path).to_pylist() for name, path in paths.items()}  # type: ignore[no-untyped-call]
    wikivoyage_hashes = {
        path: sha256_file(path)
        for path in (
            data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
            data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet",
        )
    }

    result = _repair(data_root, stem, "Q2")

    after = {name: pq.read_table(path).to_pylist() for name, path in paths.items()}  # type: ignore[no-untyped-call]
    healthy_before = next(row for row in before["polygons"] if row["wikidata"] == "Q1")
    healthy_after = next(row for row in after["polygons"] if row["wikidata"] == "Q1")
    affected = next(row for row in after["polygons"] if row["wikidata"] == "Q2")
    assert result.changed is True
    assert result.map_inputs_changed is True
    assert healthy_after == healthy_before
    assert affected["has_wikipedia"] is True
    assert affected["wikipedia_languages"] == '["en"]'
    assert affected["wikipedia_article_count"] == 1
    assert affected["text_available"] is True
    assert before["links"][0] in after["links"]
    assert before["documents"][0] in after["documents"]
    assert before["sections"][0] in after["sections"]
    assert before["facts"][0] in after["facts"]
    assert len(after["links"]) == 2
    assert len(after["documents"]) == 2
    assert len(after["sections"]) == 2
    assert len(after["facts"]) == 2
    assert {path: sha256_file(path) for path in wikivoyage_hashes} == wikivoyage_hashes


def test_repair_links_every_polygon_sharing_affected_qid(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "shared"
    _write_region(data_root, stem, ["Q1", "Q2", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)

    _repair(data_root, stem, "Q2")

    links = pq.read_table(data_root.processed_links / f"{stem}.parquet").to_pylist()  # type: ignore[no-untyped-call]
    q2_links = [row for row in links if row["wikidata"] == "Q2"]
    assert len(q2_links) == 2
    assert len({row["polygon_id"] for row in q2_links}) == 2
    assert len({row["article_id"] for row in q2_links}) == 1


def test_repair_preserves_multi_qid_osm_tag_and_links_each_entity(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "multiple"
    raw_tag = "Q8254481;Q6033432"
    qids = ("Q6033432", "Q8254481")
    _write_region(data_root, stem, [raw_tag])
    _finish_region(data_root, stem, healthy_qid=qids[0])
    entities = {qid: _entity(qid) for qid in qids}
    plan = audit_wikidata_integrity(
        data_root,
        [stem],
        _RecordingWikidataClient(entities),
    ).region(stem)

    result = repair_wikidata_region(
        data_root,
        plan,
        wikidata_client=_RecordingWikidataClient(entities),
        wikipedia_client=InMemoryWikipediaClient(
            {("enwiki", f"Title {qid}"): FetchResult("ok", _wikipedia_article(qid)) for qid in qids}
        ),
        augmentation_client=_AugmentationClient(set(qids)),
        settings=_settings(),
    )

    polygon = pq.read_table(data_root.processed_polygons / f"{stem}.parquet").to_pylist()[0]  # type: ignore[no-untyped-call]
    links = pq.read_table(data_root.processed_links / f"{stem}.parquet").to_pylist()  # type: ignore[no-untyped-call]
    assert result.changed is True
    assert result.map_inputs_changed is True
    assert result.affected_polygon_count == 1
    assert polygon["wikidata"] == raw_tag
    assert polygon["wikipedia_article_count"] == 2
    assert {link["wikidata"] for link in links} == set(qids)


def test_repair_removes_only_orphan_fact_rows(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "orphan-fact"
    _write_region(data_root, stem, ["Q1"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    facts_path = data_root.processed / "wikidata" / "facts" / f"{stem}.parquet"
    facts = pq.read_table(facts_path).to_pylist()  # type: ignore[no-untyped-call]
    orphan = dict(facts[0])
    orphan.update({"fact_id": "orphan-fact", "wikidata": "Q21847764"})
    pq.write_table(pa.Table.from_pylist([*facts, orphan], schema=fact_schema()), facts_path)
    plan = audit_wikidata_integrity(
        data_root,
        [stem],
        _RecordingWikidataClient({}),
    ).region(stem)

    augmentation_client = _AugmentationClient(set())
    result = repair_wikidata_region(
        data_root,
        plan,
        wikidata_client=_RecordingWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        augmentation_client=augmentation_client,
        settings=_settings(),
    )

    repaired = pq.read_table(facts_path).to_pylist()  # type: ignore[no-untyped-call]
    assert plan.orphan_fact_ids == ("orphan-fact",)
    assert result.changed is True
    assert result.map_inputs_changed is False
    assert repaired == facts
    assert augmentation_client.entity_calls == []


def test_repair_removes_orphan_wikipedia_document_and_its_sections(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "orphan-document"
    _write_region(data_root, stem, ["Q1"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    sections_path = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
    documents = pq.read_table(documents_path).to_pylist()  # type: ignore[no-untyped-call]
    sections = pq.read_table(sections_path).to_pylist()  # type: ignore[no-untyped-call]
    orphan_document = dict(documents[0])
    orphan_document.update(
        {
            "article_id": "Q30901095:it:6369732:150337256",
            "document_id": "Q30901095:wikipedia:it:6369732:150337256",
            "wikidata": "Q30901095",
            "language": "it",
            "site": "itwiki",
            "page_id": 6369732,
            "revision_id": 150337256,
        }
    )
    orphan_section = dict(sections[0])
    orphan_section.update(
        {
            "section_id": stable_id(orphan_document["document_id"], 0, ""),
            "document_id": orphan_document["document_id"],
            "article_id": orphan_document["article_id"],
            "wikidata": orphan_document["wikidata"],
            "language": "it",
            "site": "itwiki",
            "page_id": orphan_document["page_id"],
            "revision_id": orphan_document["revision_id"],
        }
    )
    pq.write_table(
        pa.Table.from_pylist([*documents, orphan_document], schema=pq.read_schema(documents_path)),
        documents_path,
    )
    pq.write_table(
        pa.Table.from_pylist([*sections, orphan_section], schema=section_schema()),
        sections_path,
    )

    plan = audit_wikidata_integrity(
        data_root,
        [stem],
        _RecordingWikidataClient({}),
    ).region(stem)
    result = repair_wikidata_region(
        data_root,
        plan,
        wikidata_client=_RecordingWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        augmentation_client=_AugmentationClient(set()),
        settings=_settings(),
    )

    repaired_documents = pq.read_table(documents_path).to_pylist()  # type: ignore[no-untyped-call]
    repaired_sections = pq.read_table(sections_path).to_pylist()  # type: ignore[no-untyped-call]
    assert plan.orphan_document_ids == (orphan_document["document_id"],)
    assert result.changed is True
    assert result.map_inputs_changed is True
    assert repaired_documents == documents
    assert repaired_sections == sections


def test_duplicate_document_identity_is_rejected_before_commit(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "duplicates"
    _write_region(data_root, stem, ["Q1", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    table = pq.read_table(documents_path)  # type: ignore[no-untyped-call]
    pq.write_table(pa.concat_tables([table, table]), documents_path)
    original_hash = sha256_file(documents_path)

    with pytest.raises(RecoveryRepairError, match=r"duplicate (article_id|document_id)"):
        _repair(data_root, stem, "Q2")

    assert sha256_file(documents_path) == original_hash


def test_foreign_key_failure_is_rejected_before_commit(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "foreign-key"
    _write_region(data_root, stem, ["Q1", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    links_path = data_root.processed_links / f"{stem}.parquet"
    links = pq.read_table(links_path).to_pylist()  # type: ignore[no-untyped-call]
    links[0]["article_id"] = "missing-article"
    pq.write_table(pa.Table.from_pylist(links, schema=polygon_article_schema()), links_path)
    original_hash = sha256_file(links_path)

    with pytest.raises(RecoveryRepairError, match=r"(absent article_id|missing document)"):
        _repair(data_root, stem, "Q2")

    assert sha256_file(links_path) == original_hash


def test_failure_before_commit_leaves_every_original_hash_unchanged(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "rollback"
    _write_region(data_root, stem, ["Q1", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    tracked_paths = [
        data_root.processed_polygons / f"{stem}.parquet",
        data_root.processed_links / f"{stem}.parquet",
        data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
        data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet",
        data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
        data_root.processed_manifests / "processed_pbfs.json",
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json",
    ]
    hashes = {path: sha256_file(path) for path in tracked_paths}

    def fail() -> None:
        raise RuntimeError("injected before commit")

    with pytest.raises(RuntimeError, match="injected before commit"):
        _repair(data_root, stem, "Q2", before_commit=fail)

    assert {path: sha256_file(path) for path in tracked_paths} == hashes


def test_authoritatively_missing_article_is_receipted_and_not_retried(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "missing-article"
    _write_region(data_root, stem, ["Q1", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    plan = _audit_plan(data_root, stem, "Q2")

    result = repair_wikidata_region(
        data_root,
        plan,
        wikidata_client=_RecordingWikidataClient({"Q2": _entity("Q2")}),
        wikipedia_client=InMemoryWikipediaClient({}),
        augmentation_client=_AugmentationClient({"Q2"}),
        settings=_settings(),
    )
    second = audit_wikidata_integrity(data_root, [stem], _RecordingWikidataClient({}))

    assert result.changed is True
    assert second.region(stem).affected_qids == ()
    assert second.region(stem).reused is True


def test_transient_article_failure_aborts_without_receipt_or_file_changes(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "article-failure"
    _write_region(data_root, stem, ["Q1", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    plan = _audit_plan(data_root, stem, "Q2")
    tracked = [
        data_root.processed_polygons / f"{stem}.parquet",
        data_root.processed_links / f"{stem}.parquet",
        data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
        data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet",
        data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
    ]
    hashes = {path: sha256_file(path) for path in tracked}
    failing_wikipedia = InMemoryWikipediaClient(
        {("enwiki", "Title Q2"): FetchResult("http_error", None, "temporary outage")}
    )

    with pytest.raises(RecoveryRepairError, match="Incomplete Wikipedia recovery"):
        repair_wikidata_region(
            data_root,
            plan,
            wikidata_client=_RecordingWikidataClient({"Q2": _entity("Q2")}),
            wikipedia_client=failing_wikipedia,
            augmentation_client=_AugmentationClient({"Q2"}),
            settings=_settings(),
        )

    assert {path: sha256_file(path) for path in tracked} == hashes
    receipt = data_root.cache / "wikidata_recovery" / "index.json"
    assert not receipt.exists()


def test_second_audit_after_repair_is_a_no_op(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "idempotent"
    _write_region(data_root, stem, ["Q1", "Q2"], linked_qids={"Q1"})
    _finish_region(data_root, stem)
    _repair(data_root, stem, "Q2")
    paths = [
        data_root.processed_polygons / f"{stem}.parquet",
        data_root.processed_links / f"{stem}.parquet",
        data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
        data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet",
        data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
    ]
    hashes = {path: sha256_file(path) for path in paths}

    second = audit_wikidata_integrity(data_root, [stem], _RecordingWikidataClient({}))

    assert second.region(stem).affected_qids == ()
    assert second.region(stem).reused is True
    assert {path: sha256_file(path) for path in paths} == hashes


def test_recovery_resumes_after_last_durable_qid_batch(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    stem = "resumable"
    qids = tuple(f"Q{index}" for index in range(1, 28))
    affected = qids[1:]
    ordered_affected = tuple(sorted(affected))
    _write_region(data_root, stem, list(qids), linked_qids={"Q1"})
    _finish_region(data_root, stem)
    entities = {qid: _entity(qid) for qid in affected}
    plan = audit_wikidata_integrity(
        data_root,
        [stem],
        _RecordingWikidataClient(entities),
    ).region(stem)
    first_results = {
        ("enwiki", f"Title {qid}"): FetchResult("ok", _wikipedia_article(qid, index=int(qid[1:])))
        for qid in ordered_affected[:-1]
    }
    first_results[("enwiki", f"Title {ordered_affected[-1]}")] = FetchResult(
        "http_error", None, "injected second-batch failure"
    )

    with pytest.raises(RecoveryRepairError, match="injected second-batch failure"):
        repair_wikidata_region(
            data_root,
            plan,
            wikidata_client=_RecordingWikidataClient(entities),
            wikipedia_client=InMemoryWikipediaClient(first_results),
            augmentation_client=_AugmentationClient(set(affected)),
            settings=_settings(),
        )

    class RecordingWikipedia(InMemoryWikipediaClient):
        def __init__(self) -> None:
            super().__init__(
                {
                    ("enwiki", f"Title {qid}"): FetchResult(
                        "ok", _wikipedia_article(qid, index=int(qid[1:]))
                    )
                    for qid in affected
                }
            )
            self.titles: list[str] = []

        def fetch_article(self, language: str, site: str, title: str, **kwargs: Any):
            self.titles.append(title)
            return super().fetch_article(language, site, title, **kwargs)

    wikipedia = RecordingWikipedia()
    messages: list[str] = []
    result = repair_wikidata_region(
        data_root,
        plan,
        wikidata_client=_RecordingWikidataClient(entities),
        wikipedia_client=wikipedia,
        augmentation_client=_AugmentationClient(set(affected)),
        settings=_settings(),
        log=messages.append,
    )

    assert result.changed is True
    assert wikipedia.titles == [f"Title {ordered_affected[-1]}"]
    assert any("reused durable checkpoint" in message for message in messages)
    assert any("checkpoint saved" in message for message in messages)
    assert not (data_root.cache / "wikidata_recovery" / "checkpoints" / stem).exists()


def test_recovery_fetches_different_qids_concurrently_and_keeps_order() -> None:
    barrier = threading.Barrier(2, timeout=1)

    class ConcurrentWikipedia(InMemoryWikipediaClient):
        def __init__(self) -> None:
            super().__init__({})

        def fetch_article(self, language: str, site: str, title: str, **kwargs: Any):
            barrier.wait()
            qid = title.removeprefix("Title ")
            return FetchResult("ok", _wikipedia_article(qid, int(qid[1:])))

    rows = repair_module._fetch_missing_documents(
        ("Q1", "Q2"),
        entities={"Q1": _entity("Q1"), "Q2": _entity("Q2")},
        existing_documents=[],
        wikipedia_client=ConcurrentWikipedia(),
        settings=_settings(),
    )

    assert [row["wikidata"] for row in rows] == ["Q1", "Q2"]


def test_recovery_parses_different_documents_concurrently_and_keeps_order() -> None:
    documents = repair_module._fetch_missing_documents(
        ("Q1", "Q2"),
        entities={"Q1": _entity("Q1"), "Q2": _entity("Q2")},
        existing_documents=[],
        wikipedia_client=InMemoryWikipediaClient(
            {
                ("enwiki", "Title Q1"): FetchResult("ok", _wikipedia_article("Q1", 1)),
                ("enwiki", "Title Q2"): FetchResult("ok", _wikipedia_article("Q2", 2)),
            }
        ),
        settings=_settings(),
    )
    barrier = threading.Barrier(2, timeout=1)

    class ConcurrentAugmentation(_AugmentationClient):
        def parse_html(self, project: str, language: str, revision_id: int) -> str:
            barrier.wait()
            return f"<p>Revision {revision_id}</p>"

    rows = repair_module._sections_for_new_documents(
        documents,
        {str(row["document_id"]) for row in documents},
        augmentation_client=ConcurrentAugmentation({"Q1", "Q2"}),
    )

    assert [row["wikidata"] for row in rows] == ["Q1", "Q2"]
