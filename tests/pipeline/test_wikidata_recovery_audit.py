from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import fact_schema, section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
    empty_row,
    polygon_article_schema,
    polygon_schema,
)
from osm_polygon_wikidata_only.enrichment.wikidata.cache import CachedWikidataClient
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataClient, WikidataEntity
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.pipeline._wikidata_recovery import audit as audit_mod
from osm_polygon_wikidata_only.pipeline.wikidata_recovery import (
    RECOVERY_CONTRACT_VERSION,
    RecoveryClassification,
    audit_wikidata_integrity,
)


class _RecordingWikidataClient(WikidataClient):
    def __init__(self, mapping: dict[str, WikidataEntity | None]) -> None:
        self.mapping = mapping
        self.batch_calls: list[list[str]] = []

    def get_entity(self, qid: str) -> WikidataEntity | None:
        return self.mapping[qid]

    def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
        requested = list(qids)
        self.batch_calls.append(requested)
        return [self.mapping[qid] for qid in requested]


class _FailingWikidataClient(WikidataClient):
    def get_entity(self, qid: str) -> WikidataEntity | None:
        raise RuntimeError(f"network unavailable for {qid}")

    def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
        raise RuntimeError(f"network unavailable for {qids}")


def _table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def _article_row(qid: str, index: int) -> dict[str, Any]:
    row = empty_row(ARTICLE_COLUMNS)
    page_id = 1000 + index
    revision_id = 2000 + index
    row.update(
        {
            "article_id": f"{qid}:en:{page_id}:{revision_id}",
            "wikidata": qid,
            "language": "en",
            "site": "enwiki",
            "title": f"Title {qid}",
            "url": f"https://en.wikipedia.org/wiki/Title_{qid}",
            "page_id": page_id,
            "revision_id": revision_id,
            "full_text_format": "plain_text",
            "fetch_status": "ok",
        }
    )
    return row


def _polygon_row(stem: str, qid: str, index: int) -> dict[str, Any]:
    row = empty_row(POLYGON_COLUMNS)
    row.update(
        {
            "polygon_id": f"{stem}:way:{index}",
            "region": stem,
            "source_pbf": f"{stem}.osm.pbf",
            "osm_type": "way",
            "osm_id": index,
            "wikidata": qid,
            "has_wikidata": True,
            "wikipedia_languages": "[]",
        }
    )
    return row


def _write_region(
    data_root: DataRoot,
    stem: str,
    qids: list[str],
    *,
    linked_qids: set[str] | None = None,
) -> None:
    linked = linked_qids or set()
    polygons = [_polygon_row(stem, qid, index) for index, qid in enumerate(qids, start=1)]
    documents: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    qid_document: dict[str, dict[str, Any]] = {}
    for index, qid in enumerate(dict.fromkeys(qids), start=1):
        if qid not in linked:
            continue
        document = wikipedia_document_from_article_row(_article_row(qid, index)).to_dict()
        qid_document[qid] = document
        documents.append(document)
    for polygon in polygons:
        qid = str(polygon["wikidata"])
        document = qid_document.get(qid)
        if document is None:
            continue
        link = empty_row(POLYGON_ARTICLE_COLUMNS)
        link.update(
            {
                "polygon_id": polygon["polygon_id"],
                "article_id": document["article_id"],
                "wikidata": qid,
                "language": "en",
                "source_pbf": polygon["source_pbf"],
                "region": stem,
                "osm_type": "way",
                "osm_id": polygon["osm_id"],
                "page_id": document["page_id"],
                "revision_id": document["revision_id"],
                "is_best_language": True,
            }
        )
        links.append(link)

    paths = {
        data_root.processed_polygons / f"{stem}.parquet": _table(polygons, polygon_schema()),
        data_root.processed_links / f"{stem}.parquet": _table(links, polygon_article_schema()),
        data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet": _table(
            documents, wikipedia_document_schema()
        ),
        data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet": _table(
            [], section_schema()
        ),
        data_root.processed / "wikidata" / "facts" / f"{stem}.parquet": _table([], fact_schema()),
    }
    for path, table in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def _data_root(tmp_path: Path) -> DataRoot:
    root = DataRoot(tmp_path / "dataset")
    root.ensure()
    return root


def _entity(qid: str, *, sitelinks: bool = True) -> WikidataEntity:
    return WikidataEntity(
        qid=qid,
        sitelinks={"enwiki": f"Title {qid}"} if sitelinks else {},
    )


def test_full_failure_region_is_detected_without_thresholds(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    qids = [f"Q{index}" for index in range(1, 51)]
    _write_region(data_root, "full-failure", qids)
    client = _RecordingWikidataClient({qid: _entity(qid) for qid in qids})

    result = audit_wikidata_integrity(data_root, ["full-failure"], client, batch_size=50)

    region = result.region("full-failure")
    assert region.affected_qids == tuple(sorted(qids))
    assert region.affected_polygon_count == 50
    assert client.batch_calls == [sorted(qids)]


def test_one_lost_qid_in_majority_covered_region_is_detected(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "partial", ["Q1", "Q2", "Q3"], linked_qids={"Q1", "Q2"})
    client = _RecordingWikidataClient({"Q3": _entity("Q3")})

    result = audit_wikidata_integrity(data_root, ["partial"], client)

    assert result.region("partial").affected_qids == ("Q3",)
    assert client.batch_calls == [["Q3"]]


@pytest.mark.parametrize(
    ("entity", "expected"),
    [
        (None, RecoveryClassification.AUTHORITATIVE_MISSING),
        (_entity("Q1", sitelinks=False), RecoveryClassification.AUTHORITATIVE_NO_SITELINK),
    ],
)
def test_authoritative_non_repair_outcomes_are_not_repaired(
    tmp_path: Path,
    entity: WikidataEntity | None,
    expected: RecoveryClassification,
) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "healthy-negative", ["Q1"])

    result = audit_wikidata_integrity(
        data_root,
        ["healthy-negative"],
        _RecordingWikidataClient({"Q1": entity}),
    )

    assert result.qid("Q1").state is expected
    assert result.region("healthy-negative").affected_qids == ()


def test_positive_cache_detects_stale_output_without_inner_fetch(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "cached-positive", ["Q1"])
    cache = JsonFileCache(tmp_path / "cache")
    cache.set(
        "wikidata/Q1.json",
        {
            "qid": "Q1",
            "sitelinks": {"enwiki": "Title Q1"},
            "labels": {},
            "descriptions": {},
            "aliases": {},
        },
        status="ok",
    )

    result = audit_wikidata_integrity(
        data_root,
        ["cached-positive"],
        CachedWikidataClient(_FailingWikidataClient(), cache),
    )

    assert result.region("cached-positive").affected_qids == ("Q1",)


def test_legacy_ambiguous_negative_forces_validation(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "legacy-negative", ["Q1"])
    cache = JsonFileCache(tmp_path / "cache")
    cache.set(
        "wikidata/Q1.json",
        None,
        status="error",
        response_metadata={"reason": "wikidata_not_found"},
    )
    inner = _RecordingWikidataClient({"Q1": _entity("Q1")})

    result = audit_wikidata_integrity(
        data_root,
        ["legacy-negative"],
        CachedWikidataClient(inner, cache),
    )

    assert result.region("legacy-negative").affected_qids == ("Q1",)
    assert inner.batch_calls == [["Q1"]]


def test_shared_qid_marks_every_polygon_and_region_after_one_validation(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "alpha", ["Q1", "Q1"])
    _write_region(data_root, "beta", ["Q1"])
    client = _RecordingWikidataClient({"Q1": _entity("Q1")})

    result = audit_wikidata_integrity(data_root, ["beta", "alpha"], client)

    assert result.region("alpha").affected_polygon_count == 2
    assert result.region("beta").affected_polygon_count == 1
    assert result.qid("Q1").regions == ("alpha", "beta")
    assert client.batch_calls == [["Q1"]]


def test_all_qids_are_globally_deduplicated_and_sorted_before_batches(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "one", ["Q3", "Q1", "Q2", "Q1"])
    _write_region(data_root, "two", ["Q2", "Q4"])
    client = _RecordingWikidataClient({qid: _entity(qid) for qid in ("Q1", "Q2", "Q3", "Q4")})

    audit_wikidata_integrity(data_root, ["two", "one"], client, batch_size=2)

    assert client.batch_calls == [["Q1", "Q2"], ["Q3", "Q4"]]


def test_unreadable_parquet_blocks_without_writing_receipt(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "broken", ["Q1"])
    (data_root.processed_polygons / "broken.parquet").write_bytes(b"not parquet")

    result = audit_wikidata_integrity(
        data_root,
        ["broken"],
        _RecordingWikidataClient({}),
    )

    assert result.region("broken").blocked_reason
    assert result.region("broken").classifications == ()
    assert not (data_root.cache / "wikidata_recovery" / "index.json").exists()


def test_receipt_reuse_requires_matching_hashes_and_contract(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "receipt", ["Q1"])
    first = _RecordingWikidataClient({"Q1": None})
    audit_wikidata_integrity(data_root, ["receipt"], first)

    reused = audit_wikidata_integrity(data_root, ["receipt"], _FailingWikidataClient())
    assert reused.region("receipt").reused is True

    _write_region(data_root, "receipt", ["Q1", "Q2"])
    changed = _RecordingWikidataClient({"Q1": None, "Q2": None})
    audit_wikidata_integrity(data_root, ["receipt"], changed)
    assert changed.batch_calls == [["Q1", "Q2"]]

    index_path = data_root.cache / "wikidata_recovery" / "index.json"
    text = index_path.read_text(encoding="utf-8")
    index_path.write_text(text.replace(RECOVERY_CONTRACT_VERSION, "old-contract"), encoding="utf-8")
    contract_changed = _RecordingWikidataClient({"Q1": None, "Q2": None})
    audit_wikidata_integrity(data_root, ["receipt"], contract_changed)
    assert contract_changed.batch_calls == [["Q1", "Q2"]]


def test_audit_does_not_read_or_hash_sections_and_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "bounded", ["Q1"], linked_qids={"Q1"})
    forbidden = {
        data_root.processed / "wikipedia" / "sections" / "bounded.parquet",
        data_root.processed / "wikidata" / "facts" / "bounded.parquet",
    }
    original_read_table = pq.read_table
    original_sha256 = audit_mod.sha256_file

    def guarded_read(path: object, *args: object, **kwargs: object) -> pa.Table:
        assert Path(path) not in forbidden
        return original_read_table(path, *args, **kwargs)

    def guarded_hash(path: Path) -> str:
        assert path not in forbidden
        return original_sha256(path)

    monkeypatch.setattr(audit_mod.pq, "read_table", guarded_read)
    monkeypatch.setattr(audit_mod, "sha256_file", guarded_hash)

    result = audit_wikidata_integrity(data_root, ["bounded"], _RecordingWikidataClient({}))
    assert result.region("bounded").blocked_reason == ""


def test_interrupted_validation_writes_no_completed_receipt(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "interrupted", ["Q1"])

    with pytest.raises(RuntimeError, match="network unavailable"):
        audit_wikidata_integrity(data_root, ["interrupted"], _FailingWikidataClient())

    assert not (data_root.cache / "wikidata_recovery" / "index.json").exists()


def test_audit_output_and_index_are_deterministic(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    _write_region(data_root, "zeta", ["Q2", "Q1"])
    _write_region(data_root, "alpha", ["Q3"])
    client = _RecordingWikidataClient({"Q1": None, "Q2": None, "Q3": None})

    result = audit_wikidata_integrity(data_root, ["zeta", "alpha"], client)
    first_index = (data_root.cache / "wikidata_recovery" / "index.json").read_text()
    second = audit_wikidata_integrity(data_root, ["alpha", "zeta"], _FailingWikidataClient())
    second_index = (data_root.cache / "wikidata_recovery" / "index.json").read_text()

    assert tuple(region.stem for region in result.regions) == ("alpha", "zeta")
    assert tuple(entry.qid for entry in result.qids) == ("Q1", "Q2", "Q3")
    assert result == second
    assert first_index == second_index
