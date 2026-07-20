"""Regression tests for the deterministic join-integrity enforcement.

The 9 diagnosed shard defects that motivated this module:

* ``italy-latest`` -- ``polygon_articles.parquet`` carries
  ``wikidata = Q30901095`` for ``italy-latest:way:845321022`` while
  the canonical polygons parquet carries
  ``wikidata = Q134675336``. The integrity pass MUST drop the stale
  link row and MUST NOT mutate either QID.
* Eight Wikivoyage shards
  (``australia``, ``bahamas``, ``brazil-nordeste``,
  ``canada-prince-edward-island``, ``canada-yukon``, ``chile``,
  ``mexico``, ``rheinland-pfalz``) have 1-12 wikivoyage documents
  whose ``wikidata`` QIDs are absent from the shard's polygons
  wikidata set. The integrity pass MUST drop the offending
  documents and cascade the rejection to their sections.

These tests exercise the public surface of
:mod:`osm_polygon_wikidata_only.augmentation.integrity` with hand
crafted parquets that mirror the real defect geometry. Unknown
integrity violations (e.g. polygon_articles referencing a polygon
not in the polygons parquet) MUST fail loudly and MUST NOT be
silently coerced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.integrity import (
    INTEGRITY_CONTRACT_VERSION,
    REASON_POLYGON_ARTICLES_MISMATCH,
    REASON_WIKIVOYAGE_ABSENT,
    PolygonArticlesIntegrityResult,
    WikivoyageIntegrityResult,
    enforce_all_regions,
    enforce_polygon_articles_integrity,
    enforce_wikivoyage_integrity,
)
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    SECTION_COLUMNS,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _minimal_polygon_row(polygon_id: str, wikidata: str) -> dict:
    """Build a single polygons row with the columns needed for the
    integrity checks (polygon_id + wikidata)."""
    return {
        "polygon_id": polygon_id,
        "region": polygon_id.split("-")[0],
        "source_pbf": f"{polygon_id.split(':')[0]}.osm.pbf",
        "osm_type": polygon_id.split(":")[1],
        "osm_id": int(polygon_id.split(":")[2]),
        "wikidata": wikidata,
        "name": "",
        "tags": json.dumps({}, sort_keys=True),
        "tag_keys": json.dumps([]),
        "tag_count": 0,
        "osm_primary_tag": "",
        "centroid": json.dumps({"type": "Point", "coordinates": [0.0, 0.0]}),
        "lat": 0.0,
        "lon": 0.0,
        "bbox": json.dumps([0.0, 0.0, 0.0, 0.0]),
        "geometry": "",
        "area_m2": 0.0,
        "area_km2": 0.0,
        "area_bucket": "0",
        "has_name": False,
        "has_wikidata": True,
        "has_wikipedia": False,
        "wikipedia_language_count": 0,
        "wikipedia_languages": json.dumps([]),
        "wikipedia_article_count": 0,
        "has_english_wikipedia": False,
        "has_french_wikipedia": False,
        "text_available": False,
        "best_language": "",
        "extraction_version": "test",
        "extracted_at": "2026-01-01T00:00:00Z",
    }


def _minimal_link_row(polygon_id: str, wikidata: str) -> dict:
    return {
        "polygon_id": polygon_id,
        "article_id": "article-1",
        "wikidata": wikidata,
        "language": "en",
        "source_pbf": "test.osm.pbf",
        "region": polygon_id.split("-")[0],
        "osm_type": polygon_id.split(":")[1],
        "osm_id": int(polygon_id.split(":")[2]),
        "page_id": 1,
        "revision_id": 1,
        "is_best_language": True,
    }


def _write_polygons(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="snappy")


def _write_polygon_articles(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{col: row.get(col) for col in POLYGON_ARTICLE_COLUMNS} for row in rows]
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, path, compression="snappy")


def _minimal_voyage_document(document_id: str, wikidata: str) -> dict:
    return {
        "document_id": document_id,
        "article_id": "art",
        "wikidata": wikidata,
        "project": "wikivoyage",
        "language": "en",
        "site": "enwikivoyage",
        "title": "t",
        "url": "https://en.wikivoyage.org/wiki/t",
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "full_text": "",
        "full_text_format": "plain_text",
        "article_length_chars": 0,
        "article_length_words": 0,
        "article_length_tokens_estimate": 0,
        "license": "CC-BY-SA",
        "attribution": "t",
        "source_api": "wikivoyage_document",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "h",
    }


def _minimal_voyage_section(section_id: str, document_id: str, wikidata: str) -> dict:
    return {
        "section_id": section_id,
        "document_id": document_id,
        "article_id": "art",
        "wikidata": wikidata,
        "project": "wikivoyage",
        "language": "en",
        "site": "enwikivoyage",
        "page_id": 1,
        "revision_id": 1,
        "section_index": 0,
        "heading": "h",
        "anchor": "a",
        "level": 2,
        "parent_section_id": "",
        "section_path": "/h",
        "text": "t",
        "text_length_chars": 1,
        "text_length_words": 1,
        "text_length_tokens_estimate": 1,
        "content_hash": "h",
        "license": "CC-BY-SA",
        "attribution": "t",
    }


def _write_voyage(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table({col: [] for col in columns})
    else:
        normalized = [{col: row.get(col) for col in columns} for row in rows]
        table = pa.Table.from_pylist(normalized)
    pq.write_table(table, path, compression="snappy")


def _make_data_root(tmp_path: Path) -> DataRoot:
    return DataRoot(tmp_path)


# ---------------------------------------------------------------------------
# Italy polygon_articles
# ---------------------------------------------------------------------------


def test_italy_polygon_articles_drops_stale_link_and_preserves_master_qid(tmp_path: Path):
    """Path A for italy-latest:way:845321022.

    polygons.parquet carries the canonical Q134675336. polygon_articles.parquet
    carries a stale Q30901095. The integrity pass drops the link
    row, rewrites polygon_articles, emits a deterministic rejection
    record, and NEVER rewrites either QID.
    """
    data_root = _make_data_root(tmp_path)
    stem = "italy-latest"
    polygon_id = "italy-latest:way:845321022"

    polygons = [_minimal_polygon_row(polygon_id, "Q134675336")]
    links = [_minimal_link_row(polygon_id, "Q30901095")]

    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)
    _write_polygon_articles(data_root.processed_links / f"{stem}.parquet", links)

    result = enforce_polygon_articles_integrity(data_root, stem)

    assert isinstance(result, PolygonArticlesIntegrityResult)
    assert result.shard == stem
    assert result.original_row_count == 1
    assert result.retained_row_count == 0
    assert result.rejected_row_count == 1
    assert result.rewritten is True

    # Rejection record: deterministic shape, no QID rewriting.
    assert len(result.rejections) == 1
    record = result.rejections[0]
    assert record.shard == stem
    assert record.source_table == "polygon_articles"
    assert record.identifier == polygon_id
    assert record.wikidata == "Q30901095"
    assert record.expected == "Q134675336"
    assert record.reason == REASON_POLYGON_ARTICLES_MISMATCH

    # polygon_articles parquet on disk now has zero rows (correct schema).
    rewritten = pq.read_table(data_root.processed_links / f"{stem}.parquet")
    assert rewritten.num_rows == 0
    assert list(rewritten.column_names) == list(POLYGON_ARTICLE_COLUMNS)

    # polygons parquet is unchanged (canonical Q134675336 preserved).
    polygons_table = pq.read_table(data_root.processed_polygons / f"{stem}.parquet")
    assert polygons_table.column("wikidata").to_pylist() == ["Q134675336"]


def test_polygon_articles_integrity_is_idempotent(tmp_path: Path):
    """A second run on the corrected polygon_articles must leave it
    untouched (no rows rejected, rewritten=False)."""
    data_root = _make_data_root(tmp_path)
    stem = "italy-latest"
    polygon_id = "italy-latest:way:845321022"

    polygons = [_minimal_polygon_row(polygon_id, "Q134675336")]
    # Already-correct link: polygon.wikidata == polygons.wikidata.
    links = [_minimal_link_row(polygon_id, "Q134675336")]

    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)
    links_path = data_root.processed_links / f"{stem}.parquet"
    _write_polygon_articles(links_path, links)
    before = links_path.read_bytes()

    result = enforce_polygon_articles_integrity(data_root, stem)

    assert result.rejected_row_count == 0
    assert result.rewritten is False
    assert links_path.read_bytes() == before  # byte-identical


# ---------------------------------------------------------------------------
# Unknown polygon_articles integrity violations must fail loudly
# ---------------------------------------------------------------------------


def test_polygon_articles_referencing_missing_polygon_raises(tmp_path: Path):
    """A polygon_articles row whose polygon_id is absent from polygons
    MUST NOT be silently dropped -- the join contract is total, so a
    missing polygon is a data hazard. Path A rejects nothing; it
    raises."""
    data_root = _make_data_root(tmp_path)
    stem = "italy-latest"
    polygon_id = "italy-latest:way:99999999"  # not in polygons

    # Polygons parquet carries a DIFFERENT polygon, not this one.
    polygons = [_minimal_polygon_row("italy-latest:way:1", "Q134675336")]
    links = [_minimal_link_row(polygon_id, "Q134675336")]

    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)
    _write_polygon_articles(data_root.processed_links / f"{stem}.parquet", links)

    with pytest.raises(ValueError, match="absent from polygons"):
        enforce_polygon_articles_integrity(data_root, stem)


# ---------------------------------------------------------------------------
# Wikivoyage shard integrity
# ---------------------------------------------------------------------------


WIKIVOYAGE_DEFECTIVE_SHARDS = [
    ("australia-latest", ["Q13426131"]),
    ("bahamas-latest", ["Q23666", "Q863944"]),
    ("brazil-nordeste-latest", ["Q10289", "Q10376", "Q11873418"]),
    ("canada-prince-edward-island-latest", ["Q28509"]),
    ("canada-yukon-latest", ["Q200254", "Q200255", "Q200256"]),
    ("chile-latest", ["Q298", "Q2980"]),
    ("mexico-latest", ["Q22050597", "Q22050598", "Q22050599", "Q22050600"]),
    ("rheinland-pfalz-latest", ["Q1022"]),
]


@pytest.mark.parametrize(
    ("stem", "invalid_qids"),
    WIKIVOYAGE_DEFECTIVE_SHARDS,
)
def test_wikivoyage_filters_documents_with_qids_absent_from_polygons(
    tmp_path: Path, stem: str, invalid_qids: list[str]
):
    """The eight Wikivoyage defect shards.

    polygons carries one valid QID. Wikivoyage documents include the
    valid QID plus one or more QIDs absent from polygons. The
    integrity pass drops the absent-QID documents and cascades to
    their sections. Path A: documents are dropped, never rewritten.
    """
    data_root = _make_data_root(tmp_path)
    polygon_id = f"{stem}:way:1"
    valid_qid = "Q1"

    # polygons: only the valid QID.
    polygons = [_minimal_polygon_row(polygon_id, valid_qid)]
    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)

    # documents: one valid document + one per invalid qid.
    documents = [_minimal_voyage_document(f"{valid_qid}:wikivoyage:en:1:1", valid_qid)]
    for index, qid in enumerate(invalid_qids, start=1):
        documents.append(_minimal_voyage_document(f"{qid}:wikivoyage:en:{index}:1", qid))

    # sections: one section per document, belonging to that document.
    sections: list[dict] = []
    for index, doc in enumerate(documents, start=1):
        sections.append(
            _minimal_voyage_section(
                section_id=f"sec-{index}",
                document_id=doc["document_id"],
                wikidata=doc["wikidata"],
            )
        )

    _write_voyage(
        data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
        documents,
        DOCUMENT_COLUMNS,
    )
    _write_voyage(
        data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet",
        sections,
        SECTION_COLUMNS,
    )

    result = enforce_wikivoyage_integrity(data_root, stem)

    assert isinstance(result, WikivoyageIntegrityResult)
    assert result.shard == stem
    assert result.original_document_count == 1 + len(invalid_qids)
    assert result.retained_document_count == 1
    assert result.rejected_document_count == len(invalid_qids)
    assert result.original_section_count == 1 + len(invalid_qids)
    assert result.retained_section_count == 1
    assert result.cascaded_section_count == len(invalid_qids)
    assert result.rewritten_documents is True
    assert result.rewritten_sections is True

    # Rejection records: one per rejected document, deterministic.
    assert len(result.rejections) == len(invalid_qids)
    expected_invalid_ids = {
        f"{qid}:wikivoyage:en:{index}:1" for index, qid in enumerate(invalid_qids, start=1)
    }
    actual_invalid_ids = {record.identifier for record in result.rejections}
    assert actual_invalid_ids == expected_invalid_ids
    for record in result.rejections:
        assert record.shard == stem
        assert record.source_table == "wikivoyage_documents"
        assert record.wikidata in invalid_qids
        assert record.expected is None
        assert record.reason == REASON_WIKIVOYAGE_ABSENT
        assert record.cascaded_sections == 1

    # On disk: only the valid document and its section remain.
    docs_after = pq.read_table(data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet")
    assert docs_after.num_rows == 1
    assert docs_after.column("wikidata").to_pylist() == [valid_qid]
    sections_after = pq.read_table(
        data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    )
    assert sections_after.num_rows == 1
    assert sections_after.column("document_id").to_pylist() == [f"{valid_qid}:wikivoyage:en:1:1"]


def test_wikivoyage_integrity_is_idempotent(tmp_path: Path):
    """A second run on a shard whose documents already match the
    polygons wikidata set must leave both files untouched."""
    data_root = _make_data_root(tmp_path)
    stem = "australia-latest"
    polygon_id = f"{stem}:way:1"
    valid_qid = "Q1"

    polygons = [_minimal_polygon_row(polygon_id, valid_qid)]
    documents = [_minimal_voyage_document(f"{valid_qid}:wikivoyage:en:1:1", valid_qid)]
    sections = [_minimal_voyage_section("sec-1", documents[0]["document_id"], valid_qid)]
    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)
    docs_path = data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
    sections_path = data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    _write_voyage(docs_path, documents, DOCUMENT_COLUMNS)
    _write_voyage(sections_path, sections, SECTION_COLUMNS)
    docs_before = docs_path.read_bytes()
    sections_before = sections_path.read_bytes()

    result = enforce_wikivoyage_integrity(data_root, stem)

    assert result.rejected_document_count == 0
    assert result.cascaded_section_count == 0
    assert result.rewritten_documents is False
    assert result.rewritten_sections is False
    assert docs_path.read_bytes() == docs_before
    assert sections_path.read_bytes() == sections_before


# ---------------------------------------------------------------------------
# Schema and column-order preservation
# ---------------------------------------------------------------------------


def test_polygon_articles_rewrite_preserves_schema_and_column_order(tmp_path: Path):
    data_root = _make_data_root(tmp_path)
    stem = "italy-latest"
    polygon_id = "italy-latest:way:845321022"

    polygons = [_minimal_polygon_row(polygon_id, "Q134675336")]
    # Two valid links + one stale link.
    links = [
        _minimal_link_row(polygon_id, "Q30901095"),
        _minimal_link_row("italy-latest:way:1", "Q1"),
        _minimal_link_row("italy-latest:way:2", "Q2"),
    ]
    # Polygons carries way:1 and way:2 with the matching QIDs.
    polygons.append(_minimal_polygon_row("italy-latest:way:1", "Q1"))
    polygons.append(_minimal_polygon_row("italy-latest:way:2", "Q2"))

    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)
    _write_polygon_articles(data_root.processed_links / f"{stem}.parquet", links)

    result = enforce_polygon_articles_integrity(data_root, stem)
    assert result.rejected_row_count == 1
    assert result.retained_row_count == 2

    rewritten = pq.read_table(data_root.processed_links / f"{stem}.parquet")
    assert list(rewritten.column_names) == list(POLYGON_ARTICLE_COLUMNS)


def test_wikivoyage_rewrite_preserves_schema_and_column_order(tmp_path: Path):
    data_root = _make_data_root(tmp_path)
    stem = "chile-latest"
    polygon_id = f"{stem}:way:1"
    valid_qid = "Q1"

    polygons = [_minimal_polygon_row(polygon_id, valid_qid)]
    documents = [
        _minimal_voyage_document(f"{valid_qid}:wikivoyage:en:1:1", valid_qid),
        _minimal_voyage_document("Q298:wikivoyage:en:2:1", "Q298"),
    ]
    sections = [
        _minimal_voyage_section("sec-1", documents[0]["document_id"], valid_qid),
        _minimal_voyage_section("sec-2", documents[1]["document_id"], "Q298"),
    ]
    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)
    _write_voyage(
        data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
        documents,
        DOCUMENT_COLUMNS,
    )
    _write_voyage(
        data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet",
        sections,
        SECTION_COLUMNS,
    )

    enforce_wikivoyage_integrity(data_root, stem)

    docs_after = pq.read_table(data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet")
    assert list(docs_after.column_names) == list(DOCUMENT_COLUMNS)
    sections_after = pq.read_table(
        data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
    )
    assert list(sections_after.column_names) == list(SECTION_COLUMNS)


# ---------------------------------------------------------------------------
# enforce_all_regions: aggregate audit
# ---------------------------------------------------------------------------


def test_enforce_all_regions_writes_deterministic_audit_json(tmp_path: Path):
    """enforce_all_regions runs both checks across every shard and
    writes a sorted, deterministic audit JSON."""
    data_root = _make_data_root(tmp_path)

    # italy polygon_articles defect.
    stem_italy = "italy-latest"
    polygons_italy = [_minimal_polygon_row("italy-latest:way:845321022", "Q134675336")]
    _write_polygons(data_root.processed_polygons / f"{stem_italy}.parquet", polygons_italy)
    _write_polygon_articles(
        data_root.processed_links / f"{stem_italy}.parquet",
        [_minimal_link_row("italy-latest:way:845321022", "Q30901095")],
    )

    # chile wikivoyage defect.
    stem_chile = "chile-latest"
    polygons_chile = [_minimal_polygon_row("chile-latest:way:1", "Q1")]
    _write_polygons(data_root.processed_polygons / f"{stem_chile}.parquet", polygons_chile)
    documents_chile = [
        _minimal_voyage_document("Q1:wikivoyage:en:1:1", "Q1"),
        _minimal_voyage_document("Q298:wikivoyage:en:2:1", "Q298"),
    ]
    sections_chile = [
        _minimal_voyage_section("sec-1", documents_chile[0]["document_id"], "Q1"),
        _minimal_voyage_section("sec-2", documents_chile[1]["document_id"], "Q298"),
    ]
    _write_voyage(
        data_root.processed / "wikivoyage" / "documents" / f"{stem_chile}.parquet",
        documents_chile,
        DOCUMENT_COLUMNS,
    )
    _write_voyage(
        data_root.processed / "wikivoyage" / "sections" / f"{stem_chile}.parquet",
        sections_chile,
        SECTION_COLUMNS,
    )

    report = enforce_all_regions(data_root)
    assert report.contract_version == INTEGRITY_CONTRACT_VERSION
    assert report.total_polygon_articles_rejected == 1
    assert report.total_wikivoyage_documents_rejected == 1
    assert report.total_wikivoyage_sections_cascaded == 1

    audit = json.loads(report.audit_path.read_text())
    assert audit["contract_version"] == INTEGRITY_CONTRACT_VERSION
    assert {entry["shard"] for entry in audit["polygon_articles"]} == {stem_italy}
    assert {entry["shard"] for entry in audit["wikivoyage"]} == {stem_chile}
    assert audit["totals"]["polygon_articles_rejected"] == 1
    assert audit["totals"]["wikivoyage_documents_rejected"] == 1
    assert audit["totals"]["wikivoyage_sections_cascaded"] == 1
    assert sorted(audit["totals"]["shards_with_rejections"]) == sorted([stem_chile, stem_italy])


def test_enforce_all_regions_skips_shards_without_sidecars(tmp_path: Path):
    data_root = _make_data_root(tmp_path)
    stem = "italy-latest"
    polygons = [_minimal_polygon_row("italy-latest:way:1", "Q1")]
    _write_polygons(data_root.processed_polygons / f"{stem}.parquet", polygons)

    report = enforce_all_regions(data_root)
    # No polygon_articles or wikivoyage/documents -> nothing to do.
    assert report.polygon_articles == ()
    assert report.wikivoyage == ()
    assert report.audit_path.exists()
