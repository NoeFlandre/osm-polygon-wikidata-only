from __future__ import annotations

from typing import Any

import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery import link_rows
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import RecoveryRepairError


def _legacy_link(article: str = "a1", qid: str = "Q1") -> dict[str, Any]:
    return {
        "polygon_id": "p1",
        "article_id": article,
        "wikidata": qid,
        "language": "en",
        "source_pbf": "region.osm.pbf",
        "region": "region",
        "osm_type": "way",
        "osm_id": 1,
        "page_id": 100,
        "revision_id": 1,
        "is_best_language": True,
    }


def _document(article: str = "a1", qid: str = "Q1", project: str = "wikipedia") -> dict[str, Any]:
    return {
        "article_id": article,
        "document_id": f"{qid}:{project}:en:100:1",
        "project": project,
        "wikidata": qid,
        "language": "en",
        "page_id": 100,
        "revision_id": 1,
    }


def _canonical_link(document_id: str = "Q1:wikipedia:en:100:1") -> dict[str, Any]:
    return {
        "polygon_id": "p1",
        "document_id": document_id,
        "project": "wikipedia",
        "wikidata": "Q1",
        "language": "en",
        "source_pbf": "region.osm.pbf",
        "region": "region",
        "osm_type": "way",
        "osm_id": 1,
        "page_id": 100,
        "revision_id": 1,
    }


def _polygon(wikidata: str = "Q1;Q2") -> dict[str, Any]:
    return {
        "polygon_id": "p1",
        "wikidata": wikidata,
        "source_pbf": "region.osm.pbf",
        "region": "region",
        "osm_type": "way",
        "osm_id": 1,
        "best_language": "en",
    }


def test_legacy_links_convert_to_canonical_rows_and_preserve_voyage() -> None:
    voyage = {
        **_canonical_link("Q1:wikivoyage:en:200:2"),
        "document_id": "Q1:wikivoyage:en:200:2",
        "project": "wikivoyage",
        "page_id": 200,
        "revision_id": 2,
    }

    rows = link_rows.legacy_wikipedia_links_to_canonical([_legacy_link()], [_document()], [voyage])

    assert [row["project"] for row in rows] == ["wikipedia", "wikivoyage"]
    assert rows[0]["document_id"] == "Q1:wikipedia:en:100:1"
    assert rows[1] == voyage


def test_legacy_link_conversion_rejects_missing_document() -> None:
    with pytest.raises(RecoveryRepairError, match="missing article 'a1'"):
        link_rows.legacy_wikipedia_links_to_canonical([_legacy_link()], [], [])


def test_canonical_links_convert_and_skip_affected_orphans() -> None:
    converted = link_rows.canonical_wikipedia_links_to_legacy(
        [_canonical_link()], [_document()], [_polygon()], affected_qids={"Q1"}
    )
    assert converted[0]["article_id"] == "a1"
    assert converted[0]["is_best_language"] is True

    assert (
        link_rows.canonical_wikipedia_links_to_legacy(
            [_canonical_link("Q1:wikipedia:en:100:1")],
            [],
            [_polygon()],
            affected_qids={"Q1"},
        )
        == []
    )


def test_canonical_link_conversion_rejects_unaffected_orphan() -> None:
    with pytest.raises(RecoveryRepairError, match="missing document"):
        link_rows.canonical_wikipedia_links_to_legacy(
            [_canonical_link()], [], [_polygon()], affected_qids={"Q2"}
        )


def test_canonical_conversion_ignores_wikivoyage_rows() -> None:
    voyage = {
        **_canonical_link("Q1:wikivoyage:en:200:2"),
        "document_id": "Q1:wikivoyage:en:200:2",
        "project": "wikivoyage",
    }
    assert (
        link_rows.canonical_wikipedia_links_to_legacy(
            [voyage], [_document(project="wikivoyage")], [_polygon()], affected_qids={"Q1"}
        )
        == []
    )


def test_merge_links_adds_each_missing_document_for_affected_qids() -> None:
    links = link_rows.merge_links(
        [_polygon()],
        [_legacy_link(article="a1", qid="Q1")],
        [_document(article="a1", qid="Q1"), _document(article="a2", qid="Q2")],
        affected_qids={"Q1", "Q2"},
    )

    assert [(row["wikidata"], row["article_id"]) for row in links] == [
        ("Q1", "a1"),
        ("Q2", "a2"),
    ]


def test_merge_links_ignores_unaffected_qids_and_rejects_duplicate_input() -> None:
    assert (
        link_rows.merge_links(
            [_polygon()],
            [],
            [_document(article="a1", qid="Q1"), _document(article="a2", qid="Q2")],
            affected_qids={"Q1"},
        )[0]["wikidata"]
        == "Q1"
    )

    with pytest.raises(RecoveryRepairError, match="duplicate polygon-article identity"):
        link_rows.merge_links(
            [_polygon("Q1")],
            [_legacy_link(), _legacy_link()],
            [_document()],
            affected_qids={"Q1"},
        )
