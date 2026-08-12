from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.v2.deduplication import (
    deduplicate_documents,
    deduplicate_links,
)


def test_documents_deduplicate_identical_rows_and_sort_by_identity() -> None:
    rows = [
        {"document_id": "d2", "title": "Second"},
        {"document_id": "d1", "title": "First"},
        {"document_id": "d1", "title": "First"},
    ]

    assert deduplicate_documents(rows) == [
        {"document_id": "d1", "title": "First"},
        {"document_id": "d2", "title": "Second"},
    ]


def test_documents_reject_conflicting_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="Conflicting duplicate document identity 'd1'"):
        deduplicate_documents(
            [
                {"document_id": "d1", "title": "First"},
                {"document_id": "d1", "title": "Changed"},
            ]
        )


def test_links_deduplicate_by_polygon_project_document_identity() -> None:
    rows = [
        {"polygon_id": "p2", "project": "wikipedia", "document_id": "d2", "page_id": 2},
        {"polygon_id": "p1", "project": "wikipedia", "document_id": "d1", "page_id": 1},
        {"polygon_id": "p1", "project": "wikipedia", "document_id": "d1", "page_id": 1},
    ]

    assert deduplicate_links(rows) == [
        {"polygon_id": "p1", "project": "wikipedia", "document_id": "d1", "page_id": 1},
        {"polygon_id": "p2", "project": "wikipedia", "document_id": "d2", "page_id": 2},
    ]


def test_links_reject_conflicting_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="Conflicting duplicate link identity"):
        deduplicate_links(
            [
                {
                    "polygon_id": "p1",
                    "project": "wikipedia",
                    "document_id": "d1",
                    "page_id": 1,
                },
                {
                    "polygon_id": "p1",
                    "project": "wikipedia",
                    "document_id": "d1",
                    "page_id": 2,
                },
            ]
        )
