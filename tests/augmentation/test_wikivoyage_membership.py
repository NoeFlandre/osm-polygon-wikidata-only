from __future__ import annotations

from osm_polygon_wikidata_only.augmentation._wikivoyage_membership import (
    partition_wikivoyage_documents,
)
from osm_polygon_wikidata_only.augmentation.schema import DOCUMENT_COLUMNS


def test_partition_projects_retained_rows_and_preserves_rejected_identity() -> None:
    retained = {
        "document_id": "doc:kept",
        "wikidata": "Q1",
        "title": "Kept",
        "unexpected": "drop this field",
    }
    rejected = {
        "document_id": "doc:rejected",
        "wikidata": "Q2",
        "title": "Rejected",
        "unexpected": "also drop this field",
    }

    result = partition_wikivoyage_documents([retained, rejected], {"Q1"})

    assert result.retained_rows == [{column: retained.get(column) for column in DOCUMENT_COLUMNS}]
    assert [(row.document_id, row.wikidata) for row in result.rejected_documents] == [
        ("doc:rejected", "Q2")
    ]


def test_partition_keeps_source_order_and_duplicate_rejections() -> None:
    rows = [
        {"document_id": "doc:two", "wikidata": "Q2"},
        {"document_id": "doc:one", "wikidata": "Q3"},
        {"document_id": "doc:two", "wikidata": "Q4"},
    ]

    result = partition_wikivoyage_documents(rows, set())

    assert result.retained_rows == []
    assert [(row.document_id, row.wikidata) for row in result.rejected_documents] == [
        ("doc:two", "Q2"),
        ("doc:one", "Q3"),
        ("doc:two", "Q4"),
    ]
