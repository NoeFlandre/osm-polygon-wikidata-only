from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.v2.deduplication import (
    _deduplicate_rows,
    _document_identity,
    _link_identity,
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


def test_shared_deduplicate_rows_supports_custom_identity_and_sorting() -> None:
    rows = [
        {"slug": "b", "value": 2},
        {"slug": "a", "value": 1},
        {"slug": "a", "value": 1},
    ]

    assert _deduplicate_rows(
        rows,
        identity_of=lambda row: str(row["slug"]),
        kind="row",
    ) == [
        {"slug": "a", "value": 1},
        {"slug": "b", "value": 2},
    ]


def test_document_identity_helper_requires_and_normalizes_the_key() -> None:
    assert _document_identity({"document_id": 42}) == "42"
    assert _document_identity({"document_id": "XXXX"}) == "XXXX"
    with pytest.raises(ValueError, match="missing document_id"):
        _document_identity({})


def test_link_identity_helpers_preserve_field_order_and_missing_values() -> None:
    row = {"polygon_id": 1, "project": "wikipedia", "document_id": "d1"}
    assert _link_identity(row) == ("1", "wikipedia", "d1")

    sentinel = {"polygon_id": "XXXX", "project": "XXXX", "document_id": "XXXX"}
    assert _link_identity(sentinel) == ("XXXX", "XXXX", "XXXX")

    incomplete = {"polygon_id": "p1", "project": "wikipedia"}
    with pytest.raises(ValueError, match="missing identity fields"):
        _link_identity(incomplete)


def test_documents_reject_conflicting_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="Conflicting duplicate document identity 'd1'"):
        deduplicate_documents(
            [
                {"document_id": "d1", "title": "First"},
                {"document_id": "d1", "title": "Changed"},
            ]
        )


def test_documents_require_an_identity_and_do_not_mutate_inputs() -> None:
    row = {"document_id": "d1", "title": "Original"}

    assert deduplicate_documents([row]) == [row]
    assert row == {"document_id": "d1", "title": "Original"}

    with pytest.raises(ValueError) as error:
        deduplicate_documents([{"title": "missing"}])
    assert str(error.value) == "Document row is missing document_id"


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


def test_links_require_all_identity_fields_and_do_not_mutate_inputs() -> None:
    row = {"polygon_id": "p1", "project": "wikipedia", "document_id": "d1"}

    assert deduplicate_links([row]) == [row]
    assert row == {
        "polygon_id": "p1",
        "project": "wikipedia",
        "document_id": "d1",
    }

    with pytest.raises(ValueError) as error:
        deduplicate_links([{"polygon_id": "p1", "project": "wikipedia"}])
    assert str(error.value) == ("Link row is missing identity fields: ('p1', 'wikipedia', '')")


@pytest.mark.parametrize("missing", ["polygon_id", "project", "document_id"])
def test_links_report_each_missing_identity_field(missing: str) -> None:
    row = {"polygon_id": "p1", "project": "wikipedia", "document_id": "d1"}
    row.pop(missing)

    with pytest.raises(ValueError) as error:
        deduplicate_links([row])

    expected = {
        "polygon_id": ("", "wikipedia", "d1"),
        "project": ("p1", "", "d1"),
        "document_id": ("p1", "wikipedia", ""),
    }[missing]
    assert str(error.value) == f"Link row is missing identity fields: {expected!r}"
