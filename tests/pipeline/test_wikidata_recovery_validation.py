"""Focused referential-integrity validation contracts."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import RecoveryRepairError
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.validation import validate_existing_rows


@pytest.mark.parametrize(
    ("link", "documents", "message"),
    [
        (
            {"polygon_id": "missing", "article_id": "a", "wikidata": "Q1"},
            [{"article_id": "a", "document_id": "d", "wikidata": "Q1"}],
            "link references missing polygon",
        ),
        (
            {"polygon_id": "p1", "article_id": "missing", "wikidata": "Q1"},
            [],
            "link references missing document",
        ),
        (
            {"polygon_id": "p1", "article_id": "a", "wikidata": "Q2"},
            [{"article_id": "a", "document_id": "d", "wikidata": "Q1"}],
            "link QID mismatch",
        ),
    ],
)
def test_link_referential_integrity_failures_are_explicit(
    link: dict[str, str], documents: list[dict[str, str]], message: str
) -> None:
    polygons = [{"polygon_id": "p1", "wikidata": "Q1"}]

    with pytest.raises(RecoveryRepairError, match=message):
        validate_existing_rows(polygons, [link], documents, [], [])
