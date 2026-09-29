"""Direct contracts for small recovery-audit input validators."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery.audit_inputs import (
    required_string,
    validate_polygon_qid,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.audit_types import ScanError


def test_validate_polygon_qid_requires_present_matching_qid() -> None:
    polygons = {"p1": ("Q1", "Q2")}
    validate_polygon_qid("p1", "Q2", polygons)

    with pytest.raises(ScanError, match="absent polygon_id"):
        validate_polygon_qid("missing", "Q1", polygons)
    with pytest.raises(ScanError, match="disagrees with polygon"):
        validate_polygon_qid("p1", "Q3", polygons)


def test_required_string_accepts_only_nonempty_strings() -> None:
    assert required_string({"id": "p1"}, "id", "polygons") == "p1"

    for value in ("", None, 1):
        with pytest.raises(ScanError, match="empty or non-string id"):
            required_string({"id": value}, "id", "polygons")
