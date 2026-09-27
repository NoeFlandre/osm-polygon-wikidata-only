from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.augmentation.steps import _fact_claim_entity_id


@pytest.mark.parametrize(
    "claim",
    [
        {},
        {"mainsnak": None},
        {"mainsnak": {"datavalue": {"value": None}}},
        {"mainsnak": {"datavalue": {"value": {}}}},
    ],
)
def test_fact_claim_without_entity_id_is_ignored(claim: dict[str, object]) -> None:
    assert _fact_claim_entity_id(claim) is None
