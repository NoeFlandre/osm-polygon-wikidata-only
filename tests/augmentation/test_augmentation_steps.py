from __future__ import annotations

from typing import Any

import pytest

from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.wikidata_facts import (
    _fact_claim_entity_id,
    _label_map,
    build_wikidata_facts,
)


class _LabelClient:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def entities(self, qids: list[str] | set[str], *, props: str) -> dict[str, dict[str, Any]]:
        ordered_qids = tuple(sorted(qids))
        self.calls.append((ordered_qids, props))
        return {qid: {"labels": {"en": {"value": f"Label {qid}"}}} for qid in ordered_qids}


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


def test_label_map_normalizes_values_and_ignores_invalid_entries() -> None:
    assert _label_map(
        {
            "labels": {
                "en": {"value": "English"},
                "fr": {"value": "Français"},
                "empty": {"value": ""},
                "missing": {},
                "malformed": "not a label record",
            }
        }
    ) == {"en": "English", "fr": "Français"}


def test_build_facts_sorts_fact_ids_and_advances_progress() -> None:
    client = _LabelClient()
    progress = AugmentationProgress()
    entities = {
        "Q1": {
            "id": "Q1",
            "claims": {
                "P17": [
                    {
                        "rank": "normal",
                        "mainsnak": {
                            "snaktype": "value",
                            "datatype": "wikibase-item",
                            "datavalue": {"value": {"id": "Q2"}},
                        },
                    }
                ],
                "P31": [
                    {
                        "rank": "normal",
                        "mainsnak": {
                            "snaktype": "value",
                            "datatype": "wikibase-item",
                            "datavalue": {"value": {"id": "Q3"}},
                        },
                    }
                ],
            },
        }
    }

    facts = build_wikidata_facts(client, entities=entities, progress=progress)

    assert len(client.calls) == 1
    assert client.calls[0][1] == "labels"
    assert {"P17", "P31", "Q2", "Q3"} <= set(client.calls[0][0])
    assert [fact.fact_id for fact in facts] == sorted(fact.fact_id for fact in facts)
    snapshot = progress.snapshot()
    assert (snapshot.phase, snapshot.completed, snapshot.total) == ("Wikidata facts", 1, 1)
