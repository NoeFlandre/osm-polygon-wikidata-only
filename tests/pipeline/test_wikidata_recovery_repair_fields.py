from __future__ import annotations

from osm_polygon_wikidata_only.pipeline._wikidata_recovery.repair_fields import (
    _preferred_language,
)


def test_preferred_language_uses_preference_then_input_order() -> None:
    assert _preferred_language(["zz", "en"]) == "en"
    assert _preferred_language(["zz", "xx"]) == "zz"
    assert _preferred_language([]) == ""
