from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.grid5000.sentence_controller_policy import (
    ControllerRunError,
    batch_stems,
    receipt_artifact,
)


@pytest.mark.parametrize("batch", [{}, {"stems": ("a",)}, {"stems": ["a", 1]}])
def test_batch_stems_rejects_non_string_lists(batch: dict[str, object]) -> None:
    with pytest.raises(ControllerRunError, match="batch stems are invalid"):
        batch_stems(batch)


@pytest.mark.parametrize(
    "raw",
    [None, [], {"relative_path": "x", "size": "1", "sha256": "abc"}],
)
def test_receipt_artifact_rejects_invalid_values(raw: object) -> None:
    with pytest.raises(ControllerRunError, match="receipt artifact"):
        receipt_artifact(raw)
