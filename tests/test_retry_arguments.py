"""Argument validation contracts for bounded and unbounded retries."""

import pytest

from osm_polygon_wikidata_only.utils.retry import _validate_retry_arguments


@pytest.mark.parametrize("attempts", [None, 1, 3])
def test_retry_attempts_accept_unbounded_or_positive_counts(attempts: int | None) -> None:
    _validate_retry_arguments(attempts)


@pytest.mark.parametrize("attempts", [0, -1])
def test_retry_attempts_reject_nonpositive_counts(attempts: int) -> None:
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        _validate_retry_arguments(attempts)
