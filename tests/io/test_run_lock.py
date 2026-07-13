"""Tests for exclusive unified-pipeline ownership."""

import pytest

from osm_polygon_wikidata_only.io.run_lock import RunLockError, exclusive_run_lock


def test_second_sync_lock_is_rejected(tmp_path) -> None:
    with exclusive_run_lock(tmp_path / "sync.lock"):
        with pytest.raises(RunLockError, match="already running"):
            with exclusive_run_lock(tmp_path / "sync.lock"):
                pass
