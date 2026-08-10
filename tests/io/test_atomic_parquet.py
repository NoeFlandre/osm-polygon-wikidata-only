from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.io import atomic


def test_atomic_write_parquet_replaces_target_without_partial_output(tmp_path: Path) -> None:
    target = tmp_path / "artifact.parquet"
    atomic.atomic_write_parquet(target, pa.table({"value": [1, 2]}))

    atomic.atomic_write_parquet(target, pa.table({"value": [3]}))

    assert pq.read_table(target).to_pylist() == [{"value": 3}]
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_parquet_cleans_up_after_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact.parquet"

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic.pq, "write_table", interrupted)

    with pytest.raises(KeyboardInterrupt):
        atomic.atomic_write_parquet(target, pa.table({"value": [1]}))

    assert list(tmp_path.iterdir()) == []
