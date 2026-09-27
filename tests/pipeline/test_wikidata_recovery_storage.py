from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import RecoveryRepairError
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.storage import read_table


def test_read_table_rejects_missing_artifacts_and_schema_mismatches(tmp_path: Path) -> None:
    schema = pa.schema([("polygon_id", pa.string())])
    missing = tmp_path / "missing.parquet"
    with pytest.raises(RecoveryRepairError, match="input is missing"):
        read_table(missing, schema)

    wrong_schema = tmp_path / "wrong.parquet"
    pq.write_table(pa.table({"unexpected": ["value"]}), wrong_schema)
    with pytest.raises(RecoveryRepairError, match="schema mismatch"):
        read_table(wrong_schema, schema)
