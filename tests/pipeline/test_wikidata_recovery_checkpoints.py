from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.augmentation.schema import fact_schema, section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.checkpoints import (
    RecoveryBatchArtifacts,
    RecoveryCheckpointStore,
    recovery_plan_key,
)


def _artifacts(qid: str) -> RecoveryBatchArtifacts:
    return RecoveryBatchArtifacts(qids=(qid,), documents=(), sections=(), facts=())


def test_checkpoint_round_trip_and_deterministic_path(tmp_path: Path) -> None:
    key = recovery_plan_key(
        fingerprints=(("polygons", "abc"),),
        affected_qids=("Q1", "Q2"),
        sections_hash="sections",
        settings_identity=(None, None, True),
    )
    store = RecoveryCheckpointStore(tmp_path, "region-latest", key)

    path = store.save(0, _artifacts("Q1"))

    assert path == tmp_path / "region-latest" / key / "batch-000000"
    assert store.load(0, ("Q1",)) == _artifacts("Q1")
    assert store.load(1, ("Q2",)) is None


def test_checkpoint_uses_exact_table_schemas(tmp_path: Path) -> None:
    store = RecoveryCheckpointStore(tmp_path, "region-latest", "plan")

    path = store.save(0, _artifacts("Q1"))

    import pyarrow.parquet as pq

    assert pq.read_schema(path / "documents.parquet").equals(
        wikipedia_document_schema(), check_metadata=True
    )
    assert pq.read_schema(path / "sections.parquet").equals(section_schema(), check_metadata=True)
    assert pq.read_schema(path / "facts.parquet").equals(fact_schema(), check_metadata=True)


def test_incomplete_checkpoint_is_not_reused(tmp_path: Path) -> None:
    store = RecoveryCheckpointStore(tmp_path, "region-latest", "plan")
    incomplete = tmp_path / "region-latest" / "plan" / "batch-000000"
    incomplete.mkdir(parents=True)

    assert store.load(0, ("Q1",)) is None


def test_corrupted_checkpoint_is_not_reused(tmp_path: Path) -> None:
    store = RecoveryCheckpointStore(tmp_path, "region-latest", "plan")
    path = store.save(0, _artifacts("Q1"))
    with (path / "documents.parquet").open("ab") as stream:
        stream.write(b"corrupt")

    assert store.load(0, ("Q1",)) is None


def test_clear_removes_only_the_region_checkpoint(tmp_path: Path) -> None:
    first = RecoveryCheckpointStore(tmp_path, "first", "plan")
    second = RecoveryCheckpointStore(tmp_path, "second", "plan")
    first.save(0, _artifacts("Q1"))
    second.save(0, _artifacts("Q2"))

    first.clear()

    assert not (tmp_path / "first").exists()
    assert second.load(0, ("Q2",)) == _artifacts("Q2")
