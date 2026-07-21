from __future__ import annotations

from osm_polygon_wikidata_only.pipeline._wikidata_recovery.progress import RecoveryProgress


def test_progress_formats_active_stage_and_eta() -> None:
    now = [100.0]
    progress = RecoveryProgress("gcc-latest", 4, clock=lambda: now[0])
    progress.start_batch(2, ("Q26", "Q27"))
    progress.set_stage("Wikipedia documents", total=20)
    progress.advance(5, documents=3)
    now[0] = 160.0

    message = progress.message()

    assert "gcc-latest" in message
    assert "batch 2/4" in message
    assert "Wikipedia documents 5/20" in message
    assert "documents 3" in message
    assert "60s elapsed" in message
    assert "ETA" in message


def test_checkpoint_message_is_immediate_and_factual() -> None:
    progress = RecoveryProgress("gcc-latest", 2, clock=lambda: 10.0)
    progress.start_batch(1, ("Q1",))
    progress.checkpoint_saved(documents=2, sections=5, facts=3)

    assert progress.message().startswith("Wikidata recovery progress gcc-latest: batch 1/2")
    assert "checkpoint saved" in progress.message()
    assert "documents 2; sections 5; facts 3" in progress.message()
