"""Phase 2 / Amendment 9: PublicationOp boundary tightening.

The queue-only ``snapshot_path`` state must be private to the queue
machinery -- ordinary publication assemblers must NOT set it. If the
field is retained on ``PublicationOp``, the contract must be
documented and enforced:

1. ``add_op(...)`` and ``delete_op(...)`` are the assembler-facing
   factories; both must produce ``snapshot_path=None``.
2. ``PublicationOp(action="delete", ..., snapshot_path=...)`` is
   rejected at construction.
3. Direct construction with ``snapshot_path`` is allowed only via a
   documented queue-only path (e.g. a private constructor or
   ``set_snapshot_path`` on a non-frozen subclass).
4. The upload callback must read bytes from the immutable snapshot
   when present, NOT from the canonical ``local_path``.
5. The cleanup step after a successful upload must delete the
   SNAPSHOT, leaving the canonical ``local_path`` untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_publication_op():
    from osm_polygon_wikidata_only.hf._uploader.plan import (
        PublicationOp,
        add_op,
        delete_op,
    )

    return PublicationOp, add_op, delete_op


def test_add_op_factory_does_not_set_snapshot_path(tmp_path: Path) -> None:
    """``add_op`` must default ``snapshot_path`` to ``None`` -- it is a
    publication-assembler factory, not a queue factory."""
    _PublicationOp, add_op, _delete = _import_publication_op()
    canonical = tmp_path / "data.parquet"
    canonical.write_bytes(b"DATA")
    op = add_op(canonical, path_in_repo="data.parquet")
    assert op.snapshot_path is None, (
        f"add_op must produce snapshot_path=None; got {op.snapshot_path!r}"
    )


def test_delete_op_factory_does_not_set_snapshot_path(tmp_path: Path) -> None:
    _PublicationOp, _add, delete_op = _import_publication_op()
    op = delete_op("legacy/file.parquet")
    assert op.snapshot_path is None, (
        f"delete_op must produce snapshot_path=None; got {op.snapshot_path!r}"
    )


def test_publication_op_rejects_snapshot_path_on_delete(tmp_path: Path) -> None:
    """``PublicationOp(action="delete", ..., snapshot_path=...)`` must
    raise -- a delete has no snapshot."""
    PublicationOp, _add, _delete = _import_publication_op()
    with pytest.raises(ValueError):
        PublicationOp(
            action="delete",
            path_in_repo="legacy/file.parquet",
            local_path=None,
            snapshot_path=tmp_path / "snapshot",
        )


@pytest.mark.parametrize(
    ("action", "local_path", "message"),
    [
        ("add", None, "requires a local_path"),
        ("delete", Path("canonical.parquet"), "must not carry a local_path"),
    ],
)
def test_publication_op_enforces_local_path_by_action(
    action: str, local_path: Path | None, message: str
) -> None:
    PublicationOp, _add, _delete = _import_publication_op()

    with pytest.raises(ValueError, match=message):
        PublicationOp(action=action, path_in_repo="data.parquet", local_path=local_path)


def test_ordinary_publication_assemblers_do_not_set_snapshot_path() -> None:
    """No assembler helper in ``hf.publication`` should construct a
    ``PublicationOp`` with a non-None ``snapshot_path``. The
    queue-only field must never leak out of the queue layer.
    """
    import inspect

    from osm_polygon_wikidata_only.hf import publication as pub

    source = inspect.getsource(pub)
    # The assemblers should only call add_op(...) or delete_op(...).
    # Direct ``PublicationOp(`` with a snapshot_path kwarg must not
    # appear in the publication module.
    assert "snapshot_path=" not in source, (
        "hf/publication.py must not construct PublicationOp with snapshot_path="
    )


def test_upload_uses_snapshot_bytes_when_present(tmp_path: Path) -> None:
    """The queue snapshots the canonical file at submit time and
    passes the immutable snapshot_path on the op. The upload
    callback reads from ``op.snapshot_path`` (immutable), not
    ``op.local_path`` (canonical, may mutate).
    """
    from osm_polygon_wikidata_only.hf._uploader.plan import add_op
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"CANONICAL")

    captured: dict[str, bytes] = {}

    def upload(ops, message):
        # The upload callback (as in cli/run_sync.py) must read from
        # the snapshot path when present.
        source = ops[0].snapshot_path or ops[0].local_path
        assert source is not None
        captured["bytes"] = source.read_bytes()
        captured["path"] = str(source)

    q = BackgroundUploadQueue(upload=upload, state_dir=tmp_path / "state")
    q.submit([add_op(canonical, path_in_repo="data.parquet")], "test")
    # Mutate the canonical file. The queue's snapshot must still
    # contain the original CANONICAL bytes.
    canonical.write_bytes(b"MUTATED")
    q.close_and_wait()
    assert captured.get("bytes") == b"CANONICAL", (
        f"Upload must read snapshot bytes, got {captured.get('bytes')!r}"
    )
    # The path used must be the snapshot path, NOT canonical.
    assert "/snapshots/" in captured.get("path", ""), (
        f"Upload must use the snapshot subdirectory, got path={captured.get('path')!r}"
    )
    assert "canonical.parquet" == Path(captured.get("path", "")).name, (
        f"Upload path must end with the snapshot's local filename, got path={captured.get('path')!r}"
    )


def test_cleanup_deletes_snapshot_preserves_canonical(tmp_path: Path) -> None:
    """After a successful upload, the queue-owned snapshot directory
    must be removed, but the canonical ``local_path`` must remain.
    """
    from osm_polygon_wikidata_only.hf._uploader.plan import add_op
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"DATA")

    state_dir = tmp_path / "state"
    q = BackgroundUploadQueue(upload=lambda ops, msg: None, state_dir=state_dir)
    q.submit([add_op(canonical, path_in_repo="data.parquet")], "test")
    q.close_and_wait()
    # Snapshots directory must be empty (or absent) after successful upload.
    snapshots_dir = state_dir / "snapshots"
    if snapshots_dir.is_dir():
        leftovers = list(snapshots_dir.iterdir())
        assert leftovers == [], f"Successful upload must remove all snapshots, got {leftovers}"
    # Canonical must remain.
    assert canonical.is_file(), "Canonical file must be preserved after upload"
    assert canonical.read_bytes() == b"DATA"
