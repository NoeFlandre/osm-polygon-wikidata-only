"""Tests for the sync upload queue and its publication boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from osm_polygon_wikidata_only.cli import sync_publication
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf import publication
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub


def test_enqueue_containment_retirement_logs_all_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings(repo_id="org/dataset")
    operations = [delete_op("child/polygons.parquet")]
    submitted: list[tuple[list[PublicationOp], str]] = []
    assembly_calls: list[dict[str, object]] = []

    def assemble(**kwargs: object) -> list[PublicationOp]:
        assembly_calls.append(kwargs)
        return operations

    class Queue:
        def submit(self, ops: list[PublicationOp], message: str) -> None:
            submitted.append((ops, message))

    monkeypatch.setattr(publication, "assemble_containment_retirement_upload", assemble)

    with caplog.at_level("INFO", logger="osm_polygon_wikidata_only.cli"):
        result = sync_publication.enqueue_containment_retirement(
            data_root,
            settings,
            {"parent": ("first", "second")},
            cast(Any, Queue()),
            push_enabled=True,
        )

    assert result is True
    assert assembly_calls == [
        {
            "data_root": data_root,
            "repo_id": "org/dataset",
            "parent_children": {"parent": ("first", "second")},
            "world_land_warning": None,
        }
    ]
    assert submitted == [(operations, "Retire losslessly contained regional dataset shards")]
    assert "Enqueued containment retirement for 2 child region(s)" in caplog.messages


def test_execute_upload_job_uses_snapshots_and_cleans_up_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(repo_id="org/dataset", hf_token="secret-token")
    data_root = DataRoot(tmp_path)
    hub = object()
    ops = [
        PublicationOp(
            action="add",
            path_in_repo="polygons/region.parquet",
            local_path=Path("canonical.parquet"),
            snapshot_path=Path("snapshot.parquet"),
        ),
        add_op("ordinary.parquet", path_in_repo="ordinary.parquet"),
    ]
    calls: list[tuple[str, Any]] = []

    def upload(repo_id: str, **kwargs: Any) -> None:
        calls.append(("upload", (repo_id, kwargs)))

    def cleanup(actual_root: DataRoot, actual_ops: list[PublicationOp], *, dry_run: bool) -> None:
        calls.append(("cleanup", (actual_root, actual_ops, dry_run)))

    monkeypatch.setattr(sync_publication, "upload_files", upload)
    monkeypatch.setattr(sync_publication, "post_upload_publication_cleanup", cleanup)

    sync_publication.execute_upload_job(
        data_root=data_root,
        settings=settings,
        ops=ops,
        message="sync region",
        num_threads=5,
        hub=cast(Any, hub),
        dry_run=True,
    )

    upload_repo, upload_kwargs = calls[0][1]
    assert calls[0][0] == "upload"
    assert upload_repo == "org/dataset"
    uploaded_ops = upload_kwargs["ops"]
    assert uploaded_ops == [
        PublicationOp("add", "polygons/region.parquet", Path("snapshot.parquet")),
        PublicationOp("add", "ordinary.parquet", Path("ordinary.parquet")),
    ]
    assert upload_kwargs == {
        "ops": uploaded_ops,
        "hub": hub,
        "token": "secret-token",
        "commit_message": "sync region",
        "num_threads": 5,
    }
    assert calls[1] == ("cleanup", (data_root, ops, True))


@pytest.mark.parametrize(
    ("dry_run", "injected_hub", "expected_kind"),
    [
        (False, None, "none"),
        (True, None, "stub"),
        (True, "injected", "injected"),
    ],
)
def test_build_upload_queue_uses_the_selected_hub_and_resumes_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    dry_run: bool,
    injected_hub: object | None,
    expected_kind: str,
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings(repo_id="org/dataset")
    constructed: dict[str, Any] = {}
    executed: list[dict[str, Any]] = []

    class Queue:
        def __init__(self, *, upload: Any, max_pending: int, state_dir: Path) -> None:
            constructed.update(upload=upload, max_pending=max_pending, state_dir=state_dir)

        def resume_pending(self) -> int:
            return 2

    monkeypatch.setattr(sync_publication, "BackgroundUploadQueue", Queue)

    def execute(**kwargs: Any) -> None:
        executed.append(kwargs)

    monkeypatch.setattr(sync_publication, "execute_upload_job", execute)
    with caplog.at_level("INFO", logger="osm_polygon_wikidata_only.cli"):
        queue = sync_publication.build_upload_queue(
            push=True,
            dry_run=dry_run,
            settings=settings,
            data_root=data_root,
            num_threads=4,
            _hub=cast(Any, injected_hub),
        )

    assert isinstance(queue, Queue)
    assert constructed["max_pending"] == 2
    assert constructed["state_dir"] == data_root.cache / "sync_upload_jobs"
    assert "Resumed 2 pending background upload(s)" in caplog.messages
    constructed["upload"]([add_op("region.parquet", path_in_repo="region.parquet")], "commit")
    assert len(executed) == 1
    call = executed[0]
    assert call["data_root"] is data_root
    assert call["settings"] is settings
    assert call["message"] == "commit"
    assert call["num_threads"] == 4
    assert call["dry_run"] is dry_run
    if expected_kind == "injected":
        assert call["hub"] == "injected"
    elif expected_kind == "stub":
        assert isinstance(call["hub"], StubHfHub)
    else:
        assert call["hub"] is None


def test_build_upload_queue_skips_local_only_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sync_publication,
        "BackgroundUploadQueue",
        lambda **_kwargs: pytest.fail("local-only runs must not create an upload queue"),
    )

    assert (
        sync_publication.build_upload_queue(
            push=False,
            dry_run=False,
            settings=Settings(),
            data_root=DataRoot(tmp_path),
            num_threads=2,
        )
        is None
    )
