"""The uploader must honor explicit queue snapshots and preserve other ops."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.uploader import upload_files


def test_upload_files_fails_when_explicit_snapshot_is_missing(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"LIVE-VERSION")
    missing_snapshot = tmp_path / "snapshot.parquet"
    hub = StubHfHub()

    with pytest.raises(UploadError, match=r"[Ss]napshot.*does not exist"):
        upload_files(
            "org/dataset",
            ops=[
                PublicationOp(
                    action="add",
                    path_in_repo="polygons/region.parquet",
                    local_path=canonical,
                    snapshot_path=missing_snapshot,
                )
            ],
            hub=hub,
            token="test-token",
            commit_message="publish snapshot",
        )

    assert hub.remote_content == {}
    assert hub.commits == []


def test_nonqueued_add_without_snapshot_still_uploads_local_file(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"DIRECT-CALL")
    hub = StubHfHub()

    upload_files(
        "org/dataset",
        ops=[add_op(canonical, path_in_repo="polygons/region.parquet")],
        hub=hub,
        token="test-token",
        commit_message="direct publication",
    )

    assert hub.remote_content == {"polygons/region.parquet": b"DIRECT-CALL"}


def test_delete_operation_behavior_is_unchanged() -> None:
    hub = StubHfHub(
        remote_files={"legacy/region.parquet"},
        remote_content={"legacy/region.parquet": b"OLD"},
    )

    upload_files(
        "org/dataset",
        ops=[delete_op("legacy/region.parquet")],
        hub=hub,
        token="test-token",
        commit_message="retire legacy path",
    )

    assert hub.remote_files == set()
    assert hub.remote_content == {}
