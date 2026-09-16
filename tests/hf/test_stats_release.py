"""Contract tests for the polygon statistics release path.

The release must refuse an unconfirmed repository, publish exactly the card
and the report, verify the remote revision, and stay byte-stable across runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.stats_release import (
    REMOTE_CARD_FILE,
    ReleasedFile,
    StatsReleaseError,
    release_polygon_stats,
    release_v1_polygon_stats,
    release_v2_polygon_stats,
)
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID
from osm_polygon_wikidata_only.v2.storage import write_v2_region

_REPO = "NoeFlandre/osm-polygon-wikidata-only"


def _write_card(destination: Path) -> None:
    destination.write_text("# Card\n", encoding="utf-8")


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    root = tmp_path / "processed_v2"
    write_v2_region(
        root,
        "region-latest",
        polygons=[{"polygon_id": "p1", "has_wikidata": False}],
        documents=[],
        links=[],
    )
    return root


def _release(processed: Path, staging: Path, **kwargs: object):
    return release_polygon_stats(
        processed_dir=processed,
        staging_dir=staging,
        repo_id=_REPO,
        confirm_repo=str(kwargs.pop("confirm_repo", _REPO)),
        card_writer=_write_card,
        **kwargs,  # type: ignore[arg-type]
    )


def test_dry_run_plans_the_card_and_report_without_uploading(
    processed: Path, tmp_path: Path
) -> None:
    hub = StubHfHub()

    report = _release(processed, tmp_path / "staging", hub=hub)

    assert report.repo_id == _REPO
    assert report.published is False
    assert report.revision is None
    assert report.polygon_rows == 1
    assert report.polygon_files == 1
    assert [item.path_in_repo for item in report.files] == [REMOTE_CARD_FILE, "stats.json"]
    assert hub.commits == []


def test_apply_publishes_only_the_card_and_report_then_verifies(
    processed: Path, tmp_path: Path
) -> None:
    hub = StubHfHub()
    seen: list[tuple[str, tuple[ReleasedFile, ...]]] = []

    def verifier(repo_id: str, files: tuple[ReleasedFile, ...]) -> str:
        seen.append((repo_id, files))
        return "c0ffee"

    report = _release(processed, tmp_path / "staging", apply=True, hub=hub, verifier=verifier)

    assert report.published is True
    assert report.revision == "c0ffee"
    assert len(hub.commits) == 1
    assert hub.commits[0]["paths"] == [REMOTE_CARD_FILE, "stats.json"]
    assert hub.commits[0]["repo_id"] == _REPO
    assert seen == [(_REPO, report.files)]


def test_second_release_is_a_byte_stable_no_op(processed: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    first = _release(processed, staging)
    report_path = staging / "stats.json"
    before = report_path.read_bytes()

    second = _release(processed, staging)

    assert second.files == first.files
    assert report_path.read_bytes() == before


def test_wrong_repository_confirmation_is_refused(processed: Path, tmp_path: Path) -> None:
    with pytest.raises(StatsReleaseError, match="repository confirmation"):
        _release(processed, tmp_path / "staging", confirm_repo="someone-else/dataset")


def test_a_tree_without_published_polygons_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StatsReleaseError, match="no published polygon files"):
        _release(tmp_path / "empty", tmp_path / "staging")


def test_empty_remote_revision_is_refused(processed: Path, tmp_path: Path) -> None:
    with pytest.raises(StatsReleaseError, match="empty revision"):
        _release(
            processed,
            tmp_path / "staging",
            apply=True,
            hub=StubHfHub(),
            verifier=lambda repo_id, files: "",
        )


def test_v2_release_targets_the_wikidata_and_wikipedia_dataset(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    write_v2_region(
        data_root.processed_v2,
        "region-latest",
        polygons=[{"polygon_id": "p1", "has_wikidata": False}],
        documents=[],
        links=[],
    )

    report = release_v2_polygon_stats(data_root, confirm_repo=V2_REPO_ID)

    assert report.repo_id == V2_REPO_ID
    assert report.polygon_rows == 1
    card = (data_root.cache / "stats_release_snapshots" / "v2" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "## Polygon surface and geometry" in card
    assert "[`stats.json`](stats.json)" in card


def test_v1_release_refuses_an_unconfirmed_repository(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    with pytest.raises(StatsReleaseError, match="repository confirmation"):
        release_v1_polygon_stats(data_root, confirm_repo=V2_REPO_ID)

    assert DEFAULT_REPO_ID == _REPO
