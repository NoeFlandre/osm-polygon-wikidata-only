"""Step definitions for ``publish_dry_run.feature`` (#15, #21, #115)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, scenarios, then, when

from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.stats_release import (
    REMOTE_CARD_FILE,
    ReleasedFile,
    StatsReleaseReport,
    default_remote_verifier,
    release_polygon_stats,
)
from osm_polygon_wikidata_only.v2.storage import write_v2_region

scenarios("publish_dry_run.feature")

_REPO = "NoeFlandre/osm-polygon-wikidata-only"
_COMMIT_HASH = "0123456789abcdef0123456789abcdef01234567"
_COMMIT_URL = f"https://huggingface.co/datasets/{_REPO}/commit/{_COMMIT_HASH}"
_PRIOR_CARD = b"# Card\n\n## Citation\n\nKeep this citation.\n"


class _RecordingHub:
    """Fake Hub that records the revision every file API receives."""

    def __init__(self, contents: dict[str, bytes], cache: Path) -> None:
        self.contents = contents
        self.cache = cache
        self.paths_info_revisions: list[str] = []
        self.download_revisions: list[str] = []

    def get_paths_info(
        self, repo_id: str, paths: list[str], *, revision: str, repo_type: str
    ) -> list[object]:
        del repo_id, repo_type
        self.paths_info_revisions.append(revision)
        return [SimpleNamespace(path=path, size=len(self.contents[path])) for path in paths]

    def hf_hub_download(self, repo_id: str, filename: str, *, revision: str, repo_type: str) -> str:
        del repo_id, repo_type
        self.download_revisions.append(revision)
        target = self.cache / filename
        target.write_bytes(self.contents[filename])
        return str(target)


@dataclass
class _State:
    processed: Path | None = None
    staging: Path | None = None
    hub: StubHfHub | None = None
    reports: list[StatsReleaseReport] = field(default_factory=list)
    files: tuple[ReleasedFile, ...] = ()
    recording_hub: _RecordingHub | None = None
    verified_revision: str | None = None


def _write_card(destination: Path) -> None:
    destination.write_text("# Card\n", encoding="utf-8")


def _release(state: _State, *, apply: bool) -> StatsReleaseReport:
    assert state.processed is not None and state.staging is not None
    report = release_polygon_stats(
        processed_dir=state.processed,
        staging_dir=state.staging,
        repo_id=_REPO,
        confirm_repo=_REPO,
        card_writer=_write_card,
        apply=apply,
        hub=state.hub,
        verifier=(lambda repo_id, files, *, revision: revision) if apply else None,
    )
    state.reports.append(report)
    return report


@pytest.fixture
def state() -> _State:
    return _State()


@given("a staged V2 release and a fake Hub with a prior card snapshot")
def staged_release(state: _State, tmp_path: Path) -> None:
    processed = tmp_path / "processed_v2"
    write_v2_region(
        processed,
        "region-latest",
        polygons=[{"polygon_id": "p1", "osm_type": "way", "osm_id": 1, "has_wikidata": False}],
        documents=[],
        links=[],
    )
    state.processed = processed
    state.staging = tmp_path / "staging"
    state.hub = StubHfHub(
        remote_files={REMOTE_CARD_FILE},
        remote_content={REMOTE_CARD_FILE: _PRIOR_CARD},
    )


@given("the release was already published to the fake Hub")
def already_published(state: _State) -> None:
    report = _release(state, apply=True)
    assert report.committed is True


@when("I publish the statistics with dry-run")
def publish_dry_run(state: _State) -> None:
    _release(state, apply=False)


@when("I publish the statistics again")
def publish_again(state: _State) -> None:
    _release(state, apply=True)


@then("the planned operations list only the card and the statistics report")
def planned_files(state: _State) -> None:
    report = state.reports[-1]
    assert report.published is False
    assert report.revision is None
    assert [item.path_in_repo for item in report.files] == [REMOTE_CARD_FILE, "stats.json"]


@then("nothing is uploaded or committed to the Hub")
def nothing_uploaded(state: _State) -> None:
    assert state.hub is not None
    assert state.hub.commits == []
    assert state.hub.uploads == []
    assert state.hub.remote_content == {REMOTE_CARD_FILE: _PRIOR_CARD}


@then("no file is reported as changed and no second commit is made")
def no_second_commit(state: _State) -> None:
    assert state.hub is not None
    first, second = state.reports
    assert second.no_op is True
    assert second.committed is False
    assert second.changed_files == ()
    assert second.revision == first.revision
    assert len(state.hub.commits) == 1


@given("released files already present on a fake Hub")
def released_files(state: _State, tmp_path: Path) -> None:
    contents = {REMOTE_CARD_FILE: b"card", "stats.json": b"report"}
    state.files = tuple(
        ReleasedFile(path, hashlib.sha256(data).hexdigest(), len(data))
        for path, data in contents.items()
    )
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    state.recording_hub = _RecordingHub(contents, cache)


@when("I verify the release with a full commit URL revision")
def verify_with_url(state: _State) -> None:
    state.verified_revision = default_remote_verifier(
        _REPO, state.files, revision=_COMMIT_URL, hub=state.recording_hub
    )


@then("every Hub file API receives the bare commit hash")
def bare_hash(state: _State) -> None:
    hub = state.recording_hub
    assert hub is not None
    assert state.verified_revision == _COMMIT_URL
    assert hub.paths_info_revisions
    assert hub.download_revisions
    assert set(hub.paths_info_revisions) == {_COMMIT_HASH}
    assert set(hub.download_revisions) == {_COMMIT_HASH}
