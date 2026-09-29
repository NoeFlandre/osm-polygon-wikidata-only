"""Step definitions for ``publish_dry_run.feature`` (#15, #21, #115)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pytest_bdd import given, scenarios, then, when

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.stats_release import (
    RELEASE_ASSET_FILES,
    REMOTE_CARD_FILE,
    ReleasedFile,
    StatsReleaseReport,
    default_remote_verifier,
    release_v2_polygon_stats,
)
from osm_polygon_wikidata_only.v2 import maps as v2_maps
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID
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
    data_root: DataRoot | None = None
    hub: StubHfHub | None = None
    reports: list[StatsReleaseReport] = field(default_factory=list)
    files: tuple[ReleasedFile, ...] = ()
    recording_hub: _RecordingHub | None = None
    verified_revision: str | None = None


def _release(state: _State, *, apply: bool) -> StatsReleaseReport:
    assert state.data_root is not None
    assert state.hub is not None
    report = release_v2_polygon_stats(
        state.data_root,
        confirm_repo=V2_REPO_ID,
        apply=apply,
        hub=state.hub,
        verifier=(lambda repo_id, files, *, revision: revision) if apply else None,
        generated_on="2026-09-01",
    )
    state.reports.append(report)
    return report


@pytest.fixture
def state() -> _State:
    return _State()


@given("a staged V2 release and a fake Hub with a prior card snapshot")
def staged_release(state: _State, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = DataRoot(tmp_path / "data")
    data_root.ensure()
    write_v2_region(
        data_root.processed_v2,
        "region-latest",
        polygons=[
            {
                "polygon_id": "p1",
                "osm_type": "way",
                "osm_id": 1,
                "has_wikidata": False,
                "lon": 2.35,
                "lat": 48.86,
            }
        ],
        documents=[],
        links=[],
    )

    def write_map_fixtures(
        processed_v2: Path,
        output_dir: Path,
        **_kwargs: object,
    ) -> tuple[Path, Path, Path]:
        assert processed_v2 == data_root.processed_v2
        assets = (
            output_dir / "coverage_map.png",
            output_dir / "geographic_text_presence.png",
            output_dir / "geographic_text_density.png",
        )
        for asset in assets:
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(b"deterministic V2 map fixture")
        return assets

    monkeypatch.setattr(v2_maps, "generate_v2_map_assets", write_map_fixtures)
    state.data_root = data_root
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


@then("the planned operations include the V2 card, report, and map assets")
def planned_files(state: _State) -> None:
    report = state.reports[-1]
    assert report.published is False
    assert report.revision is None
    assert report.repo_id == V2_REPO_ID
    assert {item.path_in_repo for item in report.files} == {
        REMOTE_CARD_FILE,
        "stats.json",
        *RELEASE_ASSET_FILES,
    }

    assert state.data_root is not None
    snapshot = state.data_root.cache / "stats_release_snapshots" / "v2"
    card = (snapshot / REMOTE_CARD_FILE).read_text(encoding="utf-8")
    assert "## Polygon area and geometry" in card
    assert "[`stats.json`](stats.json)" in card
    for path in RELEASE_ASSET_FILES:
        asset = snapshot / path
        assert asset.is_file()
        assert asset.stat().st_size > 0

    payload = json.loads((snapshot / "stats.json").read_text(encoding="utf-8"))
    assert payload["card_contract"] == "minimal-v2"
    assert "v2_card_stats" in payload


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
        _REPO,
        state.files,
        revision=_COMMIT_URL,
        # This spy intentionally implements only the verifier's read methods.
        hub=cast(HfHub, state.recording_hub),
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
