"""Contract tests for the polygon statistics release path.

The release must refuse an unconfirmed repository, publish exactly the card
and the report, verify the remote revision, and stay byte-stable across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.stats_release import (
    REMOTE_CARD_FILE,
    ReleasedFile,
    StatsReleaseError,
    default_remote_verifier,
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
        polygons=[{"polygon_id": "p1", "osm_type": "way", "osm_id": 1, "has_wikidata": False}],
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
    seen: list[tuple[str, tuple[ReleasedFile, ...], str]] = []

    def verifier(
        repo_id: str,
        files: tuple[ReleasedFile, ...],
        *,
        revision: str,
    ) -> str:
        seen.append((repo_id, files, revision))
        return revision

    report = _release(processed, tmp_path / "staging", apply=True, hub=hub, verifier=verifier)

    assert report.published is True
    assert report.revision == hub.commits[0]["commit_id"]
    assert len(hub.commits) == 1
    assert hub.commits[0]["paths"] == [REMOTE_CARD_FILE, "stats.json"]
    assert hub.commits[0]["repo_id"] == _REPO
    assert seen == [(_REPO, report.files, report.revision)]


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
        polygons=[{"polygon_id": "p1", "osm_type": "way", "osm_id": 1, "has_wikidata": False}],
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


def test_apply_is_idempotent_and_makes_no_second_commit(processed: Path, tmp_path: Path) -> None:
    hub = StubHfHub()

    first = _release(processed, tmp_path / "staging", apply=True, hub=hub)
    second = _release(processed, tmp_path / "staging", apply=True, hub=hub)

    assert first.committed is True
    assert second.committed is False
    assert second.no_op is True
    assert second.revision == first.revision
    assert len(hub.commits) == 1


def test_apply_verification_is_bound_to_upload_returned_revision(
    processed: Path, tmp_path: Path
) -> None:
    hub = StubHfHub()
    seen: list[str] = []

    def verifier(
        repo_id: str,
        files: tuple[ReleasedFile, ...],
        *,
        revision: str,
    ) -> str:
        del repo_id, files
        seen.append(revision)
        return revision

    report = _release(
        processed,
        tmp_path / "staging",
        apply=True,
        hub=hub,
        verifier=verifier,
    )

    assert seen == [report.revision]
    assert report.revision == hub.commits[0]["commit_id"]


def test_apply_merges_stats_into_existing_card_without_dropping_sections(
    processed: Path, tmp_path: Path
) -> None:
    existing = (
        "---\n"
        "configs:\n"
        "  - config_name: polygons\n"
        "    data_files: []\n"
        "---\n\n"
        "# Existing card\n\n"
        "## Maintainer notes\n\nKeep this section.\n\n"
        "## Polygon surface and geometry\n\nOld statistics.\n\n"
        "## Citation\n\nKeep this citation.\n"
    )

    def write_generated_card(destination: Path) -> None:
        destination.write_text(
            "# Generated card\n\n## Polygon surface and geometry\n\nNew statistics.\n",
            encoding="utf-8",
        )

    hub = StubHfHub(remote_content={REMOTE_CARD_FILE: existing.encode("utf-8")})
    release_polygon_stats(
        processed_dir=processed,
        staging_dir=tmp_path / "staging",
        repo_id=_REPO,
        confirm_repo=_REPO,
        card_writer=write_generated_card,
        apply=True,
        hub=hub,
    )

    merged = hub.remote_content[REMOTE_CARD_FILE].decode("utf-8")
    assert "configs:" in merged
    assert "## Maintainer notes" in merged
    assert "Keep this section." in merged
    assert "New statistics." in merged
    assert "Old statistics." not in merged
    assert "## Citation" in merged
    assert "Keep this citation." in merged


def test_apply_inserts_new_release_sections_before_existing_citation(
    processed: Path, tmp_path: Path
) -> None:
    existing = "# Existing card\n\n## Maintainer notes\n\nKeep this section.\n\n## Citation\n"

    def write_generated_card(destination: Path) -> None:
        destination.write_text(
            "# Generated card\n\n"
            "## Polygon surface and geometry\n\nNew geometry.\n\n"
            "## Snapshot\n\nNew snapshot.\n",
            encoding="utf-8",
        )

    hub = StubHfHub(remote_content={REMOTE_CARD_FILE: existing.encode("utf-8")})
    release_polygon_stats(
        processed_dir=processed,
        staging_dir=tmp_path / "staging",
        repo_id=_REPO,
        confirm_repo=_REPO,
        card_writer=write_generated_card,
        apply=True,
        hub=hub,
    )

    merged = hub.remote_content[REMOTE_CARD_FILE].decode("utf-8")
    assert merged.index("New geometry.") < merged.index("New snapshot.")
    assert merged.index("New snapshot.") < merged.index("## Citation")
    assert "Keep this section." in merged


def test_release_report_contains_complete_manifest_inventory_and_provenance(
    processed: Path, tmp_path: Path
) -> None:
    report = _release(
        processed,
        tmp_path / "staging",
        source_revision="osm-source@abc123",
        data_revision="processed-data@def456",
    )

    payload = report.to_payload()
    provenance = payload["provenance"]
    assert provenance["source_revision"] == "osm-source@abc123"
    assert provenance["data_revision"] == "processed-data@def456"
    assert provenance["manifest"]["path"] == "manifests/processed_pbfs.json"
    assert provenance["manifest"]["sha256"]
    assert provenance["polygons"] == [
        {
            "path": "polygons/region-latest.parquet",
            "sha256": provenance["polygons"][0]["sha256"],
            "size_bytes": provenance["polygons"][0]["size_bytes"],
            "source_pbf": "region-latest.osm.pbf",
            "row_count": 1,
        }
    ]

    stats_payload = json.loads((tmp_path / "staging" / "stats.json").read_text())
    assert stats_payload["provenance"] == provenance


def test_release_rejects_polygon_files_absent_from_manifest(
    processed: Path, tmp_path: Path
) -> None:
    (processed / "polygons" / "unlisted.parquet").write_bytes(b"not a table")

    with pytest.raises(StatsReleaseError, match="absent from the manifest"):
        _release(processed, tmp_path / "staging")


def test_v2_release_rejects_noncanonical_repo_id(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    write_v2_region(
        data_root.processed_v2,
        "region-latest",
        polygons=[{"polygon_id": "p1", "osm_type": "way", "osm_id": 1, "has_wikidata": False}],
        documents=[],
        links=[],
    )

    with pytest.raises(StatsReleaseError, match="canonical"):
        release_v2_polygon_stats(
            data_root,
            repo_id=DEFAULT_REPO_ID,
            confirm_repo=DEFAULT_REPO_ID,
        )


def test_default_remote_verifier_uses_supplied_upload_revision_and_hashes_files(
    tmp_path: Path,
) -> None:
    card = tmp_path / "README.md"
    report = tmp_path / "stats.json"
    card.write_bytes(b"card")
    report.write_bytes(b"report")
    files = (
        ReleasedFile(REMOTE_CARD_FILE, hashlib.sha256(b"card").hexdigest(), 4),
        ReleasedFile("stats.json", hashlib.sha256(b"report").hexdigest(), 6),
    )

    class FakeHub:
        def __init__(self) -> None:
            self.revisions: list[str | None] = []

        def get_paths_info(
            self,
            repo_id: str,
            paths: list[str],
            *,
            revision: str,
            repo_type: str,
        ) -> list[object]:
            del repo_id, repo_type
            self.revisions.append(revision)
            return [
                SimpleNamespace(path=path, size=(4 if path == "README.md" else 6)) for path in paths
            ]

        def hf_hub_download(
            self,
            repo_id: str,
            filename: str,
            *,
            revision: str,
            repo_type: str,
        ) -> str:
            del repo_id, repo_type
            self.revisions.append(revision)
            return str(tmp_path / filename)

    remote_card = tmp_path / REMOTE_CARD_FILE
    remote_card.write_bytes(card.read_bytes())
    remote_stats = tmp_path / "stats.json"
    remote_stats.write_bytes(report.read_bytes())
    hub = FakeHub()

    assert (
        default_remote_verifier(_REPO, files, revision="upload-commit", hub=hub) == "upload-commit"
    )
    assert hub.revisions == ["upload-commit"] * 4
