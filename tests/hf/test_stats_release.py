"""Contract tests for the polygon statistics release path.

The release must refuse an unconfirmed repository, publish exactly the card
and the report, verify the remote revision, and stay byte-stable across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.domain.schema import POLYGON_COLUMNS, empty_row, polygon_schema
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.stats_release import (
    RELEASE_ASSET_FILES,
    REMOTE_CARD_FILE,
    ReleasedFile,
    StatsReleaseError,
    _merge_release_card,
    _require_consistent_text_coverage,
    _stage_release_assets,
    default_remote_verifier,
    release_polygon_stats,
    release_v1_polygon_stats,
    release_v2_polygon_stats,
)
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID
from osm_polygon_wikidata_only.v2.storage import write_v2_region

_REPO = "NoeFlandre/osm-polygon-wikidata-only"
_COMMIT_HASH = "0123456789abcdef0123456789abcdef01234567"
_COMMIT_URL = f"https://huggingface.co/datasets/{_REPO}/commit/{_COMMIT_HASH}"


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
    assert "<summary>Polygon surface and geometry</summary>" in card
    assert "### Polygon surface and geometry" in card
    assert "[`stats.json`](stats.json)" in card


def test_v1_release_moves_the_polygon_surface_report_from_the_card_into_stats(
    tmp_path: Path,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    polygon_path = data_root.processed / "polygons" / "region-latest.parquet"
    row = empty_row(POLYGON_COLUMNS)
    row.update(
        {
            "polygon_id": "region:way:1",
            "osm_type": "way",
            "osm_id": 1,
            "source_pbf": "region-latest.osm.pbf",
            "lon": 0.0,
            "lat": 0.0,
            "area_m2": 2500.0,
            "area_km2": 0.0025,
            "bbox": json.dumps([0.0, 0.0, 1.0, 1.0]),
            "geometry": json.dumps(
                {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
                }
            ),
        }
    )
    polygon_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=polygon_schema()), polygon_path)
    manifest_path = data_root.processed / "manifests" / "processed_pbfs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "region-latest.osm.pbf": {
                    "polygons_path": "polygons/region-latest.parquet",
                    "polygon_count": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    report = release_v1_polygon_stats(data_root, confirm_repo=DEFAULT_REPO_ID)

    assert report.repo_id == DEFAULT_REPO_ID
    assert report.polygon_rows == 1
    staging = data_root.cache / "stats_release_snapshots" / "v1"
    card = (staging / "README.md").read_text(encoding="utf-8")
    payload = json.loads((staging / "stats.json").read_text(encoding="utf-8"))
    # The geometry histogram stays in the card but collapsed, so the public
    # card reads as a summary without losing any published figure.
    assert "<summary>Polygon surface and geometry</summary>" in card
    assert "### Polygon surface and geometry" in card
    assert "[`stats.json`](stats.json)" in card
    assert payload["area_m2"]["total"] == 2500.0


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


def test_apply_regenerates_data_sections_and_preserves_front_matter_and_prose(
    processed: Path, tmp_path: Path
) -> None:
    existing = (
        "---\n"
        "configs:\n"
        "  - config_name: polygons\n"
        "    data_files: []\n"
        "---\n\n"
        "# Existing card\n\n"
        "## Maintainer notes\n\nStale generated section.\n\n"
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
    # The Viewer configs block is owned by the publication and language-split
    # paths; a statistics release must never drop it.
    assert "configs:" in merged
    assert "config_name: polygons" in merged
    # Data sections are regenerated, and sections the generated card no longer
    # renders are dropped instead of surviving with stale numbers.
    assert "New statistics." in merged
    assert "Old statistics." not in merged
    assert "## Maintainer notes" not in merged
    assert "Stale generated section." not in merged
    # Author-owned prose is preserved.
    assert "## Citation" in merged
    assert "Keep this citation." in merged


def test_apply_orders_generated_sections_before_preserved_citation(
    processed: Path, tmp_path: Path
) -> None:
    existing = "# Existing card\n\n## Citation\n\nKeep this citation.\n"

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
    assert "Keep this citation." in merged


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


@pytest.mark.parametrize(
    ("revision", "paths_info_revision"),
    [
        pytest.param(_COMMIT_URL, _COMMIT_HASH, id="full-commit-url"),
        pytest.param(_COMMIT_HASH, _COMMIT_HASH, id="bare-commit-hash"),
    ],
)
def test_default_remote_verifier_normalizes_all_hub_revision_arguments(
    tmp_path: Path,
    revision: str,
    paths_info_revision: str,
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
            self.paths_info_revisions: list[str] = []
            self.download_revisions: list[str] = []

        def get_paths_info(
            self,
            repo_id: str,
            paths: list[str],
            *,
            revision: str,
            repo_type: str,
        ) -> list[object]:
            del repo_id, repo_type
            self.paths_info_revisions.append(revision)
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
            self.download_revisions.append(revision)
            return str(tmp_path / filename)

    remote_card = tmp_path / REMOTE_CARD_FILE
    remote_card.write_bytes(card.read_bytes())
    remote_stats = tmp_path / "stats.json"
    remote_stats.write_bytes(report.read_bytes())
    hub = FakeHub()

    assert default_remote_verifier(_REPO, files, revision=revision, hub=hub) == revision
    assert hub.paths_info_revisions == [paths_info_revision] * 2
    assert hub.download_revisions == [paths_info_revision] * 2


# ---------------------------------------------------------------------------
# Card consistency guard and asset staging
# ---------------------------------------------------------------------------


def _card_with(headline: str, continent_rows: str) -> str:
    return (
        "# Card\n\n"
        "## Dataset snapshot\n\n"
        "| Metric | Value |\n| --- | ---: |\n"
        f"| Polygons with successful non-empty text (unique OSM identities) | {headline} |\n\n"
        "## Geographic distribution by continent\n\n"
        "| Continent | Polygons | Wikipedia documents | Wikivoyage documents | "
        "Polygons with Wikipedia text | Polygons with Wikipedia or Wikivoyage text | "
        "Text coverage |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{continent_rows}\n"
        "## Citation\n\nCite me.\n"
    )


def test_consistency_guard_accepts_a_card_whose_continent_rows_sum_to_the_headline(
    tmp_path: Path,
) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        _card_with(
            "1,500",
            "| Europe | 10 | 10 | 0 | 900 | 900 | 50.0% |\n"
            "| Asia | 10 | 10 | 0 | 600 | 600 | 50.0% |",
        ),
        encoding="utf-8",
    )

    _require_consistent_text_coverage(card)


def test_consistency_guard_rejects_a_card_that_states_two_text_coverage_totals(
    tmp_path: Path,
) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        _card_with(
            "1,500",
            "| Europe | 10 | 10 | 0 | 900 | 900 | 50.0% |\n"
            "| Asia | 10 | 10 | 0 | 700 | 700 | 50.0% |",
        ),
        encoding="utf-8",
    )

    with pytest.raises(StatsReleaseError, match=r"1,500 but the continent table sums to 1,600"):
        _require_consistent_text_coverage(card)


def test_consistency_guard_ignores_a_card_without_a_continent_table(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "# Card\n\n## Dataset snapshot\n\n"
        "| Polygons with successful non-empty text (unique OSM identities) | 42 |\n",
        encoding="utf-8",
    )

    _require_consistent_text_coverage(card)


def test_consistency_guard_reads_the_v2_headline_bullet(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "# Card\n\n"
        "- **Polygons with non-empty Wikipedia or Wikivoyage text:** 30\n\n"
        "## Geographic distribution by continent\n\n"
        "| Continent | A | B | C | D | E | F |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        "| Europe | 1 | 1 | 1 | 1 | 31 | 9.9% |\n",
        encoding="utf-8",
    )

    with pytest.raises(StatsReleaseError, match=r"sums to 31"):
        _require_consistent_text_coverage(card)


def test_staging_rejects_an_asset_writer_that_omits_a_required_map(tmp_path: Path) -> None:
    def incomplete(destination: Path) -> dict[str, Path]:
        destination.mkdir(parents=True, exist_ok=True)
        only = destination / "coverage_map.png"
        only.write_bytes(b"png")
        return {RELEASE_ASSET_FILES[0]: only}

    with pytest.raises(StatsReleaseError, match="did not produce"):
        _stage_release_assets(tmp_path, incomplete)


def test_staging_rejects_an_asset_writer_that_returns_a_missing_file(tmp_path: Path) -> None:
    def absent(destination: Path) -> dict[str, Path]:
        return {
            path: destination / f"{index}.png" for index, path in enumerate(RELEASE_ASSET_FILES)
        }

    with pytest.raises(StatsReleaseError, match="was not rendered"):
        _stage_release_assets(tmp_path, absent)


def test_staging_without_an_asset_writer_releases_no_assets(tmp_path: Path) -> None:
    assert _stage_release_assets(tmp_path, None) == {}


def test_merge_preserves_front_matter_and_drops_sections_the_card_no_longer_renders() -> None:
    existing = (
        "---\nconfigs:\n  - config_name: kept\n---\n"
        "# Old\n\n## Dataset snapshot\n\nOLD\n\n## Gone\n\nstale\n\n## Citation\n\nCite me.\n"
    )
    generated = (
        "---\nconfigs:\n  - config_name: regenerated\n---\n# New\n\n## Dataset snapshot\n\nNEW\n"
    )

    merged = _merge_release_card(existing, generated)

    assert "config_name: kept" in merged
    assert "config_name: regenerated" not in merged
    assert "NEW" in merged
    assert "OLD" not in merged
    assert "## Gone" not in merged
    assert "Cite me." in merged


def test_merge_returns_the_generated_card_when_there_is_no_remote_card() -> None:
    assert _merge_release_card("", "# Only\n") == "# Only\n"


def test_merge_returns_the_generated_card_when_it_renders_no_sections() -> None:
    existing = "# Old\n\n## Dataset snapshot\n\nOLD\n"

    assert _merge_release_card(existing, "# New body only\n") == "# New body only\n"
