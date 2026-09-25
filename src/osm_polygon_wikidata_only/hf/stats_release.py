"""Deterministic polygon-statistics release for one published dataset.

The release path recomputes ``stats.json`` from every manifest-listed polygon
row, records the complete local inventory, and updates only release-owned
statistics sections in the dataset card. A real apply uses one Hub client for
the remote comparison, upload, and revision-bound verification.

Both published contracts use it:

* V1 -- ``NoeFlandre/osm-polygon-wikidata-only`` from ``<data-root>/processed``;
* V2 -- ``NoeFlandre/osm-polygon-wikidata-and-wikipedia`` from
  ``<data-root>/processed_v2``.

No clock or network is consulted for a dry run. A date is emitted only when
the caller supplies pinned ``generated_on`` metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.hf._polygon_geometry.validation import PolygonStatsInputError
from osm_polygon_wikidata_only.hf._stats_release.card_merge import (
    merge_release_card as _merge_release_card,
)
from osm_polygon_wikidata_only.hf._stats_release.manifest import (
    build_provenance as _build_provenance,
)
from osm_polygon_wikidata_only.hf._stats_release.models import (
    ReleasedFile,
    StatsReleaseError,
)
from osm_polygon_wikidata_only.hf._stats_release.models import (
    RemoteState as _RemoteState,
)
from osm_polygon_wikidata_only.hf._stats_release.remote import (
    RemoteVerifier,
    default_remote_verifier,
)
from osm_polygon_wikidata_only.hf._stats_release.remote import (
    changed_paths as _changed_paths,
)
from osm_polygon_wikidata_only.hf._stats_release.remote import (
    client_for_release as _client_for_release,
)
from osm_polygon_wikidata_only.hf._stats_release.remote import (
    load_remote_state as _load_remote_state,
)
from osm_polygon_wikidata_only.hf._stats_release.remote import (
    verify_uploaded_files as _verify_uploaded_files,
)
from osm_polygon_wikidata_only.hf._stats_release.text_coverage import (
    require_consistent_text_coverage as _require_consistent_text_coverage,
)
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import (
    PolygonGeometryStats,
    load_polygon_geometry_stats,
    stats_payload,
)
from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
    REMOTE_POLYGON_STATS_FILE,
)
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.atomic import atomic_write_json, atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID

if TYPE_CHECKING:
    from osm_polygon_wikidata_only.hf.publication import MinimalV1ReleaseSnapshot
    from osm_polygon_wikidata_only.v2.card_release import MinimalV2ReleaseSnapshot

REMOTE_CARD_FILE = "README.md"
RELEASE_COMMIT_MESSAGE = "Publish dataset card and polygon statistics report"
_STAGING_DIRNAME = "stats_release_snapshots"
_REMOTE_CACHE_DIRNAME = "remote_verification"
RELEASE_ASSET_FILES = (
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
)
CardWriter = Callable[[Path], None]
# Renders the public coverage assets into a staging directory and returns the
# mapping of remote ``assets/*.png`` path to the freshly rendered local file.
AssetWriter = Callable[[Path], Mapping[str, Path]]
ReportExtraBuilder = Callable[[PolygonGeometryStats], Mapping[str, Any]]


@dataclass(frozen=True)
class StatsReleaseReport:
    """Evidence for one statistics release run."""

    repo_id: str
    processed_dir: str
    files: tuple[ReleasedFile, ...]
    polygon_files: int
    polygon_rows: int
    published: bool
    revision: str | None
    provenance: dict[str, Any] = field(default_factory=dict)
    committed: bool = False
    no_op: bool = False
    changed_files: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "changed_files": list(self.changed_files),
            "committed": self.committed,
            "files": [
                {
                    "path_in_repo": item.path_in_repo,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            ],
            "no_op": self.no_op,
            "polygon_files": self.polygon_files,
            "polygon_rows": self.polygon_rows,
            "processed_dir": self.processed_dir,
            "provenance": self.provenance,
            "published": self.published,
            "repo_id": self.repo_id,
            "revision": self.revision,
        }


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    atomic_write_text(path, text)


def _stage_report(
    stats: PolygonGeometryStats,
    destination: Path,
    provenance: dict[str, Any],
    report_extra_builder: ReportExtraBuilder | None = None,
) -> None:
    payload = stats_payload(stats)
    if report_extra_builder is not None:
        payload.update(report_extra_builder(stats))
    payload["provenance"] = provenance
    atomic_write_json(destination, payload)


def _released_file(path: Path, path_in_repo: str) -> ReleasedFile:
    return ReleasedFile(
        path_in_repo=path_in_repo,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _require_exact_repo(confirm_repo: str, repo_id: str) -> None:
    if confirm_repo != repo_id:
        raise StatsReleaseError(
            f"repository confirmation must equal {repo_id!r} (got {confirm_repo!r})"
        )


def _require_canonical_repo(repo_id: str, expected: str) -> None:
    if repo_id != expected:
        raise StatsReleaseError(
            f"release target must use canonical repository {expected!r} (got {repo_id!r})"
        )


def _require_published_polygons(processed_dir: Path) -> None:
    polygons = processed_dir / "polygons"
    if not polygons.is_dir() or not any(polygons.glob("*.parquet")):
        raise StatsReleaseError(f"no published polygon files under {polygons}")


def release_polygon_stats(
    *,
    processed_dir: Path,
    staging_dir: Path,
    repo_id: str,
    confirm_repo: str,
    card_writer: CardWriter,
    asset_writer: AssetWriter | None = None,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
    token: str | None = None,
    source_revision: str | None = None,
    data_revision: str | None = None,
    report_extra_builder: ReportExtraBuilder | None = None,
) -> StatsReleaseReport:
    """Recompute, publish, and verify the card, report, and coverage assets."""
    _require_exact_repo(confirm_repo, repo_id)
    _require_published_polygons(processed_dir)
    provenance = _build_provenance(
        processed_dir,
        source_revision=source_revision,
        data_revision=data_revision,
    )
    stats = _load_release_stats(processed_dir)
    client, remote = _release_remote_state(
        apply,
        hub=hub,
        token=token,
        repo_id=repo_id,
        staging_dir=staging_dir,
        asset_writer=asset_writer,
    )
    files, local_paths = _prepare_release_files(
        stats,
        provenance,
        staging_dir=staging_dir,
        card_writer=card_writer,
        asset_writer=asset_writer,
        remote=remote,
        report_extra_builder=report_extra_builder,
    )
    if not apply:
        return _release_report(
            repo_id=repo_id,
            processed_dir=processed_dir,
            stats=stats,
            files=files,
            provenance=provenance,
            published=False,
        )
    revision, committed, no_op, changed_files = _apply_release(
        repo_id,
        files,
        remote=remote,
        client=client,
        token=token,
        verifier=verifier,
        staging_dir=staging_dir,
        local_paths=local_paths,
    )
    return _release_report(
        repo_id=repo_id,
        processed_dir=processed_dir,
        stats=stats,
        files=files,
        provenance=provenance,
        published=True,
        revision=revision,
        committed=committed,
        no_op=no_op,
        changed_files=changed_files,
    )


def _load_release_stats(processed_dir: Path) -> PolygonGeometryStats:
    try:
        return load_polygon_geometry_stats(processed_dir)
    except (OSError, ValueError, PolygonStatsInputError) as error:
        raise StatsReleaseError(str(error)) from error


def _release_remote_state(
    apply: bool,
    *,
    hub: HfHub | None,
    token: str | None,
    repo_id: str,
    staging_dir: Path,
    asset_writer: AssetWriter | None = None,
) -> tuple[HfHub | None, _RemoteState]:
    if not apply:
        return None, _RemoteState(None, {})
    client = _client_for_release(hub, token)
    tracked = (REMOTE_CARD_FILE, REMOTE_POLYGON_STATS_FILE)
    if asset_writer is not None:
        tracked = (*tracked, *RELEASE_ASSET_FILES)
    remote = _load_remote_state(
        client,
        repo_id,
        tracked,
        cache_dir=staging_dir / _REMOTE_CACHE_DIRNAME,
    )
    return client, remote


def _prepare_release_files(
    stats: PolygonGeometryStats,
    provenance: dict[str, Any],
    *,
    staging_dir: Path,
    card_writer: CardWriter,
    asset_writer: AssetWriter | None,
    remote: _RemoteState,
    report_extra_builder: ReportExtraBuilder | None,
) -> tuple[tuple[ReleasedFile, ...], dict[str, Path]]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    report_path = staging_dir / "stats.json"
    card_path = staging_dir / "README.md"
    _stage_report(stats, report_path, provenance, report_extra_builder)
    card_writer(card_path)
    _merge_remote_card(card_path, remote)
    local_paths = {
        REMOTE_CARD_FILE: card_path,
        REMOTE_POLYGON_STATS_FILE: report_path,
    }
    local_paths.update(_stage_release_assets(staging_dir, asset_writer))
    _require_consistent_text_coverage(card_path)
    return (
        tuple(_released_file(local_paths[path], path) for path in local_paths),
        local_paths,
    )


def _stage_release_assets(
    staging_dir: Path,
    asset_writer: AssetWriter | None,
) -> dict[str, Path]:
    """Render the public coverage assets that accompany this card.

    The card numbers and the map captions are produced from the same local
    Parquet tables in the same run, so a release can never ship a refreshed
    statistic beside a map rendered from an older snapshot.
    """
    if asset_writer is None:
        return {}
    rendered = dict(asset_writer(staging_dir / "assets"))
    _require_complete_assets(rendered)
    return rendered


def _require_complete_assets(rendered: Mapping[str, Path]) -> None:
    _require_every_asset_planned(rendered)
    _require_every_asset_written(rendered)


def _require_every_asset_planned(rendered: Mapping[str, Path]) -> None:
    missing = sorted(path for path in RELEASE_ASSET_FILES if path not in rendered)
    if missing:
        raise StatsReleaseError("coverage asset writer did not produce: " + ", ".join(missing))


def _require_every_asset_written(rendered: Mapping[str, Path]) -> None:
    unrendered = sorted(path for path, local in rendered.items() if not local.is_file())
    if unrendered:
        raise StatsReleaseError(f"coverage asset {unrendered[0]} was not rendered")


def _merge_remote_card(card_path: Path, remote: _RemoteState) -> None:
    if REMOTE_CARD_FILE not in remote.contents:
        return
    try:
        existing = remote.contents[REMOTE_CARD_FILE].decode("utf-8")
        generated = card_path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise StatsReleaseError(f"dataset card is not valid UTF-8: {error}") from error
    _write_text_if_changed(card_path, _merge_release_card(existing, generated))


def _apply_release(
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    *,
    remote: _RemoteState,
    client: HfHub | None,
    token: str | None,
    verifier: RemoteVerifier | None,
    staging_dir: Path,
    local_paths: Mapping[str, Path],
) -> tuple[str | None, bool, bool, tuple[str, ...]]:
    changed_files = _changed_paths(remote, files, local_paths)
    if remote.revision and not changed_files:
        return remote.revision, False, True, changed_files
    ops: list[PublicationOp] = [
        add_op(local_paths[path], path_in_repo=path) for path in changed_files
    ]
    upload_revision = upload_files(
        repo_id,
        ops=ops,
        hub=client,
        token=token,
        commit_message=RELEASE_COMMIT_MESSAGE,
    )
    if not upload_revision:
        raise StatsReleaseError("Hub upload returned an empty revision")
    revision = _verify_uploaded_files(
        repo_id,
        files,
        upload_revision=upload_revision,
        verifier=verifier,
        client=client,
        token=token,
        cache_dir=staging_dir / _REMOTE_CACHE_DIRNAME,
    )
    return revision, True, False, changed_files


def _release_report(
    *,
    repo_id: str,
    processed_dir: Path,
    stats: PolygonGeometryStats,
    files: tuple[ReleasedFile, ...],
    provenance: dict[str, Any],
    published: bool,
    revision: str | None = None,
    committed: bool = False,
    no_op: bool = False,
    changed_files: tuple[str, ...] = (),
) -> StatsReleaseReport:
    return StatsReleaseReport(
        repo_id=repo_id,
        processed_dir=str(processed_dir),
        files=files,
        polygon_files=stats.file_count,
        polygon_rows=stats.polygon_count,
        published=published,
        revision=revision,
        provenance=provenance,
        committed=committed,
        no_op=no_op,
        changed_files=changed_files,
    )


def release_v1_polygon_stats(
    data_root: DataRoot,
    *,
    confirm_repo: str,
    repo_id: str = DEFAULT_REPO_ID,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
    token: str | None = None,
    source_revision: str | None = None,
    data_revision: str | None = None,
    generated_on: str | None = None,
) -> StatsReleaseReport:
    """Release the V1 Wikidata-only card and statistics report."""
    _require_canonical_repo(repo_id, DEFAULT_REPO_ID)
    # Release-only helpers stay lazy so normal sync imports avoid card/map scanning.
    from osm_polygon_wikidata_only.hf.publication import (  # noqa: PLC0415
        build_minimal_v1_release_snapshot,
        refresh_coverage_assets,
    )

    prepared: MinimalV1ReleaseSnapshot | None = None

    def get_prepared() -> MinimalV1ReleaseSnapshot:
        nonlocal prepared
        if prepared is None:
            prepared = build_minimal_v1_release_snapshot(
                data_root,
                repo_id,
                generated_on=generated_on,
            )
        return prepared

    def write_card(destination: Path) -> None:
        # Rendering is performed only when the staged card is materialized.
        from osm_polygon_wikidata_only.hf.minimal_card import (  # noqa: PLC0415
            render_minimal_card,
        )

        _write_text_if_changed(destination, render_minimal_card(get_prepared().card))

    def write_assets(destination: Path) -> Mapping[str, Path]:
        snapshot = get_prepared().text_presence
        coverage, presence, density = refresh_coverage_assets(
            data_root=data_root,
            snapshot_stem="release",
            snapshots_dir=destination,
            world_land_warning=None,
            text_snapshot=snapshot,
        )
        return {
            REMOTE_COVERAGE_MAP_FILE: coverage,
            REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE: presence,
            REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE: density,
        }

    def report_extra_builder(_stats: PolygonGeometryStats) -> Mapping[str, Any]:
        return get_prepared().report_extra

    return release_polygon_stats(
        processed_dir=data_root.processed,
        staging_dir=data_root.cache / _STAGING_DIRNAME / "v1",
        repo_id=repo_id,
        confirm_repo=confirm_repo,
        card_writer=write_card,
        asset_writer=write_assets,
        apply=apply,
        hub=hub,
        verifier=verifier,
        token=token,
        source_revision=source_revision,
        data_revision=data_revision,
        report_extra_builder=report_extra_builder,
    )


def release_v2_polygon_stats(
    data_root: DataRoot,
    *,
    confirm_repo: str,
    repo_id: str = V2_REPO_ID,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
    token: str | None = None,
    source_revision: str | None = None,
    data_revision: str | None = None,
    generated_on: str | None = None,
) -> StatsReleaseReport:
    """Release the V2 Wikidata + Wikipedia card and statistics report."""
    _require_canonical_repo(repo_id, V2_REPO_ID)
    # V2 card/map scanners are loaded only for this explicit release operation.
    from osm_polygon_wikidata_only.v2.card import (  # noqa: PLC0415
        build_minimal_v2_release_snapshot,
    )
    from osm_polygon_wikidata_only.v2.maps import (  # noqa: PLC0415
        generate_v2_map_assets,
    )

    processed_v2 = data_root.processed_v2
    prepared: MinimalV2ReleaseSnapshot | None = None

    def get_prepared() -> MinimalV2ReleaseSnapshot:
        nonlocal prepared
        if prepared is None:
            prepared = build_minimal_v2_release_snapshot(
                processed_v2,
                v1_processed=data_root.processed,
                cache_dir=data_root.cache,
                generated_on=generated_on,
            )
        return prepared

    def write_card(destination: Path) -> None:
        # Rendering is performed only when the staged card is materialized.
        from osm_polygon_wikidata_only.hf.minimal_card import (  # noqa: PLC0415
            render_minimal_card,
        )

        _write_text_if_changed(destination, render_minimal_card(get_prepared().card))

    def write_assets(destination: Path) -> Mapping[str, Path]:
        snapshot = get_prepared().text_presence
        # ``v1_processed`` is deliberately omitted: it only adds the V2-added
        # Wikipedia-tag comparison map, which this release neither renders on
        # the compact card nor publishes, and which costs a full extra scan of
        # both the V1 and V2 polygon tables.
        coverage, presence, density = generate_v2_map_assets(
            processed_v2,
            destination,
            land_cache_dir=data_root.cache,
            text_snapshot=snapshot,
        )
        return {
            REMOTE_COVERAGE_MAP_FILE: coverage,
            REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE: presence,
            REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE: density,
        }

    def report_extra_builder(_stats: PolygonGeometryStats) -> Mapping[str, Any]:
        return get_prepared().report_extra

    return release_polygon_stats(
        processed_dir=processed_v2,
        staging_dir=data_root.cache / _STAGING_DIRNAME / "v2",
        repo_id=repo_id,
        confirm_repo=confirm_repo,
        card_writer=write_card,
        asset_writer=write_assets,
        apply=apply,
        hub=hub,
        verifier=verifier,
        token=token,
        source_revision=source_revision,
        data_revision=data_revision,
        report_extra_builder=report_extra_builder,
    )


__all__ = [
    "RELEASE_COMMIT_MESSAGE",
    "REMOTE_CARD_FILE",
    "ReleasedFile",
    "StatsReleaseError",
    "StatsReleaseReport",
    "default_remote_verifier",
    "release_polygon_stats",
    "release_v1_polygon_stats",
    "release_v2_polygon_stats",
]
