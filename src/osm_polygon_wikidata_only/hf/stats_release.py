"""Deterministic polygon-statistics release for one published dataset.

The release path is narrow on purpose. For one processed artifact tree it
recomputes the machine-readable ``stats.json`` report and the dataset card from
every published polygon row, publishes exactly those two files to the exact Hub
dataset, and verifies the remote files afterwards. Parquet tables, manifests,
maps, and unrelated Hub files are never rewritten by this path.

Both published contracts use it:

* V1 -- ``NoeFlandre/osm-polygon-wikidata-only`` from ``<data-root>/processed``;
* V2 -- ``NoeFlandre/osm-polygon-wikidata-and-wikipedia`` from
  ``<data-root>/processed_v2``.

Determinism: the report and the card come from one scan of the published
polygon table, cross-checked against the processed manifest. Staged files are
rewritten only when their bytes change, so a second release over unchanged
artifacts stages nothing and produces the same plan.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import (
    load_polygon_geometry_stats,
    stats_payload,
)
from osm_polygon_wikidata_only.hf.repo_layout import REMOTE_POLYGON_STATS_FILE
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.atomic import atomic_write_json, atomic_write_text
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID

REMOTE_CARD_FILE = "README.md"
RELEASE_COMMIT_MESSAGE = "Publish dataset card and polygon statistics report"
_STAGING_DIRNAME = "stats_release_snapshots"

CardWriter = Callable[[Path], None]


class StatsReleaseError(RuntimeError):
    """Raised when a statistics release cannot be planned or verified."""


@dataclass(frozen=True)
class ReleasedFile:
    """One published metadata file and its exact identity."""

    path_in_repo: str
    sha256: str
    size_bytes: int


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

    def to_payload(self) -> dict[str, Any]:
        return {
            "files": [
                {
                    "path_in_repo": item.path_in_repo,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            ],
            "polygon_files": self.polygon_files,
            "polygon_rows": self.polygon_rows,
            "processed_dir": self.processed_dir,
            "published": self.published,
            "repo_id": self.repo_id,
            "revision": self.revision,
        }


class RemoteVerifier(Protocol):
    """Confirm the released files exist remotely and return the revision."""

    def __call__(self, repo_id: str, files: tuple[ReleasedFile, ...]) -> str: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    atomic_write_text(path, text)


def _stage_report(processed_dir: Path, destination: Path) -> None:
    payload = stats_payload(load_polygon_geometry_stats(processed_dir))
    atomic_write_json(destination, payload)


def _released_file(path: Path, path_in_repo: str) -> ReleasedFile:
    return ReleasedFile(
        path_in_repo=path_in_repo,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _require_exact_repo(confirm_repo: str, repo_id: str) -> None:
    if confirm_repo != repo_id:
        raise StatsReleaseError(
            f"repository confirmation must equal {repo_id!r} (got {confirm_repo!r})"
        )


def _require_published_polygons(processed_dir: Path) -> None:
    polygons = processed_dir / "polygons"
    if not polygons.is_dir() or not any(polygons.glob("*.parquet")):
        raise StatsReleaseError(f"no published polygon files under {polygons}")


def default_remote_verifier(repo_id: str, files: tuple[ReleasedFile, ...]) -> str:
    """Confirm each released file's remote identity and return the revision."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id, repo_type="dataset")
    revision = str(getattr(info, "sha", "") or "")
    if not revision:
        raise StatsReleaseError(f"hub repository {repo_id} returned an empty revision")
    for item in files:
        entries = api.get_paths_info(
            repo_id,
            paths=[item.path_in_repo],
            revision=revision,
            repo_type="dataset",
        )
        entry = next(iter(entries), None)
        if entry is None:
            raise StatsReleaseError(
                f"remote file missing in revision {revision}: {item.path_in_repo}"
            )
        size = getattr(entry, "size", None)
        if size is not None and int(size) != item.size_bytes:
            raise StatsReleaseError(
                f"remote size mismatch for {item.path_in_repo}: "
                f"local={item.size_bytes}, remote={size}"
            )
        local_path = api.hf_hub_download(
            repo_id,
            item.path_in_repo,
            revision=revision,
            repo_type="dataset",
        )
        if _sha256(Path(local_path)) != item.sha256:
            raise StatsReleaseError(f"remote SHA-256 mismatch for {item.path_in_repo}")
    return revision


def release_polygon_stats(
    *,
    processed_dir: Path,
    staging_dir: Path,
    repo_id: str,
    confirm_repo: str,
    card_writer: CardWriter,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
) -> StatsReleaseReport:
    """Recompute, publish, and verify the card and report for one dataset.

    ``apply=False`` performs the full local compute-and-validate pass and
    returns the exact plan that would be uploaded, without touching the
    network.
    """
    _require_exact_repo(confirm_repo, repo_id)
    _require_published_polygons(processed_dir)
    stats = load_polygon_geometry_stats(processed_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    report_path = staging_dir / "stats.json"
    card_path = staging_dir / "README.md"
    _stage_report(processed_dir, report_path)
    card_writer(card_path)
    files = (
        _released_file(card_path, REMOTE_CARD_FILE),
        _released_file(report_path, REMOTE_POLYGON_STATS_FILE),
    )
    revision: str | None = None
    if apply:
        ops: list[PublicationOp] = [
            add_op(card_path, path_in_repo=REMOTE_CARD_FILE),
            add_op(report_path, path_in_repo=REMOTE_POLYGON_STATS_FILE),
        ]
        upload_files(
            repo_id,
            ops=ops,
            hub=hub,
            commit_message=RELEASE_COMMIT_MESSAGE,
        )
        verify_remote = verifier or default_remote_verifier
        revision = verify_remote(repo_id, files)
        if not revision:
            raise StatsReleaseError("remote verification returned an empty revision")
    return StatsReleaseReport(
        repo_id=repo_id,
        processed_dir=str(processed_dir),
        files=files,
        polygon_files=stats.file_count,
        polygon_rows=stats.polygon_count,
        published=apply,
        revision=revision,
    )


def release_v1_polygon_stats(
    data_root: DataRoot,
    *,
    confirm_repo: str,
    repo_id: str = DEFAULT_REPO_ID,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
) -> StatsReleaseReport:
    """Release the V1 Wikidata-only card and statistics report."""
    from osm_polygon_wikidata_only.hf.publication import write_readme_snapshot

    def write_card(destination: Path) -> None:
        write_readme_snapshot(data_root, repo_id, destination)

    return release_polygon_stats(
        processed_dir=data_root.processed,
        staging_dir=data_root.cache / _STAGING_DIRNAME / "v1",
        repo_id=repo_id,
        confirm_repo=confirm_repo,
        card_writer=write_card,
        apply=apply,
        hub=hub,
        verifier=verifier,
    )


def release_v2_polygon_stats(
    data_root: DataRoot,
    *,
    confirm_repo: str,
    repo_id: str = V2_REPO_ID,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
) -> StatsReleaseReport:
    """Release the V2 Wikidata + Wikipedia card and statistics report."""
    from osm_polygon_wikidata_only.v2.card import render_v2_card

    processed_v2 = data_root.processed_v2

    def write_card(destination: Path) -> None:
        _write_text_if_changed(
            destination,
            render_v2_card(processed_v2, v1_processed=data_root.processed),
        )

    return release_polygon_stats(
        processed_dir=processed_v2,
        staging_dir=data_root.cache / _STAGING_DIRNAME / "v2",
        repo_id=repo_id,
        confirm_repo=confirm_repo,
        card_writer=write_card,
        apply=apply,
        hub=hub,
        verifier=verifier,
    )


__all__ = [
    "RELEASE_COMMIT_MESSAGE",
    "REMOTE_CARD_FILE",
    "ReleasedFile",
    "StatsReleaseError",
    "StatsReleaseReport",
    "release_polygon_stats",
    "release_v1_polygon_stats",
    "release_v2_polygon_stats",
]
