"""Remote reconciliation for versioned language split publication.

This module pins each remote read to one revision, compares remote shard identities
against the local plan, removes only manifest-owned stale paths, and verifies the
result after the atomic publication commit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from osm_polygon_wikidata_only.hf._publication.hub_snapshot import read_repo_sha
from osm_polygon_wikidata_only.hf._publication.language_card import merge_language_card
from osm_polygon_wikidata_only.hf._publication.language_errors import LanguagePublicationError
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf.language_split_release import LanguageSplitVersion
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import loads as json_loads

if TYPE_CHECKING:
    from osm_polygon_wikidata_only.hf.language_split_publication_models import (
        LanguagePublicationPlan,
        LanguagePublishedFile,
    )

REMOTE_README = "README.md"
REMOTE_CACHE_DIR = "language_split_publication"
_REMOTE_README = REMOTE_README
_REMOTE_CACHE_DIR = REMOTE_CACHE_DIR


@dataclass(frozen=True, slots=True)
class RemotePublicationSnapshot:
    """Revision-pinned remote state needed to reconcile one local publication."""

    revision: str
    files: set[str]
    manifest: bytes | None
    readme: str


def _remote_snapshot(
    hub: HfHub,
    plan: LanguagePublicationPlan,
    data_root: Path,
) -> tuple[str, set[str], bytes | None]:
    revision = _remote_revision(hub, plan.repo_id)
    remote_files = _remote_files(hub, plan.repo_id, revision=revision)
    old_manifest = _remote_bytes(
        hub,
        plan.repo_id,
        plan.manifest_remote_path,
        revision=revision,
        data_root=data_root,
        required=False,
    )
    return revision, remote_files, old_manifest


def _stale_manifest_files(
    old_manifest: bytes | None,
    plan: LanguagePublicationPlan,
    remote_files: set[str],
) -> tuple[str, ...]:
    planned_paths = _plan_paths(plan)
    return tuple(
        sorted(
            path
            for path in _manifest_owned_paths(old_manifest, plan)
            if path in remote_files and path not in planned_paths
        )
    )


def _build_publication_operations(
    plan: LanguagePublicationPlan,
    stale_files: Sequence[str],
    *,
    hub: HfHub,
    revision: str,
    data_root: Path,
) -> tuple[list[PublicationOp], tuple[str, ...]]:
    remote_entries = _remote_entries(hub, plan.repo_id, _plan_paths(plan), revision=revision)
    operations = _publication_operations(
        plan.files, remote_entries, hub, plan.repo_id, revision, data_root
    )
    operations.extend(delete_op(path) for path in stale_files)
    changed_files = tuple(sorted(op.path_in_repo for op in operations))
    return operations, changed_files


def _publication_operations(
    files: Sequence[LanguagePublishedFile],
    remote_entries: dict[str, Any],
    hub: HfHub,
    repo_id: str,
    revision: str,
    data_root: Path,
) -> list[PublicationOp]:
    return [
        add_op(file.local_path, path_in_repo=file.path_in_repo)
        for file in files
        if not _remote_matches(
            file, remote_entries.get(file.path_in_repo), hub, repo_id, revision, data_root
        )
    ]


def _remote_matches(
    local: LanguagePublishedFile,
    remote: Any | None,
    hub: HfHub,
    repo_id: str,
    revision: str,
    data_root: Path,
) -> bool:
    if not _remote_size_matches(local, remote):
        return False
    return _remote_digest_matches(local, remote, hub, repo_id, revision, data_root)


def _remote_size_matches(local: LanguagePublishedFile, remote: Any | None) -> bool:
    if remote is None or local.size_bytes is None or local.sha256 is None:
        return False
    remote_size = getattr(remote, "size", None)
    return not isinstance(remote_size, int) or remote_size == local.size_bytes


def _remote_digest_matches(
    local: LanguagePublishedFile,
    remote: Any,
    hub: HfHub,
    repo_id: str,
    revision: str,
    data_root: Path,
) -> bool:
    for matcher in (_remote_lfs_match, _remote_blob_match):
        matched = matcher(local, remote)
        if matched is not None:
            return matched
    remote_sha256 = _remote_download(
        hub,
        repo_id,
        local.path_in_repo,
        sha256_file,
        revision=revision,
        data_root=data_root,
        required=False,
    )
    return remote_sha256 is not None and remote_sha256 == local.sha256


def _remote_lfs_match(local: LanguagePublishedFile, remote: Any) -> bool | None:
    lfs = getattr(remote, "lfs", None)
    lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if isinstance(lfs_sha, str) and len(lfs_sha) == 64:
        return lfs_sha == local.sha256
    return None


def _remote_blob_match(local: LanguagePublishedFile, remote: Any) -> bool | None:
    blob_id = getattr(remote, "blob_id", None)
    if isinstance(blob_id, str) and len(blob_id) == 40:
        return blob_id == local.git_sha1()
    return None


def _remote_entries(
    hub: HfHub,
    repo_id: str,
    paths: Iterable[str],
    *,
    revision: str,
) -> dict[str, Any]:
    wanted = sorted(set(paths))
    result: dict[str, Any] = {}
    for start in range(0, len(wanted), 256):
        chunk = wanted[start : start + 256]
        _read_remote_entries(
            result,
            hub,
            repo_id=repo_id,
            paths=chunk,
            revision=revision,
        )
    return result


def _read_remote_entries(
    result: dict[str, Any],
    hub: HfHub,
    *,
    repo_id: str,
    paths: list[str],
    revision: str,
) -> None:
    try:
        entries = hub.get_paths_info(
            repo_id=repo_id,
            paths=paths,
            revision=revision,
            repo_type="dataset",
        )
        for entry in entries:
            path = getattr(entry, "path", None)
            if isinstance(path, str):
                result[path] = entry
    except Exception as error:
        raise LanguagePublicationError(
            f"could not read remote paths for {repo_id}@{revision}: {error}"
        ) from error


def _remote_files(hub: HfHub, repo_id: str, *, revision: str) -> set[str]:
    try:
        return set(hub.list_repo_files(repo_id=repo_id, revision=revision, repo_type="dataset"))
    except Exception as error:
        raise LanguagePublicationError(
            f"could not list remote files for {repo_id}@{revision}: {error}"
        ) from error


def _remote_revision(hub: HfHub, repo_id: str) -> str:
    try:
        revision = read_repo_sha(hub, repo_id)
    except Exception as error:
        raise LanguagePublicationError(
            f"could not read remote revision for {repo_id}: {error}"
        ) from error
    if not isinstance(revision, str) or not revision:
        raise LanguagePublicationError(
            f"remote revision unavailable for {repo_id}; refusing an unpinned publication"
        )
    return revision


def _remote_card(
    hub: HfHub,
    repo_id: str,
    revision: str,
    remote_files: set[str],
    data_root: Path,
) -> str:
    if _REMOTE_README not in remote_files:
        return f"# {repo_id}\n"
    raw = _remote_bytes(
        hub,
        repo_id,
        _REMOTE_README,
        revision=revision,
        data_root=data_root,
        required=True,
    )
    assert raw is not None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LanguagePublicationError(f"remote README is not valid UTF-8 for {repo_id}") from error


def _remote_bytes(
    hub: HfHub,
    repo_id: str,
    path: str,
    *,
    revision: str,
    data_root: Path,
    required: bool,
) -> bytes | None:
    return _remote_download(
        hub,
        repo_id,
        path,
        Path.read_bytes,
        revision=revision,
        data_root=data_root,
        required=required,
    )


def _remote_download[T](
    hub: HfHub,
    repo_id: str,
    path: str,
    read: Callable[[Path], T],
    *,
    revision: str,
    data_root: Path,
    required: bool,
) -> T | None:
    cache_dir = data_root / "cache" / _REMOTE_CACHE_DIR
    try:
        downloaded = hub.hf_hub_download(
            repo_id,
            path,
            revision=revision,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        return read(Path(downloaded))
    except Exception as error:
        if required:
            raise LanguagePublicationError(
                f"could not download remote {path} from {repo_id}: {error}"
            ) from error
        return None


def _write_card_snapshot(data_root: Path, plan: LanguagePublicationPlan, existing: str) -> Path:
    directory = data_root / "cache" / _REMOTE_CACHE_DIR / plan.version.value
    path = directory / "README.md"
    updated = merge_language_card(
        existing,
        version=plan.version,
        configurations=plan.configurations,
        languages=plan.languages,
        configuration_languages=plan.configuration_languages,
    )
    atomic_write_text(path, updated)
    return path


def _manifest_owned_paths(raw: bytes | None, plan: LanguagePublicationPlan) -> set[str]:
    if raw is None:
        return set()
    try:
        payload = json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError):
        return set()
    paths: set[str] = set()
    _collect_manifest_paths(payload, paths)
    return {path for path in paths if _is_managed_language_path(plan, path)}


_LANGUAGE_PATH_SHAPES: dict[LanguageSplitVersion, tuple[int, str, int]] = {
    LanguageSplitVersion.V1: (3, "data", 2),
    LanguageSplitVersion.V2: (4, "language_splits", 3),
}


def _is_managed_language_path(plan: LanguagePublicationPlan, path: str) -> bool:
    parts = PurePosixPath(path).parts
    expected_length, root, filename_index = _LANGUAGE_PATH_SHAPES[plan.version]
    if len(parts) != expected_length:
        return False
    return all(
        (
            parts[0] == root,
            parts[1] in plan.configurations,
            parts[2].startswith("lang-"),
            parts[filename_index].endswith(".parquet"),
        )
    )


def _collect_manifest_paths(value: object, paths: set[str]) -> None:
    if isinstance(value, dict):
        _collect_manifest_mapping(cast(dict[object, object], value), paths)
    elif isinstance(value, list):
        _collect_manifest_list(cast(list[object], value), paths)


def _collect_manifest_mapping(value: dict[object, object], paths: set[str]) -> None:
    raw_path = value.get("path")
    if isinstance(raw_path, str) and raw_path.endswith(".parquet"):
        paths.add(raw_path)
    for child in value.values():
        _collect_manifest_paths(child, paths)


def _collect_manifest_list(value: list[object], paths: set[str]) -> None:
    for child in value:
        _collect_manifest_paths(child, paths)


def _verify_remote_release(
    hub: HfHub,
    plan: LanguagePublicationPlan,
    *,
    revision: str,
    stale_files: Sequence[str],
    data_root: Path,
) -> None:
    entries = _remote_entries(
        hub,
        plan.repo_id,
        (*_plan_paths(plan), *stale_files),
        revision=revision,
    )
    for file in plan.files:
        if not _remote_matches(
            file, entries.get(file.path_in_repo), hub, plan.repo_id, revision, data_root
        ):
            raise LanguagePublicationError(
                f"remote verification failed for {plan.repo_id}:{file.path_in_repo}"
            )
    for path in stale_files:
        if path in entries:
            raise LanguagePublicationError(
                f"remote stale language file remains for {plan.repo_id}:{path}"
            )


def _plan_paths(plan: LanguagePublicationPlan) -> set[str]:
    return {file.path_in_repo for file in plan.files}


def inspect_remote_publication(
    hub: HfHub, plan: LanguagePublicationPlan, data_root: Path
) -> RemotePublicationSnapshot:
    """Read one immutable remote snapshot for a publication plan."""
    revision, files, manifest = _remote_snapshot(hub, plan, data_root)
    readme = _remote_card(hub, plan.repo_id, revision, files, data_root)
    return RemotePublicationSnapshot(revision, files, manifest, readme)


def stale_language_files(
    snapshot: RemotePublicationSnapshot, plan: LanguagePublicationPlan
) -> tuple[str, ...]:
    """Select only prior manifest-owned language files absent from the new plan."""
    return _stale_manifest_files(snapshot.manifest, plan, snapshot.files)


def write_remote_card_snapshot(
    data_root: Path, plan: LanguagePublicationPlan, existing: str
) -> Path:
    """Merge the managed language section into a local cached README snapshot."""
    return _write_card_snapshot(data_root, plan, existing)


def build_remote_publication_operations(
    plan: LanguagePublicationPlan,
    stale_files: Iterable[str],
    *,
    hub: HfHub,
    revision: str,
    data_root: Path,
) -> tuple[list[PublicationOp], tuple[str, ...]]:
    """Compare planned files at the pinned revision and build atomic changes."""
    return _build_publication_operations(
        plan, tuple(stale_files), hub=hub, revision=revision, data_root=data_root
    )


def remote_revision(hub: HfHub, repo_id: str) -> str:
    """Return the current immutable remote commit identity or raise."""
    return _remote_revision(hub, repo_id)


def verify_remote_publication(
    hub: HfHub,
    plan: LanguagePublicationPlan,
    *,
    revision: str,
    stale_files: Iterable[str],
    data_root: Path,
) -> None:
    """Verify every planned file and ensure obsolete managed paths are gone."""
    _verify_remote_release(
        hub,
        plan,
        revision=revision,
        stale_files=tuple(stale_files),
        data_root=data_root,
    )
