"""Exact-target publication for generated V1 and V2 language partitions.

Local generation and Hub publication are deliberately separate operations.  This
module validates the requested contract, maps only generated language files to
the matching dataset, updates a managed card section without replacing the
card, and submits one atomic Hub commit per dataset.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._uploader.operations import _build_hf_api
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.token import resolve_hf_token
from osm_polygon_wikidata_only.hf.language_split_release import (
    LanguageSplitVersion,
    plan_language_split_release,
    run_language_split_release,
)
from osm_polygon_wikidata_only.hf.language_splits import (
    DATASET_V1_ID,
    DATASET_V2_ID,
    DatasetContract,
)
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads

LANGUAGE_PUBLICATION_COMMIT_MESSAGE = "Publish row-level language partitions"
V1_LANGUAGE_MANIFEST_REMOTE = "manifests/language_splits_v1.json"
V2_LANGUAGE_MANIFEST_REMOTE = "manifests/language_splits.json"
LANGUAGE_CARD_HEADING = "## Language partitions"
_REMOTE_README = "README.md"
_REMOTE_CACHE_DIR = "language_split_publication"
MAX_ATOMIC_PUBLICATION_FILES = 25_000


class LanguagePublicationError(RuntimeError):
    """Raised when a language publication cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class LanguagePublishedFile:
    """One local file and its exact remote publication identity."""

    local_path: Path
    path_in_repo: str
    size_bytes: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, object]:
        """Return deterministic evidence for this file."""
        return {
            "local_path": str(self.local_path),
            "path_in_repo": self.path_in_repo,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class LanguagePublicationPlan:
    """Complete local publication plan for one dataset contract."""

    version: LanguageSplitVersion
    repo_id: str
    processed_root: Path
    output_root: Path
    manifest_path: Path
    manifest_remote_path: str
    languages: tuple[str, ...]
    configurations: tuple[str, ...]
    files: tuple[LanguagePublishedFile, ...]

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible plan evidence."""
        return {
            "dataset_version": self.version.value,
            "repo_id": self.repo_id,
            "processed_root": str(self.processed_root),
            "output_root": str(self.output_root),
            "manifest_path": str(self.manifest_path),
            "manifest_remote_path": self.manifest_remote_path,
            "languages": list(self.languages),
            "configurations": list(self.configurations),
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class LanguagePublicationReport:
    """Evidence for one dry-run or one remote publication."""

    version: LanguageSplitVersion
    repo_id: str
    dry_run: bool
    published: bool
    committed: bool
    no_op: bool
    revision: str | None
    files: tuple[LanguagePublishedFile, ...]
    changed_files: tuple[str, ...] = ()
    stale_files: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Return deterministic release evidence."""
        return {
            "changed_files": list(self.changed_files),
            "committed": self.committed,
            "dataset_version": self.version.value,
            "dry_run": self.dry_run,
            "files": [item.to_dict() for item in self.files],
            "no_op": self.no_op,
            "published": self.published,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "stale_files": list(self.stale_files),
        }


@dataclass(frozen=True, slots=True)
class LanguagePublicationResult:
    """Combined evidence for the selected V1/V2 publication set."""

    reports: tuple[LanguagePublicationReport, ...]

    def to_payload(self) -> dict[str, object]:
        """Return deterministic JSON-compatible evidence."""
        return {
            "command": "publish-language-splits",
            "reports": [report.to_payload() for report in self.reports],
        }

    def to_json(self) -> str:
        """Serialize the publication evidence."""
        return json_dumps(self.to_payload())


def plan_language_split_publication(
    data_root: DataRoot | Path,
    *,
    dataset_version: str = "both",
    batch_size: int = 65_536,
    confirm_repos: Sequence[str] = (),
) -> tuple[LanguagePublicationPlan, ...]:
    """Validate source inventories and return a no-write publication plan."""
    root = _resolve_data_root(data_root)
    versions = _selected_versions(dataset_version)
    _validate_confirmations(versions, confirm_repos)
    release_plan = plan_language_split_release(
        root,
        dataset_version=dataset_version,
        batch_size=batch_size,
    )
    return tuple(_plan_from_version_plan(version_plan) for version_plan in release_plan.releases)


def run_language_split_publication(
    data_root: DataRoot | Path,
    *,
    dataset_version: str = "both",
    batch_size: int = 65_536,
    confirm_repos: Sequence[str] = (),
    apply: bool = False,
    dry_run: bool = False,
    hub: HfHub | None = None,
    token: str | None = None,
) -> LanguagePublicationResult:
    """Generate, publish, and independently verify selected language releases.

    A real run generates each selected contract locally, submits one atomic
    commit to its exact dataset, and performs a revision-bound remote check.
    Repeating the run with unchanged local inputs produces no second commit.
    """
    root = _resolve_data_root(data_root)
    versions = _selected_versions(dataset_version)
    _validate_confirmations(versions, confirm_repos)
    if dry_run or not apply:
        return _plan_only_publication(
            root,
            dataset_version=dataset_version,
            batch_size=batch_size,
            confirm_repos=confirm_repos,
        )
    plans = plan_language_split_publication(
        root,
        dataset_version=dataset_version,
        batch_size=batch_size,
        confirm_repos=confirm_repos,
    )
    _validate_atomic_publication_plans(plans)
    generated = run_language_split_release(
        root,
        dataset_version=dataset_version,
        batch_size=batch_size,
        dry_run=False,
    )
    client = hub or _build_hf_api(resolve_hf_token(token))
    reports = _publish_generated_versions(
        root,
        generated,
        hub=client,
        token=token,
    )
    return LanguagePublicationResult(reports=reports)


def _validate_atomic_publication_plans(
    plans: Sequence[LanguagePublicationPlan],
) -> None:
    """Reject plans that cannot fit one safe Hub commit before generation."""
    for plan in plans:
        planned_operation_files = len(plan.files) + 2
        if planned_operation_files > MAX_ATOMIC_PUBLICATION_FILES:
            raise LanguagePublicationError(
                f"single atomic commit for {plan.repo_id} would contain "
                f"{planned_operation_files:,} files, exceeding the Hugging Face "
                f"create_commit limit of {MAX_ATOMIC_PUBLICATION_FILES:,} files; "
                "no remote mutation was attempted"
            )


def _plan_only_publication(
    data_root: Path,
    *,
    dataset_version: str,
    batch_size: int,
    confirm_repos: Sequence[str],
) -> LanguagePublicationResult:
    plans = plan_language_split_publication(
        data_root,
        dataset_version=dataset_version,
        batch_size=batch_size,
        confirm_repos=confirm_repos,
    )
    reports = tuple(
        LanguagePublicationReport(
            version=plan.version,
            repo_id=plan.repo_id,
            dry_run=True,
            published=False,
            committed=False,
            no_op=False,
            revision=None,
            files=plan.files,
        )
        for plan in plans
    )
    return LanguagePublicationResult(reports=reports)


def _publish_generated_versions(
    data_root: Path,
    generated: Any,
    *,
    hub: HfHub,
    token: str | None,
) -> tuple[LanguagePublicationReport, ...]:
    return tuple(
        _publish_one_version(
            data_root,
            version_plan,
            generated_release,
            hub=hub,
            token=token,
        )
        for version_plan, generated_release in zip(
            generated.plan.releases, generated.generated, strict=True
        )
    )


def _publish_one_version(
    data_root: Path,
    version_plan: Any,
    generated: Any,
    *,
    hub: HfHub,
    token: str | None,
) -> LanguagePublicationReport:
    plan = _plan_from_generated(version_plan, generated)
    revision, remote_files, old_manifest = _remote_snapshot(hub, plan, data_root)
    stale_files = _stale_manifest_files(old_manifest, plan, remote_files)
    plan = _plan_with_card_file(
        data_root,
        plan,
        _remote_card(hub, plan.repo_id, revision, remote_files, data_root),
    )
    operations, changed_files = _build_publication_operations(
        plan,
        stale_files,
        hub=hub,
        revision=revision,
        data_root=data_root,
    )
    return _publish_or_record(
        data_root,
        plan,
        operations,
        changed_files,
        stale_files,
        hub=hub,
        token=token,
        revision=revision,
    )


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


def _plan_with_card_file(
    data_root: Path,
    plan: LanguagePublicationPlan,
    existing_card: str,
) -> LanguagePublicationPlan:
    card_path = _write_card_snapshot(data_root, plan, existing_card)
    return replace(plan, files=(*plan.files, _local_file(card_path, _REMOTE_README)))


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


def _publish_or_record(
    data_root: Path,
    plan: LanguagePublicationPlan,
    operations: list[PublicationOp],
    changed_files: tuple[str, ...],
    stale_files: Sequence[str],
    *,
    hub: HfHub,
    token: str | None,
    revision: str,
) -> LanguagePublicationReport:
    if not operations:
        return _record_no_op(
            data_root,
            plan,
            hub=hub,
            revision=revision,
            stale_files=stale_files,
        )

    commit_revision = upload_files(
        plan.repo_id,
        ops=operations,
        hub=hub,
        token=token,
        commit_message=LANGUAGE_PUBLICATION_COMMIT_MESSAGE,
        allow_noop=True,
    )
    if not commit_revision:
        latest_revision = _remote_revision(hub, plan.repo_id)
        return _record_no_op(
            data_root,
            plan,
            hub=hub,
            revision=latest_revision,
            stale_files=stale_files,
        )
    _verify_remote_release(
        hub,
        plan,
        revision=commit_revision,
        stale_files=tuple(stale_files),
        data_root=data_root,
    )
    report = LanguagePublicationReport(
        version=plan.version,
        repo_id=plan.repo_id,
        dry_run=False,
        published=True,
        committed=True,
        no_op=False,
        revision=commit_revision,
        files=plan.files,
        changed_files=changed_files,
        stale_files=tuple(stale_files),
    )
    _write_report(data_root, report)
    return report


def _record_no_op(
    data_root: Path,
    plan: LanguagePublicationPlan,
    *,
    hub: HfHub,
    revision: str,
    stale_files: Sequence[str],
) -> LanguagePublicationReport:
    """Verify and record a publication that required no remote commit."""
    _verify_remote_release(
        hub,
        plan,
        revision=revision,
        stale_files=stale_files,
        data_root=data_root,
    )
    report = LanguagePublicationReport(
        version=plan.version,
        repo_id=plan.repo_id,
        dry_run=False,
        published=True,
        committed=False,
        no_op=True,
        revision=revision,
        files=plan.files,
        changed_files=(),
        stale_files=tuple(stale_files),
    )
    _write_report(data_root, report)
    return report


def _plan_from_version_plan(version_plan: Any) -> LanguagePublicationPlan:
    """Map the validated local plan to remote paths before generation."""
    records = version_plan.to_dict(version_plan.processed_root.parent)["expected_files"]
    files = tuple(
        LanguagePublishedFile(
            local_path=version_plan.processed_root.parent / str(record["path"]),
            path_in_repo=_remote_path_for_local(
                _contract_for_version(version_plan.version),
                version_plan.processed_root.parent / str(record["path"]),
                processed_root=version_plan.processed_root,
                output_root=version_plan.output_root,
            ),
            size_bytes=None,
            sha256=None,
        )
        for record in cast(list[dict[str, object]], records)
    )
    return LanguagePublicationPlan(
        version=version_plan.version,
        repo_id=_repo_for_version(version_plan.version),
        processed_root=version_plan.processed_root,
        output_root=version_plan.output_root,
        manifest_path=version_plan.manifest_path,
        manifest_remote_path=_manifest_remote_for_version(version_plan.version),
        languages=tuple(version_plan.inventory.languages),
        configurations=tuple(
            sorted(table.configuration for table in version_plan.inventory.tables)
        ),
        files=tuple(sorted(files, key=lambda item: item.path_in_repo)),
    )


def _plan_from_generated(version_plan: Any, generated: Any) -> LanguagePublicationPlan:
    """Build a complete hashed plan from generated local files."""
    contract = _contract_for_version(version_plan.version)
    files = tuple(
        _local_file(
            _generated_local_path(version_plan, generated_file),
            _remote_path_for_local(
                contract,
                _generated_local_path(version_plan, generated_file),
                processed_root=version_plan.processed_root,
                output_root=version_plan.output_root,
            ),
        )
        for generated_file in generated.files
    )
    manifest = _local_file(
        Path(generated.manifest_path),
        _manifest_remote_for_version(version_plan.version),
    )
    return LanguagePublicationPlan(
        version=version_plan.version,
        repo_id=_repo_for_version(version_plan.version),
        processed_root=version_plan.processed_root,
        output_root=version_plan.output_root,
        manifest_path=Path(generated.manifest_path),
        manifest_remote_path=manifest.path_in_repo,
        languages=tuple(version_plan.inventory.languages),
        configurations=tuple(
            sorted(table.configuration for table in version_plan.inventory.tables)
        ),
        files=tuple(sorted((*files, manifest), key=lambda item: item.path_in_repo)),
    )


def _generated_local_path(version_plan: Any, generated_file: Any) -> Path:
    path = Path(generated_file.path)
    return (
        path
        if version_plan.version is LanguageSplitVersion.V1
        else version_plan.processed_root / path
    )


def _local_file(path: Path, path_in_repo: str) -> LanguagePublishedFile:
    resolved = path.resolve()
    if not resolved.is_file():
        raise LanguagePublicationError(f"generated language file is missing: {resolved}")
    return LanguagePublishedFile(
        local_path=resolved,
        path_in_repo=path_in_repo,
        size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    )


def _publication_operations(
    files: Sequence[LanguagePublishedFile],
    remote_entries: dict[str, Any],
    hub: HfHub,
    repo_id: str,
    revision: str,
    data_root: Path,
) -> list[PublicationOp]:
    operations: list[PublicationOp] = []
    for file in files:
        if not _remote_matches(
            file, remote_entries.get(file.path_in_repo), hub, repo_id, revision, data_root
        ):
            operations.append(add_op(file.local_path, path_in_repo=file.path_in_repo))
    return operations


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
    downloaded = _remote_bytes(
        hub,
        repo_id,
        local.path_in_repo,
        revision=revision,
        data_root=data_root,
        required=False,
    )
    return downloaded is not None and hashlib.sha256(downloaded).hexdigest() == local.sha256


def _remote_lfs_match(local: LanguagePublishedFile, remote: Any) -> bool | None:
    lfs = getattr(remote, "lfs", None)
    lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if isinstance(lfs_sha, str) and len(lfs_sha) == 64:
        return lfs_sha == local.sha256


def _remote_blob_match(local: LanguagePublishedFile, remote: Any) -> bool | None:
    blob_id = getattr(remote, "blob_id", None)
    if isinstance(blob_id, str) and len(blob_id) == 40:
        return blob_id == _git_blob_sha1(local.local_path)


def _remote_entries(
    hub: HfHub,
    repo_id: str,
    paths: Iterable[str],
    *,
    revision: str,
) -> dict[str, Any]:
    wanted = sorted(set(paths))
    result: dict[str, Any] = {}
    get_paths_info = _path_info_reader(hub, revision)
    for start in range(0, len(wanted), 256):
        chunk = wanted[start : start + 256]
        _read_remote_entries(
            result,
            get_paths_info,
            repo_id=repo_id,
            paths=chunk,
            revision=revision,
        )
    return result


def _path_info_reader(hub: HfHub, revision: str) -> Any:
    reader = getattr(cast(Any, hub), "get_paths_info", None)
    if not callable(reader):
        raise LanguagePublicationError(
            f"remote client cannot read paths at immutable revision {revision}"
        )
    return reader


def _read_remote_entries(
    result: dict[str, Any],
    get_paths_info: Any,
    *,
    repo_id: str,
    paths: list[str],
    revision: str,
) -> None:
    try:
        entries = get_paths_info(
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
        info = hub.repo_info(repo_id, repo_type="dataset")
    except Exception as error:
        raise LanguagePublicationError(
            f"could not read remote revision for {repo_id}: {error}"
        ) from error
    revision = getattr(info, "sha", None)
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
    cache_dir = data_root / "cache" / _REMOTE_CACHE_DIR
    try:
        downloaded = hub.hf_hub_download(
            repo_id,
            path,
            revision=revision,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        return Path(downloaded).read_bytes()
    except Exception as error:
        if required:
            raise LanguagePublicationError(
                f"could not download remote {path} from {repo_id}: {error}"
            ) from error
        return None


def _write_card_snapshot(data_root: Path, plan: LanguagePublicationPlan, existing: str) -> Path:
    directory = data_root / "cache" / _REMOTE_CACHE_DIR / plan.version.value
    path = directory / "README.md"
    updated = _merge_language_card(
        existing,
        version=plan.version,
        configurations=plan.configurations,
        languages=plan.languages,
    )
    atomic_write_text(path, updated)
    return path


def _merge_language_card(
    existing: str,
    *,
    version: LanguageSplitVersion,
    configurations: Sequence[str],
    languages: Sequence[str],
) -> str:
    """Replace only the managed language section and preserve other card text."""
    section = _render_language_card_section(version, configurations, languages)
    pattern = re.compile(
        rf"^{re.escape(LANGUAGE_CARD_HEADING)}\n.*?(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(existing)
    if match:
        matched = match.group(0)
        trailing_newlines = len(matched) - len(matched.rstrip("\n"))
        separator = "\n" * trailing_newlines
        replacement = section.rstrip("\n") + separator
        return existing[: match.start()] + replacement + existing[match.end() :]
    marker = re.search(r"^## Data sources & licenses\n", existing, re.MULTILINE)
    if marker:
        return existing[: marker.start()] + section + "\n" + existing[marker.start() :]
    return existing.rstrip() + "\n\n" + section


def _render_language_card_section(
    version: LanguageSplitVersion,
    configurations: Sequence[str],
    languages: Sequence[str],
) -> str:
    contract = "V1" if version is LanguageSplitVersion.V1 else "V2"
    lines = [
        LANGUAGE_CARD_HEADING,
        "",
        f"The {contract} language release is an additive, row-level partition of the published text tables.",
        "Each source row is routed by its normalized `language` value; multilingual rows are not collapsed to a polygon-level preferred language.",
        "Missing, blank, malformed, and legacy-unusable values are preserved in the explicit `lang-unknown` partition.",
        "",
        f"Validated languages: **{len(languages)}** (including `unknown`).",
        "",
        "| Configuration | Split names | Remote path |",
        "| --- | --- | --- |",
    ]
    for configuration in sorted(configurations):
        if version is LanguageSplitVersion.V1:
            path = f"data/{configuration}/lang-<language>-00000-of-00001.parquet"
        else:
            path = f"language_splits/{configuration}/lang-<language>/*.parquet"
        lines.append(f"| `{configuration}` | `lang-<language>` and `lang-unknown` | `{path}` |")
    lines.extend(
        [
            "",
            "The release manifest records the source fingerprint, schema, row counts, and SHA-256 hash for every generated file.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _write_report(data_root: Path, report: LanguagePublicationReport) -> None:
    path = data_root / "cache" / _REMOTE_CACHE_DIR / report.version.value / "release-report.json"
    atomic_write_text(path, json_dumps(report.to_payload()) + "\n")


def _plan_paths(plan: LanguagePublicationPlan) -> set[str]:
    return {file.path_in_repo for file in plan.files}


def _remote_path_for_local(
    contract: DatasetContract,
    local_path: Path,
    *,
    processed_root: Path,
    output_root: Path,
) -> str:
    """Map a generated local artifact to its contract-specific Hub path."""
    base = output_root if contract is DatasetContract.V1 else processed_root
    try:
        return local_path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError as error:
        raise LanguagePublicationError(
            f"language artifact is outside its publication root: {local_path}"
        ) from error


def _manifest_remote_for_version(version: LanguageSplitVersion) -> str:
    return (
        V1_LANGUAGE_MANIFEST_REMOTE
        if version is LanguageSplitVersion.V1
        else V2_LANGUAGE_MANIFEST_REMOTE
    )


def _repo_for_version(version: LanguageSplitVersion) -> str:
    return DATASET_V1_ID if version is LanguageSplitVersion.V1 else DATASET_V2_ID


def _contract_for_version(version: LanguageSplitVersion) -> DatasetContract:
    return DatasetContract.V1 if version is LanguageSplitVersion.V1 else DatasetContract.V2


def _selected_versions(dataset_version: str) -> tuple[LanguageSplitVersion, ...]:
    if dataset_version == "both":
        return (LanguageSplitVersion.V1, LanguageSplitVersion.V2)
    try:
        return (LanguageSplitVersion(dataset_version),)
    except ValueError as error:
        raise LanguagePublicationError(
            f"dataset_version must be one of v1, v2, both; got {dataset_version!r}"
        ) from error


def _validate_confirmations(
    versions: Sequence[LanguageSplitVersion],
    confirm_repos: Sequence[str],
) -> None:
    expected = [_repo_for_version(version) for version in versions]
    if sorted(confirm_repos) != sorted(expected):
        raise LanguagePublicationError(
            "exact --confirm-repo values are required: " + ", ".join(expected)
        )


def _resolve_data_root(data_root: DataRoot | Path) -> Path:
    root = data_root.path if isinstance(data_root, DataRoot) else Path(data_root)
    root = root.resolve()
    if not root.is_dir():
        raise LanguagePublicationError(f"data root is not a directory: {root}")
    return root


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "LANGUAGE_PUBLICATION_COMMIT_MESSAGE",
    "LanguagePublicationError",
    "LanguagePublicationPlan",
    "LanguagePublicationReport",
    "LanguagePublicationResult",
    "LanguagePublishedFile",
    "plan_language_split_publication",
    "run_language_split_publication",
]
