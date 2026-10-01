"""Exact-target publication for generated V1 and V2 language partitions.

Local generation and Hub publication are deliberately separate operations.  This
module validates the requested contract, maps only generated language files to
the matching dataset, updates a managed card section without replacing the
card, and submits one atomic Hub commit per dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._publication.data_root import resolve_data_root
from osm_polygon_wikidata_only.hf._publication.language_card import LANGUAGE_CARD_HEADING
from osm_polygon_wikidata_only.hf._publication.language_card import (
    language_sort_key as _language_sort_key,
)
from osm_polygon_wikidata_only.hf._publication.language_errors import LanguagePublicationError
from osm_polygon_wikidata_only.hf._uploader.operations import build_hf_api as _build_hf_api
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.token import resolve_hf_token
from osm_polygon_wikidata_only.hf.language_split_publication_models import (
    LanguagePublicationPlan,
    LanguagePublicationReport,
    LanguagePublicationResult,
    LanguagePublishedFile,
)
from osm_polygon_wikidata_only.hf.language_split_release import (
    LanguageSplitReleaseResult,
    LanguageSplitVersion,
    LanguageSplitVersionPlan,
    plan_language_split_release,
    run_language_split_release,
    select_language_split_versions,
)
from osm_polygon_wikidata_only.hf.language_split_remote import (
    REMOTE_CACHE_DIR,
    REMOTE_README,
    build_remote_publication_operations,
    inspect_remote_publication,
    remote_revision,
    stale_language_files,
    verify_remote_publication,
    write_remote_card_snapshot,
)
from osm_polygon_wikidata_only.hf.language_splits import (
    DATASET_V1_ID,
    DATASET_V2_ID,
    DatasetContract,
    LanguageInventory,
)
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import (
    enable_hash_cache,
    flush_hash_cache,
    sha256_file,
)
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps

if TYPE_CHECKING:
    # Type-only: the release module keeps the per-version generators lazy.
    from osm_polygon_wikidata_only.hf.v1_language_splits import (
        V1LanguageSplitRelease,
        V1PartitionFile,
    )
    from osm_polygon_wikidata_only.v2.language_splits import (
        V2LanguageSplitFile,
        V2LanguageSplitResult,
    )

    _GeneratedRelease = V1LanguageSplitRelease | V2LanguageSplitResult
    _GeneratedFile = V1PartitionFile | V2LanguageSplitFile

LANGUAGE_PUBLICATION_COMMIT_MESSAGE = "Publish row-level language partitions"
V1_LANGUAGE_MANIFEST_REMOTE = "manifests/language_splits_v1.json"
V2_LANGUAGE_MANIFEST_REMOTE = "manifests/language_splits.json"
MAX_ATOMIC_PUBLICATION_FILES = 25_000


def plan_language_split_publication(
    data_root: DataRoot | Path,
    *,
    dataset_version: str = "both",
    batch_size: int = 65_536,
    confirm_repos: Sequence[str] = (),
) -> tuple[LanguagePublicationPlan, ...]:
    """Validate source inventories and return a no-write publication plan."""
    root = resolve_data_root(data_root, LanguagePublicationError)
    # Digests of unchanged artifacts survive between runs, so a repeated
    # release does not re-read the whole corpus just to re-derive them.
    enable_hash_cache(root / "cache" / "hash_cache")
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
    root = resolve_data_root(data_root, LanguagePublicationError)
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
    flush_hash_cache()
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
    generated: LanguageSplitReleaseResult,
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
    version_plan: LanguageSplitVersionPlan,
    generated: _GeneratedRelease,
    *,
    hub: HfHub,
    token: str | None,
) -> LanguagePublicationReport:
    plan = _plan_from_generated(version_plan, generated)
    snapshot = inspect_remote_publication(hub, plan, data_root)
    stale_files = stale_language_files(snapshot, plan)
    plan = _plan_with_card_file(data_root, plan, snapshot.readme)
    operations, changed_files = build_remote_publication_operations(
        plan,
        stale_files,
        hub=hub,
        revision=snapshot.revision,
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
        revision=snapshot.revision,
    )


def _plan_with_card_file(
    data_root: Path,
    plan: LanguagePublicationPlan,
    existing_card: str,
) -> LanguagePublicationPlan:
    card_path = write_remote_card_snapshot(data_root, plan, existing_card)
    return replace(plan, files=(*plan.files, _local_file(card_path, REMOTE_README)))


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
        latest_revision = remote_revision(hub, plan.repo_id)
        return _record_no_op(
            data_root,
            plan,
            hub=hub,
            revision=latest_revision,
            stale_files=stale_files,
        )
    verify_remote_publication(
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
    verify_remote_publication(
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


def _plan_from_version_plan(version_plan: LanguageSplitVersionPlan) -> LanguagePublicationPlan:
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
        configuration_languages=_configuration_languages(version_plan.inventory),
    )


def _configuration_languages(
    inventory: LanguageInventory,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return non-empty language buckets grouped by Viewer configuration."""
    grouped: list[tuple[str, tuple[str, ...]]] = []
    for table in getattr(inventory, "tables", ()):
        buckets = getattr(table, "buckets", ())
        languages = tuple(
            sorted(
                {str(bucket.language) for bucket in buckets if getattr(bucket, "row_count", 0) > 0},
                key=_language_sort_key,
            )
        )
        if languages:
            grouped.append((str(table.configuration), languages))
    return tuple(sorted(grouped, key=lambda item: item[0]))


def _plan_from_generated(
    version_plan: LanguageSplitVersionPlan, generated: _GeneratedRelease
) -> LanguagePublicationPlan:
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
        configuration_languages=_configuration_languages(version_plan.inventory),
    )


def _generated_local_path(
    version_plan: LanguageSplitVersionPlan, generated_file: _GeneratedFile
) -> Path:
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


def _write_report(data_root: Path, report: LanguagePublicationReport) -> None:
    path = data_root / "cache" / REMOTE_CACHE_DIR / report.version.value / "release-report.json"
    atomic_write_text(path, json_dumps(report.to_payload()) + "\n")


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
    try:
        return select_language_split_versions(dataset_version)
    except ValueError as error:
        raise LanguagePublicationError(str(error)) from error


def _validate_confirmations(
    versions: Sequence[LanguageSplitVersion],
    confirm_repos: Sequence[str],
) -> None:
    expected = [_repo_for_version(version) for version in versions]
    if sorted(confirm_repos) != sorted(expected):
        raise LanguagePublicationError(
            "exact --confirm-repo values are required: " + ", ".join(expected)
        )


__all__ = [
    "LANGUAGE_CARD_HEADING",
    "LANGUAGE_PUBLICATION_COMMIT_MESSAGE",
    "LanguagePublicationError",
    "LanguagePublicationPlan",
    "LanguagePublicationReport",
    "LanguagePublicationResult",
    "LanguagePublishedFile",
    "plan_language_split_publication",
    "run_language_split_publication",
]
