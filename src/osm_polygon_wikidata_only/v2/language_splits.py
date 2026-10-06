"""Bounded, deterministic row-level language partitions for V2 artifacts.

The generator consumes the validated V2 inventory from the shared language
contract, then streams complete Parquet rows through one source file at a
time. It intentionally does not inspect or route the polygon table: a row's
own ``language`` value is the only partition key.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import cast

from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageInventory,
    LanguageTableInventory,
    LanguageTableSpec,
    build_language_inventory,
    language_table_specs,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_json
from osm_polygon_wikidata_only.v2.language_split_manifest import (
    file_sort_key as _file_sort_key,
)
from osm_polygon_wikidata_only.v2.language_split_manifest import (
    install_staged_files as _install_staged_files,
)
from osm_polygon_wikidata_only.v2.language_split_manifest import (
    manifest_payload as _manifest_payload,
)
from osm_polygon_wikidata_only.v2.language_split_manifest import (
    validate_conservation as _validate_conservation,
)
from osm_polygon_wikidata_only.v2.language_split_models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ROWS_PER_SHARD,
    LANGUAGE_SPLITS_DIRNAME,
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    V2_LANGUAGE_SPLIT_CONTRACT_VERSION,
    V2LanguageSplitError,
    V2LanguageSplitFile,
    V2LanguageSplitResult,
)
from osm_polygon_wikidata_only.v2.language_split_resume import (
    record_completed_table as _record_completed_table,
)
from osm_polygon_wikidata_only.v2.language_split_resume import (
    resume_completed_table as _resume_completed_table,
)
from osm_polygon_wikidata_only.v2.language_split_writer import (
    TableWriteContext,
)
from osm_polygon_wikidata_only.v2.language_split_writer import (
    write_table as _write_table,
)

LOGGER = logging.getLogger(__name__)


def build_v2_language_splits(
    processed_root: Path,
    *,
    output_root: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rows_per_shard: int = DEFAULT_MAX_ROWS_PER_SHARD,
) -> V2LanguageSplitResult:
    """Build V2 language partitions from validated local Parquet artifacts.

    Every source file is read in bounded record batches and every complete
    source row is written exactly once. Source files and their rows retain
    their deterministic order; each normalized language is written to
    deterministic bounded ``part-*`` shards. The output manifest is published
    last.
    """
    root, destination, inventory = _prepare_v2_split_request(
        processed_root, output_root, batch_size, max_rows_per_shard
    )
    manifest_path = root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
    inventory, ordered_files = _stage_and_install_v2_release(
        root, destination, inventory, batch_size, max_rows_per_shard, manifest_path
    )

    return V2LanguageSplitResult(
        processed_root=root,
        output_root=destination,
        manifest_path=manifest_path,
        inventory=inventory,
        files=ordered_files,
    )


def _prepare_v2_split_request(
    processed_root: Path,
    output_root: Path | None,
    batch_size: int,
    max_rows_per_shard: int,
) -> tuple[Path, Path, LanguageInventory]:
    _validate_batch_size(batch_size)
    _validate_max_rows_per_shard(max_rows_per_shard)
    root = Path(processed_root).resolve()
    destination = (
        Path(output_root).resolve() if output_root is not None else root / LANGUAGE_SPLITS_DIRNAME
    )
    _ensure_output_is_under_root(destination, root)
    if destination.exists() and not destination.is_dir():
        raise V2LanguageSplitError(f"V2 language split output is not a directory: {destination}")
    inventory = build_language_inventory(root, DatasetContract.V2)
    _ensure_output_does_not_overlap_source(destination, root, inventory)
    return root, destination, inventory


def _stage_and_install_v2_release(
    root: Path,
    destination: Path,
    inventory: LanguageInventory,
    batch_size: int,
    max_rows_per_shard: int,
    manifest_path: Path,
) -> tuple[LanguageInventory, tuple[V2LanguageSplitFile, ...]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A deterministic staging root lets an interrupted run resume: tables that
    # already finished are recorded under ``.resume`` and are not rebuilt.
    stage_root = destination.parent / f".{destination.name}-staging"
    stage_root.mkdir(exist_ok=True)
    try:
        files, staged_paths = _stage_v2_files(
            root, destination, stage_root, inventory, batch_size, max_rows_per_shard
        )
        ordered_files = tuple(sorted(files, key=_file_sort_key))
        actual_inventory = _verify_source_inventory(root, inventory)
        _validate_conservation(actual_inventory, ordered_files)
        manifest_stage = stage_root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
        atomic_write_json(
            manifest_stage, _manifest_payload(root, destination, actual_inventory, ordered_files)
        )
        staged_paths[manifest_path] = manifest_stage
        _install_staged_files(root, destination, staged_paths)
    except BaseException:
        # Keep the staging tree so the next run resumes instead of restarting.
        raise
    else:
        shutil.rmtree(stage_root, ignore_errors=True)
        return actual_inventory, ordered_files


def _verify_source_inventory(root: Path, expected: LanguageInventory) -> LanguageInventory:
    actual = build_language_inventory(root, DatasetContract.V2)
    if actual.artifact_fingerprint != expected.artifact_fingerprint:
        raise V2LanguageSplitError(
            "V2 source artifact fingerprint changed during generation: "
            f"expected={expected.artifact_fingerprint}, actual={actual.artifact_fingerprint}"
        )
    return actual


def _stage_v2_files(
    root: Path,
    destination: Path,
    stage_root: Path,
    inventory: LanguageInventory,
    batch_size: int,
    max_rows_per_shard: int,
) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
    files: list[V2LanguageSplitFile] = []
    staged_paths: dict[Path, Path] = {}
    for spec in language_table_specs(DatasetContract.V2):
        table_inventory = cast(LanguageTableInventory, inventory.table(spec.table))
        if not isinstance(table_inventory, LanguageTableInventory):
            raise V2LanguageSplitError(
                f"V2 inventory entry is not a language table: {spec.table.value}"
            )
        generated, table_staged_paths = _staged_table(
            root,
            destination,
            stage_root,
            spec,
            table_inventory,
            batch_size,
            max_rows_per_shard,
        )
        files.extend(generated)
        staged_paths.update(table_staged_paths)
    return files, staged_paths


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")


def _validate_max_rows_per_shard(max_rows_per_shard: int) -> None:
    if max_rows_per_shard < 1:
        raise ValueError(f"max_rows_per_shard must be positive, got {max_rows_per_shard}")


def _ensure_output_is_under_root(destination: Path, root: Path) -> None:
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError(f"V2 language split output must be under {root}: {destination}") from error


def _ensure_output_does_not_overlap_source(
    destination: Path,
    root: Path,
    inventory: LanguageInventory,
) -> None:
    source_paths = [root / inventory.source_manifest]
    source_paths.extend(
        root / source_file for table in inventory.tables for source_file in table.source_files
    )
    for source_path in source_paths:
        try:
            source_path.resolve().relative_to(destination)
        except ValueError:
            continue
        raise V2LanguageSplitError(
            "V2 language split output must not overlap source artifacts: "
            f"{destination} contains {source_path.resolve()}"
        )


def _staged_table(
    root: Path,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    table_inventory: LanguageTableInventory,
    batch_size: int,
    max_rows_per_shard: int,
) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
    """Stage one table, reusing a completed staging run when one is present."""
    completed = _resume_completed_table(stage_root, spec, table_inventory)
    if completed is not None:
        LOGGER.info(
            "Resuming: reusing %d staged shards for %s", len(completed[0]), spec.table.value
        )
        return completed
    context = TableWriteContext.for_inventory(
        destination, stage_root, spec, table_inventory, max_rows_per_shard
    )
    generated, staged_paths = _write_table(root, context, table_inventory, batch_size)
    _record_completed_table(stage_root, spec, table_inventory, generated, staged_paths)
    return generated, staged_paths


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_ROWS_PER_SHARD",
    "LANGUAGE_SPLITS_DIRNAME",
    "LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH",
    "V2_LANGUAGE_SPLIT_CONTRACT_VERSION",
    "V2LanguageSplitError",
    "V2LanguageSplitFile",
    "V2LanguageSplitResult",
    "build_v2_language_splits",
]
