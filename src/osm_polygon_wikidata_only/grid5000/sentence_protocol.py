"""Pure contracts shared by the Grid5000 sentence job and local controller."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.sentence_logic import sentence_schema
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

GRID5000_SENTENCE_CONTRACT_VERSION = "grid5000-sentence-v1"
DEFAULT_MAX_STEMS = 4
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024
DEFAULT_WALLTIME = "0:30"

_RUN_ID_PATTERN = re.compile(r"[a-z0-9_-]+")
_CHECKPOINT_BATCH_PATTERN = re.compile(r"batch-(\d{8})\.parquet")
_MANIFEST_INVARIANT_KEYS = (
    "contract_version",
    "segmenter",
    "model_id",
    "model_revision",
    "segmenter_version",
    "supported_languages",
    "unsupported_language_policy",
)


@dataclass(frozen=True, slots=True)
class SentenceBatch:
    """A deterministic group of stems submitted as one GPU job."""

    index: int
    stems: tuple[str, ...]
    input_bytes: int


@dataclass(frozen=True, slots=True)
class FileDigest:
    """A content digest for a file relative to a declared artifact root."""

    relative_path: str
    size: int
    sha256: str


def plan_sentence_batches(
    processed_v2: Path,
    sentence_manifest: Mapping[str, object] | None,
    *,
    max_stems: int = DEFAULT_MAX_STEMS,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> tuple[SentenceBatch, ...]:
    """Plan bounded, sorted batches without reading any Parquet rows."""
    _validate_limits(max_stems=max_stems, max_input_bytes=max_input_bytes)
    finalized_stems = load_v2_manifest(Path(processed_v2))
    if not finalized_stems:
        return ()
    completed = _completed_regions(sentence_manifest)
    pending = _pending_stems(processed_v2, finalized_stems, completed)
    return _pack_batches(pending, max_stems=max_stems, max_input_bytes=max_input_bytes)


def sentence_source_paths(processed_v2: Path, stem: str) -> tuple[Path, ...]:
    """Return the required Wikipedia and any existing Wikivoyage source files."""
    _validate_stem(stem)
    wikipedia = Path(processed_v2) / "wikipedia" / "sections" / f"{stem}.parquet"
    if not wikipedia.is_file():
        raise FileNotFoundError(f"V2 Wikipedia section source is missing: {wikipedia}")
    wikivoyage = Path(processed_v2) / "wikivoyage" / "sections" / f"{stem}.parquet"
    return (wikipedia, wikivoyage) if wikivoyage.is_file() else (wikipedia,)


def is_safe_run_id(value: str) -> bool:
    """Return whether a run identifier is safe to interpolate into a namespace."""
    return isinstance(value, str) and bool(_RUN_ID_PATTERN.fullmatch(value))


def validate_cleanup_target(run_root: Path, target: Path) -> bool:
    """Return whether *target* is a strict descendant of the run namespace."""
    try:
        Path(target).resolve().relative_to(Path(run_root).resolve())
    except ValueError:
        return False
    return Path(target).resolve() != Path(run_root).resolve()


def sha256_manifest(paths: Sequence[Path], *, root: Path) -> tuple[FileDigest, ...]:
    """Hash files in sorted relative-path order using bounded file reads."""
    root = Path(root).resolve()
    records: list[FileDigest] = []
    seen: set[str] = set()
    for path in paths:
        file_path = Path(path)
        relative_path = _relative_file_path(file_path, root)
        if relative_path in seen:
            raise ValueError(f"Duplicate artifact path: {relative_path}")
        seen.add(relative_path)
        records.append(
            FileDigest(
                relative_path=relative_path,
                size=file_path.stat().st_size,
                sha256=sha256_file(file_path),
            )
        )
    return tuple(sorted(records, key=lambda record: record.relative_path))


def validate_manifest_extension(
    local_payload: Mapping[str, object],
    incoming_payload: Mapping[str, object],
    *,
    selected_stems: Sequence[str],
) -> None:
    """Ensure an incoming sentence manifest only appends verified regions."""
    local_regions = _manifest_regions(local_payload, "local")
    incoming_regions = _manifest_regions(incoming_payload, "incoming")
    _validate_manifest_invariants(local_payload, incoming_payload)
    _validate_existing_regions(local_regions, incoming_regions)
    _validate_selected_stems(incoming_regions, selected_stems)


def _validate_existing_regions(
    local_regions: Mapping[tuple[str, str], Mapping[str, object]],
    incoming_regions: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    for identity, region in local_regions.items():
        if identity not in incoming_regions:
            raise ValueError(f"Incoming sentence manifest removed region: {identity}")
        if dict(incoming_regions[identity]) != dict(region):
            raise ValueError(f"Incoming sentence manifest changed region: {identity}")


def _validate_selected_stems(
    incoming_regions: Mapping[tuple[str, str], Mapping[str, object]],
    selected_stems: Sequence[str],
) -> None:
    requested = set(selected_stems)
    if not requested:
        raise ValueError("At least one selected sentence stem is required")
    incoming_stems = {stem for stem, _ in incoming_regions}
    missing = sorted(requested - incoming_stems)
    if missing:
        raise ValueError(f"Incoming sentence manifest is missing selected stems: {missing}")


def validate_sentence_output(path: Path, *, expected_sha256: str) -> None:
    """Validate a sentence Parquet schema and its expected content digest."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Sentence output is missing: {path}")
    try:
        with pq.ParquetFile(path) as parquet_file:
            if not parquet_file.schema_arrow.equals(sentence_schema(), check_metadata=True):
                raise ValueError(f"Sentence output has an unexpected schema: {path}")
    except (OSError, pa.ArrowException) as exc:
        raise ValueError(f"Sentence output has an invalid schema: {path}") from exc
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Sentence output SHA-256 does not match receipt: {path}")


def import_checkpoint_tree(
    incoming_root: Path,
    local_root: Path,
    *,
    expected_identity: Mapping[str, object],
) -> tuple[int, ...]:
    """Validate and atomically import one remote checkpoint directory."""
    incoming_root = Path(incoming_root)
    local_root = Path(local_root)
    metadata_path, batch_paths = _checkpoint_paths(incoming_root)
    metadata = _read_checkpoint_metadata(metadata_path)
    _validate_checkpoint_identity(metadata, expected_identity)
    indexes = _validate_checkpoint_batches(batch_paths)
    _validate_existing_checkpoint(local_root, expected_identity)
    _install_checkpoint_tree(local_root, batch_paths, metadata)
    return indexes


def _validate_limits(*, max_stems: int, max_input_bytes: int) -> None:
    if max_stems < 1:
        raise ValueError("max_stems must be positive")
    if max_input_bytes < 1:
        raise ValueError("max_input_bytes must be positive")


def _manifest_regions(
    payload: Mapping[str, object], label: str
) -> dict[tuple[str, str], Mapping[str, object]]:
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError(f"{label.capitalize()} sentence manifest regions must be a list")
    regions: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw_region in raw_regions:
        identity, region = _manifest_region_identity(raw_region, label)
        if identity in regions:
            raise ValueError(f"{label.capitalize()} sentence manifest has duplicate region")
        regions[identity] = region
    return regions


def _manifest_region_identity(
    raw_region: object, label: str
) -> tuple[tuple[str, str], Mapping[str, object]]:
    if not isinstance(raw_region, Mapping):
        raise ValueError(f"{label.capitalize()} sentence manifest region must be an object")
    region = cast(Mapping[str, object], raw_region)
    stem = region.get("stem")
    project = region.get("project")
    if not isinstance(stem, str) or not isinstance(project, str):
        raise ValueError(
            f"{label.capitalize()} sentence manifest region needs string stem and project"
        )
    return (stem, project), region


def _validate_manifest_invariants(
    local_payload: Mapping[str, object], incoming_payload: Mapping[str, object]
) -> None:
    for key in _MANIFEST_INVARIANT_KEYS:
        if (
            key not in local_payload
            or key not in incoming_payload
            or local_payload[key] != incoming_payload[key]
        ):
            raise ValueError(f"Sentence manifest {key} changed")


def _completed_regions(
    sentence_manifest: Mapping[str, object] | None,
) -> set[tuple[str, str]]:
    if sentence_manifest is None:
        return set()
    raw_regions: object = sentence_manifest.get("regions", [])
    if not isinstance(raw_regions, list):
        raise ValueError("Sentence manifest regions must be a list")
    return {_region_identity(region) for region in raw_regions}


def _region_identity(region: object) -> tuple[str, str]:
    if not isinstance(region, Mapping):
        raise ValueError("Sentence manifest region must be an object")
    stem = region.get("stem")
    project = region.get("project")
    if not isinstance(stem, str) or not isinstance(project, str):
        raise ValueError("Sentence manifest region needs string stem and project")
    return stem, project


def _pending_stems(
    processed_v2: Path,
    finalized_stems: Mapping[str, object],
    completed: set[tuple[str, str]],
) -> list[tuple[str, int]]:
    pending: list[tuple[str, int]] = []
    for stem in sorted(finalized_stems):
        source_paths = sentence_source_paths(processed_v2, stem)
        required_projects = _required_projects(source_paths)
        if _is_complete(stem, required_projects, completed):
            continue
        pending.append((stem, sum(path.stat().st_size for path in source_paths)))
    return pending


def _required_projects(source_paths: Sequence[Path]) -> set[str]:
    return {"wikipedia", "wikivoyage"} if len(source_paths) == 2 else {"wikipedia"}


def _is_complete(
    stem: str,
    required_projects: Collection[str],
    completed: set[tuple[str, str]],
) -> bool:
    return all((stem, project) in completed for project in required_projects)


def _pack_batches(
    pending: Sequence[tuple[str, int]],
    *,
    max_stems: int,
    max_input_bytes: int,
) -> tuple[SentenceBatch, ...]:
    batches: list[SentenceBatch] = []
    current_stems: list[str] = []
    current_bytes = 0
    for stem, input_bytes in pending:
        if _should_flush(
            current_stems,
            current_bytes=current_bytes,
            next_bytes=input_bytes,
            max_stems=max_stems,
            max_input_bytes=max_input_bytes,
        ):
            batches.append(_make_batch(batches, current_stems, current_bytes))
            current_stems = []
            current_bytes = 0
        current_stems.append(stem)
        current_bytes += input_bytes
    if current_stems:
        batches.append(_make_batch(batches, current_stems, current_bytes))
    return tuple(batches)


def _should_flush(
    current_stems: Sequence[str],
    *,
    current_bytes: int,
    next_bytes: int,
    max_stems: int,
    max_input_bytes: int,
) -> bool:
    return bool(
        current_stems
        and (len(current_stems) >= max_stems or current_bytes + next_bytes > max_input_bytes)
    )


def _make_batch(
    batches: Sequence[SentenceBatch],
    stems: Sequence[str],
    input_bytes: int,
) -> SentenceBatch:
    return SentenceBatch(
        index=len(batches),
        stems=tuple(stems),
        input_bytes=input_bytes,
    )


def _relative_file_path(path: Path, root: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact is outside root: {path}") from exc
    return relative.as_posix()


def _checkpoint_paths(root: Path) -> tuple[Path, tuple[Path, ...]]:
    if not root.is_dir():
        raise ValueError(f"Checkpoint root is missing: {root}")
    classified = tuple(
        _classify_checkpoint_path(root, candidate) for candidate in sorted(root.iterdir())
    )
    return _split_checkpoint_paths(classified)


def _split_checkpoint_paths(
    classified: Sequence[tuple[str, Path]],
) -> tuple[Path, tuple[Path, ...]]:
    return _checkpoint_metadata_path(classified), _checkpoint_batch_paths(classified)


def _checkpoint_metadata_path(classified: Sequence[tuple[str, Path]]) -> Path:
    for kind, path in classified:
        if kind == "metadata":
            return path
    raise ValueError("Checkpoint metadata.json is missing")


def _checkpoint_batch_paths(classified: Sequence[tuple[str, Path]]) -> tuple[Path, ...]:
    return tuple(path for kind, path in classified if kind == "batch")


def _classify_checkpoint_path(root: Path, candidate: Path) -> tuple[str, Path]:
    _validate_checkpoint_path(root, candidate)
    if candidate.name == "metadata.json":
        if not candidate.is_file():
            raise ValueError("Checkpoint metadata must be a file")
        return "metadata", candidate
    if not candidate.is_file() or _CHECKPOINT_BATCH_PATTERN.fullmatch(candidate.name) is None:
        raise ValueError(f"Unexpected checkpoint artifact: {candidate.name}")
    return "batch", candidate


def _validate_checkpoint_path(root: Path, candidate: Path) -> None:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Checkpoint artifact is outside root: {candidate}") from exc
    if candidate.is_symlink():
        raise ValueError(f"Checkpoint artifact cannot be a symlink: {candidate}")


def _read_checkpoint_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json_loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint metadata: {path}")
    return cast(dict[str, object], payload)


def _validate_checkpoint_identity(
    metadata: Mapping[str, object], expected_identity: Mapping[str, object]
) -> None:
    identity = metadata.get("identity")
    if not isinstance(identity, Mapping) or dict(identity) != dict(expected_identity):
        raise ValueError("Checkpoint identity does not match expected source/model")


def _validate_checkpoint_batches(batch_paths: Sequence[Path]) -> tuple[int, ...]:
    indexes = tuple(sorted(_checkpoint_batch_index(path) for path in batch_paths))
    if indexes != tuple(range(len(indexes))):
        raise ValueError("Checkpoint batches are not contiguous")
    for path in batch_paths:
        _validate_checkpoint_schema(path)
    return indexes


def _validate_checkpoint_schema(path: Path) -> None:
    try:
        with pq.ParquetFile(path) as parquet_file:
            if not parquet_file.schema_arrow.equals(sentence_schema(), check_metadata=True):
                raise ValueError(f"Checkpoint batch has an unexpected schema: {path}")
    except (OSError, pa.ArrowException) as exc:
        raise ValueError(f"Checkpoint batch has an invalid schema: {path}") from exc


def _checkpoint_batch_index(path: Path) -> int:
    match = _CHECKPOINT_BATCH_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint batch name: {path.name}")
    return int(match.group(1))


def _validate_existing_checkpoint(
    local_root: Path, expected_identity: Mapping[str, object]
) -> None:
    metadata_path = local_root / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = _read_checkpoint_metadata(metadata_path)
    _validate_checkpoint_identity(metadata, expected_identity)


def _install_checkpoint_tree(
    local_root: Path,
    batch_paths: Sequence[Path],
    metadata: Mapping[str, object],
) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    for source in batch_paths:
        _copy_file_atomically(source, local_root / source.name)
    atomic_write_text(local_root / "metadata.json", json_dumps(dict(metadata)) + "\n")


def _copy_file_atomically(source: Path, target: Path) -> None:
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temporary)
    try:
        with source.open("rb") as source_stream, os.fdopen(fd, "wb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_stem(stem: str) -> None:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid sentence stem: {stem!r}")


__all__ = [
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_STEMS",
    "DEFAULT_WALLTIME",
    "GRID5000_SENTENCE_CONTRACT_VERSION",
    "FileDigest",
    "SentenceBatch",
    "is_safe_run_id",
    "plan_sentence_batches",
    "sentence_source_paths",
    "sha256_manifest",
    "validate_cleanup_target",
]
