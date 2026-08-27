"""Pure contracts shared by the Grid5000 sentence job and local controller."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

GRID5000_SENTENCE_CONTRACT_VERSION = "grid5000-sentence-v1"
DEFAULT_MAX_STEMS = 4
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024
DEFAULT_WALLTIME = "0:30"

_RUN_ID_PATTERN = re.compile(r"[a-z0-9_-]+")


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


def _validate_limits(*, max_stems: int, max_input_bytes: int) -> None:
    if max_stems < 1:
        raise ValueError("max_stems must be positive")
    if max_input_bytes < 1:
        raise ValueError("max_input_bytes must be positive")


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
