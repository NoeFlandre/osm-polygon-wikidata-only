"""Validate and persist restart markers for completed V2 language split tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

from osm_polygon_wikidata_only.hf.language_splits import LanguageTableInventory, LanguageTableSpec
from osm_polygon_wikidata_only.io.atomic import atomic_write_json
from osm_polygon_wikidata_only.io.hashing import sha256_file_uncached
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.language_split_models import V2LanguageSplitFile


def _resume_marker_path(stage_root: Path, spec: LanguageTableSpec) -> Path:
    return stage_root / ".resume" / f"{spec.table.value}.json"


def table_fingerprint(
    source_root: Path,
    table_inventory: LanguageTableInventory,
    max_rows_per_shard: int,
) -> str:
    """Bind a staged table to the exact source bytes and shard layout that produce it.

    Source files are hashed without the stat-keyed digest cache: an edit that
    keeps size and mtime (same-tick rewrite, restored mtime) must still
    invalidate the staged shards. Callers compute this before staging, so a
    source edited mid-run cannot be recorded against shards built from its
    earlier bytes.
    """
    payload = {
        "inventory": table_inventory.to_dict(),
        "max_rows_per_shard": max_rows_per_shard,
        "source_sha256": {
            name: sha256_file_uncached(source_root / name) for name in table_inventory.source_files
        },
    }
    return hashlib.sha256(json_dumps(payload).encode()).hexdigest()


def record_completed_table(
    stage_root: Path,
    spec: LanguageTableSpec,
    fingerprint: str,
    files: list[V2LanguageSplitFile],
    staged_paths: dict[Path, Path],
) -> None:
    """Record a finished table so an interrupted run does not rebuild it."""
    atomic_write_json(
        _resume_marker_path(stage_root, spec),
        {
            "fingerprint": fingerprint,
            "files": [file.to_dict() for file in files],
            "staged_paths": {
                str(final): str(staged) for final, staged in sorted(staged_paths.items())
            },
        },
    )


def resume_completed_table(
    stage_root: Path,
    spec: LanguageTableSpec,
    fingerprint: str,
) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]] | None:
    """Return a previously staged table, or ``None`` when it must be rebuilt.

    The marker is honoured only when it was written from exactly this
    fingerprint (source bytes and shard size) and every staged file it names is
    still on disk.
    """
    payload = _resume_payload(stage_root, spec)
    if payload is None:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    staged_paths = _resume_staged_paths(payload.get("staged_paths"))
    files = _resume_files(payload.get("files"), spec)
    if staged_paths is None or files is None:
        return None
    return files, staged_paths


def _resume_payload(stage_root: Path, spec: LanguageTableSpec) -> dict[str, object] | None:
    """Read a resume marker, or ``None`` when it is absent or unreadable."""
    marker = _resume_marker_path(stage_root, spec)
    if not marker.is_file():
        return None
    try:
        payload = json_loads(marker.read_bytes())
    except (OSError, UnicodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _resume_staged_paths(raw: object) -> dict[Path, Path] | None:
    """Rebuild the staged-path map, rejecting it if any staged file is gone."""
    if not isinstance(raw, dict):
        return None
    staged_paths: dict[Path, Path] = {}
    for final, staged in raw.items():
        entry = _resume_staged_entry(final, staged)
        if entry is None:
            return None
        staged_paths[entry[0]] = entry[1]
    return staged_paths


def _resume_staged_entry(final: object, staged: object) -> tuple[Path, Path] | None:
    """Validate one final/staged pair, requiring the staged file to exist."""
    if not isinstance(final, str) or not isinstance(staged, str):
        return None
    staged_path = Path(staged)
    if not staged_path.is_file():
        return None
    return Path(final), staged_path


def _resume_files(raw: object, spec: LanguageTableSpec) -> list[V2LanguageSplitFile] | None:
    """Rebuild every shard record a completed table published."""
    if not isinstance(raw, list):
        return None
    files: list[V2LanguageSplitFile] = []
    for entry in raw:
        file = _resume_file(entry, spec)
        if file is None:
            return None
        files.append(file)
    return files


def _resume_file(entry: object, spec: LanguageTableSpec) -> V2LanguageSplitFile | None:
    """Rebuild one shard record, rejecting anything not shaped as written."""
    if not isinstance(entry, dict):
        return None
    values = {str(key): value for key, value in entry.items()}
    rows = _resume_file_rows(values)
    texts = _resume_file_texts(values)
    if rows is None or texts is None:
        return None
    sources, row_count = rows
    configuration, language, split, path, sha256 = texts
    return V2LanguageSplitFile(
        table=spec.table,
        configuration=configuration,
        language=language,
        split=split,
        source_files=sources,
        path=path,
        row_count=row_count,
        sha256=sha256,
    )


def _resume_file_rows(values: dict[str, object]) -> tuple[tuple[str, ...], int] | None:
    """Read the source-file list and row count a shard record must carry."""
    sources = values.get("source_files")
    row_count = values.get("row_count")
    if not isinstance(sources, list) or not isinstance(row_count, int):
        return None
    return tuple(str(name) for name in sources), row_count


def _resume_file_texts(values: dict[str, object]) -> tuple[str, str, str, str, str] | None:
    """Read the five text fields a shard record must carry."""
    names = ("configuration", "language", "split", "path", "sha256")
    texts = [values.get(name) for name in names]
    if not all(isinstance(text, str) for text in texts):
        return None
    configuration, language, split, path, sha256 = (str(text) for text in texts)
    return configuration, language, split, path, sha256
