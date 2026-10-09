"""Filesystem helpers for the upload queue's durable state directory."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.json_files import read_json

QUEUE_CONTRACT_VERSION = "bg-upload-v1"

SEQUENCE_FILE = re.compile(r"^(\d+)\.json$")
HIGHWATER_FILENAME = ".highwater"
SNAPSHOTS_SUBDIR = "snapshots"


def read_highwater(state_dir: Path) -> int:
    """Read the allocated sequence high-water mark."""
    path = state_dir / HIGHWATER_FILENAME
    if not path.is_file():
        return 0
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0
    return max(value, 0)


def write_highwater(state_dir: Path, value: int) -> None:
    """Persist the allocated sequence high-water mark atomically."""
    atomic_write_text(state_dir / HIGHWATER_FILENAME, f"{int(value)}\n")


def next_sequence_from_state_dir(state_dir: Path) -> int:
    """Return the next sequence after persisted state."""
    if not state_dir.is_dir():
        return 1
    return max(read_highwater(state_dir), scanned_sequence(state_dir)) + 1


def scanned_sequence(state_dir: Path) -> int:
    highest = 0
    for path in state_dir.glob("*.json"):
        highest = max(highest, sequence_from_state_path(path))
    return highest


def sequence_from_state_path(path: Path) -> int:
    match = SEQUENCE_FILE.match(path.name)
    if match is not None:
        return int(match.group(1))
    envelope = read_legacy_or_current_envelope(path)
    sequence = envelope.get("sequence") if envelope is not None else None
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        return 0
    return max(sequence, 0)


def independent_copy(source: Path, target: Path) -> None:
    """Copy a source file without sharing its inode."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` for malformed input."""
    raw = read_json(path)
    return raw if isinstance(raw, dict) else None


def read_legacy_or_current_envelope(path: Path) -> dict[str, Any] | None:
    """Read either the current or legacy envelope shape."""
    return read_json_object(path)


def required_string(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Envelope operation field {key!r} must be a string")
    return value


def current_sequence(payload: dict[str, Any], path: Path) -> int:
    raw_sequence = payload.get("sequence")
    if not isinstance(raw_sequence, int) or isinstance(raw_sequence, bool) or raw_sequence < 1:
        raise ValueError(f"Current envelope {path} has an invalid sequence")
    return raw_sequence


def is_legacy_envelope(payload: dict[str, Any]) -> bool:
    if "contract_version" in payload:
        return False
    return isinstance(payload.get("message"), str) and isinstance(payload.get("ops"), list)


def snapshot_dir_for_sequence(state_dir: Path, sequence: int) -> Path:
    return state_dir / SNAPSHOTS_SUBDIR / f"{sequence:06d}"


def is_inside(child: Path, parent: Path) -> bool:
    """Return whether a resolved path is inside or equal to its parent."""
    try:
        child_resolved = child.resolve(strict=False)
        parent_resolved = parent.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    if child_resolved == parent_resolved:
        return True
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError:
        return False
    return True


def remove_failed_upgrade(state_dir: Path, sequence: int, snapshot_dir: Path) -> None:
    """Remove an upgraded duplicate that cannot be queued."""
    (state_dir / f"{sequence:06d}.json").unlink(missing_ok=True)
    if snapshot_dir.is_dir() and is_inside(snapshot_dir, state_dir / SNAPSHOTS_SUBDIR):
        shutil.rmtree(snapshot_dir, ignore_errors=True)
