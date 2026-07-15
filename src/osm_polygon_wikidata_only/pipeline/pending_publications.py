"""Durable publication intent for locally migrated Wikipedia documents."""

from __future__ import annotations

import json
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_text

CONTRACT_VERSION = "pending-publications-v1"
FILENAME = "pending_migration_publications.json"


def _validate_stem(stem: str) -> str:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid pending publication stem: {stem!r}")
    return stem


def _manifest_path(data_root: DataRoot) -> Path:
    return data_root.processed_manifests / FILENAME


def load_pending_publications(data_root: DataRoot) -> set[str]:
    """Load pending publication stems from the durable manifest in processed manifests."""
    path = _manifest_path(data_root)
    if not path.exists():
        return set()

    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed pending publication manifest: {exc}") from exc

    if not isinstance(data, dict):
        raise TypeError("Pending publication manifest must be a JSON object")

    version = data.get("contract_version")
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"Invalid contract version: expected {CONTRACT_VERSION!r}, got {version!r}"
        )

    stems = data.get("stems")
    if stems is None:
        raise TypeError("Pending publication manifest is missing 'stems' field")
    if not isinstance(stems, list):
        raise TypeError("Pending publication manifest 'stems' field must be a list")

    for idx, stem in enumerate(stems):
        if not isinstance(stem, str):
            raise TypeError(f"Pending publication stem at index {idx} is not a string: {stem!r}")
        _validate_stem(stem)
    if len(stems) != len(set(stems)):
        raise ValueError("Pending publication manifest contains duplicate stems")

    return set(stems)


def save_pending_publications(data_root: DataRoot, stems: set[str]) -> None:
    """Save pending publication stems atomically to the durable manifest."""
    path = _manifest_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = {_validate_stem(stem) for stem in stems}
    payload = {
        "contract_version": CONTRACT_VERSION,
        "stems": sorted(validated),
    }
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def add_pending_publications(data_root: DataRoot, stems: set[str]) -> None:
    """Add stems to the pending publications list."""
    if not stems:
        return
    current = load_pending_publications(data_root)
    save_pending_publications(data_root, current | stems)


def remove_pending_publications(data_root: DataRoot, stems: set[str]) -> None:
    """Remove stems from the pending publications list."""
    if not stems:
        return
    current = load_pending_publications(data_root)
    save_pending_publications(data_root, current - stems)
