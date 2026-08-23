"""Small local caches that make V2 restart checks cheap and restart-safe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.fingerprints import FileStatFingerprint

_CACHE_CONTRACT_VERSION = "v2-resume-file-hashes-v1"
_HASH_LENGTH = 64


def _fingerprint(path: Path) -> dict[str, int]:
    return FileStatFingerprint.from_path(path).resume()


def _is_valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HASH_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_cache_entry(value: object) -> dict[str, Any] | None:
    """Normalize one persisted digest entry or reject it."""
    if not isinstance(value, dict):
        return None
    digest = value.get("sha256")
    fingerprint = value.get("fingerprint")
    if not _is_valid_digest(digest) or not isinstance(fingerprint, dict):
        return None
    return {"fingerprint": fingerprint, "sha256": digest}


def _load_cache_entries(files: object) -> dict[str, dict[str, Any]]:
    """Keep only valid entries from the persisted file collection."""
    if not isinstance(files, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for key, value in files.items():
        if not isinstance(key, str):
            continue
        entry = _valid_cache_entry(value)
        if entry is not None:
            entries[key] = entry
    return entries


class V2FileHashCache:
    """Persist file digests keyed by a strong local file fingerprint.

    V2 manifests already contain exact content hashes.  The cache only avoids
    rereading a file when its size, timestamps, inode, device, and birth time
    are unchanged.  A changed fingerprint always triggers a fresh hash, and a
    corrupt or missing cache is treated as empty, so the cache cannot make a
    region publishable by itself.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = self._load()
        self._dirty = False

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json_loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return {}
        if not isinstance(raw, dict) or raw.get("contract_version") != _CACHE_CONTRACT_VERSION:
            return {}
        return _load_cache_entries(raw.get("files"))

    def digest(self, path: Path) -> str:
        """Return the exact digest, reusing it only for the same fingerprint."""
        resolved = str(Path(path).resolve())
        fingerprint = _fingerprint(Path(path))
        cached = self._entries.get(resolved)
        if cached is not None and cached.get("fingerprint") == fingerprint:
            return str(cached["sha256"])
        digest = sha256_file(Path(path))
        self._entries[resolved] = {"fingerprint": fingerprint, "sha256": digest}
        self._dirty = True
        return digest

    def flush(self) -> None:
        """Atomically persist newly observed fingerprints and digests."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.path,
            json_dumps(
                {
                    "contract_version": _CACHE_CONTRACT_VERSION,
                    "files": dict(sorted(self._entries.items())),
                }
            )
            + "\n",
        )
        self._dirty = False


__all__ = ["V2FileHashCache"]
