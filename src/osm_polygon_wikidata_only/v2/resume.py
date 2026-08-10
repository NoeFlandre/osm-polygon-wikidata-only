"""Small local caches that make V2 restart checks cheap and restart-safe."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.fingerprints import FileStatFingerprint

_CACHE_CONTRACT_VERSION = "v2-resume-file-hashes-v1"
_HASH_LENGTH = 64


def _fingerprint(path: Path) -> dict[str, int]:
    return FileStatFingerprint.from_path(path).resume()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        files = raw.get("files")
        if not isinstance(files, dict):
            return {}
        entries: dict[str, dict[str, Any]] = {}
        for key, value in files.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            digest = value.get("sha256")
            fingerprint = value.get("fingerprint")
            if (
                isinstance(digest, str)
                and len(digest) == _HASH_LENGTH
                and all(character in "0123456789abcdef" for character in digest)
                and isinstance(fingerprint, dict)
            ):
                entries[key] = {"fingerprint": fingerprint, "sha256": digest}
        return entries

    def digest(self, path: Path) -> str:
        """Return the exact digest, reusing it only for the same fingerprint."""
        resolved = str(Path(path).resolve())
        fingerprint = _fingerprint(Path(path))
        cached = self._entries.get(resolved)
        if cached is not None and cached.get("fingerprint") == fingerprint:
            return str(cached["sha256"])
        digest = _sha256(Path(path))
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
