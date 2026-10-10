"""Shared file-stat fingerprints used by V2 restart, cache, and index contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileStatFingerprint:
    """Capture stable filesystem metadata without hashing file contents.

    Callers deliberately choose their serialized view with one of the adapter
    methods below.  The adapters preserve the historical contracts of the
    extraction checkpoint, resume hash cache, persistent V1 index, and the
    content-digest memo in :mod:`osm_polygon_wikidata_only.io.hashing`.
    """

    size: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int
    birthtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> FileStatFingerprint:
        """Read the metadata needed by all V2 fingerprint consumers once."""
        stat = Path(path).stat()
        return cls(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
            inode=stat.st_ino,
            device=stat.st_dev,
            birthtime_ns=int(getattr(stat, "st_birthtime_ns", 0)),
        )

    def checkpoint(self, name: str) -> dict[str, int | str]:
        """Return the extraction-checkpoint identity shape."""
        return {
            "name": name,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "inode": self.inode,
        }

    def resume(self) -> dict[str, int]:
        """Return the V2 content-hash-cache fingerprint shape."""
        return {
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "inode": self.inode,
            "device": self.device,
            "birthtime_ns": self.birthtime_ns,
        }

    def index_tuple(self) -> tuple[int, int, int, int]:
        """Return the persistent V1 index file-state shape."""
        return self.size, self.mtime_ns, self.ctime_ns, self.inode

    def hash_cache(self) -> tuple[int, int, int]:
        """Return the digest-memo key: size, inode, and mtime.

        This shape is persisted in ``hash_cache.json``, so its order and
        contents must not change.
        """
        return self.size, self.inode, self.mtime_ns


__all__ = ["FileStatFingerprint"]
