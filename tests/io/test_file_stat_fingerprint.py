from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.io import fingerprints as io_fingerprints
from osm_polygon_wikidata_only.v2 import fingerprints as v2_fingerprints


def test_v2_fingerprint_module_reexports_the_io_class() -> None:
    assert v2_fingerprints.FileStatFingerprint is io_fingerprints.FileStatFingerprint


def test_hash_cache_view_keeps_size_inode_and_mtime_in_that_order(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    stat = path.stat()

    assert io_fingerprints.FileStatFingerprint.from_path(path).hash_cache() == (
        stat.st_size,
        stat.st_ino,
        stat.st_mtime_ns,
    )
