from __future__ import annotations

import hashlib
from pathlib import Path

from osm_polygon_wikidata_only.io.hashing import sha256_file


def test_sha256_file_matches_standard_digest_for_empty_and_nonempty_files(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.bin"
    payload = tmp_path / "payload.bin"
    empty.write_bytes(b"")
    payload.write_bytes(b"a" * (2 * 1024 * 1024 + 17))

    assert sha256_file(empty) == hashlib.sha256(b"").hexdigest()
    assert sha256_file(payload) == hashlib.sha256(payload.read_bytes()).hexdigest()
