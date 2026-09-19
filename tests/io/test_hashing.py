from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.io import hashing
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


def test_digest_is_reused_while_the_file_is_untouched(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"first")
    reads = 0
    original = Path.open

    def counting_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal reads
        if self == payload:
            reads += 1
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    hashing._CACHE.clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", counting_open)
        first = sha256_file(payload)
        second = sha256_file(payload)

    assert first == second
    assert reads == 1, "an unchanged file must not be re-read"


def test_rewriting_the_file_invalidates_the_cached_digest(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"first")
    hashing._CACHE.clear()
    before = sha256_file(payload)

    payload.write_bytes(b"second")
    # A rewrite changes size and mtime, so the fingerprint no longer matches.
    after = sha256_file(payload)

    assert before == hashlib.sha256(b"first").hexdigest()
    assert after == hashlib.sha256(b"second").hexdigest()


def test_persisted_digests_are_restored_and_revalidated(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"persisted")
    hashing._CACHE.clear()
    hashing.enable_hash_cache(cache_dir)
    expected = sha256_file(payload)
    hashing.flush_hash_cache()

    hashing._CACHE.clear()
    hashing.enable_hash_cache(cache_dir)

    assert hashing._CACHE, "the persisted index should be restored"
    assert sha256_file(payload) == expected


def test_a_corrupt_persisted_index_is_ignored(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "hash_cache.json").write_text("{not json", encoding="utf-8")
    hashing._CACHE.clear()

    hashing.enable_hash_cache(cache_dir)

    assert hashing._CACHE == {}
