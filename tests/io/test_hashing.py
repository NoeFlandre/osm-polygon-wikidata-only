from __future__ import annotations

import hashlib
import logging
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


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        "[]",
        '{"contract": "other", "entries": {"a": [1, 2, 3, "d"]}}',
        '{"contract": "sha256-v1", "entries": []}',
    ],
    ids=["unparseable", "not-an-object", "wrong-contract", "entries-not-an-object"],
)
def test_an_unusable_persisted_index_is_ignored(tmp_path: Path, payload: str) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "hash_cache.json").write_text(payload, encoding="utf-8")
    hashing._CACHE.clear()

    hashing.enable_hash_cache(cache_dir)

    assert hashing._CACHE == {}


def test_a_missing_persisted_index_is_ignored(tmp_path: Path) -> None:
    hashing._CACHE.clear()

    hashing.enable_hash_cache(tmp_path / "absent")

    assert hashing._CACHE == {}


@pytest.mark.parametrize(
    "entry",
    [
        "not-a-list",
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4],
        ["x", 2, 3, "digest"],
        [1, "x", 3, "digest"],
        [1, 2, "x", "digest"],
    ],
    ids=["not-a-list", "too-short", "too-long", "digest-not-a-string", "size", "inode", "mtime"],
)
def test_a_malformed_entry_is_dropped(entry: object) -> None:
    assert hashing._restored_entry(entry) is None


def test_flushing_without_an_enabled_cache_writes_nothing(tmp_path: Path) -> None:
    hashing._CACHE_DIR = None

    hashing.flush_hash_cache()

    assert list(tmp_path.iterdir()) == []


def test_a_file_that_cannot_be_stat_ed_is_still_hashed(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"content")
    hashing._CACHE.clear()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "stat", _raise_os_error)
        digest = sha256_file(payload)

    assert digest == hashlib.sha256(b"content").hexdigest()
    assert hashing._CACHE == {}, "an unfingerprintable file must not be cached"


def _raise_os_error(*args: object, **kwargs: object) -> None:
    raise OSError("stat unavailable")


def test_a_failed_flush_is_reported_without_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hashing._CACHE.clear()
    hashing._CACHE_DIR = tmp_path / "cache"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hashing, "atomic_write_text", _raise_os_error)
        with caplog.at_level(logging.WARNING):
            hashing.flush_hash_cache()

    assert "Could not persist the hash cache" in caplog.text


def test_a_restored_entry_whose_file_changed_is_rehashed(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"before")
    hashing._CACHE.clear()
    hashing.enable_hash_cache(cache_dir)
    sha256_file(payload)
    hashing.flush_hash_cache()

    payload.write_bytes(b"after")
    hashing._CACHE.clear()
    hashing.enable_hash_cache(cache_dir)

    assert sha256_file(payload) == hashlib.sha256(b"after").hexdigest()


def test_a_malformed_entry_does_not_discard_the_valid_ones(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "hash_cache.json").write_text(
        '{"contract": "sha256-v1", "entries": {"/bad": "malformed", "/good": [1, 2, 3, "digest"]}}',
        encoding="utf-8",
    )
    hashing._CACHE.clear()

    hashing.enable_hash_cache(cache_dir)

    assert hashing._CACHE == {"/good": ((1, 2, 3), "digest")}
