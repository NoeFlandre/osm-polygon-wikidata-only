from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.v2.fingerprints import FileStatFingerprint


def test_file_stat_fingerprint_captures_required_metadata(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    path.write_bytes(b"stable artifact")
    stat = path.stat()

    fingerprint = FileStatFingerprint.from_path(path)

    assert fingerprint.size == stat.st_size
    assert fingerprint.mtime_ns == stat.st_mtime_ns
    assert fingerprint.ctime_ns == stat.st_ctime_ns
    assert fingerprint.inode == stat.st_ino
    assert fingerprint.device == stat.st_dev
    assert fingerprint.birthtime_ns == int(getattr(stat, "st_birthtime_ns", 0))


def test_file_stat_fingerprint_preserves_checkpoint_contract(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    path.write_bytes(b"stable artifact")

    assert FileStatFingerprint.from_path(path).checkpoint(path.name) == {
        "name": path.name,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "ctime_ns": path.stat().st_ctime_ns,
        "inode": path.stat().st_ino,
    }


def test_file_stat_fingerprint_preserves_resume_contract(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    path.write_bytes(b"stable artifact")
    stat = path.stat()

    assert FileStatFingerprint.from_path(path).resume() == {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
        "birthtime_ns": int(getattr(stat, "st_birthtime_ns", 0)),
    }


def test_file_stat_fingerprint_preserves_index_contract(tmp_path: Path) -> None:
    path = tmp_path / "artifact.parquet"
    path.write_bytes(b"stable artifact")
    stat = path.stat()

    assert FileStatFingerprint.from_path(path).index_tuple() == (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_ino,
    )
