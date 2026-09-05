import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.v2.resume import V2FileHashCache
from osm_polygon_wikidata_only.v2.runner import _region_artifacts_are_current
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest, write_v2_region


def test_hash_cache_loads_only_current_valid_digest_entries(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "resume-hashes.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "contract_version": "v2-resume-file-hashes-v1",
                "files": {
                    "/valid": {
                        "fingerprint": {"size": 1},
                        "sha256": "a" * 64,
                    },
                    "/bad-digest": {
                        "fingerprint": {"size": 1},
                        "sha256": "not-a-digest",
                    },
                    "/bad-shape": [],
                },
            }
        ),
        encoding="utf-8",
    )

    cache = V2FileHashCache(path)

    assert cache._entries == {"/valid": {"fingerprint": {"size": 1}, "sha256": "a" * 64}}


def test_hash_cache_treats_invalid_file_collection_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "resume-hashes.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"contract_version": "v2-resume-file-hashes-v1", "files": []}))

    assert V2FileHashCache(path)._entries == {}


@pytest.mark.parametrize("payload", ["{}", "not-json"])
def test_hash_cache_treats_invalid_cache_documents_as_empty(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "cache" / "resume-hashes.json"
    path.parent.mkdir()
    path.write_text(payload, encoding="utf-8")

    assert V2FileHashCache(path)._entries == {}


def test_region_artifact_check_reuses_persisted_file_digests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    entry = load_v2_manifest(tmp_path)["region-latest"]
    cache_path = tmp_path / "cache" / "resume-hashes.json"
    cache = V2FileHashCache(cache_path)

    import osm_polygon_wikidata_only.v2.resume as resume

    calls = 0
    original = resume.sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(resume, "sha256_file", counted)
    assert _region_artifacts_are_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=cache,
    )
    assert calls == 4
    cache.flush()
    restored = V2FileHashCache(cache_path)
    assert _region_artifacts_are_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=restored,
    )
    assert calls == 4


def test_region_artifact_check_rehashes_changed_file(tmp_path: Path) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    entry = load_v2_manifest(tmp_path)["region-latest"]
    cache = V2FileHashCache(tmp_path / "cache" / "resume-hashes.json")
    assert _region_artifacts_are_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=cache,
    )
    changed = tmp_path / "polygons" / "region-latest.parquet"
    changed.write_bytes(changed.read_bytes() + b"changed")
    assert not _region_artifacts_are_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=cache,
    )
