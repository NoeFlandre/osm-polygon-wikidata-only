from pathlib import Path

from osm_polygon_wikidata_only.v2.resume import V2FileHashCache
from osm_polygon_wikidata_only.v2.runner import _region_is_current
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest, write_v2_region


def test_region_current_check_reuses_persisted_file_digests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    entry = load_v2_manifest(tmp_path)["region-latest"]
    cache_path = tmp_path / "cache" / "resume-hashes.json"
    cache = V2FileHashCache(cache_path)

    import osm_polygon_wikidata_only.v2.resume as resume

    calls = 0
    original = resume._sha256

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(resume, "_sha256", counted)
    assert _region_is_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=cache,
    )
    cache.flush()
    restored = V2FileHashCache(cache_path)
    assert _region_is_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=restored,
    )
    assert calls == 4


def test_region_current_check_rehashes_changed_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    entry = load_v2_manifest(tmp_path)["region-latest"]
    cache = V2FileHashCache(tmp_path / "cache" / "resume-hashes.json")
    assert _region_is_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=cache,
    )
    changed = tmp_path / "polygons" / "region-latest.parquet"
    changed.write_bytes(changed.read_bytes() + b"changed")
    assert not _region_is_current(
        tmp_path,
        "region-latest",
        {"region-latest": entry},
        hash_cache=cache,
    )
