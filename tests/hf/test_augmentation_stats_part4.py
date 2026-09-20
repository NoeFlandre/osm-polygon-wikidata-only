"""Split coverage tests (part 4)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.hf.augmentation_stats_support import *


def test_cache_index_written_atomically(tmp_path: Path, monkeypatch) -> None:
    """``write_cache_index`` must use the documented atomic_write_text.

    A failed write MUST clean up its sibling temp file. We simulate
    the failure by monkey-patching ``atomic_write_text`` to raise.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod
    from osm_polygon_wikidata_only.io.atomic import atomic_write_text

    cachemod.cache_dir(tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    index_path = cachemod.index_path(tmp_path / "cache")

    # Recreate: succeed path → atomic write creates the final file.
    payload = {"a": {"v": 1}}
    cachemod.write_cache_index(tmp_path / "cache", payload)
    assert index_path.exists()
    cached = cachemod.load_cache_index(tmp_path / "cache")
    assert cached == payload

    # The write must use atomic_write_text, not direct .write_text.
    calls: list[tuple[Path, str]] = []
    real = atomic_write_text

    def tracker(path: Path, text: str, *, encoding: str = "utf-8") -> None:
        calls.append((path, text))
        return real(path, text, encoding=encoding)

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf._dataset_stats.cache.atomic_write_text",
        tracker,
    )
    cachemod.write_cache_index(tmp_path / "cache", {"b": {"v": 2}})
    assert any(call[0] == index_path for call in calls), (
        "write_cache_index must call atomic_write_text on the index path"
    )


def test_cache_index_filters_contract_and_non_mapping_entries() -> None:
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    assert cachemod._cache_entries(
        {
            "__contract_version__": cachemod.CACHE_CONTRACT_VERSION,
            "documents.parquet": {"rows": 2},
            "invalid-list": ["not", "a", "cache"],
            "invalid-null": None,
        }
    ) == {"documents.parquet": {"rows": 2}}


def test_scan_paths_skips_missing_subdirectories_and_sorts_files(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "z.parquet").touch()
    (tmp_path / "documents" / "a.parquet").touch()
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "only.parquet").touch()

    assert [
        path.relative_to(tmp_path).as_posix()
        for path in cachemod._scan_paths(tmp_path, ["missing", "sections", "documents"])
    ] == [
        "documents/a.parquet",
        "documents/z.parquet",
        "sections/only.parquet",
    ]


def test_cache_write_cleanups_sibling_on_interruption(tmp_path: Path, monkeypatch) -> None:
    """A failed replacement preserves the old index and removes the written temporary."""
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod
    from osm_polygon_wikidata_only.io import atomic

    cachemod.write_cache_index(tmp_path / "cache", {"old": {"rows": 1}})
    index_path = cachemod.index_path(tmp_path / "cache")
    original = index_path.read_bytes()

    def explode(source: Path, target: Path) -> None:
        assert source.read_bytes() != original
        assert target == index_path
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(atomic.os, "replace", explode)
    with pytest.raises(RuntimeError, match="simulated crash mid-write"):
        cachemod.write_cache_index(tmp_path / "cache", {"new": {"rows": 2}})
    assert list(index_path.parent.iterdir()) == [index_path]
    assert index_path.read_bytes() == original


def test_cache_index_has_contract_version(tmp_path: Path) -> None:
    """The cache index must declare an explicit contract version.

    Future changes to summary fields or counting rules bump the
    version. Loading a stale or missing version REBUILDS the cache
    instead of producing wrong numbers.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    cachemod.cache_dir(tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    cachemod.write_cache_index(tmp_path / "cache", {"a": {"v": 1}})
    raw = json.loads(cachemod.index_path(tmp_path / "cache").read_text())
    assert "__contract_version__" in raw


def test_cache_load_rejects_missing_version_and_rebuilds(tmp_path: Path) -> None:
    """An index whose contract version is unknown must trigger a full
    rebuild.

    We construct an index that LOOKS fingerprint-compatible with the
    live file but whose ``__contract_version__`` is missing. A correct
    implementation MUST treat the cache as incompatible and rescan
    from scratch.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    processed = _setup_processed_dir(tmp_path)
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    _write_documents(
        docs_path,
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )

    cache_dir = tmp_path / "cache"
    cachemod.cache_dir(cache_dir).mkdir(parents=True, exist_ok=True)
    # Plant a fingerprint-matching, version-MISSING entry. If the
    # loader trusts this, it returns rows=999 instead of 1.
    live_fp = cachemod._file_fingerprint(docs_path)
    cachemod.write_cache_index(
        cache_dir,
        {
            "wikipedia/documents/monaco-latest.parquet": {
                "relative_path": "wikipedia/documents/monaco-latest.parquet",
                "fingerprint": live_fp,
                "file_size_bytes": docs_path.stat().st_size,
                "kind": "documents",
                "scan_failed": False,
                "rows": 999,
                "non_empty": 999,
                "empty_or_null": 0,
                "total_chars": 11,
                "total_words": 2,
                "total_tokens_estimate": 3,
                "document_ids": ["d1"],
                "section_ids": [],
                "qids": ["Q1"],
                "languages": {"en": 1},
                "fact_rows": 0,
                "fact_ids": [],
                "subject_qids": [],
                "property_ids": [],
                "property_labels": {},
                "property_counts": {},
                "with_property_en_label": 0,
                "with_value_en_label": 0,
                "with_qualifiers": 0,
                "with_references": 0,
                "unavailable_qualifiers": 0,
                "unavailable_references": 0,
                "value_type_counts": {},
            }
        },
    )
    # Remove the contract version: write directly to the file.
    raw = json.loads(cachemod.index_path(cache_dir).read_text())
    raw.pop("__contract_version__", None)
    cachemod.index_path(cache_dir).write_text(json.dumps(raw), encoding="utf-8")

    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    real_safe_table = augmod.safe_table
    calls: list[Path] = []

    def spy(path, cols):
        calls.append(Path(path))
        return real_safe_table(path, cols)

    augmod.safe_table = spy  # type: ignore[assignment]
    try:
        stats = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]

    assert stats.wikipedia_documents.rows == 1
    assert any(c == docs_path for c in calls), (
        "Missing version must trigger a full rebuild; live fingerprint alone is not enough"
    )


def test_scan_failed_entry_is_retried_on_next_refresh(tmp_path: Path) -> None:
    """A cached ``scan_failed=True`` entry must be retried on the next
    refresh. The current code reuses it forever while the fingerprint
    is unchanged, contradicting its docstring and preventing recovery.
    Once a sidecar becomes readable, the next refresh must surface its
    rows and remove the failure flag.

    To force the cache HIT path (which is where the bug lives), the
    test plants a valid sidecar whose fingerprint matches the cached
    one for that file but whose actual content was rewritten via the
    direct cache to look like ``scan_failed=True``. The recovery must
    come from the ``scan_failed`` retry rule, not from a fingerprint
    change.
    """
    processed = _setup_processed_dir(tmp_path)
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    _write_documents(
        docs_path,
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello",
                "article_length_chars": 5,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )

    cache_dir = tmp_path / "cache"
    first = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert first.wikipedia_documents.rows == 1
    assert first.unreadable_file_count == 0

    # Inject a CACHED entry whose fingerprint matches but whose
    # ``scan_failed=True`` would otherwise mask the readable file.
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    cachemod.cache_dir(cache_dir).mkdir(parents=True, exist_ok=True)
    cached_index = cachemod.load_cache_index(cache_dir)
    target_key = "wikipedia/documents/monaco-latest.parquet"
    real_blob = cached_index[target_key]
    sabotaged = dict(real_blob)
    sabotaged["scan_failed"] = True
    sabotaged["rows"] = 0
    cached_index[target_key] = sabotaged
    cachemod.write_cache_index(cache_dir, cached_index)

    # Call again with the SAME file on disk + sabotaged cache. The
    # correct behaviour: ``scan_failed=True`` triggers a rescan, so
    # the rows become 1 again and the unreadable count stays 0.
    second = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert second.wikipedia_documents.rows == 1, (
        "Failed-scan entries must be retried on the next refresh"
    )
    assert second.unreadable_file_count == 0, (
        "A previously-failed sidecar must no longer be counted unreadable"
    )


def test_fingerprint_detects_same_size_replacement_preserving_mtime(
    tmp_path: Path, monkeypatch
) -> None:
    """Replacing a cached file must trigger a rescan even when size and mtime match."""
    import os

    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    processed = _setup_processed_dir(tmp_path)
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    _write_documents(
        docs_path,
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello world",
                "article_length_chars": 11,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            }
        ],
    )

    cache_dir = tmp_path / "cache"
    first = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    stat = docs_path.stat()
    replacement = docs_path.with_suffix(".replacement")
    replacement.write_bytes(docs_path.read_bytes())
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(docs_path)
    replaced = docs_path.stat()
    assert (replaced.st_size, replaced.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    assert replaced.st_ino != stat.st_ino

    real = augmod.safe_table
    calls: list[Path] = []

    def spy(path, cols):
        calls.append(Path(path))
        return real(path, cols)

    monkeypatch.setattr(augmod, "safe_table", spy)
    stats = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert stats.wikipedia_documents == first.wikipedia_documents
    assert calls == [docs_path]


def test_subdir_present_requires_at_least_one_readable_parquet(tmp_path: Path) -> None:
    """``subdir_present`` must reflect whether a readable valid Parquet
    sidecar exists, not just whether the directory is on disk.

    * Directory exists but contains no parquets ⇒ "No data exists
      yet." (subdir_present ``False``).
    * Directory exists with at least one readable, valid zero-row
      Parquet ⇒ "This sidecar is present but empty." (subdir_present
      ``True``).
    """
    processed = _setup_processed_dir(tmp_path)
    stats = compute_augmentation_stats(processed, cache_index_dir=tmp_path / "cache")
    assert stats.wikipedia_documents.subdir_present is False
    assert stats.wikivoyage_documents.subdir_present is False

    _write_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [],
    )
    stats = compute_augmentation_stats(processed, cache_index_dir=tmp_path / "cache")
    assert stats.wikipedia_documents.subdir_present is True
    assert stats.wikipedia_sections.subdir_present is False
    assert stats.wikivoyage_documents.subdir_present is False
    assert stats.wikidata_facts.subdir_present is False


def test_cache_module_does_not_export_make_cache_key() -> None:
    """The unused ``make_cache_key`` helper must be removed from the
    cache module so the surface reflects reality.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    assert "make_cache_key" not in getattr(cachemod, "__all__", ())
    assert not hasattr(cachemod, "make_cache_key")


def test_cache_module_docstring_matches_storage_path() -> None:
    """The cache module docstring must reflect the actual storage path
    ``<data_root>/cache/stats_cache/index.json`` and the
    (relative_path, fingerprint) lookup key.
    """
    import inspect

    from osm_polygon_wikidata_only.hf._dataset_stats import cache as cachemod

    doc = inspect.getdoc(cachemod) or ""
    assert "stats_cache/index.json" in doc, (
        f"Cache docstring must mention storage path stats_cache/index.json: {doc!r}"
    )
    assert "(relative_path, fingerprint)" in doc or (
        "relative_path" in doc and "fingerprint" in doc
    ), f"Cache docstring must explain the cache key: {doc!r}"
    assert "stats_cache/index``" not in doc and "stats_cache/index " not in doc
