"""Split coverage tests (part 2)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.pipeline.sync_reconciliation_support import *


def test_upload_failure_remains_retryable(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote missing core, trigger publish/repair
    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

    # Force upload_files failure
    def failing_upload(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("HF upload failed")

    monkeypatch.setattr(run_sync, "upload_files", failing_upload)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]

    rc = commands.main(args)
    # Failure should return non-zero
    assert rc != 0

    # No success log should be printed
    assert not any("Remote reconciliation complete" in message for message in spy.messages)


def test_dry_run_causes_no_mutation_or_retirement(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Place a pending publication stem to simulate retired state
    from osm_polygon_wikidata_only.pipeline.pending_publications import (
        add_pending_publications,
        load_pending_publications,
    )

    add_pending_publications(data_root, {stem})

    # Dry-run execution
    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0

    # Local retirement is untouched (still pending)
    assert stem in load_pending_publications(data_root)


def test_existing_paired_legacy_retirement_remains_intact(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Overwrite processed_articles AND wikipedia/documents with matching, valid data!
    from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
        build_wikipedia_document_table,
    )

    row_data = {
        "article_id": "Q1:es:1234:5678",
        "wikidata": "Q1",
        "language": "es",
        "site": "eswiki",
        "title": "Test Title",
        "url": "https://es.wikipedia.org/wiki/Test",
        "page_id": 1234,
        "revision_id": 5678,
        "revision_timestamp": "2026-07-16T12:00:00Z",
        "retrieved_at": "2026-07-16T12:00:00Z",
        "wikidata_label": "Label",
        "wikidata_description": "Description",
        "wikidata_aliases": "",
        "lead_text": "Lead",
        "extract": "Extract",
        "full_text": "Full text",
        "full_text_format": "text",
        "article_length_chars": 9,
        "article_length_words": 2,
        "article_length_tokens_estimate": 2,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "",
        "license": "CC-BY-SA",
        "attribution": "Attribution",
        "source_api": "rest",
        "fetch_status": "success",
        "fetch_error": "",
        "content_hash": "hash123",
    }

    # Write to local processed_articles
    articles_path = data_root.processed_articles / f"{stem}.parquet"
    data_root.processed_articles.mkdir(parents=True, exist_ok=True)
    art_table = pa.Table.from_pylist([row_data], schema=article_schema())
    pq.write_table(art_table, articles_path)  # type: ignore[no-untyped-call]

    # Convert to canonical document and write to wikipedia/documents
    doc_table = build_wikipedia_document_table(art_table)
    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    pq.write_table(doc_table, wikipedia_documents_path)  # type: ignore[no-untyped-call]

    # Write links table with matching article_id so assert_references_resolve passes
    links_table = pa.Table.from_pylist(
        [
            {
                "polygon_id": "1",
                "article_id": "Q1:es:1234:5678",
                "wikidata": "Q1",
                "language": "es",
                "source_pbf": f"{stem}.osm.pbf",
                "region": stem,
                "osm_type": "relation",
                "osm_id": 1,
                "page_id": 1234,
                "revision_id": 5678,
                "is_best_language": True,
            }
        ],
        schema=polygon_article_schema(),
    )
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    # The overwrite invalidated the stored augmentation-manifest hashes
    # (they still point at the pre-overwrite bytes). Refresh the manifest
    # so the region classifies as finalized (COMPLETE/PUBLISH) rather
    # than AUGMENT, which would otherwise trigger a real Wiki fetch.
    _refresh_augmentation_manifest(data_root, stem)

    # Reconciliation tests must never reach the network. The fail-loud
    # guard turns any accidental AUGMENT classification into a test
    # failure instead of a silent es.wikipedia.org request.
    recorder = _block_reconciliation_network(monkeypatch)

    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        augmentation_is_current,
    )

    assert augmentation_is_current(data_root, stem), (
        "mexico-latest was not finalized after overwriting its data files; "
        "the augmentation manifest hashes were left stale."
    )

    # Stub remote
    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        # NO dry-run so retirement executes
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0

    # Verify legacy articles file is retired (deleted locally)
    assert not (data_root.processed_articles / f"{stem}.parquet").exists()

    # The network guard must not have been tripped: no augmentation
    # transport call and no coverage-map download.
    assert recorder.wikimedia_calls == [], (
        f"Augmentation transport was called {len(recorder.wikimedia_calls)} "
        "time(s); a finalized region was mis-classified as AUGMENT."
    )
    assert recorder.urlretrieve_calls == [], (
        f"urllib.request.urlretrieve was called {len(recorder.urlretrieve_calls)} "
        "time(s); the coverage-map download was not stubbed."
    )


def test_reconciliation_network_guard_intercepts_augmentation_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reconciliation network guard must intercept the actual
    augmentation transport symbol -- ``augmentation.mediawiki.
    read_wikimedia_json`` -- not some unrelated low-level primitive.

    This prevents a future refactor from swapping the transport
    (e.g. from ``urllib.request.urlopen`` to ``requests``) and
    silently bypassing a guard that patched the wrong layer. We
    invoke the production symbol directly and assert the guard both
    recorded the call and raised, proving the boundary is the one
    the code actually imports.
    """
    import osm_polygon_wikidata_only.augmentation.mediawiki as mediawiki_mod

    recorder = _block_reconciliation_network(monkeypatch)

    # The production code imports ``read_wikimedia_json`` from
    # ``enrichment.wikimedia`` into the ``augmentation.mediawiki``
    # namespace. The guard must replace that exact binding so the
    # call the code makes is the one intercepted (a guard that
    # patched a different layer would leave this untouched).
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        read_wikimedia_json as upstream_read_wikimedia_json,
    )

    assert mediawiki_mod.read_wikimedia_json is not upstream_read_wikimedia_json

    with pytest.raises(AssertionError, match="read_wikimedia_json"):
        mediawiki_mod.read_wikimedia_json("https://es.wikipedia.org/w/api.php")

    assert len(recorder.wikimedia_calls) == 1
    assert recorder.urlretrieve_calls == []


def test_augmentation_is_current_called_exactly_once_per_stem(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    import osm_polygon_wikidata_only.augmentation.orchestrator as orch

    call_count = 0
    original_is_current = orch.augmentation_is_current

    def spy_is_current(*args: Any, **kwargs: Any) -> bool:
        nonlocal call_count
        call_count += 1
        return original_is_current(*args, **kwargs)

    monkeypatch.setattr(orch, "augmentation_is_current", spy_is_current)
    monkeypatch.setattr(run_sync, "augmentation_is_current", spy_is_current)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0
    assert call_count == 1  # Verified exactly once!


def test_token_resolver_failure_but_injected_hub_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Force resolve_hf_token to fail
    def failing_resolver(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Token resolver failed")

    monkeypatch.setattr(commands, "resolve_hf_token", failing_resolver)

    stub = StubHfHub(remote_files=set())
    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    parser = commands.build_parser()
    args = parser.parse_args(
        [
            "sync-dir",
            str(data_root.raw),
            "--data-root",
            str(tmp_path),
            "--push",
            "--dry-run",
            "--skip-existing",
        ]
    )
    settings = Settings(repo_id="test/repo", hf_token="fake", skip_existing=True)

    # Call run_sync.execute directly with injected collaborators, verifying it works
    # without raising TokenResolver errors.
    remote_inventory = RemoteInventory(
        {
            f"polygons/{stem}.parquet",
            f"polygon_articles/{stem}.parquet",
            f"wikipedia/documents/{stem}.parquet",
            f"wikipedia/sections/{stem}.parquet",
            f"wikivoyage/documents/{stem}.parquet",
            f"wikivoyage/sections/{stem}.parquet",
            f"wikidata/facts/{stem}.parquet",
        }
    )
    rc = run_sync.execute(
        args,
        data_root=data_root,
        settings=settings,
        _remote_inventory=remote_inventory,
        _hub=stub,
    )
    assert rc == 0


def test_logging_core_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]

    rc = commands.main(args)
    assert rc == 0

    assert any(
        "Remote reconciliation complete: 1 regions repaired; README and maps refreshed" in message
        for message in spy.messages
    )


def test_logging_sidecar_only_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote has core files, manifests, etc. but missing wikipedia documents
    stub_files = {
        f"polygons/{stem}.parquet",
        f"polygon_articles/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
        "README.md",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
        "assets/coverage_map.png",
        "assets/geographic_text_density.png",
    }
    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]

    rc = commands.main(args)
    assert rc == 0

    # This first run also performs the one-time unified-link migration, so a
    # final repository metadata publication is expected after the regional
    # sidecar repair drains.
    assert any(
        "Remote reconciliation complete: 1 regions repaired" in message for message in spy.messages
    )
    assert any("README and maps refreshed" in message for message in spy.messages)


def test_logging_metadata_only_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote has everything except README
    stub_files = {
        f"polygons/{stem}.parquet",
        f"polygon_articles/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
    }
    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]

    rc = commands.main(args)
    assert rc == 0

    assert any(
        "Remote reconciliation complete: README and maps refreshed" in message
        for message in spy.messages
    )
    assert not any("regions repaired" in message for message in spy.messages)


def test_logging_upload_failure(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

    # Fail the upload queue
    def failing_upload(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Upload failure")

    monkeypatch.setattr(run_sync, "upload_files", failing_upload)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]

    rc = commands.main(args)
    assert rc != 0

    # Verify aborted logging exists, and success logging is absent
    assert any(
        "Unified sync aborted:" in message or "Unified sync completed with failures" in message
        for message in spy.messages
    )
    assert not any("Remote reconciliation complete" in message for message in spy.messages)
