"""Crash/restart characterization tests for the Wikipedia migration lifecycle.

Every test uses temporary synthetic data and mocked upload boundaries.
No network, no full dataset, no timing-based sleeps.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    MigrationError,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_retirement import (
    finalize_local_retirement,
    prepare_local_retirement,
)
from osm_polygon_wikidata_only.cli.run_sync import (
    _execute_upload_job,
    _post_upload_publication_cleanup,
    _run_pre_publication_migration,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    load_pending_publications,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "processed"
STEM = "monaco-latest"

_FIXTURE_ARTICLE = FIXTURES / f"articles/{STEM}.parquet"
_FIXTURE_SECTIONS = FIXTURES / f"wikipedia/sections/{STEM}.parquet"
_FIXTURE_LINKS = FIXTURES / f"polygon_articles/{STEM}.parquet"
_FIXTURE_POLYGONS = FIXTURES / f"polygons/{STEM}.parquet"


def _copy_fixtures(data_root: DataRoot) -> None:
    for src, relative in (
        (_FIXTURE_ARTICLE, f"articles/{STEM}.parquet"),
        (_FIXTURE_LINKS, f"polygon_articles/{STEM}.parquet"),
        (_FIXTURE_POLYGONS, f"polygons/{STEM}.parquet"),
    ):
        dest = data_root.processed / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _write_manifest(data_root: DataRoot, *, canonical: bool = False) -> None:
    entry: dict[str, str] = {}
    if not canonical:
        entry["articles_path"] = f"articles/{STEM}.parquet"
    else:
        entry["wikipedia_documents_path"] = f"wikipedia/documents/{STEM}.parquet"
    (data_root.processed_manifests / "processed_pbfs.json").write_text(
        json.dumps({f"{STEM}.osm.pbf": entry}),
        encoding="utf-8",
    )


def _seed_legacy(tmp_path: Path) -> DataRoot:
    """DataRoot with only legacy article (pre-migration)."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    _copy_fixtures(data_root)
    _write_manifest(data_root, canonical=False)
    return data_root


def _seed_migrated(tmp_path: Path) -> DataRoot:
    """DataRoot after successful migration: canonical + legacy + manifest repointed."""
    data_root = _seed_legacy(tmp_path)
    _run_pre_publication_migration(data_root, {STEM})
    return data_root


def _seed_retired(tmp_path: Path) -> DataRoot:
    """DataRoot after full publication: legacy deleted, canonical only."""
    data_root = _seed_migrated(tmp_path)
    finalize_local_retirement(data_root, STEM)
    return data_root


def _canonical_retirement_ops(data_root: DataRoot) -> list[PublicationOp]:
    """Build the same-stem canonical add + legacy delete op pair for *stem*."""
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    return [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]


@pytest.fixture
def settings() -> Settings:
    return Settings(repo_id="test/repo")


# ---------------------------------------------------------------------------
# Scenario 1: Crash after pending intent written but before migration apply
# ---------------------------------------------------------------------------


def test_crash_after_intent_before_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _seed_legacy(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"

    import osm_polygon_wikidata_only.cli.run_sync as run_sync_mod

    def crash(_plan: object) -> None:
        raise RuntimeError("crash before apply")

    monkeypatch.setattr(run_sync_mod, "apply_migration", crash)

    with pytest.raises(RuntimeError, match="crash before apply"):
        _run_pre_publication_migration(data_root, {STEM})

    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)
    assert not (data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet").exists()

    # Restart: undo the monkeypatch so apply_migration runs normally.
    monkeypatch.undo()
    _run_pre_publication_migration(data_root, {STEM})

    assert (data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet").exists()
    assert legacy.exists()


# ---------------------------------------------------------------------------
# Scenario 2: Crash after canonical migration but before upload
# ---------------------------------------------------------------------------


def test_crash_after_migration_before_upload(tmp_path: Path) -> None:
    data_root = _seed_migrated(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    legacy = data_root.processed_articles / f"{STEM}.parquet"

    assert canonical.exists()
    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)

    _run_pre_publication_migration(data_root, {STEM})

    assert canonical.exists()
    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)


def test_restart_selects_canonical_only_source(tmp_path: Path) -> None:
    """After legacy is retired, augmentation reads canonical (no Wikimedia needed)."""
    data_root = _seed_retired(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    legacy = data_root.processed_articles / f"{STEM}.parquet"

    assert canonical.exists()
    assert not legacy.exists()

    from osm_polygon_wikidata_only.augmentation.steps import (
        read_source_path,
        wikipedia_source_paths,
    )

    sources = wikipedia_source_paths(data_root, STEM)
    assert sources.canonical == canonical
    assert sources.either_exists
    assert read_source_path(data_root, STEM) == canonical

    _run_pre_publication_migration(data_root, {STEM})


# ---------------------------------------------------------------------------
# Scenario 3: Upload failure — production helper never reached
# ---------------------------------------------------------------------------


def test_upload_failure_preserves_legacy_and_intent(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    ops = _canonical_retirement_ops(data_root)

    cleanup_calls: list[bool] = []
    import osm_polygon_wikidata_only.cli.run_sync as run_sync_mod

    original_cleanup = run_sync_mod._post_upload_publication_cleanup

    def tracking_cleanup(*args: object, **kwargs: object) -> None:
        cleanup_calls.append(True)
        original_cleanup(*args, **kwargs)

    monkeypatch.setattr(run_sync_mod, "_post_upload_publication_cleanup", tracking_cleanup)

    def crashing_upload(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(run_sync_mod, "upload_files", crashing_upload)

    with pytest.raises(RuntimeError, match="upload failed"):
        _execute_upload_job(
            data_root=data_root,
            settings=settings,
            ops=ops,
            message="any commit message",
            num_threads=1,
            hub=None,
            dry_run=False,
        )

    assert cleanup_calls == []
    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)
    assert canonical.exists()


# ---------------------------------------------------------------------------
# Scenario 4: Dry run — production helper short-circuits
# ---------------------------------------------------------------------------


def test_dry_run_preserves_legacy_and_intent(tmp_path: Path) -> None:
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    ops = _canonical_retirement_ops(data_root)

    _post_upload_publication_cleanup(data_root, ops, dry_run=True)

    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)
    assert canonical.exists()


# ---------------------------------------------------------------------------
# Scenario 5: Successful upload — full real-callback ordering
# ---------------------------------------------------------------------------


def test_successful_upload_retirement_and_cleanup(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    ops = _canonical_retirement_ops(data_root)

    add_ops = [op for op in ops if op.action == "add"]
    delete_ops = [op for op in ops if op.action == "delete"]
    assert len(add_ops) == 1
    assert len(delete_ops) == 1
    assert add_ops[0].path_in_repo == f"wikipedia/documents/{STEM}.parquet"
    assert delete_ops[0].path_in_repo == f"articles/{STEM}.parquet"

    prepare_local_retirement(data_root, STEM)
    entry = json.loads((data_root.processed_manifests / "processed_pbfs.json").read_text())[
        f"{STEM}.osm.pbf"
    ]
    assert entry["wikipedia_documents_path"] == f"wikipedia/documents/{STEM}.parquet"
    assert "articles_path" not in entry

    assert legacy.exists()

    import osm_polygon_wikidata_only.cli.run_sync as run_sync_mod

    monkeypatch.setattr(run_sync_mod, "upload_files", lambda *_a, **_k: None)

    _execute_upload_job(
        data_root=data_root,
        settings=settings,
        ops=ops,
        message="any commit message",
        num_threads=1,
        hub=None,
        dry_run=False,
    )

    assert not legacy.exists()
    assert STEM not in load_pending_publications(data_root)
    assert canonical.exists()


# ---------------------------------------------------------------------------
# Scenario 6: Crash after local legacy deletion but before pending cleanup
# ---------------------------------------------------------------------------


def test_crash_after_deletion_before_cleanup(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy deleted but pending intent not yet cleared → restart is idempotent."""
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    finalize_local_retirement(data_root, STEM)
    assert not legacy.exists()
    assert STEM in load_pending_publications(data_root)

    _run_pre_publication_migration(data_root, {STEM})

    assert canonical.exists()
    assert not legacy.exists()

    import osm_polygon_wikidata_only.cli.run_sync as run_sync_mod

    monkeypatch.setattr(run_sync_mod, "upload_files", lambda *_a, **_k: None)
    ops = _canonical_retirement_ops(data_root)
    _execute_upload_job(
        data_root=data_root,
        settings=settings,
        ops=ops,
        message="any commit message",
        num_threads=1,
        hub=None,
        dry_run=False,
    )

    assert STEM not in load_pending_publications(data_root)


def test_injected_cleanup_failure_leaves_intent_for_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between legacy deletion and pending cleanup must be restartable."""
    data_root = _seed_migrated(tmp_path)
    ops = _canonical_retirement_ops(data_root)

    import osm_polygon_wikidata_only.cli.run_sync as run_sync_mod

    original_remove = run_sync_mod.remove_pending_publications
    calls: list[str] = []
    crashed = {"flag": False}

    def remove_then_crash(data_root_arg: DataRoot, stems: set[str]) -> None:
        if crashed["flag"]:
            original_remove(data_root_arg, stems)
            return
        finalize_local_retirement(data_root_arg, next(iter(stems)))
        calls.append("delete")
        crashed["flag"] = True
        raise RuntimeError("crash between delete and pending cleanup")

    monkeypatch.setattr(run_sync_mod, "remove_pending_publications", remove_then_crash)

    with pytest.raises(RuntimeError, match="crash between delete and pending cleanup"):
        _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    assert calls == ["delete"]
    assert not (data_root.processed_articles / f"{STEM}.parquet").exists()
    assert STEM in load_pending_publications(data_root)

    _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    assert STEM not in load_pending_publications(data_root)


# ---------------------------------------------------------------------------
# Scenario 7: Conflicts prevent local deletion
# ---------------------------------------------------------------------------


def test_conflicting_canonical_schema_prevents_deletion(tmp_path: Path) -> None:
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    document = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    table = pq.read_table(document)  # type: ignore[no-untyped-call]
    values = table.to_pylist()
    values[0]["title"] = "tampered"
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), document)

    with pytest.raises(MigrationError, match="safe to retire"):
        finalize_local_retirement(data_root, STEM)

    assert legacy.exists()


def test_malformed_canonical_schema_prevents_deletion(tmp_path: Path) -> None:
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    document = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    bogus = pa.schema([pa.field("junk", pa.string())])
    pq.write_table(pa.Table.from_pylist([{"junk": "x"}], schema=bogus), document)

    with pytest.raises(MigrationError, match=r"safe to retire|valid canonical"):
        finalize_local_retirement(data_root, STEM)

    assert legacy.exists()


def test_unresolved_section_prevents_deletion(tmp_path: Path) -> None:
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    sections = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"

    sections.parent.mkdir(parents=True, exist_ok=True)
    if not sections.exists():
        shutil.copy2(_FIXTURE_SECTIONS, sections)
    table = pq.read_table(sections)  # type: ignore[no-untyped-call]
    values = table.to_pylist()
    values[0]["document_id"] = "Q999:wikipedia:en:1:1"
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), sections)

    with pytest.raises(MigrationError, match="sections unresolved"):
        finalize_local_retirement(data_root, STEM)

    assert legacy.exists()


def test_unresolved_polygon_link_prevents_deletion(tmp_path: Path) -> None:
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    links = data_root.processed_links / f"{STEM}.parquet"

    table = pq.read_table(links)  # type: ignore[no-untyped-call]
    values = table.to_pylist()
    values[0]["article_id"] = "Q999:en:999:999"
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), links)

    with pytest.raises(MigrationError, match="polygon links unresolved"):
        finalize_local_retirement(data_root, STEM)

    assert legacy.exists()


# ---------------------------------------------------------------------------
# Scenario 8: Commit message has no influence on acknowledgement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    ["", "any", "custom commit", "Sync complete region monaco-latest"],
)
def test_commit_message_does_not_affect_acknowledgement(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """``_execute_upload_job`` never inspects the commit message."""
    data_root = _seed_migrated(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    ops = _canonical_retirement_ops(data_root)

    import osm_polygon_wikidata_only.cli.run_sync as run_sync_mod

    monkeypatch.setattr(run_sync_mod, "upload_files", lambda *_a, **_k: None)

    _execute_upload_job(
        data_root=data_root,
        settings=settings,
        ops=ops,
        message=message,
        num_threads=1,
        hub=None,
        dry_run=False,
    )

    assert not legacy.exists()
    assert STEM not in load_pending_publications(data_root)
    assert canonical.exists()


# ---------------------------------------------------------------------------
# Unsafe plan aborts before runtime/network construction
# ---------------------------------------------------------------------------


def test_unsafe_plan_aborts_before_build_wikimedia_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_wikimedia_runtime`` must never be called when the plan is unsafe."""
    data_root = _seed_legacy(tmp_path)
    docs_dir = data_root.processed / "wikipedia" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    articles_dir = data_root.processed_articles
    shutil.copy2(FIXTURES / f"articles/{STEM}.parquet", articles_dir / "orphan.parquet")
    bogus = docs_dir / "orphan.parquet"
    bogus_schema = pa.schema([pa.field("junk", pa.string())])
    pq.write_table(pa.Table.from_pylist([{"junk": "x"}], schema=bogus_schema), bogus)

    runtime_calls: list[bool] = []

    def spy_runtime(*_args: object, **_kwargs: object) -> object:
        runtime_calls.append(True)
        return None

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.cli.run_sync.build_wikimedia_runtime", spy_runtime
    )

    with pytest.raises(MigrationError, match="not safe to apply"):
        _run_pre_publication_migration(data_root, {"orphan"})

    assert runtime_calls == []
