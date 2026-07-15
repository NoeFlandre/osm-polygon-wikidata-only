"""Sequencing and failure-boundary tests for the pre-publication migration coordinator."""

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
from osm_polygon_wikidata_only.cli.run_sync import _run_pre_publication_migration
from osm_polygon_wikidata_only.config.paths import DataRoot

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "processed"
STEM = "monaco-latest"


def _seed_legacy_only(tmp_path: Path) -> DataRoot:
    """Seed a DataRoot whose only Wikipedia data is a legacy article."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for relative in (
        f"articles/{STEM}.parquet",
        f"polygon_articles/{STEM}.parquet",
    ):
        destination = data_root.processed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURES / relative, destination)
    (data_root.processed_manifests / "processed_pbfs.json").write_text(
        json.dumps({f"{STEM}.osm.pbf": {"articles_path": f"articles/{STEM}.parquet"}}),
        encoding="utf-8",
    )
    return data_root


# ---------------------------------------------------------------------------
# Successful migration
# ---------------------------------------------------------------------------


def test_migration_creates_canonical_and_persists_intent(tmp_path: Path) -> None:
    data_root = _seed_legacy_only(tmp_path)

    _run_pre_publication_migration(data_root, {STEM})

    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    assert canonical.exists()
    from osm_polygon_wikidata_only.pipeline.pending_publications import (
        load_pending_publications,
    )

    assert STEM in load_pending_publications(data_root)


def test_migration_repoints_manifest_to_canonical(tmp_path: Path) -> None:
    data_root = _seed_legacy_only(tmp_path)

    _run_pre_publication_migration(data_root, {STEM})

    entry = json.loads((data_root.processed_manifests / "processed_pbfs.json").read_text())[
        f"{STEM}.osm.pbf"
    ]
    assert entry.get("wikipedia_documents_path") == f"wikipedia/documents/{STEM}.parquet"
    assert "articles_path" not in entry


def test_migration_preserves_legacy_during_staging(tmp_path: Path) -> None:
    data_root = _seed_legacy_only(tmp_path)

    _run_pre_publication_migration(data_root, {STEM})

    legacy = data_root.processed_articles / f"{STEM}.parquet"
    assert legacy.exists(), "Legacy article must survive until confirmed publication"


def test_migration_idempotent_on_restart(tmp_path: Path) -> None:
    data_root = _seed_legacy_only(tmp_path)

    _run_pre_publication_migration(data_root, {STEM})
    _run_pre_publication_migration(data_root, {STEM})

    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    assert canonical.exists()


# ---------------------------------------------------------------------------
# Failure boundary: unsafe plan aborts before mutation
# ---------------------------------------------------------------------------


def test_unsafe_plan_aborts_before_mutation(tmp_path: Path) -> None:
    """A document-with-conflicting-schema (blocked stem) must abort before apply."""
    data_root = _seed_legacy_only(tmp_path)
    docs_dir = data_root.processed / "wikipedia" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    articles_dir = data_root.processed_articles
    # Create a legacy article AND a document with an unexpected schema for a new stem.
    shutil.copy2(FIXTURES / f"articles/{STEM}.parquet", articles_dir / "orphan.parquet")
    bogus = docs_dir / "orphan.parquet"
    bogus_schema = pa.schema([pa.field("junk", pa.string())])
    pq.write_table(pa.Table.from_pylist([{"junk": "x"}], schema=bogus_schema), bogus)  # type: ignore[no-untyped-call]

    with pytest.raises(MigrationError, match="not safe to apply"):
        _run_pre_publication_migration(data_root, {"orphan"})

    # The conflicting document must not have been overwritten.
    actual_schema = pq.read_schema(bogus)  # type: ignore[no-untyped-call]
    assert actual_schema.names == ["junk"]
    from osm_polygon_wikidata_only.pipeline.pending_publications import (
        load_pending_publications,
    )

    assert "orphan" not in load_pending_publications(data_root)


# ---------------------------------------------------------------------------
# Ordering: intent persisted before apply
# ---------------------------------------------------------------------------


def test_intent_persisted_before_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _seed_legacy_only(tmp_path)

    # Inject a failure in apply_migration (patched in run_sync's namespace,
    # which is where the helper calls it) so we can check whether intent was
    # already persisted at that point.
    import osm_polygon_wikidata_only.cli.run_sync as run_sync_mod

    call_log: list[str] = []

    def failing_apply(plan):
        call_log.append("apply")
        from osm_polygon_wikidata_only.pipeline.pending_publications import (
            load_pending_publications,
        )

        if STEM in load_pending_publications(data_root):
            call_log.append("intent_before_apply")
        raise RuntimeError("injected crash")

    monkeypatch.setattr(run_sync_mod, "apply_migration", failing_apply)

    with pytest.raises(RuntimeError, match="injected crash"):
        _run_pre_publication_migration(data_root, {STEM})

    # intent_before_apply is appended only if the intent was already
    # persisted when apply_migration was invoked.
    assert "intent_before_apply" in call_log


# ---------------------------------------------------------------------------
# Scoping: only legacy-backed stems are migrated
# ---------------------------------------------------------------------------


def test_scopes_to_legacy_stems_only(tmp_path: Path) -> None:
    data_root = _seed_legacy_only(tmp_path)

    # Pass an input stem that has no legacy article at all.
    _run_pre_publication_migration(data_root, {"nonexistent-stem"})

    # Nothing should have been created for the nonexistent stem.
    assert not (
        data_root.processed / "wikipedia" / "documents" / "nonexistent-stem.parquet"
    ).exists()
    from osm_polygon_wikidata_only.pipeline.pending_publications import (
        load_pending_publications,
    )

    assert "nonexistent-stem" not in load_pending_publications(data_root)
