"""Tests verifying that ``_post_upload_publication_cleanup`` requires paired remote retirement.

Cleanup is authorized only when the operation list contains BOTH:

* ``add wikipedia/documents/<stem>.parquet``
* ``delete articles/<stem>.parquet``

for the exact same stem. An add without its matching delete, a delete
for another stem, a delete without an add, and mixed multi-stem
operation lists are all rejected. Cleanup for correctly-paired stems
must still run, and pending intent clears only after every selected
stem's local retirement succeeds.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.cli.run_sync import _post_upload_publication_cleanup
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    add_pending_publications,
    load_pending_publications,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "processed"
STEM = "monaco-latest"
OTHER_STEM = "andorra-latest"


def _write_minimal_article_parquet(path: Path) -> None:
    """Write a minimal valid article parquet to satisfy ``find_table`` checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from osm_polygon_wikidata_only.domain.schema import article_schema

    schema = article_schema()
    table = pa.Table.from_pylist(
        [
            {
                "article_id": "Q1:en:1:1",
                "wikidata": "Q1",
                "language": "en",
                "site": "enwiki",
                "title": "T",
                "url": "https://en.wikipedia.org/wiki/T",
                "page_id": 1,
                "revision_id": 1,
                "revision_timestamp": "2026-07-15T00:00:00Z",
                "retrieved_at": "2026-07-15T00:00:00Z",
                "wikidata_label": "",
                "wikidata_description": "",
                "wikidata_aliases": "",
                "lead_text": "",
                "extract": "",
                "full_text": "",
                "full_text_format": "plain_text",
                "article_length_chars": 0,
                "article_length_words": 0,
                "article_length_tokens_estimate": 0,
                "thumbnail_url": "",
                "thumbnail_width": 0,
                "thumbnail_height": 0,
                "categories": "",
                "license": "",
                "attribution": "",
                "source_api": "",
                "fetch_status": "ok",
                "fetch_error": "",
                "content_hash": "",
            }
        ],
        schema=schema,
    )
    pq.write_table(table, path)


def _seed_migrated_two_stems(tmp_path: Path) -> DataRoot:
    """DataRoot with two legacy-article stems pending publication."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for stem in (STEM, OTHER_STEM):
        _write_minimal_article_parquet(data_root.processed_articles / f"{stem}.parquet")
        docs = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
        docs.parent.mkdir(parents=True, exist_ok=True)
        from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
            build_wikipedia_document_table,
        )

        article_path = data_root.processed_articles / f"{stem}.parquet"
        article_table = pq.read_table(article_path)
        canonical = build_wikipedia_document_table(article_table)
        pq.write_table(canonical, docs)
    add_pending_publications(data_root, {STEM, OTHER_STEM})
    return data_root


# ---------------------------------------------------------------------------
# Add without matching delete — no cleanup
# ---------------------------------------------------------------------------


def test_add_without_matching_delete_does_not_cleanup(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops: list[PublicationOp] = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet")
    ]

    _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)
    assert canonical.exists()


# ---------------------------------------------------------------------------
# Delete for another stem — no authorization
# ---------------------------------------------------------------------------


def test_delete_for_other_stem_does_not_cleanup_first_stem(
    tmp_path: Path,
) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops: list[PublicationOp] = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{OTHER_STEM}.parquet"),
    ]

    _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)
    assert OTHER_STEM in load_pending_publications(data_root)


# ---------------------------------------------------------------------------
# Delete without add — no authorization
# ---------------------------------------------------------------------------


def test_delete_without_add_does_not_cleanup(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"

    ops: list[PublicationOp] = [
        delete_op(f"articles/{STEM}.parquet"),
    ]

    _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    assert legacy.exists()
    assert STEM in load_pending_publications(data_root)


# ---------------------------------------------------------------------------
# Mixed multi-stem — only correctly paired stems are cleaned
# ---------------------------------------------------------------------------


def test_mixed_multi_stem_only_paired_stems_are_cleaned(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    stem_legacy = data_root.processed_articles / f"{STEM}.parquet"
    stem_canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    other_legacy = data_root.processed_articles / f"{OTHER_STEM}.parquet"
    other_canonical = data_root.processed / "wikipedia" / "documents" / f"{OTHER_STEM}.parquet"

    ops: list[PublicationOp] = [
        add_op(stem_canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
        # OTHER_STEM is missing its add — only delete is present.
        delete_op(f"articles/{OTHER_STEM}.parquet"),
    ]

    _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    # STEM is correctly paired — cleaned.
    assert not stem_legacy.exists()
    # OTHER_STEM lacks its add — NOT cleaned.
    assert other_legacy.exists()
    # Pending intent reflects only the cleaned stems.
    assert STEM not in load_pending_publications(data_root)
    assert OTHER_STEM in load_pending_publications(data_root)
    assert stem_canonical.exists()
    assert other_canonical.exists()


# ---------------------------------------------------------------------------
# Both stems fully paired — both are cleaned
# ---------------------------------------------------------------------------


def test_two_stems_fully_paired_are_both_cleaned(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    stem_legacy = data_root.processed_articles / f"{STEM}.parquet"
    stem_canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    other_legacy = data_root.processed_articles / f"{OTHER_STEM}.parquet"
    other_canonical = data_root.processed / "wikipedia" / "documents" / f"{OTHER_STEM}.parquet"

    ops: list[PublicationOp] = [
        add_op(stem_canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
        add_op(other_canonical, path_in_repo=f"wikipedia/documents/{OTHER_STEM}.parquet"),
        delete_op(f"articles/{OTHER_STEM}.parquet"),
    ]

    _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    assert not stem_legacy.exists()
    assert not other_legacy.exists()
    assert STEM not in load_pending_publications(data_root)
    assert OTHER_STEM not in load_pending_publications(data_root)


# ---------------------------------------------------------------------------
# Empty ops list — no cleanup
# ---------------------------------------------------------------------------


def test_empty_ops_does_not_cleanup(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)

    _post_upload_publication_cleanup(data_root, [], dry_run=False)

    pending = load_pending_publications(data_root)
    assert STEM in pending
    assert OTHER_STEM in pending
