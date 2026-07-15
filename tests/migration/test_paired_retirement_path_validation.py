"""Tests verifying the strict-path contract of ``_paired_retirement_stems``.

A canonical add is honored only when:

* ``path_in_repo`` is exactly ``wikipedia/documents/<stem>.parquet``
  (no nested paths, no traversal, no lookalike prefixes).
* ``local_path`` resolves to exactly
  ``data_root.processed/wikipedia/documents/<stem>.parquet``.

A legacy delete is honored only when ``path_in_repo`` is exactly
``articles/<stem>.parquet`` (same restrictions).

Duplicate or conflicting canonical add operations for the same stem
prevent cleanup for that stem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli.run_sync import (
    _paired_retirement_stems,
    _post_upload_publication_cleanup,
)
from osm_polygon_wikidata_only.hf._uploader.plan import add_op, delete_op
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    add_pending_publications,
    load_pending_publications,
)
from tests.migration.test_paired_retirement_validation import (
    STEM,
    _seed_migrated_two_stems,
)

# ---------------------------------------------------------------------------
# Exact-paired positive cases
# ---------------------------------------------------------------------------


def test_valid_exact_pair_is_authorized(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == {STEM}


def test_valid_two_stem_pairs_are_authorized(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    stem_canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    other_canonical = data_root.processed / "wikipedia" / "documents" / "andorra-latest.parquet"

    ops = [
        add_op(stem_canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
        add_op(other_canonical, path_in_repo="wikipedia/documents/andorra-latest.parquet"),
        delete_op("articles/andorra-latest.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == {STEM, "andorra-latest"}


# ---------------------------------------------------------------------------
# Wrong local_path — fail closed
# ---------------------------------------------------------------------------


def test_canonical_add_with_wrong_local_path_rejected(tmp_path: Path) -> None:
    """local_path resolves to a file outside the expected canonical location."""
    data_root = _seed_migrated_two_stems(tmp_path)
    wrong_local = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"

    ops = [
        add_op(wrong_local, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


def test_canonical_add_pointing_to_external_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    external = Path("/tmp/external-canonical.parquet")
    external.write_text("x", encoding="utf-8")

    ops = [
        add_op(external, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


def test_canonical_add_with_traversal_local_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    traversal = (
        data_root.processed / "wikipedia" / "documents" / ".." / "sections" / f"{STEM}.parquet"
    )

    ops = [
        add_op(traversal, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


# ---------------------------------------------------------------------------
# Nested remote paths — fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nested_path",
    [
        "wikipedia/documents/sub/monaco-latest.parquet",
        "wikipedia/documents/monaco-latest.parquet.bak",
        "wikipedia/documents/.parquet",
        "wikipedia/documents/",
    ],
)
def test_canonical_add_nested_path_rejected(tmp_path: Path, nested_path: str) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=nested_path),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


@pytest.mark.parametrize(
    "nested_delete",
    [
        "articles/sub/monaco-latest.parquet",
        "articles/monaco-latest.parquet.bak",
        "articles/.parquet",
    ],
)
def test_legacy_delete_nested_path_rejected(tmp_path: Path, nested_delete: str) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(nested_delete),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


# ---------------------------------------------------------------------------
# Traversal remote paths — fail closed
# ---------------------------------------------------------------------------


def test_canonical_add_traversal_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo="wikipedia/documents/../evil.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


def test_legacy_delete_traversal_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op("articles/../evil.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


# ---------------------------------------------------------------------------
# Lookalike prefixes — fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lookalike_path",
    [
        "wikipedia/documentsX/monaco-latest.parquet",
        "wikipediaX/documents/monaco-latest.parquet",
        "wikipedia/documents-monaco-latest.parquet",
        "xwikipedia/documents/monaco-latest.parquet",
        "wikipedia//documents/monaco-latest.parquet",
    ],
)
def test_canonical_add_lookalike_prefix_rejected(tmp_path: Path, lookalike_path: str) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=lookalike_path),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


@pytest.mark.parametrize(
    "lookalike_delete",
    [
        "articlesX/monaco-latest.parquet",
        "xarticles/monaco-latest.parquet",
        "articles-monaco-latest.parquet",
        "articles//monaco-latest.parquet",
    ],
)
def test_legacy_delete_lookalike_prefix_rejected(tmp_path: Path, lookalike_delete: str) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(lookalike_delete),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


# ---------------------------------------------------------------------------
# Empty / invalid stems — fail closed
# ---------------------------------------------------------------------------


def test_canonical_add_empty_stem_rejected(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo="wikipedia/documents/.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


def test_legacy_delete_empty_stem_rejected(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op("articles/.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


def test_canonical_add_no_parquet_suffix_rejected(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


# ---------------------------------------------------------------------------
# Duplicate or conflicting canonical adds — fail closed for that stem
# ---------------------------------------------------------------------------


def test_duplicate_conflicting_canonical_adds_block_cleanup(tmp_path: Path) -> None:
    """Two canonical adds for the same stem, one pointing to the wrong file,
    prevent cleanup for that stem entirely."""
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    wrong = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        add_op(wrong, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


def test_duplicate_identical_canonical_adds_block_cleanup(tmp_path: Path) -> None:
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"

    ops = [
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        add_op(canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == set()


def test_duplicate_conflicting_add_does_not_block_other_stems(tmp_path: Path) -> None:
    """A conflicting add for STEM does not prevent OTHER_STEM from being authorized."""
    data_root = _seed_migrated_two_stems(tmp_path)
    stem_canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    other_canonical = data_root.processed / "wikipedia" / "documents" / "andorra-latest.parquet"
    wrong = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"

    ops = [
        add_op(stem_canonical, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        add_op(wrong, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
        add_op(other_canonical, path_in_repo="wikipedia/documents/andorra-latest.parquet"),
        delete_op("articles/andorra-latest.parquet"),
    ]

    assert _paired_retirement_stems(data_root, ops) == {"andorra-latest"}


# ---------------------------------------------------------------------------
# Integration with _post_upload_publication_cleanup
# ---------------------------------------------------------------------------


def test_post_upload_cleanup_respects_hardened_pairing(tmp_path: Path) -> None:
    """Cleanup must not retire a stem when the canonical add fails pairing checks."""
    data_root = _seed_migrated_two_stems(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    wrong = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"

    ops = [
        add_op(wrong, path_in_repo=f"wikipedia/documents/{STEM}.parquet"),
        delete_op(f"articles/{STEM}.parquet"),
    ]

    add_pending_publications(data_root, {STEM})

    _post_upload_publication_cleanup(data_root, ops, dry_run=False)

    assert (data_root.processed_articles / f"{STEM}.parquet").exists()
    assert STEM in load_pending_publications(data_root)
    assert canonical.exists()
