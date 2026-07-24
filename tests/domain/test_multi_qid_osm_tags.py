"""Phase 2 / Amendment 2: Multi-QID OSM tags.

The domain layer MUST reuse the project's canonical
:func:`qids_from_osm_tag` parser from
:mod:`osm_polygon_wikidata_only.enrichment.wikidata.parsing` -- no
ad-hoc QID regex. Values like ``Q8254481;Q6033432`` are valid. The
canonical builder, validator, migration, and Wikivoyage integrity
normalization must verify a link/document QID is a member of the
polygon's parsed QID set. ``Q0`` and other invalid QIDs are rejected
by the existing strict validator.
"""

from __future__ import annotations

import inspect
import re

import pytest


def _import_module(path: str, name: str):
    try:
        import importlib

        mod = importlib.import_module(path)
    except ImportError as exc:
        pytest.fail(f"{path} import failed: {exc}")
    return mod


# ---------------------------------------------------------------------------
# 1. No new QID regex inside the domain layer
# ---------------------------------------------------------------------------


def test_domain_layer_does_not_define_its_own_qid_regex() -> None:
    src = inspect.getsource(
        _import_module("osm_polygon_wikidata_only.domain.polygon_document_links", "pdl")
    )
    # The module must not define a fresh QID regex like ^Q\\d+$.
    assert not re.search(r"_VALID_QID\\s*=\\s*re\\.compile", src), (
        "polygon_document_links must not define a private QID regex; "
        "reuse the canonical qids_from_osm_tag / is_valid_qid instead"
    )


def test_domain_layer_imports_canonical_qid_parser() -> None:
    mod = _import_module("osm_polygon_wikidata_only.domain.polygon_document_links", "pdl")
    # The module should expose or use the canonical parser symbols.
    src = inspect.getsource(mod)
    assert "qids_from_osm_tag" in src or "is_valid_qid" in src, (
        "polygon_document_links must import the canonical qids_from_osm_tag / is_valid_qid"
    )


# ---------------------------------------------------------------------------
# 2. Multi-QID OSM tag values are accepted
# ---------------------------------------------------------------------------


def test_multi_qid_osm_tag_parses_distinct_qids() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag

    assert qids_from_osm_tag("Q8254481;Q6033432") == ("Q8254481", "Q6033432")


def test_qid_zero_is_rejected_by_strict_validator() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import is_valid_qid

    assert not is_valid_qid("Q0")
    assert not is_valid_qid("Q00")
    assert is_valid_qid("Q1")
    assert is_valid_qid("Q8254481")


# ---------------------------------------------------------------------------
# 3. Canonical builder: link QID must be a member of the polygon's QID set
# ---------------------------------------------------------------------------


def _poly(polygon_id: str, wikidata_tag: str) -> dict:
    return {
        "polygon_id": polygon_id,
        "wikidata": wikidata_tag,  # OSM tag value (may be semicolon-separated)
        "source_pbf": "x-latest.osm.pbf",
        "region": "x",
        "osm_type": "way",
        "osm_id": 1,
    }


def _doc(document_id: str, wikidata: str) -> dict:
    return {
        "document_id": document_id,
        "wikidata": wikidata,
        "language": "en",
        "page_id": 100,
        "revision_id": 1,
        "project": "wikipedia",
    }


def test_builder_accepts_multi_qid_polygon_tag() -> None:
    """A polygon tagged ``Q1;Q2`` is joinable to a Q1 document AND a Q2 document."""
    mod = _import_module("osm_polygon_wikidata_only.domain.polygon_document_links", "pdl")
    rows = mod.build_polygon_document_links(
        polygons=[_poly("p1", "Q1;Q2")],
        wikipedia_documents=[
            _doc("Q1:wikipedia:en:100:1", "Q1"),
            _doc("Q2:wikipedia:en:200:2", "Q2"),
        ],
    )
    doc_qids = sorted(r["wikidata"] for r in rows)
    assert doc_qids == ["Q1", "Q2"], f"Expected 2 rows joining to Q1 and Q2; got {doc_qids}"


def test_builder_rejects_document_qid_not_in_polygon_set() -> None:
    """A document with Q3 (absent from polygon tag) must not produce a link."""
    mod = _import_module("osm_polygon_wikidata_only.domain.polygon_document_links", "pdl")
    rows = mod.build_polygon_document_links(
        polygons=[_poly("p1", "Q1;Q2")],
        wikipedia_documents=[
            _doc("Q1:wikipedia:en:100:1", "Q1"),
            _doc("Q3:wikipedia:en:300:3", "Q3"),
        ],
    )
    doc_qids = sorted(r["wikidata"] for r in rows)
    assert doc_qids == ["Q1"], f"Q3 absent from polygon tag -- must NOT join; got {doc_qids}"


# ---------------------------------------------------------------------------
# 4. Migration: canonical table inherits the canonical parser
# ---------------------------------------------------------------------------


def test_migration_uses_canonical_qid_parser() -> None:
    mod = _import_module("osm_polygon_wikidata_only.pipeline.link_migration", "lm")
    src = inspect.getsource(mod)
    assert "qids_from_osm_tag" in src, (
        "link_migration must use the canonical qids_from_osm_tag parser"
    )


# ---------------------------------------------------------------------------
# 5. Wikivoyage integrity normalization: rejects documents outside the
#    polygon's parsed QID set
# ---------------------------------------------------------------------------


def test_wikivoyage_integrity_uses_canonical_qid_parser() -> None:
    from osm_polygon_wikidata_only.augmentation import rejection_ledger as rl

    src = inspect.getsource(rl)
    assert "qids_from_osm_tag" in src, (
        "Wikivoyage integrity normalization must use the canonical qids_from_osm_tag parser"
    )
