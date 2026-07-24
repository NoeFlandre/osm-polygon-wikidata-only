"""Phase 2 / Group A: canonical schema and pure construction/validation.

Red tests for ``osm_polygon_wikidata_only.domain.polygon_document_links``.
The builder joins polygons to documents by QID; documents carry their
own project ("wikipedia" or "wikivoyage"). The caller never picks the
project, so a document with project="wikisource" cannot accidentally
end up in the table.
"""

from __future__ import annotations

import pytest

EXPECTED_COLUMNS: tuple[str, ...] = (
    "polygon_id",
    "document_id",
    "project",
    "wikidata",
    "language",
    "source_pbf",
    "region",
    "osm_type",
    "osm_id",
    "page_id",
    "revision_id",
)
EXPECTED_PROJECTS: frozenset[str] = frozenset({"wikipedia", "wikivoyage"})


def _import_module():
    try:
        from osm_polygon_wikidata_only.domain import polygon_document_links as mod
    except ImportError as exc:
        pytest.fail(
            "Expected osm_polygon_wikidata_only.domain.polygon_document_links to exist "
            f"(Phase 2 group A: canonical link schema/construction); got ImportError: {exc}"
        )
    return mod


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_module_exposes_public_api() -> None:
    mod = _import_module()
    for name in (
        "polygon_document_link_schema",
        "build_polygon_document_links",
        "validate_polygon_document_links",
    ):
        assert hasattr(mod, name), f"Missing public API: polygon_document_links.{name}"


def test_schema_columns_in_exact_order() -> None:
    pa = pytest.importorskip("pyarrow")
    mod = _import_module()
    schema = mod.polygon_document_link_schema()
    assert tuple(schema.names) == EXPECTED_COLUMNS, (
        f"Expected canonical columns in order {EXPECTED_COLUMNS}, got {tuple(schema.names)}"
    )
    assert isinstance(schema, pa.Schema), f"Schema must be a pyarrow.Schema, got {type(schema)}"


def test_schema_describes_every_column() -> None:
    mod = _import_module()
    schema = mod.polygon_document_link_schema()
    for field in schema:
        desc = field.metadata.get(b"description") if field.metadata else None
        assert desc, f"Column {field.name} missing non-empty description metadata"


def test_schema_int_columns_for_ids() -> None:
    pa = pytest.importorskip("pyarrow")
    mod = _import_module()
    schema = mod.polygon_document_link_schema()
    for name in ("osm_id", "page_id", "revision_id"):
        assert schema.field(name).type == pa.int64(), (
            f"Column {name} must be int64, got {schema.field(name).type}"
        )


def test_schema_project_column_is_string() -> None:
    pa = pytest.importorskip("pyarrow")
    mod = _import_module()
    schema = mod.polygon_document_link_schema()
    assert schema.field("project").type == pa.string(), (
        f"Column project must be string, got {schema.field('project').type}"
    )


# ---------------------------------------------------------------------------
# Builder: success paths
# ---------------------------------------------------------------------------


def test_build_emits_one_link_per_polygon_document_qid_match() -> None:
    mod = _import_module()
    polygon = {
        "polygon_id": "monaco-latest:relation:1234",
        "wikidata": "Q1",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1234,
    }
    links = mod.build_polygon_document_links(
        polygons=[polygon],
        wikipedia_documents=[
            {
                "document_id": "Q1:wikipedia:en:1:1",
                "project": "wikipedia",
                "wikidata": "Q1",
                "language": "en",
                "page_id": 1,
                "revision_id": 1,
                "full_text": "anything",
            }
        ],
    )
    assert len(links) == 1, f"Expected exactly one link, got {len(links)}"
    link = links[0]
    assert link["polygon_id"] == "monaco-latest:relation:1234"
    assert link["document_id"] == "Q1:wikipedia:en:1:1"
    assert link["project"] == "wikipedia"
    assert link["wikidata"] == "Q1"
    assert link["language"] == "en"
    assert link["source_pbf"] == "monaco-latest.osm.pbf"
    assert link["region"] == "monaco"
    assert link["osm_type"] == "relation"
    assert link["osm_id"] == 1234
    assert link["page_id"] == 1
    assert link["revision_id"] == 1


def test_build_emits_wikivoyage_link_from_wikivoyage_documents() -> None:
    mod = _import_module()
    polygon = {"polygon_id": "p:relation:1", "wikidata": "Q42"}
    voyage_doc = {
        "document_id": "Q42:wikivoyage:en:9:9",
        "project": "wikivoyage",
        "wikidata": "Q42",
        "language": "en",
        "page_id": 9,
        "revision_id": 9,
        "full_text": "anything",
    }
    links = mod.build_polygon_document_links(polygons=[polygon], wikivoyage_documents=[voyage_doc])
    assert len(links) == 1
    assert links[0]["project"] == "wikivoyage"
    assert links[0]["document_id"] == "Q42:wikivoyage:en:9:9"


def test_build_joins_multiple_languages_for_same_polygon() -> None:
    mod = _import_module()
    polygon = {
        "polygon_id": "monaco-latest:relation:1",
        "wikidata": "Q1",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
    }
    docs = [
        {
            "document_id": "Q1:wikipedia:en:1:1",
            "project": "wikipedia",
            "wikidata": "Q1",
            "language": "en",
            "page_id": 1,
            "revision_id": 1,
        },
        {
            "document_id": "Q1:wikipedia:fr:2:2",
            "project": "wikipedia",
            "wikidata": "Q1",
            "language": "fr",
            "page_id": 2,
            "revision_id": 2,
        },
        {
            "document_id": "Q1:wikipedia:de:3:3",
            "project": "wikipedia",
            "wikidata": "Q1",
            "language": "de",
            "page_id": 3,
            "revision_id": 3,
        },
    ]
    links = mod.build_polygon_document_links(polygons=[polygon], wikipedia_documents=docs)
    assert len(links) == 3
    languages = sorted(link["language"] for link in links)
    assert languages == ["de", "en", "fr"], f"Expected de/en/fr, got {languages}"


def test_build_emits_link_even_when_full_text_is_empty() -> None:
    """An empty full_text is not itself a builder error: the join is by QID."""
    mod = _import_module()
    polygon = {
        "polygon_id": "italy-latest:relation:1",
        "wikidata": "Q38",
        "source_pbf": "italy-latest.osm.pbf",
        "region": "italy",
        "osm_type": "relation",
        "osm_id": 1,
    }
    doc = {
        "document_id": "Q38:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q38",
        "language": "en",
        "page_id": 1,
        "revision_id": 1,
        "full_text": "",
    }
    links = mod.build_polygon_document_links(polygons=[polygon], wikipedia_documents=[doc])
    assert len(links) == 1
    assert links[0]["wikidata"] == "Q38"


def test_build_emits_link_for_each_polygon_sharing_qid() -> None:
    """Multiple polygons that share the same QID each get a link."""
    mod = _import_module()
    polygons = [
        {
            "polygon_id": "italy-latest:relation:1",
            "wikidata": "Q38",
            "source_pbf": "italy-latest.osm.pbf",
            "region": "italy",
            "osm_type": "relation",
            "osm_id": 1,
        },
        {
            "polygon_id": "italy-latest:relation:2",
            "wikidata": "Q38",
            "source_pbf": "italy-latest.osm.pbf",
            "region": "italy",
            "osm_type": "relation",
            "osm_id": 2,
        },
        {
            "polygon_id": "italy-latest:relation:3",
            "wikidata": "Q38",
            "source_pbf": "italy-latest.osm.pbf",
            "region": "italy",
            "osm_type": "relation",
            "osm_id": 3,
        },
    ]
    doc = {
        "document_id": "Q38:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q38",
        "language": "en",
        "page_id": 1,
        "revision_id": 1,
    }
    links = mod.build_polygon_document_links(polygons=polygons, wikipedia_documents=[doc])
    assert len(links) == 3, f"Expected 3 links (one per polygon sharing Q38), got {len(links)}"
    polygon_ids = sorted(link["polygon_id"] for link in links)
    assert polygon_ids == [
        "italy-latest:relation:1",
        "italy-latest:relation:2",
        "italy-latest:relation:3",
    ]


def test_build_emits_links_for_both_projects_simultaneously() -> None:
    mod = _import_module()
    polygon = {
        "polygon_id": "monaco-latest:relation:1",
        "wikidata": "Q1",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
    }
    wiki_doc = {
        "document_id": "Q1:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q1",
        "language": "en",
        "page_id": 1,
        "revision_id": 1,
    }
    voyage_doc = {
        "document_id": "Q1:wikivoyage:en:2:2",
        "project": "wikivoyage",
        "wikidata": "Q1",
        "language": "en",
        "page_id": 2,
        "revision_id": 2,
    }
    links = mod.build_polygon_document_links(
        polygons=[polygon],
        wikipedia_documents=[wiki_doc],
        wikivoyage_documents=[voyage_doc],
    )
    assert len(links) == 2
    projects = sorted(link["project"] for link in links)
    assert projects == ["wikipedia", "wikivoyage"]


def test_build_produces_no_link_when_document_qid_has_no_polygon() -> None:
    """A document whose QID has no polygon is just emitted as zero links."""
    mod = _import_module()
    polygon = {
        "polygon_id": "monaco-latest:relation:1",
        "wikidata": "Q1",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
    }
    orphan_doc = {
        "document_id": "Q99:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q99",
        "language": "en",
        "page_id": 1,
        "revision_id": 1,
    }
    links = mod.build_polygon_document_links(polygons=[polygon], wikipedia_documents=[orphan_doc])
    assert links == [], f"Expected no links for orphan QID, got {links}"


# ---------------------------------------------------------------------------
# Strict validation: must fail loudly, never silently coerce
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_project() -> None:
    mod = _import_module()
    bad = {
        "polygon_id": "p:relation:1",
        "document_id": "Q1:wikipedia:en:1:1",
        "project": "wikisource",
        "wikidata": "Q1",
        "language": "en",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
    }
    with pytest.raises(ValueError):
        mod.validate_polygon_document_links([bad])


def test_validate_rejects_link_field_disagreeing_with_target_document() -> None:
    """If a link's wikidata/language mismatch the canonical document, raise."""
    mod = _import_module()
    bad = {
        "polygon_id": "monaco-latest:relation:1",
        "document_id": "Q1:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q99",
        "language": "en",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
    }
    with pytest.raises(ValueError):
        mod.validate_polygon_document_links([bad])


def test_validate_rejects_malformed_document_id_project_disagreement() -> None:
    """document_id declares 'wikipedia' but project column says 'wikivoyage'."""
    mod = _import_module()
    bad = {
        "polygon_id": "monaco-latest:relation:1",
        "document_id": "Q1:wikipedia:en:1:1",
        "project": "wikivoyage",
        "wikidata": "Q1",
        "language": "en",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
    }
    with pytest.raises(ValueError):
        mod.validate_polygon_document_links([bad])


def test_validate_rejects_duplicate_identity_with_conflicting_values() -> None:
    """A duplicate identity with conflicting values must NOT be silently merged."""
    mod = _import_module()
    base = {
        "polygon_id": "monaco-latest:relation:1",
        "document_id": "Q1:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q1",
        "language": "en",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
    }
    conflicting = dict(base, page_id=999)
    with pytest.raises(ValueError):
        mod.validate_polygon_document_links([base, conflicting])


def test_validate_dedups_exact_duplicate_identities() -> None:
    """Byte-identical duplicates are collapsed; only the first is retained."""
    mod = _import_module()
    row = {
        "polygon_id": "monaco-latest:relation:1",
        "document_id": "Q1:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q1",
        "language": "en",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
    }
    validated = mod.validate_polygon_document_links([row, dict(row)])
    assert validated == [row], f"Expected exactly one row after dedup, got {validated}"


def test_validate_sorts_deterministically_by_identity() -> None:
    mod = _import_module()
    row_a = {
        "polygon_id": "monaco-latest:relation:1",
        "document_id": "Q1:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q1",
        "language": "en",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
    }
    row_b = dict(row_a, document_id="Q1:wikipedia:fr:2:2", language="fr", page_id=2, revision_id=2)
    row_c = {
        "polygon_id": "italy-latest:relation:1",
        "document_id": "Q38:wikipedia:en:1:1",
        "project": "wikipedia",
        "wikidata": "Q38",
        "language": "en",
        "source_pbf": "italy-latest.osm.pbf",
        "region": "italy",
        "osm_type": "relation",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
    }
    out = mod.validate_polygon_document_links([row_b, row_c, row_a])
    expected_order = [row_c, row_a, row_b]
    assert out == expected_order, (
        f"Expected deterministic sort by (polygon_id, project, document_id); got {out}"
    )


def test_build_rejects_unknown_project_on_document() -> None:
    """A document whose project is not wikipedia/wikivoyage cannot be joined."""
    mod = _import_module()
    polygon = {
        "polygon_id": "monaco-latest:relation:1",
        "wikidata": "Q1",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
    }
    bad = {
        "document_id": "Q1:wikipedia:en:1:1",
        "wikidata": "Q1",
        "language": "en",
        "page_id": 1,
        "revision_id": 1,
        "project": "wikisource",
    }
    with pytest.raises(ValueError):
        mod.build_polygon_document_links(polygons=[polygon], wikipedia_documents=[bad])


def test_build_rejects_documents_with_invalid_qid() -> None:
    """A document whose wikidata is not a valid QID must raise, not silently emit."""
    mod = _import_module()
    polygon = {
        "polygon_id": "monaco-latest:relation:1",
        "wikidata": "Q1",
        "source_pbf": "monaco-latest.osm.pbf",
        "region": "monaco",
        "osm_type": "relation",
        "osm_id": 1,
    }
    bad = {
        "document_id": "BADID",
        "wikidata": "not-a-qid",
        "language": "en",
        "page_id": 1,
        "revision_id": 1,
        "project": "wikipedia",
    }
    with pytest.raises(ValueError):
        mod.build_polygon_document_links(polygons=[polygon], wikipedia_documents=[bad])
