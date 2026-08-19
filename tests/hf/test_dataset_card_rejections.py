"""Contracts for the dataset-card join-integrity block."""

from __future__ import annotations

from osm_polygon_wikidata_only.hf.dataset_card import (
    _audit_totals,
    _rejection_counts,
    _render_affected_shards,
    render_rejections_section,
)


def test_rejection_counts_default_missing_values_to_zero() -> None:
    assert _rejection_counts({}) == (0, 0, 0)


def test_render_affected_shards_adds_sorted_markdown_line() -> None:
    parts = ["header"]
    assert _render_affected_shards(parts, ["alpha", "zeta"]) == [
        "header",
        "Affected shards: `alpha`, `zeta`.\n",
    ]


def test_render_rejections_section_has_stable_empty_and_affected_forms() -> None:
    assert "| `polygon_articles` rows with mismatched wikidata | 0 |" in render_rejections_section(
        {"totals": {}}
    )
    rendered = render_rejections_section(
        {
            "contract_version": "join-integrity-v2",
            "totals": {
                "polygon_articles_rejected": 2,
                "wikivoyage_documents_rejected": 3,
                "wikivoyage_sections_cascaded": 4,
                "shards_with_rejections": ["zeta", "alpha"],
            },
        }
    )
    assert "Contract version `join-integrity-v2`." in rendered
    assert "Affected shards: `alpha`, `zeta`." in rendered


def test_audit_totals_returns_empty_mapping_for_non_mapping_input() -> None:
    assert _audit_totals([]) == {}
