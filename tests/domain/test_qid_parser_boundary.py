"""Compatibility contract for the neutral domain QID parser."""

from __future__ import annotations


def test_enrichment_qid_helpers_reexport_the_domain_implementation() -> None:
    from osm_polygon_wikidata_only.domain.wikidata_qids import (
        is_valid_qid as domain_is_valid_qid,
    )
    from osm_polygon_wikidata_only.domain.wikidata_qids import (
        qids_from_osm_tag as domain_qids_from_osm_tag,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        is_valid_qid as compatibility_is_valid_qid,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        qids_from_osm_tag as compatibility_qids_from_osm_tag,
    )

    assert compatibility_is_valid_qid is domain_is_valid_qid
    assert compatibility_qids_from_osm_tag is domain_qids_from_osm_tag
    assert domain_qids_from_osm_tag("Q2; Q1;Q2") == ("Q2", "Q1")


def test_qid_parser_rejects_an_empty_tag() -> None:
    from osm_polygon_wikidata_only.domain.wikidata_qids import qids_from_osm_tag

    assert qids_from_osm_tag("") == ()


def test_qid_parser_rejects_a_malformed_component_without_partial_results() -> None:
    from osm_polygon_wikidata_only.domain.wikidata_qids import qids_from_osm_tag

    assert qids_from_osm_tag("Q2;not-a-qid;Q1") == ()
    assert qids_from_osm_tag("Q2;;Q1") == ()
