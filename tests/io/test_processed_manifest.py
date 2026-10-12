"""Region-mapping reads for processed-PBF manifests (issue #165)."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.io.processed_manifest import processed_manifest_entries


class _Refused(Exception):
    """A refusal raised through a caller-supplied factory."""


def _read(
    raw: object,
    *,
    regions_required: bool = False,
    allow_empty: bool = True,
) -> dict[object, object]:
    return processed_manifest_entries(
        raw,
        on_not_object=lambda: _Refused("not an object"),
        on_regions_invalid=lambda: _Refused("regions invalid"),
        regions_required=regions_required,
        allow_empty=allow_empty,
    )


def test_a_flat_document_is_its_own_region_mapping() -> None:
    document = {"region": {"polygons_path": "polygons/region.parquet"}}

    assert _read(document) == document


def test_a_nested_regions_mapping_replaces_the_document() -> None:
    assert _read({"contract": 1, "regions": {"region": {}}}) == {"region": {}}


@pytest.mark.parametrize("raw", [None, [], "manifest", 1])
def test_a_non_object_document_raises_the_not_object_factory(raw: object) -> None:
    with pytest.raises(_Refused, match="not an object"):
        _read(raw)


@pytest.mark.parametrize("value", [None, [], "region", 0])
def test_a_present_regions_value_must_be_an_object(value: object) -> None:
    with pytest.raises(_Refused, match="regions invalid"):
        _read({"regions": value})


def test_required_regions_must_sit_under_the_regions_key() -> None:
    with pytest.raises(_Refused, match="regions invalid"):
        _read({"region": {}}, regions_required=True)
    assert _read({"regions": {"region": {}}}, regions_required=True) == {"region": {}}


def test_empty_mappings_are_accepted_by_default() -> None:
    assert _read({}) == {}
    assert _read({"regions": {}}) == {}


@pytest.mark.parametrize("document", [{}, {"regions": {}}])
def test_empty_mappings_are_refused_when_disallowed(document: dict[str, object]) -> None:
    with pytest.raises(_Refused, match="regions invalid"):
        _read(document, allow_empty=False)


def test_non_empty_mappings_pass_when_empty_is_disallowed() -> None:
    assert _read({"regions": {"region": {}}}, allow_empty=False) == {"region": {}}
