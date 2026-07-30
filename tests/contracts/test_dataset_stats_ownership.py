"""Focused ownership contracts for augmentation statistics."""

from osm_polygon_wikidata_only.hf._dataset_stats import augmentation, summary_codec


def test_augmentation_stats_uses_shared_summary_codec() -> None:
    assert augmentation._summary_to_json is summary_codec.summary_to_json
    assert augmentation._summary_from_json is summary_codec.summary_from_json
