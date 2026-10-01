"""The publication facade keeps exposing the release snapshot helpers."""

from __future__ import annotations

from osm_polygon_wikidata_only.hf import publication
from osm_polygon_wikidata_only.hf._publication import readme_snapshot


def test_facade_re_exports_the_release_snapshot_helpers() -> None:
    assert publication.MinimalV1ReleaseSnapshot is readme_snapshot.MinimalV1ReleaseSnapshot
    assert (
        publication.build_minimal_v1_release_snapshot
        is readme_snapshot.build_minimal_v1_release_snapshot
    )
    assert "MinimalV1ReleaseSnapshot" in publication.__all__
    assert "build_minimal_v1_release_snapshot" in publication.__all__
