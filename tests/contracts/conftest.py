"""Shared fixtures for publication-contract test modules."""

from __future__ import annotations

import pytest


@pytest.fixture
def stub_combined_text_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep publication contracts isolated from real Parquet rendering.

    Opt-in rather than ``autouse``: sibling contract modules such as the
    golden-output and augmentation suites exercise the real text-presence
    readers, so this stub must not reach them.
    """
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._load_text_presence",
        lambda _root: object(),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_presence",
        lambda _root, dest, **_kw: dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _cache: None,
    )
