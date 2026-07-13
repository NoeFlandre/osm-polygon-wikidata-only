"""Dataset statistics: compute factual stats from processed data and render them as markdown.

This module is a thin compatibility facade. The implementation lives
in :mod:`osm_polygon_wikidata_only.hf._dataset_stats` and the three
public names below are re-exported unchanged.

The :class:`DatasetStats` snapshot is computed from the processed
parquet files (columnar pruning keeps it fast); the
:func:`render_stats_section` function turns that snapshot into the
factual README sections: dataset snapshot table, Wikipedia coverage
funnel, and language distribution.

All values are computed from the data, never hardcoded. The tests in
``tests/test_dataset_stats.py`` cross-check the computed values
against a manual count over known fixture data.
"""

from __future__ import annotations

from ._dataset_stats.aggregation import compute_dataset_stats
from ._dataset_stats.models import DatasetStats
from ._dataset_stats.rendering import render_stats_section

__all__ = [
    "DatasetStats",
    "compute_dataset_stats",
    "render_stats_section",
]
