"""Dataset statistics computed from the processed Parquet files.

This module is a thin compatibility facade. The implementation lives in
:mod:`osm_polygon_wikidata_only.hf._dataset_stats` and the two public names
below are re-exported unchanged.

The :class:`DatasetStats` snapshot is computed from the processed parquet
files (columnar pruning keeps it fast). Presentation belongs to the card
renderers and to the machine-readable ``stats.json`` report, not here.

All values are computed from the data, never hardcoded. The tests in
``tests/test_dataset_stats.py`` cross-check the computed values against a
manual count over known fixture data.
"""

from __future__ import annotations

from ._dataset_stats.aggregation import compute_dataset_stats
from ._dataset_stats.models import DatasetStats

__all__ = [
    "DatasetStats",
    "compute_dataset_stats",
]
