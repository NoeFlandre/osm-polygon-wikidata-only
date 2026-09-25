"""Let the nested preprocessing suite be collected from the repository root.

The preprocessing project has its own environment (``just preprocessing-check``).
When pytest runs from the repository root instead, the package is not installed
and its ``duckdb`` dependency may be absent. Put ``preprocessing/src`` on the
import path, and skip collecting these tests when ``duckdb`` is unavailable so
the root run reports no collection errors.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("osm_polygon_wikidata_only_preprocessing") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

collect_ignore_glob = ["test_*.py"] if importlib.util.find_spec("duckdb") is None else []
