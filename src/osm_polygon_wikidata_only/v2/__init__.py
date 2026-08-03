"""Version 2 dataset support.

The package is deliberately separate from the V1 pipeline. Importing it does
not change V1 defaults or publication paths.
"""

from .config import (
    V2_CACHE_CONTRACT_VERSION,
    V2_CONTRACT_VERSION,
    V2_REPO_ID,
    DatasetVersion,
)

__all__ = [
    "V2_CACHE_CONTRACT_VERSION",
    "V2_CONTRACT_VERSION",
    "V2_REPO_ID",
    "DatasetVersion",
]
