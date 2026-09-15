"""Pure order statistics, histogram bucketing, and summary construction.

Every function here takes NumPy sample arrays and returns the frozen
containers in :mod:`.models`. Nothing reads the filesystem, so the
summaries are reproducible from the samples alone.

Percentiles use NumPy's linear interpolation between the two closest
ranks. Values are rounded to :data:`ROUNDING_DIGITS` decimals so the
rendered card and the JSON report stay byte-stable for unchanged input
across platforms with different floating-point formatting.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .models import AreaHistogramBucket, AreaSummary, Distribution, SourceAreaSummary

# Enough precision for square metres and degrees, few enough digits that
# the last bits of a float never reach the published output.
ROUNDING_DIGITS = 6

# Half-open log-scale decade edges, in square metres, held as their
# powers of ten so both the edge and its label come from one source.
_HISTOGRAM_EXPONENTS: tuple[int, ...] = tuple(range(13))


def rounded(value: float) -> float:
    """Round one published float to the documented precision."""
    return round(float(value), ROUNDING_DIGITS)


def build_distribution(values: np.ndarray) -> Distribution:
    """Return the order statistics of ``values`` (empty input yields zeros)."""
    if values.size == 0:
        return Distribution()
    percentiles = np.percentile(values, [50.0, 95.0, 99.0], method="linear")
    return Distribution(
        minimum=rounded(values.min()),
        maximum=rounded(values.max()),
        mean=rounded(values.mean()),
        median=rounded(percentiles[0]),
        p95=rounded(percentiles[1]),
        p99=rounded(percentiles[2]),
    )


def build_area_summary(areas: np.ndarray, *, null_count: int) -> AreaSummary:
    """Return the surface summary over the recorded ``area_m2`` values."""
    if areas.size == 0:
        return AreaSummary(null_count=null_count)
    percentiles = np.percentile(areas, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0], method="linear")
    return AreaSummary(
        total_m2=rounded(areas.sum()),
        minimum_m2=rounded(areas.min()),
        maximum_m2=rounded(areas.max()),
        mean_m2=rounded(areas.mean()),
        median_m2=rounded(percentiles[3]),
        p1_m2=rounded(percentiles[0]),
        p5_m2=rounded(percentiles[1]),
        p25_m2=rounded(percentiles[2]),
        p75_m2=rounded(percentiles[4]),
        p95_m2=rounded(percentiles[5]),
        p99_m2=rounded(percentiles[6]),
        non_positive_count=int(np.count_nonzero(areas <= 0.0)),
        below_one_m2_count=int(np.count_nonzero((areas > 0.0) & (areas < 1.0))),
        null_count=null_count,
    )


def build_area_histogram(areas: np.ndarray) -> tuple[AreaHistogramBucket, ...]:
    """Bucket ``areas`` into the stable degree-of-ten histogram.

    The bucket list is identical for every input, including an empty
    one, so a diff of two reports compares like with like.
    """
    buckets = [
        AreaHistogramBucket(
            label="non_positive",
            lower_m2=None,
            upper_m2=0.0,
            count=int(np.count_nonzero(areas <= 0.0)),
        )
    ]
    buckets.extend(_decade_buckets(areas))
    return tuple(buckets)


def _decade_buckets(areas: np.ndarray) -> list[AreaHistogramBucket]:
    positive = areas[areas > 0.0]
    buckets = [
        AreaHistogramBucket(
            label="0_to_1_m2",
            lower_m2=0.0,
            upper_m2=1.0,
            count=int(np.count_nonzero(positive < 1.0)),
        )
    ]
    for lower_exponent, upper_exponent in pairwise(_HISTOGRAM_EXPONENTS):
        lower = _edge(lower_exponent)
        upper = _edge(upper_exponent)
        buckets.append(
            AreaHistogramBucket(
                label=f"1e{lower_exponent}_to_1e{upper_exponent}_m2",
                lower_m2=lower,
                upper_m2=upper,
                count=int(np.count_nonzero((positive >= lower) & (positive < upper))),
            )
        )
    final_exponent = _HISTOGRAM_EXPONENTS[-1]
    final = _edge(final_exponent)
    buckets.append(
        AreaHistogramBucket(
            label=f"ge_1e{final_exponent}_m2",
            lower_m2=final,
            upper_m2=None,
            count=int(np.count_nonzero(positive >= final)),
        )
    )
    return buckets


def _edge(exponent: int) -> float:
    """Return the square-metre value of one decade edge."""
    return float(10**exponent)


def build_source_summary(
    source_pbf: str, *, polygon_count: int, areas: np.ndarray
) -> SourceAreaSummary:
    """Return one ``source_pbf`` breakdown row.

    ``polygon_count`` is every published row of that source. ``areas``
    holds only its finite ``area_m2`` samples, so a source whose areas
    are all missing still reports its real row count with zeroed area
    aggregates.
    """
    if areas.size == 0:
        return SourceAreaSummary(source_pbf, polygon_count, 0.0, 0.0, 0.0)
    return SourceAreaSummary(
        source_pbf=source_pbf,
        polygon_count=polygon_count,
        total_area_m2=rounded(areas.sum()),
        median_area_m2=rounded(np.percentile(areas, 50.0, method="linear")),
        maximum_area_m2=rounded(areas.max()),
    )


__all__ = [
    "ROUNDING_DIGITS",
    "build_area_histogram",
    "build_area_summary",
    "build_distribution",
    "build_source_summary",
    "rounded",
]
