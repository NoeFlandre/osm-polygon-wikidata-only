# Local Pipeline Performance Design

## Goal

Reduce CPU, memory, and filesystem overhead without changing polygon
eligibility, enrichment selection, output rows, Parquet schemas, or manifests.

## Design

`process_pbf` will consume `PBFReader.iter_polygon_candidates` directly,
converting each candidate immediately and preserving source order and the
existing limit. A compatibility fallback keeps collect-only test readers
working.

Article metadata is immutable for a `(qid, language, page, revision)` tuple.
It will be built once and referenced by every polygon link, eliminating
repeated hashes, text metrics, and JSON serialisation for duplicate QIDs.
Flat dataclass rows will use shallow mappings rather than deep `asdict` copies,
and Parquet writing will not copy an already materialised row list.

`orchestrate` will load the manifest once, use that mapping for skip decisions,
and pass it through successful PBF writes. Direct `process_pbf` calls retain
their existing load/save behavior.

## Invariants and tests

TDD tests will prove stream order and limits, repeated-QID row equivalence,
iterator-safe Parquet output, and one manifest load per directory run. The
full suite, Ruff, formatter, and mypy will run without a real PBF or network.
