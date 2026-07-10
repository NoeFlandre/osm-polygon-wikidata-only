# Development guide

## Setup

The project supports Python 3.12 and uses `uv` for reproducible environments:

```bash
uv sync --frozen
```

Production data belongs under `OSM_POLYGON_DATA_ROOT`; tests use temporary
directories and in-memory clients. The automated suite must not access the
network or require a real PBF collection.

## Test-driven changes

Use red-green-refactor: add one focused failing test, confirm the expected
failure, implement the smallest behavior, rerun the focused test, then run the
full gate. For structural changes, first add characterization coverage around
the boundary, move one responsibility, and prove output equivalence.

## Quality gate

```bash
uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
```

Strict typing applies to `src/`. Decoded third-party JSON may begin as `Any`,
but public and internal boundaries should narrow it immediately.

## Release checklist

Verify the quality gate, inspect wheel contents for `py.typed` and the license,
confirm CLI help from the built artifact, review dataset compatibility, and
update the version intentionally. Publishing packages or datasets is a
maintainer action and is not part of ordinary test execution.
