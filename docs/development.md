# Development guide

## Setup

The project supports Python 3.12 and uses `uv` for reproducible environments:

```bash
uv sync --frozen
```

Production data belongs under `OSM_POLYGON_DATA_ROOT`; tests use temporary
directories and in-memory clients. The automated suite must not access the
network or require a real PBF collection.

## Wikimedia authentication environment

Production operators can provide a Bot Password using the complete generated
username and secret:

```bash
export WIKIMEDIA_BOT_USERNAME='AccountName@osm-polygon-pipeline'
read -rs WIKIMEDIA_BOT_PASSWORD
export WIKIMEDIA_BOT_PASSWORD
```

The pair is optional but all-or-nothing. With neither variable, clients remain
anonymous at 180 requests per minute. With both, dependency construction shares
one authenticated session and adaptive scheduler across Wikidata and Wikipedia.
The authenticated ceiling defaults to 1,200 requests per minute and can be
overridden with a positive number:

```bash
export WIKIMEDIA_REQUESTS_PER_MINUTE=600
```

Never use live credentials in tests. Pass an explicit environment mapping and
an injected fake opener/session, as `tests/test_wikimedia_auth.py` and
`tests/test_dependencies.py` do. Tests must assert that exception messages and
representations do not contain passwords. Do not add network-dependent login
tests to the automated suite.

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
