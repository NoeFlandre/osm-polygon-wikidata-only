# Development guide

## Setup

The project supports Python 3.12 and uses `uv` for reproducible environments:

```bash
uv sync --frozen
```

Install [`just`](https://just.systems/) as the project command runner, then
install the repository's fast pre-commit hooks:

```bash
uv run pre-commit install
just --list
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
an injected fake opener/session, as `tests/enrichment/test_wikimedia_auth.py` and
`tests/cli/test_dependencies.py` do. Tests must assert that exception messages and
representations do not contain passwords. Do not add network-dependent login
tests to the automated suite.

## Test-driven changes

Use red-green-refactor: add one focused failing test, confirm the expected
failure, implement the smallest behavior, rerun the focused test, then run the
full gate. For structural changes, first add characterization coverage around
the boundary, move one responsibility, and prove output equivalence.

## Quality gate

```bash
just check
```

The recipes use uv to run pytest with coverage, Ruff lint and format checks, ty
over `src` and maintained `scripts`, the strict MkDocs build, the package build,
and the whitespace gate. GitHub Actions invokes the same quality recipes and
publishes the documentation site from `main`. Pre-commit intentionally runs
only the fast Ruff and ty subset; `just check` remains the complete gate.
Concretely, the gate owns `uv run ruff check src tests scripts`,
`uv run ruff format --check src tests scripts`, `uv run ty check src scripts`, `uv build`, and
`git diff --check`; contributors do not need to maintain a separate command
sequence.

Build the documentation site locally without starting a server:

```bash
just docs
```

Strict typing applies to `src/`. Decoded third-party JSON may begin as `Any`,
but public and internal boundaries should narrow it immediately.

## Operator audit

`osm-polygon-wikidata-only-audit-remote` is a read-only maintenance command:

```bash
uv run osm-polygon-wikidata-only-audit-remote \
  --data-root "$OSM_POLYGON_DATA_ROOT"
```

Typer owns this command's option parsing, Rich renders its report, and tqdm
shows interactive progress while local augmentation state is checked. tqdm is
disabled automatically when stderr is not a terminal, keeping captured logs
clean. The command calculates a reconciliation plan but never uploads, deletes,
or rewrites data. These interface libraries are deliberately not used by the
stable argparse production CLI or its `sync-dir` logging.

## Release checklist

Verify the quality gate, inspect wheel contents for `py.typed` and the license,
confirm CLI help from the built artifact, review dataset compatibility, and
update the version intentionally. Publishing packages or datasets is a
maintainer action and is not part of ordinary test execution.
