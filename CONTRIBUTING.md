# Contributing

Thank you for your help with `osm-polygon-wikidata-only`. The project accepts small changes with good tests. A change must keep the dataset complete and the output deterministic.

## Development workflow

1. Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` from the repository root.
3. Write a test that fails and shows the required behavior.
4. Make the smallest change that passes the test.
5. Refactor the code while the tests pass.
6. Run the complete local quality gate. [`docs/development.md`](docs/development.md) describes it.

Before you request a review, run `just quality-gauntlet`. It is the standard fail-fast sequence. The sequence has these stages: baseline, Ruff, ty, tests, coverage floor, architecture checks, CRAP, mutation tests, smoke test, and diff review.

The CRAP stage (`just crap-report`) applies two rules to each function:

- The cyclomatic complexity must be 5 or less.
- A function with a complexity of 3 or more must have at least 80% coverage.

The stage prints the CRAP scores for information. Each file that radon analyses must be in the coverage report. If a file is not in the report, the gate fails with the error "missing from the coverage report".

## Rules for a pull request

Do not commit PBFs, Parquet files, caches, tokens, or generated datasets. Keep each pull request focused. Describe the compatibility effects. A change to polygon filtering, schemas, completeness, or publication behavior is not an ordinary refactor. Start a separate design discussion for such a change.

When you contribute, you agree to license your contribution under the Apache License 2.0 of the repository.
