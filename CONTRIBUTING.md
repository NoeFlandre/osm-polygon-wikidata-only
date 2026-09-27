# Contributing

Thank you for improving `osm-polygon-wikidata-only`. The project values small,
well-tested changes that preserve dataset completeness and deterministic output.

## Development workflow

1. Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` from the repository root.
3. Write a failing test that expresses the required behavior.
4. Make the smallest change that passes it, then refactor while green.
5. Run the complete local quality gate documented in
   [`docs/development.md`](docs/development.md).

Before requesting review, run `just quality-gauntlet`. It is the canonical,
fail-fast sequence: baseline, Ruff, ty, tests, acceptance tests, architecture
checks, CRAP, mutation tests, smoke test, and diff review.

The CRAP stage (`just crap-report`) enforces two explicit rules per function:
cyclomatic complexity must be at most 5, and any function with complexity 3 or
more must have at least 80% coverage. CRAP scores are printed for context. Every
file analysed by radon must appear in the coverage report, otherwise the gate
fails with a "missing from the coverage report" error.

Do not commit PBFs, Parquet files, caches, tokens, or generated datasets. Keep
pull requests focused and explain compatibility effects explicitly. Changes to
polygon filtering, schemas, completeness, or publication semantics require a
separate design discussion; they are not ordinary refactors.

By contributing, you agree that your contribution is licensed under the
repository's Apache License 2.0.
