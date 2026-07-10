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

Do not commit PBFs, Parquet files, caches, tokens, or generated datasets. Keep
pull requests focused and explain compatibility effects explicitly. Changes to
polygon filtering, schemas, completeness, or publication semantics require a
separate design discussion; they are not ordinary refactors.

By contributing, you agree that your contribution is licensed under the
repository's Apache License 2.0.
