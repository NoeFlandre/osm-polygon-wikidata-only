# Project Tooling Stack Design

## Goal

Make uv, Ruff, ty, pytest, pre-commit, Typer, Rich, tqdm, Just, and GitHub
Actions deliberate, documented parts of the project without changing the
stable processing CLI or dataset behavior.

## Compatibility boundary

The existing `osm-polygon-wikidata-only` command remains argparse-based.
Its commands, options, help output, exit codes, logging, progress heartbeats,
and data behavior are frozen contracts. In particular, `sync-dir` will not
gain interactive progress bars or Rich logging.

Typer, Rich, and tqdm instead belong to the existing read-only remote
reconciliation audit:

- Typer owns the audit command's options and help.
- Rich renders readable headings, summaries, and error messages.
- tqdm reports the potentially long per-region local augmentation scan, and
  is disabled automatically when stderr is not an interactive terminal.

The audit is installed as `osm-polygon-wikidata-only-audit-remote`; the
existing `scripts/audit_remote.py` remains a thin compatibility entry point.

## Quality command ownership

The `Justfile` is the single human-facing command catalog. It provides:

- `sync`
- `test`
- `coverage`
- `lint`
- `format`
- `format-check`
- `typecheck`
- `build`
- `check`

Every Python command is executed through uv. `check` is non-mutating and
matches CI: frozen dependency sync, tests with coverage, Ruff lint and format
checks, ty, build, and `git diff --check`.

Pre-commit uses repository-local hooks that call the same uv-managed tools.
Fast commit-time checks are Ruff lint, Ruff format check, and ty. The complete
pytest/build matrix remains in `just check` and GitHub Actions.

GitHub Actions installs uv and Just, performs a frozen sync, and invokes
individual Just recipes so failures retain clear step names while command
definitions stay centralized.

## Safety and testing

Tests first freeze:

- direct dependencies and operator entry point;
- the complete Just recipe surface and uv-only Python commands;
- pre-commit hook stages and commands;
- CI's use of Just;
- Typer audit help, read-only planning output, progress iteration, and
  actionable failures;
- the unchanged production argparse help and command contracts.

No test or implementation may access live Hugging Face, Wikimedia, or the
external data root. The full existing suite, coverage, Ruff, ty, frozen sync,
build, pre-commit, Just, and whitespace gates must pass before commit and push.
