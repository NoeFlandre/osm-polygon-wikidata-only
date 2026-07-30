# Project Tooling Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make uv, Ruff, ty, pytest, pre-commit, Typer, Rich, tqdm, Just, and GitHub Actions first-class project tooling without changing the production data CLI.

**Architecture:** Preserve the argparse processing CLI and implement a focused Typer/Rich/tqdm operator UI around the existing read-only remote audit. Centralize developer commands in a Justfile, call those recipes from local hooks and CI, and freeze the stack with packaging, documentation, and CLI tests.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, pre-commit, Typer, Rich, tqdm, Just, GitHub Actions.

---

### Task 1: Freeze the tooling contracts

**Files:**
- Modify: `tests/test_packaging.py`
- Modify: `tests/test_documentation.py`
- Create: `tests/cli/test_audit_remote.py`

- [x] Add tests asserting the direct runtime/dev dependencies, installed audit command, Just recipes, pre-commit hooks, CI recipe calls, documentation, audit help, deterministic report, progress traversal, and error exits.
- [x] Run the focused tests and confirm they fail because the requested stack and audit module are absent.

### Task 2: Add the operator audit command

**Files:**
- Create: `src/osm_polygon_wikidata_only/cli/audit_remote.py`
- Modify: `scripts/audit_remote.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [x] Implement the smallest Typer command that delegates to the existing reconciliation model, uses Rich for output, iterates sorted stems through tqdm, and never mutates local or remote state.
- [x] Keep the script as a thin wrapper and add the installed entry point.
- [x] Run the audit tests and existing frozen CLI contract tests until green.

### Task 3: Centralize quality commands

**Files:**
- Create: `Justfile`
- Create: `.pre-commit-config.yaml`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [x] Add non-mutating Just quality recipes and a separate formatting recipe.
- [x] Add local pre-commit hooks for Ruff lint, Ruff format checking, and ty.
- [x] Make CI install Just and invoke the shared recipes with clear step names.
- [x] Run the packaging contract tests until green.

### Task 4: Document and verify the public workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/development.md`
- Modify: `docs/architecture.md`

- [x] Document installation, the audit command, every quality tool, the Just recipes, pre-commit installation, and CI parity without exposing private paths.
- [x] Run documentation and focused tooling tests.
- [x] Run `just check`, `uv run pre-commit run --all-files`, and the complete pytest/Ruff/ty/build/whitespace verification matrix.
- [x] Review the diff for production CLI or dataset contract changes, commit coherently on main, and push only after all gates pass.
