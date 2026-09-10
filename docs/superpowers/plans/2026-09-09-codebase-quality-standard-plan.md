# Codebase Quality Standard Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Improve repository quality and reproducibility without changing public behavior, data formats, or remote/data state.

**Architecture:** Keep the root package and nested preprocessing package as separate distribution boundaries. Add a small declarative quality-scope manifest consumed by validation/tests and generated command arguments, while preserving existing command entry points. Add artifact-level packaging and site-assembly checks where source-tree tests are insufficient.

**Tech Stack:** Python 3.12, pytest, Ruff, ty, uv lockfiles, Hatchling, MkDocs Material, Just, GitHub Actions.

---

### Task 1: Establish a reproducible baseline

**Files:**
- Modify: docs/development.md only to record the exact runnable gate order.
- Test: existing tracked test and quality commands.

- [ ] Step 1: Capture the staged/unstaged boundary and run read-only checks.

    git diff --cached --check
    git diff --check
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q
    uv run --no-project ruff check src tests scripts preprocessing/src preprocessing/tests
    uv run --no-project ruff format --check src tests scripts preprocessing/src preprocessing/tests

Expected: no merge conflicts; any dependency or network failure is recorded with its command and is not treated as a code failure.

### Task 2: Bring the nested preprocessing package into representative gates

**Files:**
- Modify: .github/workflows/ci.yml
- Modify: preprocessing/pyproject.toml only when required for clean package/test invocation.
- Modify: preprocessing/tests/test_cli.py and related tests only to remove bare-import coupling.
- Create: preprocessing/tests/conftest.py if shared fixtures/helpers are required.
- Modify: Justfile with a preprocessing-check recipe.
- Modify: docs/development.md

- [ ] Step 1: Add a failing CI/recipe contract test.

Extend configuration tests to require a frozen-lock preprocessing check that runs its tests, Ruff, ty, and wheel smoke separately from the root environment.

- [ ] Step 2: Run the contract test and verify it fails on the missing recipe/job.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q tests/test_documentation.py tests/quality/test_packaging.py

Expected: FAIL identifying the absent preprocessing gate.

- [ ] Step 3: Add the separate frozen-lock recipe and CI job.

The recipe must execute from preprocessing with uv sync --frozen, run its own tests, Ruff, ty, and build a wheel. The CI job must call only that recipe, not the root package environment.

- [ ] Step 4: Remove bare helper imports without changing test behavior.

Move shared test helpers into preprocessing/tests/conftest.py or import them through the package test namespace. Add a regression test for the CLI entry point from a clean installed package.

- [ ] Step 5: Run preprocessing checks and configuration tests.

    just preprocessing-check
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q tests/test_documentation.py tests/quality/test_packaging.py

Expected: PASS when locked dependencies are available.

### Task 3: Verify installed artifacts and deterministic documentation assembly

**Files:**
- Modify: tests/test_packaging.py
- Create: scripts/build_docs_site.py
- Modify: Justfile
- Modify: .github/workflows/docs.yml
- Modify: tests/test_documentation.py
- Modify: docs/development.md

- [ ] Step 1: Add RED tests for installed resources and site assembly.

Build a wheel into a temporary directory, install it into an isolated target, import the package from outside the checkout, and require the packaged GeoJSON and hero assets to resolve. Require the site assembly helper to copy the three presentation files and assets to the exact Pages paths.

- [ ] Step 2: Run the new tests and confirm they fail before implementation.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q tests/test_packaging.py tests/test_documentation.py

Expected: FAIL on the missing isolated-install/site-assembly behavior.

- [ ] Step 3: Implement the isolated wheel/resource smoke and one site-assembly helper.

Keep current package resource paths and Pages output paths unchanged. Make the helper deterministic, fail closed on missing assets, and avoid reading production data.

- [ ] Step 4: Make local docs and CI call the same helper.

just docs and .github/workflows/docs.yml must invoke the same assembly path; the workflow may upload the resulting directory but must not duplicate copy commands.

- [ ] Step 5: Run the full local quality gate.

    just quality-gauntlet
    just preprocessing-check
    uv build
    just docs
    git diff --check

Expected: full suite, lint, format, typing, docs, package build, configured CRAP scopes, and mutation gate pass. Report exact blockers if a gate cannot run because the environment lacks a dependency or network access.

- [ ] Step 6: Perform adversarial review and commit the validated result.

Review the final diff for duplicated scope declarations, new wrappers, broad exceptions, machine-specific paths, hidden global state, stale docs, and gates that are declared but not executed. Then commit:

    git add .github/workflows/ci.yml .github/workflows/docs.yml Justfile pyproject.toml docs/development.md scripts/quality tests preprocessing
    git commit -m "refactor: strengthen repository quality boundaries"
