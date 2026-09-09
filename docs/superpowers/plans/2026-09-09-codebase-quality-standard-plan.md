# Codebase Quality Standard Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Improve repository quality and reproducibility without changing public behavior, data formats, or remote/data state.

**Architecture:** Keep the root package and nested preprocessing package as separate distribution boundaries. Add a small declarative quality-scope manifest consumed by validation/tests and generated command arguments, while preserving existing command entry points. Add artifact-level packaging and site-assembly checks where source-tree tests are insufficient.

**Tech Stack:** Python 3.12, pytest, Ruff, ty, uv lockfiles, Hatchling, MkDocs Material, Just, GitHub Actions.

---

### Task 1: Establish a reproducible baseline and preserve the pilot boundary

**Files:**
- Modify: tests/quality/test_ner_quality_scope.py only if the baseline exposes a stale assertion.
- Modify: docs/development.md only to record the exact runnable gate order.
- Test: existing tracked test and quality commands.

- [ ] Step 1: Capture the staged/unstaged boundary and run read-only checks.

    git diff --cached --check
    git diff --check
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q
    uv run --no-project ruff check src tests scripts preprocessing/src preprocessing/tests
    uv run --no-project ruff format --check src tests scripts preprocessing/src preprocessing/tests

Expected: no merge conflicts; any dependency or network failure is recorded with its command and is not treated as a code failure.

- [ ] Step 2: Write one regression assertion for the pilot boundary if needed.

The assertion must require the pilot model/revision and reject the later model path, without importing deleted modules or changing runtime behavior.

- [ ] Step 3: Re-run the focused NER and quality-scope tests.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q tests/ner tests/grid5000/test_ner_controller.py tests/quality/test_ner_quality_scope.py

Expected: all selected tests pass.

- [ ] Step 4: Commit only the baseline/documentation adjustment if one was needed.

    git add tests/quality/test_ner_quality_scope.py docs/development.md
    git commit -m "test: pin the pilot quality boundary"

### Task 2: Centralize and validate deterministic quality scopes

**Files:**
- Create: scripts/quality/scope_manifest.py
- Modify: pyproject.toml
- Modify: Justfile
- Modify: tests/quality/test_ner_quality_scope.py
- Create: tests/quality/test_scope_manifest.py
- Test: tests/quality/test_scope_manifest.py

- [ ] Step 1: Add RED tests for exact manifest validation.

The tests must cover every source path exists, every test path exists, no path appears twice, and the pilot scope contains only the committed pilot modules. They must import the new manifest module so they fail before that module exists.

- [ ] Step 2: Run the manifest tests and confirm the expected import failure.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q tests/quality/test_scope_manifest.py

Expected: collection fails because scripts.quality.scope_manifest does not yet exist.

- [ ] Step 3: Implement the smallest typed manifest API.

Define immutable scope records with name, source_paths, and test_paths, plus a validator that resolves paths relative to the repository root and raises one clear ValueError per invalid scope. Keep existing scope names and paths as data; do not change mutation or CRAP semantics.

- [ ] Step 4: Run the manifest tests and existing quality-scope tests.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project pytest -q tests/quality/test_scope_manifest.py tests/quality/test_ner_quality_scope.py

Expected: PASS.

- [ ] Step 5: Replace duplicated NER scope literals with the manifest.

Use the manifest in the NER scope contract tests and generate relevant Just/Mutmut argument lists without changing the public just crap-ner or just mutation commands. Keep non-NER scopes unchanged until an equivalent manifest entry and regression test exists.

- [ ] Step 6: Run lint, format, and the affected quality recipes.

    uv run --no-project ruff check scripts/quality tests/quality
    uv run --no-project ruff format --check scripts/quality tests/quality
    just crap-ner

Expected: all available checks pass; unavailable dependency resolution is reported precisely.

### Task 3: Bring the nested preprocessing package into representative gates

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

### Task 4: Verify installed artifacts and deterministic documentation assembly

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

