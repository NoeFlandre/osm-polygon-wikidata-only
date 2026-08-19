# Mutation and CRAP Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand deterministic enrichment tests and add a reproducible mutation-testing and CRAP-score gate with zero surviving mutants in the selected pure parsing scope.

**Architecture:** Keep production behavior unchanged. Strengthen the existing Wikipedia/Wikidata parsing and text-cleaning tests, then add a small standard-library CRAP reporter that combines function-level coverage JSON with Radon cyclomatic complexity JSON. Mutmut will mutate only those three pure modules, run the existing test suite without coverage overhead, and fail on any status other than `killed`; the scope is explicit so the bounded local Mac run remains resource-safe.

**Tech Stack:** Python 3.12, pytest/pytest-cov, mutmut 3.7, Radon 6, Ruff, ty, uv, just.

---

### Task 1: Add the CRAP reporter contract tests

**Files:**
- Create: `tests/quality/test_crap_score.py`
- Create: `scripts/quality/crap_score.py`

- [ ] **Step 1: Write the failing tests**

Add tests for the pure functions that the reporter will expose:

```python
from scripts.quality.crap_score import crap_score, evaluate_threshold


def test_crap_score_uses_standard_formula() -> None:
    assert crap_score(10, 0.75) == 10.15625


def test_evaluate_threshold_returns_worst_functions() -> None:
    entries = [
        {"name": "safe", "crap": 4.0},
        {"name": "risky", "crap": 31.0},
    ]
    assert evaluate_threshold(entries, maximum=30.0) == [entries[1]]
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache uv run pytest --no-cov -q tests/quality/test_crap_score.py
```

Expected: collection fails because `scripts.quality.crap_score` does not yet exist.

- [ ] **Step 3: Implement the smallest reporter**

Implement `crap_score(complexity: int, coverage: float) -> float`, `evaluate_threshold(...)`, JSON loading for coverage and Radon reports, a concise table, and a CLI accepting `--coverage`, `--complexity`, and `--maximum`. Use statement coverage from coverage.py’s per-function `summary.percent_statements_covered`; reject malformed or missing function records with a non-zero error.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run the same command and expect both tests to pass. Then run Ruff and ty on the new script/tests.

---

### Task 2: Expand pure parsing and cleaning tests

**Files:**
- Modify: `tests/enrichment/test_wikipedia_split.py`
- Modify: `tests/enrichment/test_wikidata_split.py`
- Modify: `tests/enrichment/test_enrichment.py` only where a missing text-cleaning branch has no existing focused assertion.

- [ ] **Step 1: Add one focused test at a time for currently uncovered branches**

Cover malformed revision/page payloads, parse text in both string and `{"*": ...}` forms, missing/invalid batch query data, normalized and redirect alias chains, redirect cycles, missing pages, full URL fallback quoting, missing thumbnail metadata, empty extracts, malformed Wikidata entities, aliases, non-language sitelinks, legacy `be_x_old`, and whitespace/template/HTML cleaning edge cases.

- [ ] **Step 2: Run each focused test file and verify RED before implementation changes**

These are characterization tests for existing behavior. If a new assertion fails, correct the fixture/assertion to match the documented current contract; do not alter production behavior to make a characterization test pass.

- [ ] **Step 3: Run the focused tests GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache uv run pytest --no-cov -q tests/enrichment/test_wikipedia_split.py tests/enrichment/test_wikidata_split.py tests/enrichment/test_enrichment.py
```

---

### Task 3: Add reproducible mutation and CRAP commands

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `justfile`
- Modify: `docs/development.md`
- Create: `scripts/quality/mutation_gate.py`
- Create: `tests/quality/test_mutation_gate.py`

- [ ] **Step 1: Add the mutation-gate RED tests**

Test that a report containing only `killed` statuses succeeds, while `survived`, `no tests`, `timeout`, `suspicious`, or `not checked` statuses fail with the mutant names in the error. Keep this script stdlib-only so it does not add runtime dependencies.

- [ ] **Step 2: Run the mutation-gate tests and confirm RED**

Run the focused quality tests with `--no-cov`; collection must fail before the script exists.

- [ ] **Step 3: Implement the minimal gate and configuration**

Add dev dependencies `mutmut>=3.7,<4` and `radon>=6,<7`. Configure mutmut with the three explicit pure-module paths, `mutate_only_covered_lines = true`, and `pytest_add_cli_args = ["--no-cov", "-q"]`. Add `just mutation` using `--max-children 2` and `just crap` that runs focused coverage, Radon JSON, and the reporter with maximum CRAP `30.0`. Add `just quality-advanced` to run both. Document that this is an advanced bounded scope, not a claim about every orchestration module.

- [ ] **Step 4: Run the focused quality tests GREEN**

Run the quality test files with `--no-cov`, then run Ruff/ty. Do not include generated `mutants/`, coverage JSON, or Radon JSON in Git.

---

### Task 4: Run advanced quality end to end

**Files:**
- No source changes expected; generated reports remain ignored/untracked.

- [ ] **Step 1: Run the focused coverage and CRAP gate**

Run `UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache just crap`; require a zero exit code and maximum CRAP below `30.0`.

- [ ] **Step 2: Run mutation testing with bounded concurrency**

Run `UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache just mutation`; require every generated mutant in the configured scope to be `killed` and no survivors, timeouts, suspicious, skipped, or untested mutants.

- [ ] **Step 3: Run the full regression gate**

Run `UV_CACHE_DIR=/tmp/osm-wikidata-uv-cache just check`, inspect `git diff --check`, and verify the pre-existing `presentations/` untracked files remain untouched.

- [ ] **Step 4: Review the final reports**

Record test totals, coverage, mutation counts, worst CRAP score, exact target modules, and any non-blocking skipped tests in the final response. Do not claim whole-repository mutation cleanliness beyond the configured pure-module scope.
