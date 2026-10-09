set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

export QUALITY_RUNTIME_DIR := env_var_or_default("QUALITY_RUNTIME_DIR", "../quality-runtime")
QUALITY_RUNTIME_PATH := absolute_path(QUALITY_RUNTIME_DIR)
export QUALITY_REPORT_DIR := absolute_path(env_var_or_default("QUALITY_REPORT_DIR", QUALITY_RUNTIME_PATH + "/reports"))
export QUALITY_CACHE_DIR := absolute_path(env_var_or_default("QUALITY_CACHE_DIR", QUALITY_RUNTIME_PATH + "/cache"))
export UV_CACHE_DIR := absolute_path(env_var_or_default("UV_CACHE_DIR", QUALITY_CACHE_DIR + "/uv-cache"))
export QUALITY_TMP_DIR := absolute_path(env_var_or_default("QUALITY_TMP_DIR", QUALITY_RUNTIME_PATH + "/tmp"))
export TMPDIR := QUALITY_TMP_DIR
export RUFF_CACHE_DIR := absolute_path(env_var_or_default("RUFF_CACHE_DIR", QUALITY_CACHE_DIR + "/ruff"))
export HYPOTHESIS_STORAGE_DIRECTORY := absolute_path(env_var_or_default("HYPOTHESIS_STORAGE_DIRECTORY", QUALITY_CACHE_DIR + "/hypothesis"))
export MPLCONFIGDIR := absolute_path(env_var_or_default("MPLCONFIGDIR", QUALITY_CACHE_DIR + "/matplotlib"))
export UV_PYTHON_INSTALL_DIR := absolute_path(env_var_or_default("UV_PYTHON_INSTALL_DIR", QUALITY_CACHE_DIR + "/python"))
COVERAGE_FLOOR_EXEMPTIONS := ""
CRAP_FLAGS := "--maximum 6 --max-complexity 5 --min-coverage 0.8 --min-coverage-complexity 3"
MUTMUT_MAX_CHILDREN := env_var_or_default("MUTMUT_MAX_CHILDREN", "2")

default:
    @just --list

quality-runtime:
    @mkdir -p "{{ QUALITY_RUNTIME_PATH }}" "{{ QUALITY_REPORT_DIR }}" "{{ QUALITY_CACHE_DIR }}" "{{ QUALITY_TMP_DIR }}"

preprocessing-check: quality-runtime
    uv sync --frozen --directory preprocessing
    COVERAGE_FILE="{{ QUALITY_REPORT_DIR }}/preprocessing.coverage" uv run --directory preprocessing --frozen python -m pytest -q -p no:cacheprovider --basetemp="{{ TMPDIR }}/preprocessing-pytest" --cov=osm_polygon_wikidata_only_preprocessing --cov-branch --cov-fail-under=0 --cov-report=term-missing
    uv run python -m coverage json --data-file="{{ QUALITY_REPORT_DIR }}/preprocessing.coverage" -o "{{ QUALITY_REPORT_DIR }}/preprocessing-coverage.json"
    uv run --frozen ruff check preprocessing/src preprocessing/tests
    uv run --frozen ruff format --check preprocessing/src preprocessing/tests
    uv run --frozen ty check --project preprocessing --python preprocessing/.venv preprocessing/src
    just preprocessing-package-smoke

preprocessing-package-smoke: quality-runtime
    @smoke_dir=$(mktemp -d "{{ TMPDIR }}/preprocessing-package-smoke.XXXXXX") && \
        trap 'rm -rf "$smoke_dir"' EXIT && \
        UV_OFFLINE=1 uv build --directory preprocessing --out-dir "$smoke_dir/dist" && \
        UV_OFFLINE=1 uv venv --python "$(uv run python -c 'import sys; print(sys.executable)')" "$smoke_dir/venv" && \
        UV_OFFLINE=1 uv pip install --no-deps --python "$smoke_dir/venv/bin/python" "$smoke_dir/dist/"*.whl && \
        "$smoke_dir/venv/bin/python" scripts/quality/package_smoke.py \
        --distribution osm-polygon-wikidata-only-preprocessing \
        --package osm_polygon_wikidata_only_preprocessing \
        --entry-point osm-polygon-wikidata-only-preprocessing
sync: quality-runtime
    uv sync --frozen

# The un-instrumented full run also enforces the per-test time budget: coverage
# tracing inflates durations several-fold, so the budget is measured here.
baseline: quality-runtime
    uv sync --frozen
    uv run python -m pytest --no-cov -p no:cacheprovider --basetemp="{{ TMPDIR }}/baseline-pytest" -q -n auto --dist loadfile --durations=20 --junitxml="{{ QUALITY_REPORT_DIR }}/junit.xml"
    uv run python scripts/quality/slow_tests.py --junit-xml "{{ QUALITY_REPORT_DIR }}/junit.xml" --budget 3
    @git status --short --branch

test: quality-runtime
    uv run python -m pytest --no-cov -p no:cacheprovider --basetemp="{{ TMPDIR }}/test-pytest" -q

ruff: quality-runtime
    uv run ruff check src tests scripts preprocessing/src preprocessing/tests
    uv run ruff format --check src tests scripts preprocessing/src preprocessing/tests

coverage: quality-runtime
    COVERAGE_FILE="{{ QUALITY_REPORT_DIR }}/coverage-coverage" uv run python -m pytest --cov=osm_polygon_wikidata_only --cov=scripts --cov-report=term-missing --cov-report="json:{{ QUALITY_REPORT_DIR }}/coverage.json" -p no:cacheprovider --basetemp="{{ TMPDIR }}/coverage-pytest" -q

# The gauntlet's single coverage run over all of tests/ (testpaths = tests).
tests: quality-runtime
    COVERAGE_FILE="{{ QUALITY_REPORT_DIR }}/coverage-tests" uv run python -m pytest --cov=osm_polygon_wikidata_only --cov=scripts --cov-report=term-missing --cov-report="json:{{ QUALITY_REPORT_DIR }}/coverage.json" -p no:cacheprovider --basetemp="{{ TMPDIR }}/tests-pytest" -q -n auto --dist loadfile

# Per-file line-coverage floor over the root coverage produced by `just tests`.
# Exemptions are thin entry points that only run in a subprocess, where
# in-process coverage cannot observe them.
coverage-floor: quality-runtime
    @test -s "{{ QUALITY_REPORT_DIR }}/coverage.json" || { echo "Run just tests first to generate root coverage." >&2; exit 1; }
    uv run python scripts/quality/coverage_floor.py --coverage "{{ QUALITY_REPORT_DIR }}/coverage.json" --minimum 85 {{ COVERAGE_FLOOR_EXEMPTIONS }}

# Time the hot pure functions and the workload benchmarks (benchmarks/ is not
# part of the normal test run).
bench:
    uv run python -m pytest benchmarks --no-cov -p no:cacheprovider -q

# Write the pytest-benchmark JSON that CI uploads and compares.
bench-json out="bench.json":
    uv run python -m pytest benchmarks --no-cov -p no:cacheprovider -q --benchmark-json={{ out }}

# Compare a run with the committed baseline (benchmarks/baseline.json, recorded
# on the CI runner type). Fails on a >25% median regression only in enforce mode.
bench-compare mode="warn" out="bench.json":
    uv run python scripts/quality/bench_compare.py --baseline benchmarks/baseline.json --current {{ out }} --threshold 0.25 --mode {{ mode }}

# Local-only focused runs. The gauntlet does not repeat them: their files are
# part of the full `tests` run, which is the one coverage run in CI.
property-tests: quality-runtime
    uv run python -m pytest -q --no-cov -p no:cacheprovider --basetemp="{{ TMPDIR }}/property-pytest" -n auto --dist loadfile tests/property

acceptance-tests: quality-runtime
    uv run python -m pytest -q --no-cov -p no:cacheprovider --basetemp="{{ TMPDIR }}/acceptance-pytest" -n auto --dist loadfile tests/acceptance tests/pipeline/test_end_to_end.py tests/pipeline/test_sync_recovery_integration.py

# Local-only focused run of the architecture contract files; also part of `tests`.
architecture-tests: quality-runtime
    uv run python -m pytest -q --no-cov -p no:cacheprovider --basetemp="{{ TMPDIR }}/architecture-pytest" tests/contracts tests/test_mkdocs.py tests/test_documentation.py tests/test_docker.py

architecture-checks: quality-runtime
    just build
    just docs
    just package-smoke
    just preprocessing-check
    uv run python scripts/quality/architecture.py
    uv run python scripts/quality/architecture.py --source-root preprocessing/src/osm_polygon_wikidata_only_preprocessing --package osm_polygon_wikidata_only_preprocessing

# Canonical full-source CRAP reporting consumes coverage produced by the
# preceding root `tests` and nested preprocessing quality stages.
# Gate semantics (issue #116): every function must have cyclomatic complexity
# <= 5, and functions with complexity >= 3 must reach >= 80% coverage.
# CRAP scores are still printed for review. A radon path absent from the
# coverage report is a hard, explicit error.
crap-report: quality-runtime
    @test -s "{{ QUALITY_REPORT_DIR }}/coverage.json" || { echo "Run just tests first to generate root coverage." >&2; exit 1; }
    @test -s "{{ QUALITY_REPORT_DIR }}/preprocessing-coverage.json" || { echo "Run just preprocessing-check first to generate preprocessing coverage." >&2; exit 1; }
    uv run python -m radon cc --show-closures -j src scripts > "{{ QUALITY_REPORT_DIR }}/complexity.json"
    uv run python scripts/quality/crap_score.py --coverage "{{ QUALITY_REPORT_DIR }}/coverage.json" --complexity "{{ QUALITY_REPORT_DIR }}/complexity.json" {{ CRAP_FLAGS }}
    cd preprocessing && uv run --project .. python -m radon cc --show-closures -j src > "{{ QUALITY_REPORT_DIR }}/preprocessing-complexity.json"
    uv run python scripts/quality/crap_score.py --coverage "{{ QUALITY_REPORT_DIR }}/preprocessing-coverage.json" --complexity "{{ QUALITY_REPORT_DIR }}/preprocessing-complexity.json" {{ CRAP_FLAGS }}

# Standalone CRAP runs refresh both coverage reports before reporting.
crap-all: quality-runtime
    just tests
    just preprocessing-check
    just crap-report

# Compatibility names retained for tooling that used the historical scopes.
crap: crap-all
crap-sync: crap-all
crap-upload: crap-all
crap-quality: crap-all
crap-geography: crap-all
crap-geography-inputs: crap-all
crap-stats: crap-all
crap-sat: crap-all
crap-job: crap-all
crap-inventory: crap-all
crap-atomic: crap-all
crap-preprocessing: crap-all

# Validate the source-to-test mapping and pytest collection before mutmut runs.
# The gate refuses unreviewed survivors, timeouts, and untested mutants;
# equivalents need exact source-bound reviews.
mutation:
    uv run python scripts/quality/mutation_scope.py --max-children "{{ MUTMUT_MAX_CHILDREN }}"
    uv run python scripts/quality/mutation_scope.py results --all=true | uv run python -m scripts.quality.mutation_gate --equivalents scripts/quality/mutation_equivalents.json

smoke-test: quality-runtime
    uv run osm-polygon-wikidata-only --help
    uv run osm-polygon-wikidata-only sync-dir --help

diff-review:
    git diff --check
    @test -z "$(git diff --name-only --diff-filter=U)"
    @git status --short --branch

quality-gauntlet: quality-runtime
    uv run python scripts/quality/qa_gauntlet.py

# Fast local feedback: lint, types, the no-coverage suite, and diff hygiene.
# The full gauntlet remains the release-quality completion gate.
quality-fast: quality-runtime
    just ruff
    just ty
    just baseline
    just diff-review

# Compatibility alias retained for historical references in tooling.
qa-gauntlet: quality-gauntlet

# Run opt-in quality-strength checks; these are intentionally separate
# from `just check` because mutation testing is substantially slower.
quality-strength: crap-all mutation

quality-advanced: crap-all mutation

lint: quality-runtime
    uv run ruff check src tests scripts preprocessing/src preprocessing/tests

format: quality-runtime
    uv run ruff format src tests scripts preprocessing/src preprocessing/tests

format-check: quality-runtime
    uv run ruff format --check src tests scripts preprocessing/src preprocessing/tests

typecheck: quality-runtime
    uv run ty check src scripts tests

ty: quality-runtime
    uv run ty check src scripts tests

# Recompute and validate both dataset cards and reports without uploading.
release-stats-dry-run data_root: quality-runtime
    uv run osm-polygon-wikidata-only release-stats \
        --data-root "{{ data_root }}" \
        --confirm-repo NoeFlandre/osm-polygon-wikidata-only \
        --confirm-repo NoeFlandre/osm-polygon-wikidata-and-wikipedia

# Recompute, publish, and verify both dataset cards and reports.
release-stats data_root: quality-runtime
    uv run osm-polygon-wikidata-only release-stats \
        --data-root "{{ data_root }}" \
        --confirm-repo NoeFlandre/osm-polygon-wikidata-only \
        --confirm-repo NoeFlandre/osm-polygon-wikidata-and-wikipedia \
        --apply

# Fail on any known vulnerability in either locked dependency set.
audit: quality-runtime
    uv export --frozen --all-groups --all-extras --no-emit-project --quiet -o "{{ QUALITY_TMP_DIR }}/audit-requirements.txt"
    uvx pip-audit==2.9.0 --strict --disable-pip --require-hashes -r "{{ QUALITY_TMP_DIR }}/audit-requirements.txt"
    uv export --frozen --directory preprocessing --all-groups --no-emit-project --quiet -o "{{ QUALITY_TMP_DIR }}/audit-preprocessing-requirements.txt"
    uvx pip-audit==2.9.0 --strict --disable-pip --require-hashes -r "{{ QUALITY_TMP_DIR }}/audit-preprocessing-requirements.txt"

build: quality-runtime
    uv build

package-smoke: quality-runtime
    @smoke_dir=$(mktemp -d "{{ TMPDIR }}/package-smoke.XXXXXX") && \
        trap 'rm -rf "$smoke_dir"' EXIT && \
        UV_OFFLINE=1 uv build --out-dir "$smoke_dir/dist" && \
        UV_OFFLINE=1 uv venv --python "$(uv run python -c 'import sys; print(sys.executable)')" "$smoke_dir/venv" && \
        UV_OFFLINE=1 uv pip install --no-deps --python "$smoke_dir/venv/bin/python" "$smoke_dir/dist/"*.whl && \
        "$smoke_dir/venv/bin/python" scripts/quality/package_smoke.py \
        --distribution osm-polygon-wikidata-only \
        --package osm_polygon_wikidata_only \
        --resource hf/ne_110m_admin_0_countries.geojson \
        --resource assets/dataset_hero.png \
        --resource assets/dataset_hero_v2.png \
        --resource cli/grid5000.py \
        --resource cli/audit_containment.py \
        --entry-point osm-polygon-wikidata-only \
        --entry-point osm-polygon-wikidata-only-enforce-integrity \
        --entry-point osm-polygon-wikidata-only-audit-remote \
        --entry-point osm-polygon-wikidata-only-trackio \
        --entry-point osm-polygon-wikidata-and-wikipedia-trackio
docs: quality-runtime
    uv run python -m mkdocs build --strict --site-dir "{{ QUALITY_REPORT_DIR }}/site"
    uv run python scripts/assemble_docs_site.py --site-dir "{{ QUALITY_REPORT_DIR }}/site"

trackio: quality-runtime
    uv run osm-polygon-wikidata-only-trackio

# Build the minimal non-root runtime image; this never touches a data root.
docker-build:
    docker build --target runtime --tag osm-polygon-wikidata-only:local .

# Run the harmless default help command; this never touches a data root.
docker-help: docker-build
    docker run --rm osm-polygon-wikidata-only:local --help

# Build and run the development test target; this uses no production data.
docker-test:
    docker build --target development --tag osm-polygon-wikidata-only:dev .
    docker run --rm osm-polygon-wikidata-only:dev

# Run the development quality checks; this uses no production data.
docker-check:
    docker build --target development --tag osm-polygon-wikidata-only:dev .
    docker run --rm osm-polygon-wikidata-only:dev bash -lc \
        'task_dir=/tmp/osm-polygon-wikidata-only-docker-check && \
        mkdir -p "$task_dir/tmp" "$task_dir/uv-cache" && \
        export TMPDIR="$task_dir/tmp" UV_CACHE_DIR="$task_dir/uv-cache" && \
        uv run pytest -q && \
        uv run ruff check src tests scripts && \
        uv run ruff format --check src tests scripts && \
        uv run ty check src scripts tests'

# Opt-in data/publish operation for a host root containing `raw/` and resumable state.
docker-run data_root: docker-build
    docker run --rm -it \
        --user "$(id -u):$(id -g)" \
        --env HOME=/app/.quality-runtime/home \
        --mount "type=bind,src={{ data_root }},dst=/data" \
        --mount "type=bind,src={{ data_root }}/raw,dst=/data/raw,readonly" \
        --env HF_TOKEN \
        --env WIKIMEDIA_BOT_USERNAME \
        --env WIKIMEDIA_BOT_PASSWORD \
        --env WIKIMEDIA_REQUESTS_PER_MINUTE \
        osm-polygon-wikidata-only:local \
        sync-dir /data/raw --data-root /data --skip-existing --push

check: quality-gauntlet
