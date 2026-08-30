set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

UV_CACHE_DIR := "/tmp/osm-polygon-wikidata-only-uv"

default:
    @just --list

sync:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv sync --frozen

baseline:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv sync --frozen
    @git status --short --branch

test:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest -q

ruff:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run ruff check src tests scripts
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run ruff format --check src tests scripts

coverage:
    COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-coverage-$$ UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q

tests:
    COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-coverage-$$ UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q

acceptance-tests:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest -q tests/pipeline/test_end_to_end.py tests/pipeline/test_sync_recovery_integration.py

architecture-checks:
    just build
    just docs
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest -q tests/contracts tests/test_mkdocs.py tests/test_documentation.py tests/test_docker.py

# Enforce a CRAP score below 6 for the v2 data-integrity helper scope.
crap:
    COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-coverage-$$ UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest tests/domain/test_filters.py tests/v2/test_deduplication.py tests/v2/test_fingerprints.py tests/v2/test_wikipedia_tags.py tests/v2/test_sentence_logic.py tests/v2/test_sentence_runner.py tests/grid5000/test_sentence_protocol.py tests/grid5000/test_sentence_artifacts.py --cov=osm_polygon_wikidata_only.domain.filters --cov=osm_polygon_wikidata_only.v2.deduplication --cov=osm_polygon_wikidata_only.v2.fingerprints --cov=osm_polygon_wikidata_only.v2.wikipedia_tags --cov=osm_polygon_wikidata_only.v2.sentence_logic --cov=osm_polygon_wikidata_only.v2.sentence_runner --cov=osm_polygon_wikidata_only.grid5000.sentence_protocol --cov-branch --cov-fail-under=0 --cov-report=lcov:/tmp/osm-polygon-wikidata-only-crap.lcov -q
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run crap4py --lcov /tmp/osm-polygon-wikidata-only-crap.lcov --max-crap 5.99 src/osm_polygon_wikidata_only/domain/filters.py src/osm_polygon_wikidata_only/v2/deduplication.py src/osm_polygon_wikidata_only/v2/fingerprints.py src/osm_polygon_wikidata_only/v2/wikipedia_tags.py src/osm_polygon_wikidata_only/v2/sentence_logic.py src/osm_polygon_wikidata_only/v2/sentence_runner.py src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py

# Report function-level CRAP scores for the pure parsing/cleaning and sync
# application scopes. Reports stay in /tmp so Mac storage remains bounded.
crap-sync:
    COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-coverage-$$ UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest -q tests/enrichment/test_parsing_quality.py tests/enrichment/test_enrichment.py tests/cli/test_sync_application.py --cov=osm_polygon_wikidata_only.enrichment.wikipedia.parsing --cov=osm_polygon_wikidata_only.enrichment.wikidata.parsing --cov=osm_polygon_wikidata_only.enrichment.text_cleaning --cov=osm_polygon_wikidata_only.cli.sync_application --cov-fail-under=0 --cov-report=json:/tmp/osm-polygon-wikidata-crap-coverage.json
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run radon cc -j src/osm_polygon_wikidata_only/enrichment/wikipedia/parsing.py src/osm_polygon_wikidata_only/enrichment/wikidata/parsing.py src/osm_polygon_wikidata_only/enrichment/text_cleaning.py src/osm_polygon_wikidata_only/cli/sync_application.py > /tmp/osm-polygon-wikidata-crap-complexity.json
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run python scripts/quality/crap_score.py --coverage /tmp/osm-polygon-wikidata-crap-coverage.json --complexity /tmp/osm-polygon-wikidata-crap-complexity.json --maximum 6

# Enforce a CRAP score below 6 for durable upload-queue helpers.
crap-upload:
    COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-coverage-$$ UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest -q tests/io/test_upload_queue_durability.py tests/io/test_upload_queue_amendment_8.py tests/io/test_upload_queue_real_legacy.py tests/hf/test_upload_operation_helpers.py tests/hf/test_upload_state.py --cov=osm_polygon_wikidata_only.hf.upload_queue --cov=osm_polygon_wikidata_only.hf._upload_state --cov-branch --cov-fail-under=0 --cov-report=json:/tmp/osm-polygon-wikidata-upload-crap-coverage.json
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run radon cc -j src/osm_polygon_wikidata_only/hf/upload_queue.py src/osm_polygon_wikidata_only/hf/_upload_state.py > /tmp/osm-polygon-wikidata-upload-crap-complexity.json
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run python scripts/quality/crap_score.py --coverage /tmp/osm-polygon-wikidata-upload-crap-coverage.json --complexity /tmp/osm-polygon-wikidata-upload-crap-complexity.json --maximum 6

# Enforce a CRAP score below 6 for the quality-reporting scripts themselves.
crap-quality:
    COVERAGE_FILE=/tmp/osm-polygon-wikidata-quality-crap-coverage-$$ UV_CACHE_DIR={{UV_CACHE_DIR}} uv run pytest -q tests/quality/test_crap_score.py tests/quality/test_mutation_gate.py tests/quality/test_audit_containment.py --cov=scripts.quality.crap_score --cov=scripts.quality.mutation_gate --cov=scripts.audit_containment --cov-branch --cov-fail-under=0 --cov-report=json:/tmp/osm-polygon-wikidata-quality-crap-coverage.json
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run radon cc -j scripts/quality/crap_score.py scripts/quality/mutation_gate.py scripts/audit_containment.py > /tmp/osm-polygon-wikidata-quality-crap-complexity.json
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run python scripts/quality/crap_score.py --coverage /tmp/osm-polygon-wikidata-quality-crap-coverage.json --complexity /tmp/osm-polygon-wikidata-quality-crap-complexity.json --maximum 6

crap-all: crap crap-sync crap-upload crap-quality

# Run mutmut with two workers to keep peak Mac memory bounded. The explicit
# source scope contains only pure deterministic helpers, and the gate refuses
# any survivor, timeout, or untested mutant.
mutation:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run mutmut run --max-children 2
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run mutmut results --all=true | UV_CACHE_DIR={{UV_CACHE_DIR}} uv run python scripts/quality/mutation_gate.py

smoke-test:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run osm-polygon-wikidata-only --help
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run osm-polygon-wikidata-only sync-dir --help

diff-review:
    git diff --check
    @test -z "$(git diff --name-only --diff-filter=U)"
    @git status --short --branch

quality-gauntlet:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run python scripts/quality/qa_gauntlet.py

# Compatibility alias retained for historical references in tooling.
qa-gauntlet: quality-gauntlet

# Run opt-in quality-strength checks; these are intentionally separate
# from `just check` because mutation testing is substantially slower.
quality-strength: mutation crap crap-sync crap-upload crap-quality

quality-advanced: crap crap-sync crap-upload crap-quality mutation

lint:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run ruff check src tests scripts

format:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run ruff format src tests scripts

format-check:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run ruff format --check src tests scripts

typecheck:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run ty check src scripts

ty:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run ty check src scripts

build:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv build

docs:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run mkdocs build --strict --site-dir /tmp/osm-polygon-wikidata-only-site

trackio:
    UV_CACHE_DIR={{UV_CACHE_DIR}} uv run osm-polygon-wikidata-only-trackio

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
        'UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run pytest -q && UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run ruff check src tests scripts && UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run ruff format --check src tests scripts && UV_CACHE_DIR=/tmp/osm-polygon-wikidata-only-uv uv run ty check src scripts'

# Opt-in data/publish operation for a host root containing `raw/` and resumable state.
docker-run data_root: docker-build
    docker run --rm -it \
        --user "$(id -u):$(id -g)" \
        --env HOME=/tmp \
        --mount "type=bind,src={{data_root}},dst=/data" \
        --mount "type=bind,src={{data_root}}/raw,dst=/data/raw,readonly" \
        --env HF_TOKEN \
        --env WIKIMEDIA_BOT_USERNAME \
        --env WIKIMEDIA_BOT_PASSWORD \
        --env WIKIMEDIA_REQUESTS_PER_MINUTE \
        osm-polygon-wikidata-only:local \
        sync-dir /data/raw --data-root /data --skip-existing --push

check: quality-gauntlet
