set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

sync:
    uv sync --frozen

test:
    uv run pytest -q

coverage:
    uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q

# Enforce a CRAP score below 6 for the v2 data-integrity helper scope.
crap:
    uv run pytest tests/domain/test_filters.py tests/v2/test_deduplication.py tests/v2/test_fingerprints.py tests/v2/test_wikipedia_tags.py --cov=osm_polygon_wikidata_only.domain.filters --cov=osm_polygon_wikidata_only.v2.deduplication --cov=osm_polygon_wikidata_only.v2.fingerprints --cov=osm_polygon_wikidata_only.v2.wikipedia_tags --cov-branch --cov-report=lcov:/tmp/osm-polygon-wikidata-only-crap.lcov -q
    uv run crap4py --lcov /tmp/osm-polygon-wikidata-only-crap.lcov --max-crap 5.99 src/osm_polygon_wikidata_only/domain/filters.py src/osm_polygon_wikidata_only/v2/deduplication.py src/osm_polygon_wikidata_only/v2/fingerprints.py src/osm_polygon_wikidata_only/v2/wikipedia_tags.py

# Report function-level CRAP scores for the pure parsing/cleaning and sync
# application scopes. Reports stay in /tmp so Mac storage remains bounded.
crap-sync:
    uv run pytest -q tests/enrichment/test_parsing_quality.py tests/enrichment/test_enrichment.py tests/cli/test_sync_application.py --cov=osm_polygon_wikidata_only.enrichment.wikipedia.parsing --cov=osm_polygon_wikidata_only.enrichment.wikidata.parsing --cov=osm_polygon_wikidata_only.enrichment.text_cleaning --cov=osm_polygon_wikidata_only.cli.sync_application --cov-report=json:/tmp/osm-polygon-wikidata-crap-coverage.json
    uv run radon cc -j src/osm_polygon_wikidata_only/enrichment/wikipedia/parsing.py src/osm_polygon_wikidata_only/enrichment/wikidata/parsing.py src/osm_polygon_wikidata_only/enrichment/text_cleaning.py src/osm_polygon_wikidata_only/cli/sync_application.py > /tmp/osm-polygon-wikidata-crap-complexity.json
    uv run python scripts/quality/crap_score.py --coverage /tmp/osm-polygon-wikidata-crap-coverage.json --complexity /tmp/osm-polygon-wikidata-crap-complexity.json --maximum 6

# Enforce a CRAP score below 6 for durable upload-queue helpers.
crap-upload:
    uv run pytest -q tests/io/test_upload_queue_durability.py tests/io/test_upload_queue_amendment_8.py tests/io/test_upload_queue_real_legacy.py tests/hf/test_upload_operation_helpers.py --cov=osm_polygon_wikidata_only.hf.upload_queue --cov-branch --cov-report=json:/tmp/osm-polygon-wikidata-upload-crap-coverage.json
    uv run radon cc -j src/osm_polygon_wikidata_only/hf/upload_queue.py > /tmp/osm-polygon-wikidata-upload-crap-complexity.json
    uv run python scripts/quality/crap_score.py --coverage /tmp/osm-polygon-wikidata-upload-crap-coverage.json --complexity /tmp/osm-polygon-wikidata-upload-crap-complexity.json --maximum 6

# Run mutmut with two workers to keep peak Mac memory bounded. The explicit
# source scope contains only pure deterministic helpers, and the gate refuses
# any survivor, timeout, or untested mutant.
mutation:
    uv run mutmut run --max-children 2
    uv run mutmut results --all=true | uv run python scripts/quality/mutation_gate.py

# Run opt-in quality-strength checks; these are intentionally separate
# from `just check` because mutation testing is substantially slower.
quality-strength: mutation crap crap-sync crap-upload

quality-advanced: crap crap-sync crap-upload mutation

lint:
    uv run ruff check src tests scripts

format:
    uv run ruff format src tests scripts

format-check:
    uv run ruff format --check src tests scripts

typecheck:
    uv run ty check src scripts

build:
    uv build

docs:
    uv run mkdocs build --strict --site-dir /tmp/osm-polygon-wikidata-only-site

trackio:
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
        'uv run pytest -q && uv run ruff check src tests scripts && uv run ruff format --check src tests scripts && uv run ty check src scripts'

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

check: sync coverage lint format-check typecheck build docs
    git diff --check
