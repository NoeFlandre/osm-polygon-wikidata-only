# Development guide

## Setup

The project supports Python 3.12. It uses [`uv`](https://docs.astral.sh/uv/) for the locked environment:

```bash
uv sync --frozen
uv run pre-commit install
just --list
```

The source checkout has the code, the tests, and the documentation. Keep the PBFs, Parquet files, credentials, generated cards, and other run output in a data root that the operator selects. The data root is outside the checkout.

Keep also the quality reports, the temporary files, the local tool state, and the generated documentation outside the checkout. `QUALITY_RUNTIME_DIR` defaults to the portable path `../quality-runtime`. `QUALITY_TMP_DIR` defaults to its `tmp` directory. When you share a runtime location, set them explicitly:

```bash
export QUALITY_RUNTIME_DIR="${QUALITY_RUNTIME_DIR:-../quality-runtime}"
export QUALITY_TMP_DIR="${QUALITY_TMP_DIR:-${QUALITY_RUNTIME_DIR}/tmp}"
just quality-runtime
```

Install the optional V2 sentence stage separately. The normal pipeline then does not pull a large model runtime:

```bash
uv sync --extra sentence-splitting
```

The [sentence-splitting guide](sentence-splitting.md) describes its exact routing contract and its resume contract.

The GPU extra is only for the Grid5000 compute-node job:

```bash
uv sync --extra sentence-splitting-gpu
```

The [Grid5000 sentence operations guide](grid5000-sentence-splitting.md) describes these items: the local controller, the short-job policy, the CUDA requirement, the token boundary, and the resume and publish contract. The external data root stays authoritative. The controller stages only the bounded batch inputs. It keeps the HF authentication local.

## Docker reproducibility

The checked-in `Dockerfile` has the targets `runtime` and `development`. Both targets use the locked `uv.lock` environment. The runtime image runs as the non-root, unprivileged user `app`. It has no source data and no credentials. The `docker build` command selects a target. It does not change the data root of the host.

Build the image and run the harmless default command:

```bash
just docker-build
just docker-help
```

Run the development checks without production data:

```bash
just docker-test
just docker-check
```

The development image does not declare an ambient `OSM_POLYGON_DATA_ROOT`. The test suite resolves its own temporary roots. Only the runtime image declares that contract, with the `/data` volume behind it. The image also runs `pytest -m "not repository"`. This is because `presentations/` is intentionally not in the build context. The `quality` CI job runs those repository-completeness contracts against a full checkout.

### Docker Compose runtime

The root file `compose.yaml` builds the `runtime` target. Its default command is `--help`. When you start it, it does not process data and does not publish. Compose loads the optional `.env` file only at runtime. Start from the example. Keep the real file private:

```bash
cp .env.example .env
mkdir -p ../osm-polygon-data/raw  # bind sources must exist
```

Set `OSM_POLYGON_DATA_ROOT` in `.env` to an absolute host directory outside the source checkout. Compose uses that directory as a bind mount at `/data`. The default `../osm-polygon-data` is also outside the checkout. The CLI writes the restart state and the generated files there.

Set `HOST_UID` and `HOST_GID` to the numeric IDs of the owner of the host directory. The non-root container can then write to the mount. Compose overlays the host directory `raw/` as read-only at `/data/raw`. On Linux, the values are `id -u` and `id -g`. For one command, the Compose environment variables can override the example defaults:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose run --rm pipeline --help
```

To process local Geofabrik files, put them in `$OSM_POLYGON_DATA_ROOT/raw`. Then override the harmless default command:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
  docker compose run --rm pipeline sync-dir /data/raw --data-root /data --skip-existing
```

Set `HF_TOKEN` in `.env` only when a command explicitly includes `--push`. Compose passes it at runtime through the optional `env_file`. For a direct `docker run`, pass it with `--env-file .env`. Never put it in a Dockerfile or in an image. Compose does not add `--push` to commands. The health check of the image runs the version command of the local CLI. It does not contact Hugging Face, Wikimedia, or another service.

To run the opt-in workflow, supply a host data root that has `raw/`:

```bash
just docker-run /path/to/osm-polygon-data
```

The recipe uses the explicit `--mount` form of Docker. It mounts the data root at `/data`. It mounts `/data/raw` as read-only. It passes `HF_TOKEN` and the optional Wikimedia credentials only at runtime. When you remove the container, the restart state of the host stays. To resume, press `Ctrl-C` and run the same command again. The Docker builds, the help, and the tests do not read a real PBF. They do not make requests to Hugging Face or Wikimedia.

## Wikimedia credentials

Wikimedia authentication is optional. The [README authentication section](https://github.com/NoeFlandre/osm-polygon-wikidata-only#wikimedia-bot-password-authentication) describes how to create and revoke a Bot Password with the fewest privileges. Keep the password out of source files, logs, issues, and pull requests.

The pair `WIKIMEDIA_BOT_USERNAME` and `WIKIMEDIA_BOT_PASSWORD` is all or nothing. Supply the password in a secure way. Never log it. Do not commit it. The environment variable `WIKIMEDIA_REQUESTS_PER_MINUTE` selects the client-side request ceiling. The service limits of Wikimedia still apply to that ceiling.

The test suite never uses live credentials. The authentication tests pass explicit environment mappings and fake transports. For example, see `tests/enrichment/test_wikimedia_auth.py` and `tests/cli/test_dependencies.py`. They assert that the errors do not repeat secrets.

## Tests and quality checks

Use red-green-refactor for a change to behavior or configuration:

1. Add one focused test that fails.
2. Confirm the expected failure.
3. Implement the smallest change.
4. Refactor while the test passes.

Contract tests must check the observable CLI, schema, workflow, or documentation behavior. They must not check private implementation details.

The deterministic gate before completion is:

```bash
just quality-gauntlet
```

For fast local feedback, use:

```bash
just quality-fast
```

This recipe runs Ruff, `ty`, the test suite with no coverage, and the diff checks. It is a development loop. It does not replace the full completion gate below.

The gauntlet runs the current quality recipes once, in a fixed fail-fast order:

```bash
just baseline
just ruff
just ty
just tests
just coverage-floor
just property-tests
just acceptance-tests
just architecture-checks
just crap-report
just mutation
just smoke-test
just diff-review
```

`just coverage-floor` reads the root `coverage.json` that `just tests` writes. It fails when a measured file has a line coverage below 85%. A healthy aggregate thus cannot hide a module with almost no coverage.

`just property-tests` runs the deterministic Hypothesis properties. They test lossless sentence routing and invariance at the batch boundary. `just acceptance-tests` runs the pytest-bdd resumability scenario and the local pipeline integration tests. These checks use the real local Parquet, JSON, and manifest formats. They stub the external clients. They do not need a live network or GPU services. The tests compare a resumed fixture run with a clean replay. A retry thus cannot change rows, offsets, or routing silently.

`just architecture-checks` runs the local import-graph rules for cycles, domain purity, and the direction from the pipeline to the CLI. It also runs the package checks, documentation checks, and CLI contract checks. The smoke stage checks both public CLI help paths. It does not read a data root. It does not make a network request. The Docker runtime has its own recipe `docker-help` and its own CI container contract. `diff-review` runs `git diff --check` and a short branch status check.

`just mutation` deletes the generated `mutants/` tree before each run. `pyproject.toml` configures the mutation scope. The scope uses `mutate_only_covered_lines`. The mutant population thus depends on the own coverage attribution of mutmut. When the incremental mutmut state was reused after a source edit, the run generated 2801 mutants. A clean tree produced 3038 mutants. The gate is weaker in the first case, but it does not fail. A new generation from scratch keeps the reported mutant count reproducible. The cost is a full run each time. The gate refuses each result that is not killed. It has no configured equivalence exemptions.

The root coverage report combines `osm_polygon_wikidata_only` and `scripts`. It measures branches. It enforces the configured total coverage floor of 90%. The preprocessing coverage is reported separately before its full-source CRAP check. The aggregate floor does not replace the CRAP threshold for each function.

### Nested preprocessing package

The repository has a separate locked distribution under `preprocessing/`. Its tests and its wheel build use `preprocessing/uv.lock`. The root Ruff and ty installations lint and type-check its source. They do not merge the two package environments. Before you change that package, run the boundary gate:

    just preprocessing-check

The gate runs the frozen preprocessing tests, Ruff, ty, and an offline isolated wheel build and install smoke test. It does not read production data. It does not publish artifacts.

`just check` and the short alias `just qa-gauntlet` run the same deterministic completion gate locally:

```bash
just check
```

### Mutation and complexity gates

The standard gate runs the full-source CRAP after the root tests and the preprocessing checks. The preprocessing checks are in `architecture-checks`. `just crap-report` joins those coverage reports with the Radon reports for `src`, `scripts`, and `preprocessing/src`. `--show-closures` includes the nested functions. These rules apply:

- Each function must have a cyclomatic complexity of 5 or less.
- Each function with a complexity of 3 or more must have at least 80% coverage.
- A source function that is missing from a coverage report fails the gate.

The gate keeps the CRAP scores visible for information. `just crap-all` is the standalone variant. It refreshes both coverage reports before it reports. The historical `crap-*` aliases delegate to that same full-source run. They are compatibility names. They are not separate focused inventories. The complexity cap of 5 and the coverage floor of 80% for functions with a complexity of 3 or more keep each reported CRAP score below 6. The maximum under those limits is 5.20.

`just mutation` runs mutmut over the explicit scope. The scope has the deterministic helpers, the quality tools, and the offline publication orchestration. The selected publication tests stub the external upload effects. They exercise the queueing, deferral, and submission decisions. The gate rejects these results: unreviewed survivors, timeouts, untested results, and other statuses that are not killed. Only the reviewed equivalence mechanism can pass exact source-bound equivalent mutations.

The mutation scope does not include network clients, large data, live GPU work, and other external effects. Use focused integration checks or operational checks for them. The static Ruff and ty checks limit the source shape and the types. They do not prove that the runtime side effects are safe.

The reports and the temporary files stay under the configured quality runtime. Mutation uses two workers by default. This bounds the local memory. CI sets four workers for the same deterministic population. This does not change the gate. mutmut 3.7 does not support the HTMLParser trampoline mutations. You cannot act on them.

```bash
just crap
just mutation
just quality-advanced
```

Before you open a pull request, run `uv run pre-commit run --all-files`. The hooks run the fast subset of Ruff and `ty` on purpose. `just check` and GitHub Actions both use the complete `just quality-gauntlet` gate.

In CI, the jobs have these functions:

- The `quality` job runs the gauntlet. The gauntlet already includes the strict docs build, the wheel and sdist build, the package smoke install, and the preprocessing checks.
- The `container` job builds and smoke-tests the images.
- The `all-green` job fails unless all other jobs succeed.

Protect `main` with `all-green` as the only required status check. The Documentation workflow also builds on pull requests. It deploys Pages only from `main`. A new push to a pull request cancels the superseded runs.

The `security` CI job runs `just audit`. This recipe exports both lockfiles with hashes. It fails on each known vulnerability that `pip-audit --strict` reports. Run it locally before you update dependencies. The CodeQL workflow analyses the Python sources on pull requests, on pushes to `main`, and weekly.

## Benchmarks and the slow-test budget

`benchmarks/` is outside the default test run. It has the micro-benchmarks for the hot paths of each polygon. It also has the workload benchmarks in `benchmarks/test_workloads.py`. They build seeded synthetic inputs (see `benchmarks/_workloads.py`). They call only public functions:

| Area | Benchmark input |
|---|---|
| Coverage map | `generate_coverage_map` with the land layer, 10,000 points |
| Parquet staging | containment `_canonical_manifest_stats`, 200,000 polygon rows |
| Language splits | `partition_row_indices` plus `take`, 1,000,000 rows, 50 languages |
| Link migration | `plan_link_migration` and `apply_link_migration`, 100,000 links |
| CLI startup | `--version` in a fresh interpreter |

The containment cases and the language-partitioning cases also assert a peak of the Arrow memory pool in a fresh process (50 MiB and 40 MiB). `tracemalloc` cannot see Arrow buffers. A memory regression thus fails even when the elapsed time changes very little.

```bash
just bench                      # run and print timings
just bench-json bench.json      # write the pytest-benchmark JSON
just bench-compare warn         # compare with benchmarks/baseline.json
```

`scripts/quality/bench_compare.py` flags a benchmark when its median is more than 25% slower than `benchmarks/baseline.json`. The CI job `benchmarks` runs in `warn` mode. It reports the regressions. It does not fail the build. It reports a missing baseline and does not fail. Record the baseline on the CI runner type. Do not record it on a developer machine. After one week of stable warn-only runs, set `BENCH_MODE` to `enforce` in `.github/workflows/ci.yml`.

`just baseline` (the full run with no instrumentation) also writes a JUnit report. `scripts/quality/slow_tests.py` fails it when a single test takes more than 3 seconds. Each run prints the 20 slowest tests. Coverage increases the durations several times. The budget is thus not measured under coverage. If a test legitimately needs more time, list it with its own limit in `scripts/quality/slow_test_budgets.json`. CI runners are about three times slower than a laptop. A test that takes about one second locally thus has a limit of 10 seconds in that file.

The CI job prints `bench.json` to its job summary. To record the baseline, copy it from there into `benchmarks/baseline.json`.

## Test strength checks

The standard gate already runs the full-source CRAP and the scoped mutation gate. The opt-in recipe `just quality-strength` repeats the standalone refresh of the full-source CRAP and the mutation checks. It does not add a narrower module inventory.

```bash
just quality-strength
```

The report shows the reviewed source-bound equivalent mutations separately from the killed mutants. Incomplete mutation results and unreviewed mutation results still fail. The mutation scope does not include the live network, GPU, publication, and large-data behavior. The full-source CRAP covers the configured source inventory, including the nested functions.

## Documentation and contribution

To build the site without a server, run:

```bash
just docs
```

The strict build writes the site and its assembled public artifacts below the configured quality runtime. You can point the runtime at a portable location that the operator owns. No generated site and no report belongs in the checkout.

These rules apply to the documentation:

- The navigation targets must exist under `docs/`.
- The links and images must resolve in a clean checkout.
- The public examples must use the current CLI options.
- Write the text in ASD-STE100 (Simplified Technical English). Add each new project term to the [glossary](glossary.md).

The Pages workflow builds with `--strict`. It uploads only the generated site. It deploys that artifact with the fewest permissions.

Read the [contributing guide](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CONTRIBUTING.md) before you propose changes. Keep the pull requests small. Explain each compatibility effect on schemas or CLI options. Never commit source data or credentials.

## Read-only operator audit

The command `osm-polygon-wikidata-only-audit-remote` reports the differences between the local publication and the remote publication. It does not upload or delete files:

```bash
uv run osm-polygon-wikidata-only-audit-remote \
  --data-root "$OSM_POLYGON_DATA_ROOT"
```

Typer parses this command. Rich renders the report. tqdm shows the progress only when stderr is interactive. The command is separate from the stable argparse processing CLI. It does not change the dataset output.

## Release checklist

Do these steps before a release:

1. Run the complete gate.
2. Examine the wheel for `py.typed` and the license.
3. Verify the CLI help from the built artifact.
4. Review the dataset schemas and the attribution.
5. Update the version intentionally.

A maintainer publishes the software and the datasets. The ordinary tests do not publish anything.

## Shared sentence-model capabilities

See [Shared SaT capabilities](sat-capabilities.md) for ownership, offline pins,
cross-repository drift checks, and the reviewed update procedure.
