# Development guide

## Setup

The project supports Python 3.12 and uses [`uv`](https://docs.astral.sh/uv/)
for the locked environment:

```bash
uv sync --frozen
uv run pre-commit install
just --list
```

The source checkout contains code, tests, and documentation. Keep PBFs, Parquet
files, credentials, generated cards, and other run output in an
operator-selected data root outside the checkout.

The optional V2 sentence stage is installed separately so the normal pipeline
does not pull a large model runtime:

```bash
uv sync --extra sentence-splitting
```

Its exact routing and resumability contract is documented in the
[sentence-splitting guide](sentence-splitting.md).

The GPU extra is reserved for the Grid5000 compute-node job:

```bash
uv sync --extra sentence-splitting-gpu
```

The local controller, short-job policy, CUDA requirement, token boundary, and
resume/publish contract are documented in the
[Grid5000 sentence operations guide](grid5000-sentence-splitting.md). The
external data root remains authoritative; the controller stages only bounded
batch inputs and keeps HF authentication local.

## Docker reproducibility

The checked-in `Dockerfile` has `runtime` and `development` targets. Both are
built from the locked `uv.lock` environment; the runtime image runs as a
non-root, unprivileged `app` user and contains no source data or credentials.
The `docker build` command selects either target without changing the host data
root.

Build and exercise the harmless default command:

```bash
just docker-build
just docker-help
```

Run development checks without production data:

```bash
just docker-test
just docker-check
```

To run the opt-in workflow, provide a host data root containing `raw/`:

```bash
just docker-run /path/to/osm-polygon-data
```

The recipe uses Docker's explicit `--mount` form, mounts the data root at
`/data`, and mounts `/data/raw` read-only. It passes `HF_TOKEN` and optional
Wikimedia credentials only at runtime. Removing the container does not remove
the host's resumable state; press `Ctrl-C` and rerun the same command to resume.
Docker builds, help, and tests do not read a real PBF or make Hugging
Face/Wikimedia requests.

## Wikimedia credentials

Wikimedia authentication is optional. The [README authentication section](https://github.com/NoeFlandre/osm-polygon-wikidata-only#wikimedia-bot-password-authentication)
describes how to create and revoke a least-privilege Bot Password. Keep the
password out of source files, logs, issues, and pull requests. The
`WIKIMEDIA_BOT_USERNAME`/`WIKIMEDIA_BOT_PASSWORD` pair is all-or-nothing:
supply the password securely, never log it, and do not commit it. The
`WIKIMEDIA_REQUESTS_PER_MINUTE` environment variable selects the client-side
request ceiling; that ceiling remains subject to Wikimedia's service limits.

The test suite never uses live credentials. Authentication tests pass explicit
environment mappings and fake transports (for example,
`tests/enrichment/test_wikimedia_auth.py` and
`tests/cli/test_dependencies.py`) and assert that errors do not echo secrets.

## Tests and quality checks

Use red-green-refactor for behavior or configuration changes: add one focused
failing test, confirm the expected failure, implement the smallest change, and
then refactor while the test remains green. Contract tests should check the
observable CLI, schema, workflow, or documentation behavior rather than private
implementation details.

The deterministic pre-completion gate is:

```bash
just quality-gauntlet
```

The command runs the following stages exactly once, in this order, and stops
at the first failure:

```bash
just baseline
just ruff
just ty
just tests
just acceptance-tests
just architecture-checks
just crap-all
just mutation
just smoke-test
just diff-review
```

`crap-all` combines the bounded CRAP scopes for domain/V2 helpers, parsing and
sync helpers, the durable upload queue, quality-reporting scripts, geographic
rendering and parquet inputs, DatasetStats aggregation, and the SaT adapter.
The sentence runner is included in the domain/V2 scope. The architecture stage also
builds the package and strict MkDocs site before running its contracts. The
smoke stage checks both public CLI help paths without reading a data root or
making a network request. The Docker runtime has its own `docker-help` recipe
and CI container contract.
`diff-review` runs `git diff --check` and a short branch status check.

`just check` and the short alias `just qa-gauntlet` run the same deterministic
completion gate locally:

```bash
just check
```

### Mutation and complexity gates

The advanced gate covers the pure, deterministic Wikipedia and Wikidata parser
helpers and the durable upload queue. V2 comparison/checkpoint, geographic
parquet/rendering, DatasetStats, and optional SaT dependency boundaries remain
under focused tests and CRAP gates; they are intentionally outside mutation
testing because they cross file-system or external-runtime boundaries. The
mutation run keeps reports in `/tmp` and uses two workers to limit peak Mac
memory without reading production data.
HTML cleaning remains covered by the CRAP and branch-coverage gates; mutmut 3.7 cannot execute mutations inside
its `HTMLParser` subclass trampoline reliably, so those unsupported class
mutations are not counted as actionable results:

```bash
just crap
just mutation
just quality-advanced
```

`just crap`, `just crap-sync`, `just crap-upload`, `just crap-quality`,
`just crap-geography`, `just crap-geography-inputs`, `just crap-stats`, and
`just crap-sat` join coverage.py and Radon function reports and fail when any
function reaches CRAP 6; every measured score must therefore be below 6.
`just mutation` runs mutmut over the explicit deterministic module scope and
fails unless every generated mutant is killed. These gates
are deliberately narrow: network clients, large data files, and external
publication are outside the mutation run.

Run `uv run pre-commit run --all-files` before opening a pull request. The
hooks intentionally run the fast Ruff and `ty` subset; `just check` and
GitHub Actions both use the complete `just quality-gauntlet` gate.

## Test strength checks

The normal gate measures regression coverage across the whole package. The
opt-in `just quality-strength` recipe adds focused checks for identity helpers,
parser/sync helpers, and durable upload-queue behavior:

```bash
just mutation
just crap
just crap-sync
just crap-upload
just crap-quality
just crap-geography
just crap-geography-inputs
just crap-stats
just crap-sat
```

`mutmut` deliberately changes those helpers and requires every generated
mutant to be killed by the focused tests. `crap4py` reports the CRAP score,
which combines cyclomatic complexity and line coverage, and fails if any
targeted function reaches 6 or more. Keeping this gate focused makes the result
exhaustive and repeatable without pretending that a single mutation run can
meaningfully cover the entire I/O-heavy pipeline.

## Documentation and contribution

Build the site without starting a server:

```bash
just docs
```

Navigation targets must exist under `docs/`, links and images must resolve in a
clean checkout, and public examples must use current CLI options. The Pages
workflow builds with `--strict`, uploads only the generated site, and deploys
that artifact with least-privilege permissions.

Please read the repository's [contributing guide](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CONTRIBUTING.md)
before proposing changes. Keep pull requests small, explain any compatibility
effect on schemas or CLI options, and never commit source data or credentials.

## Read-only operator audit

The `osm-polygon-wikidata-only-audit-remote` command reports local/remote
publication differences without uploading or deleting files:

```bash
uv run osm-polygon-wikidata-only-audit-remote \
  --data-root "$OSM_POLYGON_DATA_ROOT"
```

Typer parses this command, Rich renders the report, and tqdm shows progress only
when stderr is interactive. It is separate from the stable argparse processing
CLI and does not change dataset output.

## Release checklist

Before a release, run the complete gate, inspect the wheel for `py.typed` and
the license, verify CLI help from the built artifact, review dataset schemas and
attribution, and update the version intentionally. Publishing software or
datasets is a maintainer action; ordinary tests do not publish anything.
