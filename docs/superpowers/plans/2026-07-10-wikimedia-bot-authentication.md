# Wikimedia Bot Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authenticate read-only Wikimedia API traffic with optional Bot Password credentials, safely increase authenticated throughput, and document operator setup.

**Architecture:** Add one credential loader and one shared, cookie-preserving HTTP session under `enrichment`. CLI dependency construction composes that session with the existing process-wide scheduler and injects both into Wikidata and Wikipedia clients. The scheduler keeps the three-request concurrency limit, gains deterministic success ramp-up and throttling backoff, and remains fixed at 180 requests per minute when anonymous.

**Tech Stack:** Python 3.12, stdlib `urllib`/`http.cookiejar`, dataclasses, threading, pytest, Ruff, mypy, uv.

---

## File structure

- Create `src/osm_polygon_wikidata_only/enrichment/wikimedia_auth.py`: credential loading, sanitized authentication errors, per-host cookie sessions, and the MediaWiki Bot Password handshake.
- Modify `src/osm_polygon_wikidata_only/utils/request_scheduler.py`: observable adaptive pacing while preserving the current constructor and concurrency behavior.
- Modify `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`: route transport through the injected shared session.
- Modify `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`: route transport through the injected shared session.
- Modify `src/osm_polygon_wikidata_only/cli/dependencies.py`: load credentials, validate the optional rate ceiling, and construct one shared session/scheduler/client pair.
- Create `tests/test_wikimedia_auth.py`: focused credentials and cookie-session behavior tests.
- Modify `tests/test_utils.py`: deterministic scheduler ramp/backoff tests.
- Modify `tests/test_enrichment.py`: transport-injection characterization tests.
- Create `tests/test_dependencies.py`: composition, anonymous compatibility, and invalid-environment tests.
- Modify `README.md`, `docs/development.md`, `docs/architecture.md`, and `SECURITY.md`: public Bot Password creation, export, behavior, troubleshooting, architecture, and secret-handling guidance.

### Task 1: Credentials and authenticated host sessions

**Files:**
- Create: `tests/test_wikimedia_auth.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikimedia_auth.py`

- [ ] **Step 1: Write failing credential tests**

Add tests that call `load_wikimedia_credentials()` with explicit mappings and require: `None` for neither variable, a `WikimediaCredentials` value for both, and `WikimediaConfigurationError` naming only the missing variable for partial configuration. Assert `repr(credentials)` and every raised error omit the sample password.

```python
def test_complete_bot_password_environment_loads_credentials() -> None:
    credentials = load_wikimedia_credentials(
        {
            "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
            "WIKIMEDIA_BOT_PASSWORD": "secret-value",
        }
    )
    assert credentials is not None
    assert credentials.username == "NoeFlandre@pipeline"
    assert "secret-value" not in repr(credentials)
```

- [ ] **Step 2: Run the focused credential tests and verify RED**

Run: `uv run pytest tests/test_wikimedia_auth.py -q`

Expected: collection fails because `enrichment.wikimedia_auth` does not exist.

- [ ] **Step 3: Implement minimal credential loading**

Create frozen `WikimediaCredentials(repr=False)`, `WikimediaConfigurationError`, `WikimediaAuthenticationError`, constants for all three environment names, and `load_wikimedia_credentials(environ: Mapping[str, str] | None = None)`. Strip usernames, treat blank values as absent, require an all-or-nothing pair, and provide a constant redacted `__repr__`.

- [ ] **Step 4: Run the focused credential tests and verify GREEN**

Run: `uv run pytest tests/test_wikimedia_auth.py -q`

Expected: all credential tests pass.

- [ ] **Step 5: Write failing session tests**

Using an injected fake opener factory and real `urllib.request.Request` objects, specify:

- anonymous `WikimediaSession.read()` performs no login;
- authenticated first use requests a login token, POSTs `lgname`, `lgpassword`, and `lgtoken`, then performs the requested GET through the same opener;
- two requests to one host log in once;
- requests to two hosts log in once per host;
- concurrent first use of one host logs in once;
- rejected or malformed login responses raise `WikimediaAuthenticationError` containing the hostname but neither password nor raw response.

The fake response implements `read()`, `headers`, and context-manager methods so tests exercise the real session parsing and cookie/opener boundary.

- [ ] **Step 6: Run the session tests and verify RED**

Run: `uv run pytest tests/test_wikimedia_auth.py -q`

Expected: failures show `WikimediaSession` and its host/session behavior are missing.

- [ ] **Step 7: Implement the minimal shared session**

Implement a private per-host state containing a `CookieJar`-backed opener, lock, and authenticated flag. `WikimediaSession.read(request)` derives the host, lazily calls the token/login endpoints under that host's lock, schedules every network operation, and returns `(body, content_encoding)`. Build POST data with `urllib.parse.urlencode(...).encode()` and never interpolate credentials into URLs or messages.

- [ ] **Step 8: Run the session tests and verify GREEN**

Run: `uv run pytest tests/test_wikimedia_auth.py -q`

Expected: all credentials and session tests pass.

- [ ] **Step 9: Refactor while green and commit**

Run: `uv run ruff check tests/test_wikimedia_auth.py src/osm_polygon_wikidata_only/enrichment/wikimedia_auth.py && uv run mypy src/osm_polygon_wikidata_only/enrichment/wikimedia_auth.py`

Commit: `feat: add secure Wikimedia bot sessions`

### Task 2: Deterministic authenticated rate adaptation

**Files:**
- Modify: `tests/test_utils.py`
- Modify: `src/osm_polygon_wikidata_only/utils/request_scheduler.py`

- [ ] **Step 1: Write failing scheduler tests**

Add deterministic tests using injected clock/sleep that require:

- `current_requests_per_minute` starts at the configured initial rate;
- `report_success()` raises the rate only after `successes_per_increase` successes and never above `max_requests_per_minute`;
- a fixed scheduler with no higher maximum remains at 180;
- `report_throttled(delay)` applies global cooldown and halves the active rate without dropping below `minimum_requests_per_minute`;
- successful windows gradually restore a throttled authenticated scheduler.

- [ ] **Step 2: Run the new scheduler tests and verify RED**

Run: `uv run pytest tests/test_utils.py -q`

Expected: failures identify the missing adaptive constructor arguments, properties, and reporting methods.

- [ ] **Step 3: Implement minimal adaptive scheduler state**

Extend the constructor compatibly with `max_requests_per_minute: float | None = None`, `minimum_requests_per_minute: float = 60`, and `successes_per_increase: int = 100`. Store an active rate, recompute request intervals under the existing lock, increase by 25% per successful window, cap at the configured maximum, and halve on throttling. `report_throttled()` also calls the cooldown behavior atomically.

- [ ] **Step 4: Run scheduler tests and verify GREEN**

Run: `uv run pytest tests/test_utils.py -q`

Expected: all utility tests pass, including the existing three-request concurrency test.

- [ ] **Step 5: Refactor while green and commit**

Run: `uv run ruff check tests/test_utils.py src/osm_polygon_wikidata_only/utils/request_scheduler.py && uv run mypy src/osm_polygon_wikidata_only/utils/request_scheduler.py`

Commit: `feat: adapt authenticated Wikimedia request pacing`

### Task 3: Route both clients through the shared session

**Files:**
- Modify: `tests/test_enrichment.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikimedia_auth.py`

- [ ] **Step 1: Write failing client transport tests**

Create a typed fake session whose `read(Request)` records URLs and returns JSON bytes. Inject it into each HTTP client and assert `_http_get()` parses the fake response, requests preserve User-Agent/Accept/gzip headers, and no direct network access occurs. Add a 429 test proving each client calls `scheduler.report_throttled(retry_after)` instead of only deferring.

- [ ] **Step 2: Run the focused client tests and verify RED**

Run: `uv run pytest tests/test_enrichment.py -q`

Expected: constructors reject `session=` and current `_http_get()` bypasses the fake transport.

- [ ] **Step 3: Implement minimal session injection**

Define a small `WikimediaHttpSession` protocol with `read(Request) -> tuple[bytes, str]`. Accept `session` in both client constructors, defaulting to an anonymous `WikimediaSession` using the client's scheduler and timeout. Replace direct `urlopen()` closures with `self._session.read(req)`. Report successful reads in the session and replace client 429 `defer()` calls with `report_throttled()`.

- [ ] **Step 4: Run focused enrichment tests and verify GREEN**

Run: `uv run pytest tests/test_enrichment.py tests/test_wikimedia_auth.py tests/test_utils.py -q`

Expected: all focused tests pass and existing parsing/cache behavior is unchanged.

- [ ] **Step 5: Refactor while green and commit**

Run: `uv run ruff check tests/test_enrichment.py src/osm_polygon_wikidata_only/enrichment && uv run mypy src/osm_polygon_wikidata_only/enrichment`

Commit: `refactor: share authenticated Wikimedia transport`

### Task 4: Compose credentials and rate ceilings at CLI startup

**Files:**
- Create: `tests/test_dependencies.py`
- Modify: `src/osm_polygon_wikidata_only/cli/dependencies.py`

- [ ] **Step 1: Write failing dependency-construction tests**

Patch constructors with recording fakes and specify that `build_clients(..., environ=...)`:

- constructs anonymous clients with a fixed 180 request-per-minute scheduler when credentials are absent;
- constructs one shared authenticated session and scheduler for both clients when both credentials exist;
- starts authenticated pacing at the settings rate and caps it at 1,200 by default;
- applies a positive numeric `WIKIMEDIA_REQUESTS_PER_MINUTE` override;
- raises a sanitized configuration error for partial, blank, nonnumeric, zero, or negative overrides;
- constructs each concrete HTTP client only once when cache is enabled.

- [ ] **Step 2: Run dependency tests and verify RED**

Run: `uv run pytest tests/test_dependencies.py -q`

Expected: `build_clients` rejects `environ=` and does not compose a shared session.

- [ ] **Step 3: Implement minimal startup composition**

Add optional `environ: Mapping[str, str] | None = None`, load credentials once, parse the optional ceiling with a sanitized helper, construct one scheduler and one session, and inject both into exactly one Wikidata and one Wikipedia HTTP client before adding cache decorators. Log mode, username, and ceiling only.

- [ ] **Step 4: Run dependency tests and verify GREEN**

Run: `uv run pytest tests/test_dependencies.py tests/test_enrichment.py tests/test_wikimedia_auth.py tests/test_utils.py -q`

Expected: all composition and focused regression tests pass.

- [ ] **Step 5: Refactor while green and commit**

Run: `uv run ruff check tests/test_dependencies.py src/osm_polygon_wikidata_only/cli/dependencies.py && uv run mypy src/osm_polygon_wikidata_only/cli/dependencies.py`

Commit: `feat: enable bot authentication from environment`

### Task 5: Public setup and security documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/development.md`
- Modify: `docs/architecture.md`
- Modify: `SECURITY.md`
- Modify: `tests/test_documentation.py`

- [ ] **Step 1: Write failing documentation assertions**

Add a test that reads the public docs and requires the official `Special:BotPasswords` URL, exact `WIKIMEDIA_BOT_USERNAME`, `WIKIMEDIA_BOT_PASSWORD`, and `WIKIMEDIA_REQUESTS_PER_MINUTE` names, the resumable pipeline command, revocation instructions, and an explicit warning not to commit or paste the secret. Assert no documentation contains the test/example password used by code tests.

- [ ] **Step 2: Run documentation tests and verify RED**

Run: `uv run pytest tests/test_documentation.py -q`

Expected: failures list the missing Bot Password setup and security text.

- [ ] **Step 3: Document the operator workflow**

In README, add a production authentication section before the full-dataset command with exact Meta-Wiki navigation, a read-only grant recommendation, one-time credential copying, safe shell exports using placeholder values, optional ceiling tuning, startup behavior, the complete processing command, troubleshooting, and revocation. Link only official Wikimedia Bot Password, API login, and rate-limit pages. State that account standing determines the actual tier and 429 handling remains authoritative.

In `docs/development.md`, describe the environment contract and local testing without live secrets. In `docs/architecture.md`, replace the anonymous-only scheduler statement with the shared per-host authenticated-session flow. In `SECURITY.md`, forbid committed credentials, browser-cookie reuse, or secrets in issue reports and explain immediate Bot Password revocation.

- [ ] **Step 4: Run documentation tests and verify GREEN**

Run: `uv run pytest tests/test_documentation.py -q`

Expected: all documentation tests pass.

- [ ] **Step 5: Refactor while green and commit**

Run: `uv run ruff check tests/test_documentation.py && git diff --check`

Commit: `docs: explain Wikimedia bot authentication setup`

### Task 6: Full verification and integration

**Files:**
- Modify only files required by failures found in the complete gate.

- [ ] **Step 1: Run the complete test and coverage gate**

Run: `uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q`

Expected: all tests pass and coverage is at least 80%.

- [ ] **Step 2: Run static quality checks**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
git diff --check
```

Expected: all commands exit zero with no warnings requiring action.

- [ ] **Step 3: Build and inspect the package**

Run: `uv build`

Expected: source distribution and wheel build successfully.

- [ ] **Step 4: Run sanitized CLI smoke checks**

Run the CLI help with no credentials, then with only a fake username and a command that reaches dependency construction. Confirm the latter exits nonzero naming `WIKIMEDIA_BOT_PASSWORD` without printing any fake secret. Do not contact Wikimedia during this smoke check.

- [ ] **Step 5: Review the final diff against the specification**

Confirm every acceptance criterion in `docs/superpowers/specs/2026-07-10-wikimedia-bot-authentication-design.md` has a corresponding passing test or documentation assertion. Confirm `git grep` finds no credential value and no unrelated behavior changes.

- [ ] **Step 6: Commit verification fixes if any**

Commit only if the verification gate required code changes: `fix: complete Wikimedia authentication verification`.

- [ ] **Step 7: Finish and integrate**

Use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`, and finally `superpowers:finishing-a-development-branch`. Merge the reviewed branch into `main`, rerun the complete quality gate on `main`, and remove the worktree and feature branch only after the merge is proven clean.
