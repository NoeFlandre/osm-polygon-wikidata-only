# Wikimedia Bot Authentication Design

## Goal

Authenticate API requests with an optional Wikimedia Bot Password so the pipeline
can use the request allowance associated with the operator's Wikimedia account.
Keep anonymous operation working, preserve Wikimedia's concurrency ceiling, and
never expose credentials in logs, cache files, generated datasets, or errors.

## Scope

This change adds Bot Password authentication, authenticated request pacing, and
operator documentation. It does not add OAuth, a secrets manager, browser-cookie
reuse, account creation, bot-flag applications, or automatic edits to Wikimedia.
The pipeline remains API-only and read-only.

## Configuration

The application reads these environment variables at startup:

- `WIKIMEDIA_BOT_USERNAME`: the complete generated Bot Password username,
  including its bot suffix.
- `WIKIMEDIA_BOT_PASSWORD`: the generated Bot Password secret.
- `WIKIMEDIA_REQUESTS_PER_MINUTE`: an optional positive numeric ceiling for the
  process-wide scheduler.

When neither credential is set, the current anonymous behavior and 180 requests
per minute default remain unchanged. When both are set, authenticated mode starts
at 180 requests per minute and may increase gradually to a default ceiling of
1,200 requests per minute. An explicit `WIKIMEDIA_REQUESTS_PER_MINUTE` replaces
that ceiling. When only one credential is set, startup fails with a configuration
error. Failed authentication also fails closed rather than silently continuing
anonymously.

Credentials are held only in process memory. Their values must never appear in
`repr`, exception text, structured logs, caches, test snapshots, dataset files,
or the Hugging Face dataset card.

## Components

### Credential loader

A focused immutable credential type represents the username and password. A
loader validates the environment as an all-or-nothing pair and returns either
credentials or `None`. Public status information contains only whether
authentication is enabled and the authenticated username.

### Wikimedia session

A shared Wikimedia HTTP session owns cookie jars and URL openers. For each API
host, it performs the MediaWiki Bot Password login handshake:

1. Request a login token from `action=query&meta=tokens&type=login`.
2. Submit `action=login` with `lgname`, `lgpassword`, and `lgtoken` as form data.
3. Require a `Success` response.
4. Preserve the host's cookies for subsequent API calls.

Wikidata and each language-specific Wikipedia API are distinct hosts, so the
session authenticates lazily once per host. A per-host lock prevents concurrent
first requests from performing duplicate logins. Failed host authentication is
reported without response bodies or credential values that could leak secrets.
Anonymous sessions skip the handshake.

The existing Wikidata and Wikipedia clients depend on this session instead of
calling `urllib.request.urlopen` directly. The session remains transport-focused;
the clients retain responsibility for API parameters, response decoding,
retries, and domain mapping.

### Adaptive request scheduling

The existing scheduler continues to enforce at most three concurrent Wikimedia
requests process-wide. Authentication requests use the same scheduler and count
toward its rate.

In authenticated mode, successful request windows raise the rate gradually from
180 requests per minute toward the configured ceiling. A Wikimedia HTTP 429 or
`Retry-After` response triggers the existing global cooldown and reduces the
current rate. Later successful windows restore it gradually. Anonymous mode does
not raise its current 180 requests per minute ceiling.

The scheduler exposes only observable rate state needed for deterministic tests
and informational logs. It does not attempt to infer account age, edit count, or
rights from undocumented heuristics.

## Data Flow

CLI dependency construction loads credentials once, chooses the appropriate rate
ceiling, then creates one scheduler and one Wikimedia session. Both enrichment
clients receive those shared dependencies. A client builds an API request and
passes it to the session; the session ensures the request's host is authenticated,
runs network work through the scheduler, and returns the response. Existing retry
logic reports throttling and success to the scheduler.

At startup the CLI logs anonymous or authenticated mode, the username when
authenticated, and the rate ceiling. It never logs the password. Per-host login
is logged at debug level using only the hostname.

## Error Handling

- Partial credential configuration raises a concise startup error naming the
  missing environment variable.
- Invalid or rejected credentials raise a dedicated authentication error that
  identifies the API host but excludes secrets and raw response bodies.
- Malformed login-token and login-result responses raise the same sanitized
  authentication error.
- Network failures continue through the existing retry behavior when transient;
  an explicit authentication rejection is not retried as an anonymous request.
- HTTP 429 responses preserve `Retry-After` handling and globally slow all
  Wikimedia requests.

## Documentation

The public README and development documentation explain how to:

1. Sign in to Wikimedia and open `Special:BotPasswords` on Meta-Wiki.
2. Create a read-only Bot Password grant suitable for API data retrieval.
3. Copy the generated username and password once.
4. Export both environment variables without committing them to a shell script
   or repository file.
5. optionally set the request-rate ceiling.
6. Run the pipeline and recognize the sanitized authentication startup message.
7. Troubleshoot partial variables, rejected credentials, revocation, and HTTP
   429 cooldowns.

The documentation links to Wikimedia's official Bot Password, API login, and API
rate-limit documentation and states that authentication does not guarantee a
particular tier; the account's standing and Wikimedia policy determine it.

## Test Strategy

Implementation follows red-green-refactor cycles:

- Unit tests first specify absent, complete, and partial credential environments,
  plus positive rate-ceiling validation.
- Session tests first specify the token/login sequence, POST body, cookie-preserving
  opener reuse, lazy once-per-host login, concurrency safety, anonymous bypass,
  and sanitized failures.
- Client integration tests first prove both API clients route through the shared
  session without changing response behavior.
- Scheduler tests first specify authenticated ramp-up, configured ceilings,
  anonymous stability, and 429 reduction using injected clocks and sleepers.
- CLI dependency tests first specify shared session/scheduler construction and
  startup configuration failures.
- Documentation checks ensure the public setup includes the exact environment
  variable names and contains no example secret.
- The full test, lint, format, type-check, coverage, and package-build suite runs
  before integration.

## Acceptance Criteria

- With both Bot Password variables set, Wikidata and Wikipedia requests use
  authenticated, cookie-preserving sessions.
- Each API host logs in at most once per process, including under concurrent
  first use.
- With neither variable set, existing anonymous behavior is unchanged.
- Partial or rejected credentials stop processing with a sanitized actionable
  error.
- Global concurrency never exceeds three.
- Authenticated traffic can ramp toward 1,200 requests per minute by default and
  backs off globally when throttled.
- Reader-facing documentation accurately covers Bot Password setup, secure export,
  verification, revocation, tuning, and rate-limit caveats.
- All existing behavior and quality checks remain green.

## Authoritative References

- <https://www.mediawiki.org/wiki/Manual:Bot_passwords>
- <https://www.mediawiki.org/wiki/API:Login>
- <https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits>
- <https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits/FAQ>
