# Transient Wikimedia Network Resilience Design

## Problem

Long-running syncs abort after the finite retry budget when DNS or connectivity is unavailable for roughly one minute. Restarting is safe because responses are cached, but it repeats orchestration and requires manual intervention.

## Design

Production Wikimedia clients use an unbounded attempt count only for explicitly classified transient failures. The classifier recognizes temporary HTTP statuses (408, 425, 429, 500, 502, 503, 504), DNS resolution errors, timeouts, connection failures, and selected network-related `OSError` codes. Permanent HTTP errors, malformed responses, authentication/configuration failures, and programming errors remain fail-fast.

The existing exponential backoff remains capped, so outages do not cause busy loops. Retry callbacks emit sparse warnings at the first failure and periodically thereafter, without URLs, credentials, or request bodies. `KeyboardInterrupt` and other `BaseException` subclasses are never caught, preserving user cancellation.

Tests may continue to configure a finite attempt count. The production default becomes unbounded, so the same retry utility and client code support deterministic tests and resilient real runs without a second scheduler.

## Compatibility

Successful responses, cache keys, request order, scheduler behavior, schemas, output rows, and publication behavior are unchanged. The only behavioral change is that classified transient network failures wait and recover instead of aborting the pipeline.
