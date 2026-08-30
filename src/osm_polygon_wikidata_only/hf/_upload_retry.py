"""Deterministic retry policy for upload callbacks."""

from __future__ import annotations

from collections.abc import Callable


def _run_upload_attempts[OperationInput](
    upload: Callable[[OperationInput, str], None],
    ops: OperationInput,
    message: str,
    *,
    attempts: int,
) -> None:
    """Retry an upload and re-raise the final callback failure."""
    for attempt in range(attempts):
        try:
            upload(ops, message)
        except Exception:
            if attempt == attempts - 1:
                raise
        else:
            return
