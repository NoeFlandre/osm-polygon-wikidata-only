"""Shared argument checks for pipeline entry points."""

from __future__ import annotations


def require_positive_batch_size(batch_size: int, error_type: type[Exception]) -> None:
    """Raise ``error_type`` unless ``batch_size`` is at least one."""
    if batch_size < 1:
        raise error_type(f"batch_size must be positive, got {batch_size}")


__all__ = ["require_positive_batch_size"]
