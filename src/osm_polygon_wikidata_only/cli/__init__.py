"""Command-line interface (process-pbf, process-dir, push)."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Load the legacy command API only when a caller requests it."""
    if name in {"build_parser", "main"}:
        commands = __import__(f"{__name__}.commands", fromlist=[name])
        value = getattr(commands, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), "build_parser", "main"])


__all__ = ["build_parser", "main"]
