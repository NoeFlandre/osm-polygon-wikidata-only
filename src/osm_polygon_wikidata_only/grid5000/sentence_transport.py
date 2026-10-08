"""SSH/rsync transport for Grid5000 frontend and run-owned files."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from .sentence_controller_policy import (
    FRONTEND_COMMAND_TIMEOUT_S,
    REMOTE_NAMESPACE,
    SSH_NON_INTERACTIVE_OPTIONS,
    TRANSFER_TIMEOUT_S,
    ControllerCommandTimeoutError,
    ControllerRunError,
    required_executable,
)


class Grid5000Transport(Protocol):
    """Frontend and file-transfer operations owned by the local controller."""

    def run_frontend(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one lightweight command on the configured site frontend."""

    def upload_tree(self, local_root: Path, remote_root: str) -> None:
        """Upload a local staging tree into a remote run-owned directory."""

    def download_tree(self, remote_root: str, local_root: Path) -> None:
        """Download a remote result tree into a local temporary directory."""

    def remove_tree(self, remote_root: str) -> None:
        """Remove one exact run-owned remote directory."""


class SubprocessGrid5000Transport:
    """SSH/rsync transport restricted to frontend and run-owned paths."""

    def __init__(
        self,
        site: str,
        *,
        executable_resolver: Callable[[str], str] = required_executable,
    ) -> None:
        self.site = site
        self._remote_home: str | None = None
        self._executable_resolver = executable_resolver

    def run_frontend(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        remote_args = tuple(args)
        if remote_args and remote_args[0] == "oarsub":
            remote_args = (" ".join(shlex.quote(argument) for argument in remote_args),)
        return self._run(
            [
                self._executable_resolver("ssh"),
                *SSH_NON_INTERACTIVE_OPTIONS,
                self.site,
                *remote_args,
            ],
            timeout_s=FRONTEND_COMMAND_TIMEOUT_S,
        )

    def upload_tree(self, local_root: Path, remote_root: str) -> None:
        resolved_root = self._resolve_remote_path(remote_root)
        result = self._run(
            [
                self._executable_resolver("rsync"),
                "-a",
                "-e",
                self._rsync_remote_shell(),
                f"{local_root}/",
                f"{self.site}:{resolved_root}/",
            ],
            timeout_s=TRANSFER_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise ControllerRunError(f"Grid5000 upload failed: {(result.stderr or '').strip()}")

    def download_tree(self, remote_root: str, local_root: Path) -> None:
        local_root.mkdir(parents=True, exist_ok=True)
        resolved_root = self._resolve_remote_path(remote_root)
        result = self._run(
            [
                self._executable_resolver("rsync"),
                "-a",
                "-e",
                self._rsync_remote_shell(),
                f"{self.site}:{resolved_root}/",
                f"{local_root}/",
            ],
            timeout_s=TRANSFER_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise ControllerRunError(f"Grid5000 download failed: {(result.stderr or '').strip()}")

    def remove_tree(self, remote_root: str) -> None:
        if not remote_root.startswith(REMOTE_NAMESPACE + "/"):
            raise ControllerRunError("Refusing to remove an outside Grid5000 namespace")
        result = self.run_frontend(("rm", "-rf", self._resolve_remote_path(remote_root)))
        if result.returncode != 0:
            raise ControllerRunError("Grid5000 cleanup failed")

    def _rsync_remote_shell(self) -> str:
        """Return the rsync ``-e`` value: ssh with the same non-interactive options."""
        return shlex.join([self._executable_resolver("ssh"), *SSH_NON_INTERACTIVE_OPTIONS])

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout_s: float,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(  # noqa: S603 - commands are controller-generated argv lists
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            raise ControllerCommandTimeoutError(
                f"Grid5000 command timed out after {timeout_s:g}s: {Path(command[0]).name}"
            ) from error

    def _resolve_remote_path(self, remote_path: str) -> str:
        if not remote_path.startswith("$HOME/"):
            raise ControllerRunError("Refusing a Grid5000 path outside the remote home")
        remote_home = self._resolve_remote_home()
        return f"{remote_home}{remote_path[len('$HOME') :]}"

    def _resolve_remote_home(self) -> str:
        if self._remote_home is not None:
            return self._remote_home
        result = self.run_frontend(("printf", "%s", "$HOME"))
        remote_home = validated_remote_home(result)
        self._remote_home = remote_home
        return self._remote_home


def validated_remote_home(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        raise ControllerRunError("Could not resolve the Grid5000 remote home")
    remote_home = (result.stdout or "").strip()
    return validate_remote_home(remote_home)


def validate_remote_home(remote_home: str) -> str:
    if not remote_home.startswith("/") or any(char.isspace() for char in remote_home):
        raise ControllerRunError("Grid5000 remote home is invalid")
    return remote_home


__all__ = [
    "Grid5000Transport",
    "SubprocessGrid5000Transport",
    "validate_remote_home",
    "validated_remote_home",
]
