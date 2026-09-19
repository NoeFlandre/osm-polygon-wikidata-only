"""SSH/rsync transport for Grid5000 frontend and run-owned files."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from .sentence_controller_policy import (
    REMOTE_NAMESPACE,
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
        return subprocess.run(  # noqa: S603 - args are controller-generated frontend commands
            [self._executable_resolver("ssh"), self.site, *remote_args],
            check=False,
            capture_output=True,
            text=True,
        )

    def upload_tree(self, local_root: Path, remote_root: str) -> None:
        resolved_root = self._resolve_remote_path(remote_root)
        result = subprocess.run(  # noqa: S603 - remote_root is a validated run namespace
            [
                self._executable_resolver("rsync"),
                "-a",
                f"{local_root}/",
                f"{self.site}:{resolved_root}/",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ControllerRunError(f"Grid5000 upload failed: {(result.stderr or '').strip()}")

    def download_tree(self, remote_root: str, local_root: Path) -> None:
        local_root.mkdir(parents=True, exist_ok=True)
        resolved_root = self._resolve_remote_path(remote_root)
        result = subprocess.run(  # noqa: S603 - remote_root is a validated run namespace
            [
                self._executable_resolver("rsync"),
                "-a",
                f"{self.site}:{resolved_root}/",
                f"{local_root}/",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ControllerRunError(f"Grid5000 download failed: {(result.stderr or '').strip()}")

    def remove_tree(self, remote_root: str) -> None:
        if not remote_root.startswith(REMOTE_NAMESPACE + "/"):
            raise ControllerRunError("Refusing to remove an outside Grid5000 namespace")
        result = self.run_frontend(("rm", "-rf", self._resolve_remote_path(remote_root)))
        if result.returncode != 0:
            raise ControllerRunError("Grid5000 cleanup failed")

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
