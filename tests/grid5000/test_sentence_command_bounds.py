"""Non-interactive, time-bounded ssh/rsync/git commands for the Grid5000 controller."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from osm_polygon_wikidata_only.grid5000 import sentence_controller_policy, sentence_transport
from osm_polygon_wikidata_only.grid5000.sentence_controller_policy import (
    FRONTEND_COMMAND_TIMEOUT_S,
    GIT_QUERY_TIMEOUT_S,
    TRANSFER_TIMEOUT_S,
    ControllerCommandTimeoutError,
    ControllerRunError,
)

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
RSYNC_REMOTE_SHELL = "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
HOME_PROBE = ["printf", "%s", "$HOME"]


class _RecordingRunner:
    """Fake subprocess.run that records each argv and keyword set."""

    def __init__(self, *, timeout_for: Callable[[list[str]], bool] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self._timeout_for = timeout_for

    def __call__(self, args, **kwargs) -> CompletedProcess[str]:
        argv = list(args)
        self.calls.append((argv, dict(kwargs)))
        if self._timeout_for is not None and self._timeout_for(argv):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if argv[-3:] == HOME_PROBE:
            return CompletedProcess(argv, 0, stdout="/home/test-user\n", stderr="")
        return CompletedProcess(argv, 0, stdout="", stderr="")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> _RecordingRunner:
    recorder = _RecordingRunner()
    monkeypatch.setattr(sentence_transport.subprocess, "run", recorder)
    return recorder


def _transport() -> sentence_transport.SubprocessGrid5000Transport:
    return sentence_transport.SubprocessGrid5000Transport(
        "grenoble", executable_resolver=lambda name: name
    )


def test_frontend_command_is_non_interactive_and_bounded(runner: _RecordingRunner) -> None:
    _transport().run_frontend(("oarstat", "-s", "-j", "12345"))

    assert runner.calls == [
        (
            ["ssh", *SSH_OPTIONS, "grenoble", "oarstat", "-s", "-j", "12345"],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 300.0,
            },
        )
    ]
    assert FRONTEND_COMMAND_TIMEOUT_S == 300.0


def test_rsync_upload_is_non_interactive_and_bounded(
    tmp_path: Path, runner: _RecordingRunner
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    _transport().upload_tree(staging, "$HOME/project")

    rsync_argv, rsync_kwargs = runner.calls[-1]
    assert rsync_argv == [
        "rsync",
        "-a",
        "-e",
        RSYNC_REMOTE_SHELL,
        f"{staging}/",
        "grenoble:/home/test-user/project/",
    ]
    assert rsync_kwargs["timeout"] == TRANSFER_TIMEOUT_S == 3600.0


def test_rsync_download_is_non_interactive_and_bounded(
    tmp_path: Path, runner: _RecordingRunner
) -> None:
    local_root = tmp_path / "received"

    _transport().download_tree("$HOME/project/result", local_root)

    rsync_argv, rsync_kwargs = runner.calls[-1]
    assert rsync_argv == [
        "rsync",
        "-a",
        "-e",
        RSYNC_REMOTE_SHELL,
        "grenoble:/home/test-user/project/result/",
        f"{local_root}/",
    ]
    assert rsync_kwargs["timeout"] == TRANSFER_TIMEOUT_S


def test_every_remote_command_carries_a_timeout(tmp_path: Path, runner: _RecordingRunner) -> None:
    transport = _transport()
    staging = tmp_path / "staging"
    staging.mkdir()

    transport.run_frontend(("usagepolicycheck", "-t"))
    transport.upload_tree(staging, "$HOME/project")
    transport.download_tree("$HOME/project/result", tmp_path / "received")
    transport.remove_tree("$HOME/osm-polygon-wikidata-only-grid5000/run-1")

    assert runner.calls
    assert all(isinstance(kwargs.get("timeout"), float) for _argv, kwargs in runner.calls)


def _timeout_on(*executables: str) -> Callable[[list[str]], bool]:
    return lambda argv: argv[0] in executables and argv[-3:] != HOME_PROBE


@pytest.mark.parametrize(
    ("operation", "executable", "bound"),
    [
        ("frontend", "ssh", FRONTEND_COMMAND_TIMEOUT_S),
        ("upload", "rsync", TRANSFER_TIMEOUT_S),
        ("download", "rsync", TRANSFER_TIMEOUT_S),
        ("remove", "ssh", FRONTEND_COMMAND_TIMEOUT_S),
    ],
)
def test_stalled_remote_command_raises_controller_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    executable: str,
    bound: float,
) -> None:
    recorder = _RecordingRunner(timeout_for=_timeout_on(executable))
    monkeypatch.setattr(sentence_transport.subprocess, "run", recorder)
    transport = _transport()
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ControllerCommandTimeoutError, match=f"timed out after {bound:g}s") as info:
        if operation == "frontend":
            transport.run_frontend(("oarstat", "-s", "-j", "1"))
        elif operation == "upload":
            transport.upload_tree(staging, "$HOME/project")
        elif operation == "download":
            transport.download_tree("$HOME/project/result", tmp_path / "received")
        else:
            transport.remove_tree("$HOME/osm-polygon-wikidata-only-grid5000/run-1")

    assert isinstance(info.value, ControllerRunError)
    assert isinstance(info.value.__cause__, subprocess.TimeoutExpired)


def test_git_source_commit_is_bounded_and_reports_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def returns_revision(args, **kwargs):
        calls.append(kwargs)
        return CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(sentence_controller_policy.subprocess, "run", returns_revision)
    assert (
        sentence_controller_policy.git_source_commit(
            tmp_path, executable_resolver=lambda name: name
        )
        == "abc123"
    )
    assert calls[0]["timeout"] == GIT_QUERY_TIMEOUT_S == 30.0

    def stalls(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(sentence_controller_policy.subprocess, "run", stalls)
    with pytest.raises(ControllerRunError, match="source commit query timed out"):
        sentence_controller_policy.git_source_commit(
            tmp_path, executable_resolver=lambda name: name
        )
