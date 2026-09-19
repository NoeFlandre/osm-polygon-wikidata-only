"""Split coverage tests (part 3)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.pipeline.sync_reconciliation_support import *


def test_log_remote_reconciliation_summary_unit() -> None:
    """The summary helper accepts an injected log callable.

    Verifies the four documented branches without any caplog
    interaction, so the assertions are deterministic regardless
    of module-level logger configuration.
    """
    spy = _LoggerSpy()

    # Core + metadata refresh with repaired regions
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps={"mexico-latest"},
        core_repaired=True,
        metadata_repaired=False,
        log=spy.info,
    )
    # Maps refreshed but no repaired regions
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps=set(),
        core_repaired=True,
        metadata_repaired=False,
        log=spy.info,
    )
    # Repaired regions but maps NOT refreshed
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps={"andorra-latest", "mexico-latest"},
        core_repaired=False,
        metadata_repaired=False,
        log=spy.info,
    )
    # Converged
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps=set(),
        core_repaired=False,
        metadata_repaired=False,
        log=spy.info,
    )

    assert spy.messages[0] == (
        "Remote reconciliation complete: 1 regions repaired; README and maps refreshed"
    )
    assert spy.messages[1] == ("Remote reconciliation complete: README and maps refreshed")
    assert spy.messages[2] == "Remote reconciliation complete: 2 regions repaired"
    assert spy.messages[3] == "Remote reconciliation complete: converged"

def test_no_real_network_during_logging_core_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``test_logging_core_repair`` path must make zero real
    network calls.

    ``ensure_world_land`` (the symbol actually called by the
    publication module) is patched to a deterministic stub, and
    ``urllib.request.urlretrieve`` is patched to raise so any
    unstubbed network path becomes a deterministic test failure
    rather than a silent download. The assertions verify both
    sides of that contract:

    * the stubbed boundary was invoked (proving the test
      exercises the production code path that would have made
      a real HTTP request), and
    * ``urlretrieve`` was never invoked (proving the stub
      actually replaced the network primitive instead of relying
      on production code to swallow an ``AssertionError`` raised
      from a different module binding).
    """
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    recorder = _block_network(monkeypatch)
    _install_logger_spy(monkeypatch)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0

    # The publication code path reached the stubbed boundary at
    # least once -- this is the call that, in production, would
    # have invoked ``urllib.request.urlretrieve`` to download the
    # Natural Earth land GeoJSON.
    assert recorder.calls, (
        "Stubbed ensure_world_land was never invoked; "
        "publication code path did not exercise the network boundary."
    )
    # The urlretrieve guard was never tripped -- the stub actually
    # replaced the network primitive for the duration of the test.
    assert recorder.urlretrieve_calls == [], (
        "urllib.request.urlretrieve was invoked "
        f"{len(recorder.urlretrieve_calls)} time(s); the stub did not "
        "fully replace the network boundary."
    )
