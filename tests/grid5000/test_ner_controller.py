"""Offline safety and recovery tests for the serial NER pilot."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.ner.pipeline import INPUT_COLUMNS, Contract
from tests.fixtures.case_sensitive_paths import case_sensitive_paths as case_sensitive_paths

UNCERTAIN_SUBMISSION = "OAR submission result is uncertain; operator recovery required before retry"
PUBLICATION_FAILED = "Geographic NER publication failed; run remains ready_to_publish"
STAGE_IMMUTABLE = "Geographic NER staging manifest is immutable after submission"
SOURCE_ROWS_MISMATCH = "Geographic NER source row count does not match staged input"
ARTIFACT_ORDER_INVALID = "Geographic NER artifact order is invalid"
PROCESSED_ROWS_REQUIRED = "Geographic NER processed row count is required"
ROW_COUNT_MESSAGES = {
    "source_rows": SOURCE_ROWS_MISMATCH,
    "processed_rows": PROCESSED_ROWS_REQUIRED,
}
LEDGER_SHAPE_MESSAGES = (
    "Geographic NER ledger has no valid state",
    "Geographic NER ledger has no staging digest",
    "Geographic NER ledger has no staging digest",
    "Geographic NER ledger has no staging digest",
)
JOB_FAILURE_MESSAGES = (
    "Grid5000 geographic NER job failed; output was retrieved and remains resumable",
    "Grid5000 job failed; output is retained and remains resumable",
)


def raises_exactly(message, kind=RuntimeError):
    """Pin an operator-facing diagnostic so message-only mutations cannot survive."""
    return pytest.raises(kind, match=rf"^{re.escape(message)}$")


def controller_module():
    return importlib.import_module("osm_polygon_wikidata_only.grid5000.ner_controller")


_RUNTIME_FILES = (
    "requirements/geographic-ner-gpu.txt",
    "src/osm_polygon_wikidata_only/__init__.py",
    "src/osm_polygon_wikidata_only/ner/__init__.py",
    "src/osm_polygon_wikidata_only/ner/job.py",
    "src/osm_polygon_wikidata_only/ner/pipeline.py",
    "src/osm_polygon_wikidata_only/ner/otter.py",
    "src/osm_polygon_wikidata_only/io/__init__.py",
    "src/osm_polygon_wikidata_only/io/atomic.py",
    "src/osm_polygon_wikidata_only/io/hashing.py",
    "src/osm_polygon_wikidata_only/io/run_lock.py",
    "src/osm_polygon_wikidata_only/utils/__init__.py",
    "src/osm_polygon_wikidata_only/utils/json.py",
)


class Transport:
    def __init__(self, output: Path):
        self.output = output
        self.commands = []
        self.uploads = []
        self.downloads = []
        self.submission = "OAR_JOB_ID=42"
        self.submission_code = 0
        self.statuses = ["state = Terminated\nexit_code = 0"]
        self.policy = "OK"
        self.policy_code = 0
        self.home = "/home/test"
        self.interrupt = False

    def run_frontend(self, args):
        self.commands.append(tuple(args))
        output = ""
        code = 0
        if args[0] == "printf":
            output = self.home
        elif args[0] == "usagepolicycheck":
            output, code = self.policy, self.policy_code
        elif args[0] == "oarsub":
            output, code = self.submission, self.submission_code
        elif args[0] == "oarstat":
            if self.interrupt:
                raise KeyboardInterrupt
            if not self.statuses:
                raise AssertionError("unexpected extra OAR status poll")
            output = self.statuses.pop(0)
        return subprocess.CompletedProcess(args, code, output, "")

    def upload_tree(self, local_root, remote_root):
        self.uploads.append((local_root, remote_root))

    def download_tree(self, remote_root, local_root):
        self.downloads.append(remote_root)
        shutil.copytree(self.output, local_root, dirs_exist_ok=True)

    def remove_tree(self, remote_root):
        pytest.fail("Pilot must retain remote output")

    @property
    def submits(self):
        return [args for args in self.commands if args[0] == "oarsub"]


@pytest.fixture
def pilot(tmp_path):
    stage = tmp_path / "stage with spaces"
    stage.mkdir()
    source = pa.table(
        {
            "sentence_id": ["s1"],
            "document_id": ["d1"],
            "project": ["wikipedia"],
            "language": ["en"],
            "text": ["Paris"],
            "segmentation_status": ["split"],
        }
    )
    pq.write_table(source, stage / "input.parquet")
    contract = Contract(languages=("en",))
    (stage / "contract.json").write_text(json.dumps(asdict(contract)))
    (stage / "run.sh").write_text('cd "$(dirname "$0")"\n')
    (stage / "code").mkdir()
    (stage / "code" / "worker.py").write_text("# worker")
    for relative in _RUNTIME_FILES:
        runtime_file = stage / "code" / relative
        runtime_file.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "example==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"
            if relative == "requirements/geographic-ner-gpu.txt"
            else "# runtime\n"
        )
        runtime_file.write_text(content)
    output = tmp_path / "remote-output"
    output.mkdir()
    pq.write_table(pa.table({"text": ["Paris"]}), output / "batch-000000.parquet")
    receipt = {
        "contract": asdict(contract),
        "contract_id": contract.identity,
        "source_sha256": hashlib.sha256((stage / "input.parquet").read_bytes()).hexdigest(),
        "source_rows": 1,
        "batch_size": 128,
        "processed_rows": 1,
        "status": "completed",
        "artifacts": [
            {
                "path": "batch-000000.parquet",
                "sha256": hashlib.sha256(
                    (output / "batch-000000.parquet").read_bytes()
                ).hexdigest(),
                "rows": 1,
            }
        ],
    }
    (output / "receipt.json").write_text(json.dumps(receipt))
    return stage, tmp_path / "durable", Transport(output)


def run(pilot, **kwargs):
    stage, directory, transport = pilot
    return controller_module().run_grid5000_ner_controller(
        staging_dir=stage,
        run_dir=directory,
        run_id="pilot-1",
        transport=transport,
        repo_id="owner/dataset",
        sleep=lambda seconds: None,
        **kwargs,
    )


def direct_controller(pilot, *, initialize=True):
    stage, directory, transport = pilot
    controller = controller_module().Grid5000NerController(
        stage,
        directory,
        run_id="pilot-1",
        site="rennes",
        queue="besteffort",
        gpu_model="A40",
        period="day",
        repo_id="owner/dataset",
        transport=transport,
        sleep=lambda seconds: None,
        publish=False,
        publisher=lambda *args, **kwargs: "commit",
    )
    directory.mkdir(parents=True, exist_ok=True)
    if initialize:
        controller.ledger = controller._load_or_create_ledger()
    return controller


def test_completed_repeat_never_submits_or_uploads_again(pilot, case_sensitive_paths):
    assert run(pilot)["state"] == "completed"
    assert (pilot[1] / "output" / "receipt.json").is_file()
    assert run(pilot)["state"] == "completed"
    transport = pilot[2]
    assert len(transport.submits) == len(transport.uploads) == 1
    command = transport.submits[0]
    assert (
        command[-1]
        == "bash /home/test/osm-polygon-wikidata-only-grid5000/geographic-ner/pilot-1/run.sh"
    )
    assert command[command.index("-l") + 1] == "host=1/gpu=1,walltime=0:20"
    assert command[command.index("-t") + 1] == "day"
    assert [c for c in transport.commands if c[0] == "usagepolicycheck"] == [
        ("usagepolicycheck", "-t", "--sites", "rennes")
    ] * 3


def test_interrupt_resumes_known_job(pilot):
    pilot[2].interrupt = True
    with pytest.raises(KeyboardInterrupt):
        run(pilot)
    ledger = json.loads((pilot[1] / "ledger.json").read_text())
    assert ledger["job_id"] == "42"
    pilot[2].interrupt = False
    assert run(pilot)["state"] == "completed"
    assert len(pilot[2].submits) == 1
    assert ("oarstat", "-f", "-j", "42") in pilot[2].commands


def test_uncertain_submission_requires_recovery(pilot):
    pilot[2].submission = "connection lost"
    for _ in range(2):
        with raises_exactly(UNCERTAIN_SUBMISSION):
            run(pilot)
    assert len(pilot[2].submits) == 1
    assert json.loads((pilot[1] / "ledger.json").read_text())["state"] == "submitting"


def test_nonzero_submission_with_job_id_requires_recovery(pilot):
    pilot[2].submission_code = 1
    with raises_exactly(UNCERTAIN_SUBMISSION):
        run(pilot)
    assert len(pilot[2].submits) == 1
    assert json.loads((pilot[1] / "ledger.json").read_text())["state"] == "submitting"

    with raises_exactly(UNCERTAIN_SUBMISSION):
        run(pilot)
    assert len(pilot[2].submits) == 1


def test_failed_publication_retries_only_publication(pilot):
    calls = []

    def publish(output_dir, *, repo_id, run_id):
        calls.append((output_dir, repo_id, run_id))
        if len(calls) == 1:
            raise OSError("Hub offline")
        return "commit"

    with raises_exactly(PUBLICATION_FAILED):
        run(pilot, publish=True, publisher=publish)
    failed = json.loads((pilot[1] / "ledger.json").read_text())
    assert failed["state"] == "ready_to_publish"
    assert failed["error"] == "publisher_failure"
    before = list(pilot[2].commands)
    assert run(pilot, publish=True, publisher=publish)["state"] == "published"
    assert pilot[2].commands == before
    assert len(calls) == 2
    assert calls[0][1:] == ("owner/dataset", "pilot-1")
    ledger = json.loads((pilot[1] / "ledger.json").read_text())
    assert ledger["state"] == "published"
    assert ledger["commit"] == "commit"
    assert isinstance(ledger["published_at"], str) and ledger["published_at"]
    assert ledger["error"] is None


def test_failed_publication_preserves_the_ledger_schema(pilot):
    before = run(pilot)

    def fail(*args, **kwargs):
        raise OSError("Hub offline")

    with raises_exactly(PUBLICATION_FAILED):
        run(pilot, publish=True, publisher=fail)
    after = json.loads((pilot[1] / "ledger.json").read_text())
    assert after.keys() == before.keys()
    assert after["state"] == "ready_to_publish"
    assert after["error"] == "publisher_failure"


def test_prepare_submission_increments_attempt_and_clears_stale_state(pilot):
    controller = direct_controller(pilot)
    controller.ledger.update({"attempt": 2, "state": "paused", "job_id": "42", "error": "old"})

    controller._prepare_submission()

    assert controller.ledger == {
        **controller.ledger,
        "attempt": 3,
        "state": "submitting",
        "job_id": None,
        "error": None,
    }


def test_submission_without_an_attempt_counter_starts_at_one(pilot):
    controller = direct_controller(pilot)
    del controller.ledger["attempt"]
    controller._prepare_submission()
    assert json.loads(controller.ledger_path.read_text())["attempt"] == 1


def test_uncertain_submission_records_recovery_metadata(pilot):
    controller = direct_controller(pilot)

    controller._write_submission_uncertain(OSError("network down"))

    assert controller.ledger["state"] == "submitting"
    assert controller.ledger["job_id"] is None
    assert controller.ledger["error"] == "submission_uncertain"
    assert controller.ledger["submission_exception"] == "OSError"


def test_new_ledger_contains_resumable_defaults(pilot, monkeypatch):
    monkeypatch.setattr(controller_module(), "_timestamp", lambda: "2026-09-08T12:00:00+00:00")
    controller = direct_controller(pilot)

    assert controller.ledger["state"] == "planned"
    assert controller.ledger["job_id"] is None
    assert controller.ledger["attempt"] == 0
    assert controller.ledger["uploaded"] is False
    assert controller.ledger["remote_home"] is None
    assert controller.ledger["error"] is None
    assert controller.ledger["created_at"] == "2026-09-08T12:00:00+00:00"
    assert controller.ledger["updated_at"] == "2026-09-08T12:00:00+00:00"
    assert controller.ledger["remote_root"] == (
        "$HOME/osm-polygon-wikidata-only-grid5000/geographic-ner/pilot-1"
    )


def test_remote_home_resolution_uses_literal_home_probe_and_rejects_unsafe_value(pilot):
    controller = direct_controller(pilot)

    assert controller._resolve_remote_home() == "/home/test"
    assert controller.transport.commands[-1] == ("printf", "%s", "$HOME")
    controller.transport.home = "/home/Test.User"
    assert controller._resolve_remote_home() == "/home/Test.User"

    controller.transport.home = "/tmp/../escape"
    with raises_exactly("Grid5000 remote home is invalid"):
        controller._resolve_remote_home()


@pytest.mark.parametrize(
    "policy,code",
    [
        ("Error", 0),
        ("Error: quota", 0),
        ("account-wide daytime quota exceeded", 0),
        ("DNS resolution error", 0),
        ("violation", 0),
        ("Access denied", 0),
        ("OK", 1),
    ],
)
def test_policy_false_negative_blocks_submission(pilot, policy, code):
    pilot[2].policy, pilot[2].policy_code = policy, code
    with raises_exactly(f"Grid5000 usage policy check failed for site rennes: {policy}"):
        run(pilot)
    assert not pilot[2].submits


def test_corrupt_artifact_blocks_publication(pilot):
    (pilot[2].output / "batch-000000.parquet").write_bytes(b"corrupt")
    with raises_exactly("Geographic NER artifact hash mismatch: batch-000000.parquet"):
        run(pilot, publish=True, publisher=lambda *args, **kwargs: pytest.fail("published"))
    assert len(pilot[2].downloads) == 1


def test_failed_job_recovers_output_and_does_not_loop(pilot):
    pilot[2].statuses = ["state = Error\nexit_code = 1"]
    for attempt, message in enumerate(JOB_FAILURE_MESSAGES):
        assert attempt < len(JOB_FAILURE_MESSAGES)
        with raises_exactly(message):
            run(pilot)
    assert (pilot[1] / "output" / "batch-000000.parquet").exists()
    assert len(pilot[2].submits) == 1


def test_terminal_job_is_not_polled_again(pilot):
    assert run(pilot)["state"] == "completed"
    assert [command for command in pilot[2].commands if command[0] == "oarstat"] == [
        ("oarstat", "-f", "-j", "42")
    ]


def test_paused_continuation_does_not_replace_stage(pilot):
    pilot[2].statuses = [
        "state = Terminated\nexit_code = 0",
        "state = Terminated\nexit_code = 0",
    ]
    path = pilot[2].output / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "paused"
    path.write_text(json.dumps(receipt))
    assert run(pilot)["state"] == "paused"
    assert len(pilot[2].submits) == 1
    receipt["status"] = "completed"
    path.write_text(json.dumps(receipt))
    assert run(pilot)["state"] == "completed"
    assert len(pilot[2].submits) == 2
    assert len(pilot[2].uploads) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("run_id", "../bad"),
        ("site", "rennes;id"),
        ("gpu_model", "A40' OR 1=1"),
        ("queue", "-x"),
        ("period", "weekly"),
    ],
)
def test_unsafe_arguments_never_reach_transport(pilot, key, value):
    args = dict(
        staging_dir=pilot[0],
        run_dir=pilot[1],
        run_id="pilot-1",
        repo_id="owner/data",
        transport=pilot[2],
    )
    args[key] = value
    with raises_exactly(f"Unsafe Grid5000 {key}: {value!r}", ValueError):
        controller_module().run_grid5000_ner_controller(**args)
    assert not pilot[2].commands


def test_immutable_stage_rejected_after_submission(pilot):
    pilot[2].interrupt = True
    with pytest.raises(KeyboardInterrupt):
        run(pilot)
    (pilot[0] / "code" / "worker.py").write_text("# changed")
    with raises_exactly(STAGE_IMMUTABLE):
        run(pilot)
    assert len(pilot[2].submits) == 1


def test_tampered_stage_manifest_is_rejected(pilot):
    pilot[2].interrupt = True
    with pytest.raises(KeyboardInterrupt):
        run(pilot)
    pilot[2].interrupt = False
    ledger_path = pilot[1] / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["stage_manifest"] = []
    ledger_path.write_text(json.dumps(ledger))

    with raises_exactly(STAGE_IMMUTABLE):
        run(pilot)
    assert len(pilot[2].submits) == 1


def test_direct_controller_rejects_incomplete_runtime_tree_before_transport(pilot):
    (pilot[0] / "code/src/osm_polygon_wikidata_only/ner/pipeline.py").unlink()

    with raises_exactly(
        "Staging code is missing required geographic NER runtime files:"
        " src/osm_polygon_wikidata_only/ner/pipeline.py"
    ):
        run(pilot)

    assert not pilot[2].commands


def test_direct_controller_rejects_unhashed_runtime_lock_before_transport(pilot):
    (pilot[0] / "code/requirements/geographic-ner-gpu.txt").write_text("example==1.0\n")

    with raises_exactly("Staged GPU requirements lock must hash every pinned package"):
        run(pilot)

    assert not pilot[2].commands


def test_staging_file_walk_includes_worker_code(pilot):
    module = controller_module()
    stage = pilot[0]

    files = module._staging_files(stage)
    code_files = module._staged_code_files(stage / "code")

    assert stage / "code" / "worker.py" in files
    assert stage / "code" / "worker.py" in code_files


def test_controller_entrypoint_preserves_injected_transport(pilot, monkeypatch):
    module = controller_module()
    captured = {}

    class RecordingController:
        def __init__(self, *args, **kwargs):
            captured["transport"] = kwargs["transport"]

        def run(self):
            return {"state": "captured"}

    monkeypatch.setattr(module, "Grid5000NerController", RecordingController)
    result = module.run_grid5000_ner_controller(
        pilot[0],
        pilot[1],
        "pilot-1",
        "owner/dataset",
        transport=pilot[2],
        sleep=lambda seconds: None,
    )

    assert result == {"state": "captured"}
    assert captured["transport"] is pilot[2]


def test_geographic_ner_cli_forwards_defaults_and_publish(tmp_path, monkeypatch, capsys):
    from scripts import grid5000_geographic_ner

    captured = {}

    def fake_run(staging_dir, run_dir, run_id, repo_id, **kwargs):
        captured.update(
            staging_dir=staging_dir,
            run_dir=run_dir,
            run_id=run_id,
            repo_id=repo_id,
            **kwargs,
        )
        return {"state": "published", "run_id": run_id}

    monkeypatch.setattr(grid5000_geographic_ner, "run_grid5000_ner_controller", fake_run)
    assert (
        grid5000_geographic_ner.main(
            [
                "--staging-dir",
                str(tmp_path / "stage"),
                "--run-dir",
                str(tmp_path / "run"),
                "--run-id",
                "pilot-1",
                "--repo-id",
                "owner/data",
                "--publish",
            ]
        )
        == 0
    )
    assert captured == {
        "staging_dir": tmp_path / "stage",
        "run_dir": tmp_path / "run",
        "run_id": "pilot-1",
        "repo_id": "owner/data",
        "site": "rennes",
        "queue": "besteffort",
        "gpu_model": "A40",
        "period": "day",
        "publish": True,
    }
    assert json.loads(capsys.readouterr().out) == {"run_id": "pilot-1", "state": "published"}


def test_geographic_ner_cli_prepares_complete_staging_tree(pilot, tmp_path, capsys):
    from scripts import grid5000_geographic_ner

    source_root = Path(__file__).resolve().parents[2]
    stage, _, _ = pilot
    prepared = tmp_path / "prepared-stage"
    assert (
        grid5000_geographic_ner.main(
            [
                "--staging-dir",
                str(prepared),
                "--source",
                str(stage / "input.parquet"),
                "--contract",
                str(stage / "contract.json"),
                "--source-root",
                str(source_root),
                "--requirements-lock",
                str(source_root / "requirements/geographic-ner-gpu.txt"),
                "--prepare-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "state": "prepared",
        "staging_dir": str(prepared),
    }
    assert {
        "input.parquet",
        "contract.json",
        "code/requirements/geographic-ner-gpu.txt",
        "code/src/osm_polygon_wikidata_only/ner/job.py",
        "code/src/osm_polygon_wikidata_only/ner/pipeline.py",
        "code/src/osm_polygon_wikidata_only/ner/publication.py",
        "code/src/osm_polygon_wikidata_only/io/atomic.py",
        "code/src/osm_polygon_wikidata_only/utils/json.py",
    } <= {path.relative_to(prepared).as_posix() for path in prepared.rglob("*") if path.is_file()}
    assert not (prepared / "run.sh").exists()


def test_geographic_ner_staging_preflight_rejects_incomplete_inputs(pilot, tmp_path):
    from scripts import grid5000_geographic_ner

    stage, _, _ = pilot
    source_root = Path(__file__).resolve().parents[2]
    bad_input = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"text": ["Paris"]}), bad_input)
    with raises_exactly(
        "Geographic NER input is missing columns: "
        "['document_id', 'language', 'project', 'segmentation_status', 'sentence_id']",
        ValueError,
    ):
        grid5000_geographic_ner.prepare_geographic_ner_staging(
            tmp_path / "missing-columns",
            source=bad_input,
            contract=stage / "contract.json",
            source_root=source_root,
        )

    empty_input = tmp_path / "empty.parquet"
    pq.write_table(pa.table({column: [] for column in INPUT_COLUMNS}), empty_input)
    with raises_exactly("Geographic NER input must contain at least one row", ValueError):
        grid5000_geographic_ner.prepare_geographic_ner_staging(
            tmp_path / "empty-input",
            source=empty_input,
            contract=stage / "contract.json",
            source_root=source_root,
        )

    input_link = tmp_path / "input-link.parquet"
    input_link.symlink_to(stage / "input.parquet")
    with pytest.raises(
        ValueError, match=r"^Geographic NER input is missing or unsafe: .*/input-link\.parquet$"
    ):
        grid5000_geographic_ner.prepare_geographic_ner_staging(
            tmp_path / "symlink-input",
            source=input_link,
            contract=stage / "contract.json",
            source_root=source_root,
        )

    bad_lock = tmp_path / "requirements.txt"
    bad_lock.write_text("torch==2.8.0\n")
    with raises_exactly("GPU requirements lock must hash every pinned package", ValueError):
        grid5000_geographic_ner.prepare_geographic_ner_staging(
            tmp_path / "unhashed-lock",
            source=stage / "input.parquet",
            contract=stage / "contract.json",
            source_root=source_root,
            requirements_lock=bad_lock,
        )

    with raises_exactly("Source root is missing the geographic NER runtime", ValueError):
        grid5000_geographic_ner._validate_source_tree(tmp_path / "missing-source")

    copied_tree = tmp_path / "copied-tree"
    source_tree = tmp_path / "source-tree"
    (source_tree / "nested").mkdir(parents=True)
    (source_tree / "nested" / "worker.py").write_text("# worker")
    grid5000_geographic_ner._copy_tree(source_tree, copied_tree)
    assert (copied_tree / "nested/worker.py").read_text() == "# worker"
    link = source_tree / "link.py"
    link.symlink_to(source_tree / "nested/worker.py")
    with pytest.raises(
        ValueError, match=r"^Symlinks are not allowed in staged source: .*/source-tree$"
    ):
        grid5000_geographic_ner._copy_tree(source_tree, tmp_path / "symlink-tree")

    with raises_exactly(
        "--source/--input and --contract are required for staging preparation", SystemExit
    ):
        grid5000_geographic_ner.main(
            ["--staging-dir", str(tmp_path / "no-input"), "--prepare-only"]
        )
    with raises_exactly(
        "--run-dir, --run-id, and --repo-id are required to run the controller", SystemExit
    ):
        grid5000_geographic_ner.main(["--staging-dir", str(tmp_path / "no-run")])


def test_controller_pipeline_and_real_publisher_resume_offline(tmp_path, monkeypatch):
    from osm_polygon_wikidata_only.ner.pipeline import run_shard
    from osm_polygon_wikidata_only.ner.publication import publish_ner_shard
    from scripts import grid5000_geographic_ner

    source = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence_id": [f"s-{index}" for index in range(129)],
                "document_id": [f"d-{index}" for index in range(129)],
                "project": ["wikipedia"] * 129,
                "language": ["en"] * 129,
                "text": [f"Place {index}" for index in range(129)],
                "segmentation_status": ["split"] * 129,
            }
        ),
        source,
    )
    contract_path = tmp_path / "contract.json"
    contract = Contract(languages=("en",))
    contract_path.write_text(json.dumps(asdict(contract)))
    staging = tmp_path / "staging"
    grid5000_geographic_ner.prepare_geographic_ner_staging(
        staging,
        source=source,
        contract=contract_path,
        source_root=Path(__file__).resolve().parents[2],
    )

    class FakeExtractor:
        def predict(self, texts):
            return [
                [
                    {
                        "text": text,
                        "label": "named geographic location",
                        "start": 0,
                        "end": len(text),
                        "score": 0.9,
                    }
                ]
                for text in texts
            ]

    class PipelineTransport:
        def __init__(self):
            self.commands = []
            self.remote = tmp_path / "remote"
            self.submissions = 0

        def run_frontend(self, args):
            self.commands.append(tuple(args))
            if args[0] == "printf":
                return subprocess.CompletedProcess(args, 0, "/home/test\n", "")
            if args[0] == "usagepolicycheck":
                return subprocess.CompletedProcess(args, 0, "OK\n", "")
            if args[0] == "oarsub":
                self.submissions += 1
                raw = json.loads((self.remote / "contract.json").read_text())
                raw["languages"] = tuple(raw["languages"])
                if self.submissions == 1:
                    ticks = iter((0.0, 2.0))
                    run_shard(
                        self.remote / "input.parquet",
                        self.remote / "output",
                        FakeExtractor(),
                        Contract(**raw),
                        batch_size=128,
                        deadline=1.0,
                        clock=lambda: next(ticks, 2.0),
                    )
                else:
                    run_shard(
                        self.remote / "input.parquet",
                        self.remote / "output",
                        FakeExtractor(),
                        Contract(**raw),
                        batch_size=128,
                        deadline=math.inf,
                    )
                return subprocess.CompletedProcess(args, 0, "OAR_JOB_ID=42\n", "")
            if args[0] == "oarstat":
                return subprocess.CompletedProcess(
                    args, 0, "state = Terminated\nexit_code = 0\n", ""
                )
            return subprocess.CompletedProcess(args, 0, "", "")

        def upload_tree(self, local_root, remote_root):
            shutil.copytree(local_root, self.remote, dirs_exist_ok=True)

        def download_tree(self, remote_root, local_root):
            shutil.copytree(self.remote / "output", local_root, dirs_exist_ok=True)

        def remove_tree(self, remote_root):
            pytest.fail("Offline pilot must retain the remote namespace")

    transport = PipelineTransport()
    state = SimpleNamespace(
        revisions={"parent": {"README.md": b"original", "sentences/data.parquet": b"keep"}},
        fail_once=True,
    )
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="parent")
    api.list_repo_files.side_effect = lambda repo_id, **kwargs: list(
        state.revisions[kwargs["revision"]]
    )

    def download(repo_id, filename, *, revision, local_dir, **kwargs):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(state.revisions[revision][filename])
        return str(target)

    def commit(repo_id, operations, *, parent_commit, **kwargs):
        if state.fail_once:
            state.fail_once = False
            raise RuntimeError("fake Hub outage")
        result = dict(state.revisions[parent_commit])
        for operation in operations:
            with operation.as_file() as stream:
                result[operation.path_in_repo] = stream.read()
        state.revisions["commit"] = result
        return huggingface_hub.CommitInfo(
            commit_url="https://huggingface.co/datasets/a/b/commit/commit",
            commit_message="test",
            commit_description="",
            oid="commit",
        )

    api.create_commit.side_effect = commit
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    kwargs = dict(
        staging_dir=staging,
        run_dir=tmp_path / "run",
        run_id="pilot-1",
        repo_id="a/b",
        transport=transport,
        sleep=lambda seconds: None,
        publish=True,
        publisher=publish_ner_shard,
    )
    assert controller_module().run_grid5000_ner_controller(**kwargs)["state"] == "paused"
    first_receipt = json.loads((tmp_path / "run/output/receipt.json").read_text())
    assert first_receipt["status"] == "paused"
    assert first_receipt["processed_rows"] == 128

    with raises_exactly(PUBLICATION_FAILED):
        controller_module().run_grid5000_ner_controller(**kwargs)
    assert json.loads((tmp_path / "run/ledger.json").read_text())["state"] == "ready_to_publish"
    commands_before_publication_retry = list(transport.commands)

    assert controller_module().run_grid5000_ner_controller(**kwargs)["state"] == "published"
    assert transport.commands == commands_before_publication_retry
    final_receipt = json.loads((tmp_path / "run/output/receipt.json").read_text())
    assert final_receipt["status"] == "completed"
    assert final_receipt["processed_rows"] == final_receipt["source_rows"] == 129
    assert set(state.revisions["commit"]) == {
        "README.md",
        "sentences/data.parquet",
        "geographic_ner/pilot-1/receipt.json",
        "geographic_ner/pilot-1/batch-000000.parquet",
        "geographic_ner/pilot-1/batch-000001.parquet",
    }


def test_upload_contains_only_worker_inputs_and_exact_budget(pilot):
    stage, _, transport = pilot
    (stage / "unowned-secret.txt").write_text("must not leave the staging tree")
    captured = {}

    def capture(local_root, remote_root):
        captured["files"] = {
            path.relative_to(local_root).as_posix()
            for path in local_root.rglob("*")
            if path.is_file()
        }
        captured["run_sh"] = (local_root / "run.sh").read_text()
        captured["remote_root"] = remote_root

    transport.upload_tree = capture
    assert run(pilot)["state"] == "completed"
    assert "unowned-secret.txt" not in captured["files"]
    assert captured["files"] == {
        "input.parquet",
        "contract.json",
        "code/worker.py",
        "run.sh",
        *(f"code/{relative}" for relative in _RUNTIME_FILES),
    }
    assert "python -m osm_polygon_wikidata_only.ner.job" in captured["run_sh"]
    assert "--source input.parquet" in captured["run_sh"]
    assert "--output-dir output" in captured["run_sh"]
    assert "--contract contract.json" in captured["run_sh"]
    assert "--model-cache shared-cache" in captured["run_sh"]
    assert "--seconds 1020" in captured["run_sh"]
    assert "--batch-size 128" in captured["run_sh"]
    assert "--inference-batch-size 16" in captured["run_sh"]
    assert (
        captured["remote_root"] == "$HOME/osm-polygon-wikidata-only-grid5000/geographic-ner/pilot-1"
    )


def test_forged_parquet_row_count_blocks_publication(pilot):
    output = pilot[2].output
    artifact = output / "batch-000000.parquet"
    pq.write_table(pa.table({"text": ["Paris"]}), artifact)
    receipt = json.loads((output / "receipt.json").read_text())
    receipt["artifacts"] = [
        {
            "path": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "rows": 2,
        }
    ]
    receipt["source_rows"] = 2
    receipt["processed_rows"] = 2
    (output / "receipt.json").write_text(json.dumps(receipt))
    with raises_exactly(SOURCE_ROWS_MISMATCH):
        run(pilot, publish=True, publisher=lambda *args, **kwargs: pytest.fail("published"))


@pytest.mark.parametrize("batch_size", [None, 0, -1, True, 256])
def test_receipt_requires_the_positive_pilot_batch_size(pilot, batch_size):
    receipt_path = pilot[2].output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if batch_size is None:
        receipt.pop("batch_size", None)
    else:
        receipt["batch_size"] = batch_size
    receipt_path.write_text(json.dumps(receipt))

    with raises_exactly("Geographic NER receipt batch size is not the pilot batch size"):
        run(pilot)


@pytest.mark.parametrize("missing_field", ["source_rows", "processed_rows"])
def test_receipt_requires_pipeline_row_counts(pilot, missing_field):
    receipt_path = pilot[2].output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop(missing_field)
    receipt_path.write_text(json.dumps(receipt))

    with raises_exactly(ROW_COUNT_MESSAGES[missing_field]):
        run(pilot)


def test_receipt_source_rows_must_match_staged_input(pilot):
    receipt_path = pilot[2].output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(status="paused", source_rows=2, processed_rows=1)
    receipt_path.write_text(json.dumps(receipt))

    with raises_exactly(SOURCE_ROWS_MISMATCH):
        run(pilot)


def test_receipt_artifacts_require_pipeline_row_metadata(pilot):
    receipt_path = pilot[2].output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][0].pop("rows")
    receipt_path.write_text(json.dumps(receipt))

    with raises_exactly("Geographic NER artifact row count is required: batch-000000.parquet"):
        run(pilot)


def test_receipt_embedded_contract_is_bound_to_staging_contract_even_when_id_matches(pilot):
    receipt_path = pilot[2].output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["contract"]["threshold"] = 0.99
    receipt_path.write_text(json.dumps(receipt))

    with raises_exactly("Geographic NER receipt contract configuration mismatch"):
        run(pilot)


def test_batch_size_is_immutable_in_the_controller_ledger(pilot):
    run(pilot)
    ledger_path = pilot[1] / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    assert ledger["batch_size"] == 128
    ledger["batch_size"] = 256
    ledger_path.write_text(json.dumps(ledger))

    with raises_exactly("Immutable geographic NER ledger field changed: batch_size"):
        run(pilot)


def test_completed_run_cannot_regress_to_paused_output(pilot):
    assert run(pilot)["state"] == "completed"
    commands = list(pilot[2].commands)
    receipt_path = pilot[2].output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "paused"
    receipt_path.write_text(json.dumps(receipt))

    assert run(pilot)["state"] == "completed"
    assert pilot[2].commands == commands


def test_completed_ledger_state_cannot_be_set_to_paused(pilot):
    module = controller_module()
    stage, run_dir, transport = pilot
    controller = module.Grid5000NerController(
        stage,
        run_dir,
        run_id="pilot-1",
        site="rennes",
        queue="besteffort",
        gpu_model="A40",
        period="day",
        repo_id="owner/dataset",
        transport=transport,
        sleep=lambda seconds: None,
        publish=False,
        publisher=lambda *args, **kwargs: "commit",
    )
    controller.ledger = {"state": "completed"}
    with raises_exactly("Completed geographic NER run cannot regress to paused"):
        controller._set_state("paused")
    assert controller.ledger["state"] == "completed"


def test_artifacts_must_use_contiguous_pipeline_batch_names(pilot):
    receipt_path = pilot[2].output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    batch = pilot[2].output / "batch-000000.parquet"
    batch.rename(pilot[2].output / "part.parquet")
    receipt["artifacts"][0]["path"] = "part.parquet"
    receipt_path.write_text(json.dumps(receipt))

    with raises_exactly(ARTIFACT_ORDER_INVALID):
        run(pilot)


def test_unexpected_remote_output_file_blocks_retrieval(pilot):
    (pilot[2].output / "unexpected.txt").write_text("not in the receipt")
    with raises_exactly("Unexpected geographic NER output file: unexpected.txt"):
        run(pilot)


def test_oar_status_parser_infers_states_without_key_value():
    module = controller_module()
    for text, expected in (
        ("waiting for a resource", "waiting"),
        ("launching on node", "launching"),
        ("running", "running"),
        ("terminated", "terminated"),
        ("finishing", "finishing"),
        ("error", "error"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("no scheduler state", "unknown"),
    ):
        assert module._infer_state(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("state = Terminated\nexit_code = 0", ("terminated", 0)),
        ("state = Error\nexit_code = 1", ("error", 1)),
    ],
)
def test_oar_status_parser_preserves_reported_state_case(text, expected):
    module = controller_module()
    result = SimpleNamespace(stdout=text, stderr="")

    assert module._parse_job_status(result) == expected


def test_receipt_row_accounting_accepts_partial_metadata_and_rejects_conflicts():
    module = controller_module()
    module._validate_processed_rows(None, [1])
    module._validate_processed_rows(1, [None])
    module._validate_processed_rows(2, [1, 1])
    with raises_exactly("Geographic NER artifact rows do not match the receipt"):
        module._validate_processed_rows(3, [1, 1])

    module._validate_source_rows({"status": "paused"}, None, None)
    module._validate_source_rows({"status": "paused"}, None, 2)
    module._validate_source_rows({"status": "paused"}, 1, 2)
    module._validate_source_rows({"status": "completed"}, 2, 2)
    with raises_exactly("Geographic NER processed rows exceed source rows"):
        module._validate_source_rows({"status": "paused"}, 3, 2)
    with raises_exactly("Completed geographic NER receipt has incomplete rows"):
        module._validate_source_rows({"status": "completed"}, 1, 2)


def test_receipt_counts_validate_required_values_and_batch_order():
    module = controller_module()
    module._validate_receipt_counts({"status": "paused", "processed_rows": 0, "source_rows": 1}, [])
    module._validate_receipt_counts(
        {"status": "completed", "processed_rows": 2, "source_rows": 2}, [1, 1]
    )
    module._validate_artifact_order([])
    with raises_exactly(ARTIFACT_ORDER_INVALID):
        module._validate_artifact_order(["part.parquet"])
    module._validate_artifact_order(["batch-000000.parquet", "batch-000001.parquet"])
    with raises_exactly(ARTIFACT_ORDER_INVALID):
        module._validate_artifact_order(["batch-000001.parquet"])
    with raises_exactly("Geographic NER processed row count is invalid"):
        module._validate_receipt_counts({"status": "paused", "processed_rows": -1}, [None])
    with raises_exactly(PROCESSED_ROWS_REQUIRED):
        module._validate_receipt_counts({"status": "paused", "source_rows": 1}, [None])


def test_artifact_size_and_ledger_shape_are_fail_closed(tmp_path, pilot):
    module = controller_module()
    artifact = tmp_path / "part.bin"
    artifact.write_bytes(b"result")
    module._validate_artifact_size(artifact, "part.bin", {"size": 6})
    module._validate_artifact_size(artifact, "part.bin", {})
    with raises_exactly("Geographic NER artifact size mismatch: part.bin"):
        module._validate_artifact_size(artifact, "part.bin", {"size": 5})
    invalid_batch = tmp_path / "batch-000000.parquet"
    invalid_batch.write_bytes(b"not-a-parquet-artifact")
    with raises_exactly("Geographic NER artifact is not Parquet: batch-000000.parquet"):
        module._validate_parquet_footer(invalid_batch, invalid_batch.name, 1)

    stage, run_dir, transport = pilot
    controller = module.Grid5000NerController(
        stage,
        run_dir,
        run_id="pilot-1",
        site="rennes",
        queue="besteffort",
        gpu_model="A40",
        period="day",
        repo_id="owner/dataset",
        transport=transport,
        sleep=lambda seconds: None,
        publish=False,
        publisher=lambda *args, **kwargs: "commit",
    )
    valid = {"state": "planned", "stage_manifest": [], "stage_sha256": "a" * 64}
    controller._validate_ledger_shape(valid)
    invalid_ledgers = (
        {**valid, "state": None},
        {**valid, "stage_manifest": None},
        {**valid, "stage_manifest": ["not-an-object"]},
        {**valid, "stage_sha256": "not-a-digest"},
    )
    for invalid, message in zip(invalid_ledgers, LEDGER_SHAPE_MESSAGES, strict=True):
        with raises_exactly(message):
            controller._validate_ledger_shape(invalid)


def test_run_script_puts_the_locked_environment_bin_on_path() -> None:
    script = controller_module()._run_script()

    assert 'export PATH="$ENV_DIR/bin:$PATH"\n' in script


def test_run_script_retries_an_interrupted_dependency_install(tmp_path):
    """An existing Python executable is not evidence that pip finished."""
    root = tmp_path / "run"
    lock = root / "code/requirements/geographic-ner-gpu.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("example==1.0\n")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    environment = root / "environment-cache" / digest
    python = environment / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf worker >> "$RUN_LOG"\n')
    python.chmod(0o755)
    uv = root / "uv-bootstrap/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$RUN_LOG"\n')
    uv.chmod(0o755)
    script = root / "run.sh"
    script.write_text(controller_module()._run_script())
    log = root / "calls.log"

    import os

    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "RUN_LOG": str(log), "PIP_NO_INDEX": "1", "UV_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "pip install --python" in calls
    assert "--require-hashes -r code/requirements/geographic-ner-gpu.txt" in calls
    assert calls.endswith("worker")
    assert (environment / ".ready").is_file()

    log.write_text("")
    subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "RUN_LOG": str(log), "PIP_NO_INDEX": "1", "UV_OFFLINE": "1"},
        check=True,
        timeout=5,
    )
    assert log.read_text() == "worker"


def test_run_script_content_is_pinned_exactly() -> None:
    """The submitted GPU script is security relevant: pin it byte for byte."""
    expected = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RUN_ROOT="$(cd "$(dirname "$0")" && pwd -P)"\n'
        'cd "$RUN_ROOT"\n'
        'UV_BOOTSTRAP="$RUN_ROOT/uv-bootstrap"\n'
        'UV_CACHE_DIR="$RUN_ROOT/uv-cache"\n'
        'ENV_CACHE="$RUN_ROOT/environment-cache"\n'
        'UV_BIN="$UV_BOOTSTRAP/bin/uv"\n'
        'if [ ! -x "$UV_BIN" ]; then\n'
        '  mkdir -p "$UV_BOOTSTRAP" "$UV_CACHE_DIR"\n'
        "  if command -v python3 >/dev/null 2>&1; then\n"
        '    python3 -m venv "$UV_BOOTSTRAP"\n'
        "  elif command -v uv >/dev/null 2>&1; then\n"
        '    uv venv --python 3.12 "$UV_BOOTSTRAP"\n'
        "  else\n"
        '    echo "Python 3 or uv is required for the GPU bootstrap" >&2\n'
        "    exit 1\n"
        "  fi\n"
        '  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="$UV_CACHE_DIR"     "$UV_BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check     --no-input "uv==0.11.16"\n'
        "fi\n"
        'LOCK_SHA="$(sha256sum code/requirements/geographic-ner-gpu.txt | cut -d" " -f1)"\n'
        'ENV_DIR="$ENV_CACHE/$LOCK_SHA"\n'
        'if [ ! -x "$ENV_DIR/bin/python" ]; then\n'
        '  mkdir -p "$ENV_CACHE"\n'
        '  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="$UV_CACHE_DIR"     "$UV_BIN" venv --python 3.12 "$ENV_DIR"\n'
        "fi\n"
        'if [ ! -f "$ENV_DIR/.ready" ]; then\n'
        '  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="$UV_CACHE_DIR"     "$UV_BIN" pip install --python "$ENV_DIR/bin/python" --require-hashes     -r code/requirements/geographic-ner-gpu.txt\n'
        '  touch "$ENV_DIR/.ready"\n'
        "fi\n"
        'export PATH="$ENV_DIR/bin:$PATH"\n'
        'export PYTHONPATH="$RUN_ROOT/code/src:$RUN_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"\n'
        "exec python -m osm_polygon_wikidata_only.ner.job --source input.parquet --output-dir output --contract contract.json --model-cache shared-cache --seconds 1020 --batch-size 128 --inference-batch-size 16\n"
    )

    assert controller_module()._run_script() == "".join(expected)


def test_run_script_keeps_its_safety_critical_invariants() -> None:
    script = controller_module()._run_script()

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert script.count("env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN") == 3
    assert "--require-hashes" in script
    assert '"uv==0.11.16"' in script
    assert 'export PATH="$ENV_DIR/bin:$PATH"\n' in script
    assert script.endswith("--seconds 1020 --batch-size 128 --inference-batch-size 16\n")


def test_default_publisher_delegates_to_the_real_publisher_without_a_token(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    def fake_publish(output_dir, *, repo_id, run_id, token):
        recorded.update(output_dir=output_dir, repo_id=repo_id, run_id=run_id, token=token)
        return "commit-sha"

    monkeypatch.setattr("osm_polygon_wikidata_only.ner.publication.publish_ner_shard", fake_publish)

    result = controller_module()._default_publisher(tmp_path, repo_id="a/b", run_id="pilot-1")

    assert result == "commit-sha"
    assert recorded == {
        "output_dir": tmp_path,
        "repo_id": "a/b",
        "run_id": "pilot-1",
        "token": None,
    }


@pytest.mark.parametrize("value", [None, 1, "", "x" * 64, "a" * 63, "a" * 65])
def test_artifact_digest_rejects_invalid_metadata(value):
    with raises_exactly("Geographic NER artifact metadata is invalid"):
        controller_module()._artifact_digest(value)


def test_artifact_digest_normalizes_hex_case():
    assert controller_module()._artifact_digest("ABCDEF01" * 8) == "abcdef01" * 8


@pytest.mark.parametrize("value", [None, 1, "", "/tmp/file", "a/../file", "a?b", "a\nb"])
def test_artifact_path_metadata_rejects_unsafe_names(value):
    with raises_exactly("Geographic NER artifact metadata is invalid"):
        controller_module()._artifact_relative(value)


@pytest.mark.parametrize("value", [None, [], "artifact"])
def test_artifact_metadata_requires_an_object(value):
    with raises_exactly("Geographic NER artifact is not an object"):
        controller_module()._artifact_metadata(value)


@pytest.mark.parametrize("name", ["source", "processed"])
@pytest.mark.parametrize("value", [True, False, -1, 1.5, "0"])
def test_required_row_count_rejects_non_integer_or_negative_values(name, value):
    with raises_exactly(f"Geographic NER {name} row count is invalid"):
        controller_module()._required_count(value, name)


def test_zero_row_counts_are_valid_for_paused_empty_work():
    module = controller_module()
    assert module._required_count(0, "source") == 0
    module._validate_receipt_counts({"status": "paused", "processed_rows": 0, "source_rows": 0}, [])


@pytest.mark.parametrize(
    "reader, message",
    [
        ("_read_receipt", "Invalid geographic NER receipt"),
        ("_read_mapping", "Invalid geographic NER ledger"),
        ("_read_contract_payload", "Invalid staged geographic NER contract"),
    ],
)
@pytest.mark.parametrize("content", [None, b"{", b"\xff"])
def test_json_readers_report_missing_corrupt_or_non_utf8_input(tmp_path, reader, message, content):
    path = tmp_path / "broken.json"
    if content is not None:
        path.write_bytes(content)
    with raises_exactly(f"{message}: {path}"):
        getattr(controller_module(), reader)(path)


@pytest.mark.parametrize(
    "reader, message",
    [
        ("_read_receipt", "Geographic NER receipt must be an object"),
        ("_read_mapping", "Geographic NER ledger must be an object"),
        ("_read_contract_payload", "Invalid staged geographic NER contract"),
    ],
)
@pytest.mark.parametrize("value", [[], None, "text", 1])
def test_json_readers_reject_non_object_roots(tmp_path, reader, message, value):
    path = tmp_path / "wrong-shape.json"
    path.write_text(json.dumps(value))
    with raises_exactly(message):
        getattr(controller_module(), reader)(path)


@pytest.mark.parametrize("value", [None, [], ["bad"]])
def test_downloaded_output_artifact_list_is_validated(value):
    module = controller_module()
    if value == []:
        assert module._downloaded_output_paths({"artifacts": value}) == {
            "receipt.json",
            "predictions.sqlite3",
            "run.lock",
        }
        return
    message = (
        "Geographic NER artifact is not an object"
        if isinstance(value, list)
        else "Geographic NER receipt artifacts must be a list"
    )
    with raises_exactly(message):
        module._downloaded_output_paths({"artifacts": value})


@pytest.mark.parametrize("value", [None, [], {}])
def test_receipt_artifacts_need_a_list_and_completed_work(tmp_path, value):
    message = (
        "Geographic NER receipt has no completed artifacts"
        if value == []
        else "Geographic NER receipt artifacts must be a list"
    )
    with raises_exactly(message):
        controller_module()._validate_receipt_artifacts(
            tmp_path, {"status": "completed", "artifacts": value}
        )


def test_artifact_row_validation_preserves_zero_and_reports_bad_types(tmp_path):
    module = controller_module()
    path = tmp_path / "batch-000000.parquet"
    pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), path)
    assert module._artifact_rows(path, path.name, {"rows": 0}) == 0
    for value in (True, -1, "0"):
        with raises_exactly(f"Geographic NER artifact row count is invalid: {path.name}"):
            module._artifact_rows(path, path.name, {"rows": value})
    with raises_exactly(f"Geographic NER Parquet row count mismatch: {path.name}"):
        module._artifact_rows(path, path.name, {"rows": 1})


def test_artifact_resolving_rejects_missing_and_symlink_targets(tmp_path):
    module = controller_module()
    with raises_exactly("Geographic NER artifact is missing: missing.parquet"):
        module._artifact_path(tmp_path, "missing.parquet")
    target = tmp_path / "real.parquet"
    target.write_bytes(b"data")
    (tmp_path / "link.parquet").symlink_to(target)
    with raises_exactly("Geographic NER artifact is missing: link.parquet"):
        module._artifact_path(tmp_path, "link.parquet")
    with raises_exactly("Artifact escapes output directory: ../escape.parquet"):
        module._artifact_path(tmp_path, "../escape.parquet")


def test_uninitialized_controller_fails_explicitly(pilot):
    controller = direct_controller(pilot, initialize=False)
    with raises_exactly("Geographic NER ledger is not initialized"):
        controller._require_ledger()


def test_unsupported_ledger_state_cannot_submit(pilot):
    controller = direct_controller(pilot)
    with raises_exactly("Unsupported geographic NER ledger state: 'unexpected'"):
        controller._start_or_resume_job("unexpected")
    assert not pilot[2].submits


@pytest.mark.parametrize("job_id", [None, 42, ""])
def test_known_run_without_job_id_never_submits_again(pilot, job_id):
    controller = direct_controller(pilot)
    controller.ledger["job_id"] = job_id
    with raises_exactly(
        "Known geographic NER run has no OAR job ID; refusing duplicate submission"
    ):
        controller._known_job_id()
    assert not pilot[2].submits


def test_unknown_status_retains_job_for_reconciliation(pilot):
    pilot[2].statuses = ["unrecognized response"]
    with raises_exactly("Could not inspect OAR job 42; run remains resumable"):
        run(pilot)
    ledger = json.loads((pilot[1] / "ledger.json").read_text())
    assert ledger["state"] == "running"
    assert ledger["job_id"] == "42"


def test_runtime_lock_reports_unreadable_bytes(tmp_path):
    path = tmp_path / "lock"
    with raises_exactly("Staged GPU requirements lock is unreadable"):
        controller_module()._runtime_lock_blocks(path)
    path.write_bytes(b"\xff")
    with raises_exactly("Staged GPU requirements lock is unreadable"):
        controller_module()._runtime_lock_blocks(path)


def test_completed_then_published_run_is_an_offline_noop(pilot):
    assert run(pilot)["state"] == "completed"
    assert (
        run(pilot, publish=True, publisher=lambda *args, **kwargs: "commit")["state"] == "published"
    )
    before = list(pilot[2].commands)
    assert run(pilot)["state"] == "published"
    assert pilot[2].commands == before


def test_planned_uploaded_run_reuses_stage_and_cached_home(pilot):
    controller = direct_controller(pilot)
    controller._stage_for_submission()
    controller.ledger["remote_home"] = "/home/cached"
    controller._write_ledger()
    uploads = len(pilot[2].uploads)
    assert run(pilot)["state"] == "completed"
    assert len(pilot[2].uploads) == uploads
    assert not any(command[0] == "printf" for command in pilot[2].commands)
    assert (
        pilot[2].submits[0][-1]
        == "bash /home/cached/osm-polygon-wikidata-only-grid5000/geographic-ner/pilot-1/run.sh"
    )


def test_remote_home_is_cached_in_the_ledger(pilot):
    controller = direct_controller(pilot)
    assert controller._remote_home() == "/home/test"
    assert controller._remote_home() == "/home/test"
    assert pilot[2].commands == [("printf", "%s", "$HOME")]
    assert controller.ledger["remote_home"] == "/home/test"


def test_job_submission_and_namespace_commands_are_exact(pilot):
    controller = direct_controller(pilot)
    assert controller._submission_command("/home/test") == (
        "oarsub",
        "-q",
        "besteffort",
        "-p",
        "gpu_model='A40'",
        "-l",
        "host=1/gpu=1,walltime=0:20",
        "-t",
        "day",
        "bash /home/test/osm-polygon-wikidata-only-grid5000/geographic-ner/pilot-1/run.sh",
    )
    controller._ensure_remote_namespace()
    root = "$HOME/osm-polygon-wikidata-only-grid5000/geographic-ner/pilot-1"
    assert pilot[2].commands == [
        (
            "mkdir",
            "-p",
            root,
            f"{root}/output",
            f"{root}/shared-cache",
            f"{root}/uv-cache",
            f"{root}/environment-cache",
        )
    ]


def test_controller_creates_a_nested_run_directory(pilot):
    stage, directory, transport = pilot
    nested = directory / "nested/run"
    assert run((stage, nested, transport))["state"] == "completed"
    assert (nested / "ledger.json").is_file()


def test_submission_transport_failure_preserves_the_exception(pilot, monkeypatch):
    controller = direct_controller(pilot)
    controller.ledger["job_id"] = "previous"
    failure = OSError("connection dropped")

    def fail(args):
        raise failure

    monkeypatch.setattr(pilot[2], "run_frontend", fail)
    with raises_exactly(UNCERTAIN_SUBMISSION) as raised:
        controller._submit_job(("oarsub",))
    assert raised.value.__cause__ is failure
    assert controller.ledger["state"] == "submitting"
    assert controller.ledger["job_id"] is None
    assert controller.ledger["submission_exception"] == "OSError"


def test_recorded_job_clears_old_error(pilot):
    controller = direct_controller(pilot)
    controller.ledger["error"] = "old"
    controller._record_submitted_job("55")
    recorded = json.loads(controller.ledger_path.read_text())
    assert recorded["state"] == "running"
    assert recorded["error"] is None
    assert recorded["job_id"] == "55"


def test_frontend_errors_fail_closed_unless_explicitly_inspected(pilot, monkeypatch):
    controller = direct_controller(pilot)
    result = subprocess.CompletedProcess(("mkdir",), 1, "", "denied")
    monkeypatch.setattr(pilot[2], "run_frontend", lambda args: result)
    with raises_exactly("Grid5000 frontend command failed: mkdir"):
        controller._run_frontend(("mkdir",))
    assert controller._run_frontend(("mkdir",), allow_failure=True) is result
    with raises_exactly("Could not resolve the Grid5000 remote home"):
        controller._resolve_remote_home()


def test_publication_controller_error_is_preserved(pilot):
    error = controller_module().ControllerRunError("retry later")

    def fail(*args, **kwargs):
        raise error

    with raises_exactly("retry later") as raised:
        run(pilot, publish=True, publisher=fail)
    assert raised.value is error
    assert json.loads((pilot[1] / "ledger.json").read_text())["state"] == "ready_to_publish"


@pytest.mark.parametrize("digest", ["bad", "a" * 63, "a" * 65, "g" * 64])
def test_preparation_and_controller_reject_malformed_package_hashes(tmp_path, digest):
    from scripts import grid5000_geographic_ner as cli

    lock = tmp_path / "requirements.txt"
    lock.write_text(f"example==1.0 --hash=sha256:{digest}\n")
    with raises_exactly("GPU requirements lock must hash every pinned package", ValueError):
        cli._validate_hashed_lock(lock)
    with raises_exactly("Staged GPU requirements lock must hash every pinned package"):
        controller_module()._validate_runtime_lock(lock)


def test_nonzero_exit_from_a_terminated_job_is_retained_as_failure(pilot):
    pilot[2].statuses = ["state = Terminated\nexit_code = 1"]
    with raises_exactly(JOB_FAILURE_MESSAGES[0]):
        run(pilot)
    ledger = json.loads((pilot[1] / "ledger.json").read_text())
    assert ledger["state"] == "failed"
    assert ledger["error"] == "remote_job_failed"
    assert (pilot[1] / "output/receipt.json").exists()


def test_nonterminal_job_polls_at_the_bounded_interval(pilot):
    controller = direct_controller(pilot)
    pilot[2].statuses = ["state = Running", "state = Terminated\nexit_code = 0"]
    delays = []
    controller.sleep = delays.append
    assert controller._wait_for_job("42") == ("terminated", 0)
    assert delays == [10.0]
    assert pilot[2].commands == [("oarstat", "-f", "-j", "42")] * 2


def test_status_command_failure_retains_recovery_diagnostic(pilot, monkeypatch):
    controller = direct_controller(pilot)
    responses = iter([subprocess.CompletedProcess(("oarstat",), 1, "", "unavailable")])
    monkeypatch.setattr(
        pilot[2],
        "run_frontend",
        lambda args: next(responses),
    )
    with raises_exactly("Could not inspect OAR job 42; run remains resumable"):
        controller._wait_for_job("42")


@pytest.mark.parametrize("corruption", ["unexpected", "hash"])
def test_retrieval_corruption_records_invalid_output(pilot, corruption):
    output = pilot[2].output
    if corruption == "unexpected":
        (output / "extra.txt").write_text("unexpected")
        message = "Unexpected geographic NER output file: extra.txt"
    else:
        (output / "batch-000000.parquet").write_bytes(b"corrupt")
        message = "Geographic NER artifact hash mismatch: batch-000000.parquet"
    with raises_exactly(message):
        run(pilot)
    ledger = json.loads((pilot[1] / "ledger.json").read_text())
    assert ledger["state"] == "failed"
    assert ledger["error"] == "invalid_output"


@pytest.mark.parametrize(
    "method,state", [("_mark_ready_to_publish", "ready_to_publish"), ("_set_state", "paused")]
)
def test_successful_state_transition_clears_prior_error(pilot, method, state):
    controller = direct_controller(pilot)
    controller.ledger["error"] = "previous failure"
    if method == "_set_state":
        controller._set_state(state)
    else:
        controller._mark_ready_to_publish()
    ledger = json.loads(controller.ledger_path.read_text())
    assert ledger["state"] == state
    assert ledger["error"] is None


def test_upload_rejects_changes_before_snapshot(pilot):
    controller = direct_controller(pilot)
    (pilot[0] / "code/worker.py").write_text("changed")
    with raises_exactly("Geographic NER staging changed before upload"):
        controller._upload_stage()
    assert not pilot[2].uploads


def test_upload_rejects_changes_during_snapshot(pilot, monkeypatch):
    controller = direct_controller(pilot)
    module = controller_module()
    original = module._copy_stage_inputs

    def copy_and_change(source, target):
        original(source, target)
        (source / "code/worker.py").write_text("changed")

    monkeypatch.setattr(module, "_copy_stage_inputs", copy_and_change)
    with raises_exactly("Geographic NER staging changed during upload snapshot"):
        controller._upload_stage()
    assert not pilot[2].uploads


def test_transport_snapshots_stay_under_the_run_directory(pilot, monkeypatch, case_sensitive_paths):
    controller = direct_controller(pilot)
    original = pilot[2].download_tree

    def download(remote_root, local_root):
        assert remote_root == f"{controller.remote_root}/output"
        assert local_root.parent == controller.run_dir
        assert local_root.name.startswith(".ner-receive-")
        original(remote_root, local_root / "output")

    monkeypatch.setattr(pilot[2], "download_tree", download)
    assert controller.run()["state"] == "completed"
    staged, remote = pilot[2].uploads[0]
    assert remote == controller.remote_root
    assert staged.parent == controller.run_dir
    assert staged.name.startswith(".ner-stage-")
    assert not staged.exists()


@pytest.mark.parametrize("function", ["_copy_tree", "_validate_downloaded_output_tree"])
def test_missing_download_tree_is_reported(tmp_path, function):
    args = (
        (tmp_path / "missing", tmp_path / "target")
        if function == "_copy_tree"
        else (tmp_path / "missing",)
    )
    with raises_exactly("Downloaded Grid5000 output directory is missing"):
        getattr(controller_module(), function)(*args)


def test_checked_copy_creates_parents_and_rejects_missing_or_symlink_source(tmp_path):
    module = controller_module()
    source = tmp_path / "source"
    target = tmp_path / "nested/output/copied"
    with raises_exactly(f"Required staged file is missing or unsafe: {source}"):
        module._copy_checked(source, target)
    source.write_bytes(b"original")
    module._copy_checked(source, target)
    assert target.read_bytes() == b"original"
    link = tmp_path / "link"
    link.symlink_to(source)
    with raises_exactly(f"Required staged file is missing or unsafe: {link}"):
        module._copy_checked(link, target)


def test_checked_tree_copy_is_idempotent_and_refuses_symlinks(tmp_path):
    module = controller_module()
    source = tmp_path / "source"
    target = tmp_path / "target"
    with raises_exactly(f"Required staged code directory is missing or unsafe: {source}"):
        module._copy_checked_tree(source, target)
    (source / "nested").mkdir(parents=True)
    (source / "nested/code.py").write_text("data")
    module._copy_checked_tree(source, target)
    module._copy_checked_tree(source, target)
    assert (target / "nested/code.py").read_text() == "data"
    link = source / "linked.py"
    link.symlink_to(source / "nested/code.py")
    with raises_exactly(f"Symlinks are not allowed in staged code: {link}"):
        module._copy_checked_tree(source, target)


def test_parquet_footer_failures_keep_the_artifact_name(tmp_path):
    module = controller_module()
    path = tmp_path / "batch-000000.parquet"
    with raises_exactly(f"Could not read artifact footer: {path.name}"):
        module._parquet_markers(path, path.name)
    path.write_bytes(b"PAR1invalidPAR1")
    with raises_exactly(f"Could not validate Parquet artifact: {path.name}"):
        module._read_parquet_rows(path, path.name)


def test_source_metadata_failure_is_explicit(tmp_path):
    with raises_exactly("Could not read staged geographic NER input"):
        controller_module()._source_row_count(tmp_path / "absent.parquet")


@pytest.mark.parametrize("period", ["day", "night"])
def test_cli_preparation_paths_and_alias_have_exact_types(period):
    from scripts import grid5000_geographic_ner as cli

    args = cli._parser().parse_args(
        [
            "--staging-dir",
            "stage",
            "--input",
            "input",
            "--contract",
            "contract",
            "--source-root",
            "root",
            "--requirements-lock",
            "lock",
            "--period",
            period,
        ]
    )
    assert args.source == Path("input")
    assert args.contract == Path("contract")
    assert args.source_root == Path("root")
    assert args.requirements_lock == Path("lock")
    assert args.period == period


@pytest.mark.parametrize("args", [[], ["--staging-dir", "stage", "--period", "invalid"]])
def test_cli_rejects_missing_stage_or_invalid_period(args):
    from scripts import grid5000_geographic_ner as cli

    with pytest.raises(SystemExit) as error:
        cli._parser().parse_args(args)
    assert error.value.code == 2


def test_cli_help_describes_the_pilot():
    from scripts import grid5000_geographic_ner as cli

    assert "Run and resume the serial Grid5000 geographic NER pilot." in cli._parser().format_help()


def test_cli_contract_rejects_malformed_language_shapes_with_cause(tmp_path):
    from scripts import grid5000_geographic_ner as cli

    path = tmp_path / "contract.json"
    for content in ("null", "[]", '{"languages":"en"}'):
        path.write_text(content)
        with raises_exactly(f"Invalid geographic NER contract: {path}", ValueError) as raised:
            cli._read_contract(path)
        assert str(raised.value.__cause__) == "contract languages must be a JSON list"


def test_cli_lock_reader_reports_bad_inputs(tmp_path):
    from scripts import grid5000_geographic_ner as cli

    path = tmp_path / "lock"
    with raises_exactly(f"GPU requirements lock is missing or unsafe: {path}", ValueError):
        cli._read_lock(path)
    path.write_bytes(b"\xff")
    with raises_exactly(f"GPU requirements lock is unreadable: {path}", ValueError):
        cli._read_lock(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with raises_exactly(f"GPU requirements lock is missing or unsafe: {link}", ValueError):
        cli._read_lock(link)


def test_cli_copy_excludes_compiled_caches_and_rejects_symlinks(tmp_path):
    from scripts import grid5000_geographic_ner as cli

    source = tmp_path / "code"
    (source / "__pycache__").mkdir(parents=True)
    (source / "__pycache__/data").write_text("cache")
    (source / "main.pyc").write_text("cache")
    (source / "main.py").write_text("code")
    target = tmp_path / "out"
    cli._copy_tree(source, target)
    assert sorted(path.name for path in target.iterdir()) == ["main.py"]
    assert (target / "main.py").read_text() == "code"
    link = tmp_path / "linked"
    link.symlink_to(source, target_is_directory=True)
    with raises_exactly(f"Required staging directory is missing or unsafe: {link}", ValueError):
        cli._copy_tree(link, tmp_path / "second")


def test_cli_contract_serialization_is_canonical_utf8_in_nested_directory(tmp_path):
    from scripts import grid5000_geographic_ner as cli

    contract = Contract(languages=("en",), implementation_revision="é")
    path = tmp_path / "nested/contracts/contract.json"
    cli._write_contract(path, contract)
    expected = (
        json.dumps(asdict(contract), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    assert path.read_bytes() == expected.encode("utf-8")


def test_cli_stages_from_an_explicit_source_root_under_nested_parent(
    pilot, tmp_path, monkeypatch, case_sensitive_paths
):
    from scripts import grid5000_geographic_ner as cli

    stage, _, _ = pilot
    root = tmp_path / "custom-source"
    package = root / "src/osm_polygon_wikidata_only"
    original = Path(__file__).resolve().parents[2] / "src/osm_polygon_wikidata_only"
    package.mkdir(parents=True)
    for relative in cli._SOURCE_FILES:
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / relative, destination)
    shutil.copytree(original / "ner", package / "ner")
    (package / "ner/custom.py").write_text("# custom root marker\n")
    destination = tmp_path / "nested/outputs/staging"
    original_replace = cli.os.replace

    def replace(source, target):
        assert source.parent == destination.parent
        assert source.name.startswith(".staging-")
        assert str(source / "code/src/osm_polygon_wikidata_only/ner/custom.py") in {
            str(path) for path in source.rglob("*.py")
        }
        original_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", replace)
    result = cli.prepare_geographic_ner_staging(
        destination,
        source=stage / "input.parquet",
        contract=stage / "contract.json",
        source_root=root,
        requirements_lock=stage / "code/requirements/geographic-ner-gpu.txt",
    )
    assert result == destination
    assert (
        destination / "code/src/osm_polygon_wikidata_only/ner/custom.py"
    ).read_text() == "# custom root marker\n"
    with raises_exactly(f"Staging directory already exists: {destination}", ValueError):
        cli.prepare_geographic_ner_staging(
            destination,
            source=stage / "input.parquet",
            contract=stage / "contract.json",
            source_root=root,
            requirements_lock=stage / "code/requirements/geographic-ner-gpu.txt",
        )


def test_stage_records_bind_relative_name_size_and_digest(tmp_path):
    module = controller_module()
    source = tmp_path / "file"
    source.write_bytes(b"data")
    assert module._stage_file_record(tmp_path, source) == {
        "path": "file",
        "size": 4,
        "sha256": hashlib.sha256(b"data").hexdigest(),
    }
    link = tmp_path / "link"
    link.symlink_to(source)
    with raises_exactly(f"Symlinks are not allowed in staging: {link}"):
        module._stage_file_record(tmp_path, link)


@pytest.mark.parametrize("record", [{}, {"path": 1, "sha256": "a"}, {"path": "file", "sha256": 1}])
def test_stage_entry_fields_are_typed(record):
    with raises_exactly("Staging manifest is invalid"):
        controller_module()._stage_entry_values(record)


def test_stage_copy_hash_is_checked_after_snapshot(tmp_path):
    module = controller_module()
    source = tmp_path / "file"
    source.write_bytes(b"data")
    with raises_exactly("Staging snapshot changed: file"):
        module._verify_stage_entry(tmp_path, {"path": "file", "sha256": "a" * 64})


def test_worker_script_is_executable_without_group_or_other_write(pilot, tmp_path):
    target = tmp_path / "copied"
    target.mkdir()
    controller_module()._copy_stage_inputs(pilot[0], target)
    assert (target / "run.sh").stat().st_mode & 0o777 == 0o755


def test_ledger_write_refreshes_timestamp(pilot, monkeypatch):
    module = controller_module()
    controller = direct_controller(pilot)
    monkeypatch.setattr(module, "_timestamp", lambda: "2026-09-07T12:00:00+00:00")
    controller._write_ledger()
    assert (
        json.loads(controller.ledger_path.read_text())["updated_at"] == "2026-09-07T12:00:00+00:00"
    )


@pytest.mark.parametrize("function", ["_validate_downloaded_output_item", "_copy_output_item"])
def test_downloaded_symlinks_are_never_followed(tmp_path, function):
    module = controller_module()
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target"
    target.write_text("data")
    link = source / "link"
    link.symlink_to(target)
    with raises_exactly("Symlinks are not allowed in downloaded output: link"):
        if function == "_validate_downloaded_output_item":
            module._validate_downloaded_output_item(source, link, {"link"})
        else:
            module._copy_output_item(source, tmp_path / "destination", link)


def test_output_copy_creates_nested_directories_and_can_repeat(tmp_path):
    module = controller_module()
    source = tmp_path / "source"
    directory = source / "nested/deeper"
    directory.mkdir(parents=True)
    (directory / "file").write_bytes(b"data")
    target = tmp_path / "target"
    module._copy_output_item(source, target, directory)
    module._copy_output_item(source, target, directory)
    module._copy_output_item(source, target, directory / "file")
    assert (target / "nested/deeper/file").read_bytes() == b"data"


def test_runtime_metadata_missing_is_not_zero_rows(tmp_path, monkeypatch):
    module = controller_module()
    monkeypatch.setattr(pq, "ParquetFile", lambda path: SimpleNamespace(metadata=None))
    with raises_exactly("Staged geographic NER input has no Parquet metadata"):
        module._source_row_count(tmp_path / "input")


def test_contract_id_validation_reports_invalid_contract(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"languages": ["en"], "threshold": 99}))
    with raises_exactly("Invalid staged geographic NER contract"):
        controller_module()._contract_ids(path)


def test_receipt_identity_checks_both_source_and_contract_hashes(pilot):
    module = controller_module()
    receipt = json.loads((pilot[2].output / "receipt.json").read_text())
    with raises_exactly("Geographic NER source hash mismatch"):
        module._validate_receipt_identity({**receipt, "source_sha256": "bad"}, pilot[0])
    with raises_exactly("Geographic NER contract hash mismatch"):
        module._validate_receipt_identity({**receipt, "contract_id": "bad"}, pilot[0])


def test_duplicate_artifact_diagnostic_preserves_name(pilot):
    module = controller_module()
    receipt = json.loads((pilot[2].output / "receipt.json").read_text())
    record = receipt["artifacts"][0]
    with raises_exactly("Duplicate geographic NER artifact: batch-000000.parquet"):
        module._validate_artifact(pilot[2].output, record, {record["path"]})


def test_stage_manifest_has_a_canonical_digest_and_optional_script(pilot):
    stage = pilot[0]
    manifest, digest = controller_module()._stage_manifest(stage)
    assert "run.sh" in {entry["path"] for entry in manifest}
    expected = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert digest == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("missing", ["input.parquet", "contract.json", "code"])
def test_required_staging_inputs_are_checked_before_runtime(pilot, missing):
    path = pilot[0] / missing
    path.rename(pilot[0] / (missing + ".away"))
    with raises_exactly("Staging requires input.parquet, contract.json, and code/"):
        controller_module()._required_staging_files(pilot[0])


def test_stage_directory_and_multiple_runtime_omissions_have_clear_errors(tmp_path, pilot):
    module = controller_module()
    missing = tmp_path / "missing"
    with raises_exactly(f"Staging directory is missing: {missing}"):
        module._stage_manifest(missing)
    for name in ("ner/job.py", "ner/otter.py"):
        (pilot[0] / "code/src/osm_polygon_wikidata_only" / name).unlink()
    with raises_exactly(
        "Staging code is missing required geographic NER runtime files: src/osm_polygon_wikidata_only/ner/job.py, src/osm_polygon_wikidata_only/ner/otter.py"
    ):
        module._validate_runtime_tree(pilot[0])


def test_invalid_receipt_status_and_source_accounting_are_rejected(tmp_path):
    module = controller_module()
    with raises_exactly("Geographic NER receipt is incomplete"):
        module._validate_receipt_header({"status": "failed", "batch_size": 128})
    for value in (None, "0", -1):
        suffix = "required" if value is None else "invalid"
        with raises_exactly(f"Geographic NER source row count is {suffix}"):
            module._validate_receipt_counts({"processed_rows": 0, "source_rows": value}, [])
    with raises_exactly("Geographic NER artifact rows do not match the receipt"):
        module._validate_receipt_counts(
            {"processed_rows": 2, "source_rows": 3, "status": "paused"}, [1]
        )
    with raises_exactly("Geographic NER processed rows exceed source rows"):
        module._validate_receipt_counts(
            {"processed_rows": 2, "source_rows": 1, "status": "paused"}, [2]
        )


def test_artifact_validation_records_seen_names_and_checks_optional_size(pilot):
    module = controller_module()
    record = json.loads((pilot[2].output / "receipt.json").read_text())["artifacts"][0]
    seen = set()
    assert module._validate_artifact(pilot[2].output, record, seen) == ("batch-000000.parquet", 1)
    assert seen == {"batch-000000.parquet"}
    with raises_exactly("Geographic NER artifact size mismatch: batch-000000.parquet"):
        module._validate_artifact(pilot[2].output, {**record, "size": 0}, set())


def test_footer_validation_reports_unreadable_and_corrupt_parquet(tmp_path):
    module = controller_module()
    path = tmp_path / "batch-000000.parquet"
    with raises_exactly(f"Could not read artifact footer: {path.name}"):
        module._validate_parquet_footer(path, path.name, 1)
    path.write_bytes(b"PAR1brokenPAR1")
    with raises_exactly(f"Could not validate Parquet artifact: {path.name}"):
        module._validate_parquet_footer(path, path.name, 1)


def test_unavailable_parquet_library_has_an_operator_error(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with raises_exactly("Could not validate Parquet artifact: batch-000000.parquet"):
        controller_module()._read_parquet_rows(
            tmp_path / "batch-000000.parquet", "batch-000000.parquet"
        )


def test_explicit_oar_state_takes_precedence_over_other_status_words():
    result = subprocess.CompletedProcess([], 0, "state = Suspended\nprevious state was running", "")
    assert controller_module()._parse_job_status(result) == ("suspended", None)


def test_finishing_job_with_zero_exit_is_successful():
    assert controller_module()._job_succeeded("finishing", 0)
    assert not controller_module()._job_succeeded("finishing", 1)


def test_result_text_preserves_stderr_and_handles_missing_stdout():
    assert (
        controller_module()._result_text(SimpleNamespace(stdout=None, stderr="failure"))
        == "\nfailure"
    )


def test_permission_denied_policy_output_is_rejected():
    assert controller_module()._policy_failure("Permission denied")


def test_timestamps_are_utc_aware():
    from datetime import datetime, timedelta

    value = datetime.fromisoformat(controller_module()._timestamp())
    assert value.utcoffset() == timedelta(0)


def test_nondefault_night_policy_and_invalid_repo_are_validated(pilot):
    module = controller_module()
    module._validate_config("pilot", "rennes", "besteffort", "A40", "night", "a/b")
    with raises_exactly("Unsafe Grid5000 repo_id: 'bad'", ValueError):
        module._validate_config("pilot", "rennes", "besteffort", "A40", "day", "bad")


def test_wrapper_preserves_sleep_and_custom_site(pilot, monkeypatch):
    module = controller_module()
    stage, directory, transport = pilot
    calls = []

    def make_transport(site):
        assert site == "nancy"
        return transport

    monkeypatch.setattr(module, "SubprocessGrid5000Transport", make_transport)
    transport.statuses = ["state = Running", "state = Terminated\nexit_code = 0"]
    result = module.run_grid5000_ner_controller(
        stage, directory, "pilot-1", "owner/dataset", site="nancy", sleep=calls.append
    )
    assert result["state"] == "completed"
    assert calls == [10.0]


def test_cli_uppercase_package_and_hex_hash_are_supported(tmp_path):
    from scripts import grid5000_geographic_ner as cli

    text = f"PyYAML==6.0 --hash=sha256:{'A' * 64}\nZlib==1.0 --hash=sha256:{'B' * 64}\n"
    lock = tmp_path / "lock"
    lock.write_text(text)
    assert len(cli._split_lock_blocks(text)) == 2
    assert len(controller_module()._runtime_lock_blocks(lock)) == 2
    cli._validate_hashed_lock(lock)


def test_cli_input_metadata_errors_and_missing_metadata_are_not_valid_rows(tmp_path, monkeypatch):
    from scripts import grid5000_geographic_ner as cli

    path = tmp_path / "input.parquet"
    path.write_bytes(b"invalid")
    with raises_exactly(f"Invalid geographic NER input: {path}", ValueError):
        cli._input_metadata(path)
    monkeypatch.setattr(
        pq,
        "ParquetFile",
        lambda path: SimpleNamespace(
            schema_arrow=SimpleNamespace(names=list(INPUT_COLUMNS)), metadata=None
        ),
    )
    assert cli._input_metadata(path) == (set(INPUT_COLUMNS), 0)


def test_cli_validates_missing_runtime_and_file_symlinks(tmp_path):
    from scripts import grid5000_geographic_ner as cli

    with raises_exactly("Source root is missing the geographic NER runtime", ValueError):
        cli._validate_source_tree(tmp_path)
    source = tmp_path / "missing"
    with raises_exactly(f"Required staging file is missing or unsafe: {source}", ValueError):
        cli._copy_file(source, tmp_path / "target")
    source.write_text("data")
    link = tmp_path / "link"
    link.symlink_to(source)
    with raises_exactly(f"Required staging file is missing or unsafe: {link}", ValueError):
        cli._copy_file(link, tmp_path / "target")


@pytest.mark.parametrize("option", ["--source", "--contract"])
def test_cli_partial_preparation_never_falls_through_to_submission(option, monkeypatch):
    from scripts import grid5000_geographic_ner as cli

    monkeypatch.setattr(
        cli, "run_grid5000_ner_controller", lambda *args, **kwargs: pytest.fail("must not submit")
    )
    with raises_exactly(
        "--source/--input and --contract are required for staging preparation", SystemExit
    ):
        cli.main(["--staging-dir", "stage", option, "only-one-input"])


def test_cli_preparation_passes_the_operator_source_and_lock_paths(monkeypatch):
    from scripts import grid5000_geographic_ner as cli

    args = cli._parser().parse_args(
        [
            "--staging-dir",
            "stage",
            "--source",
            "input",
            "--contract",
            "contract",
            "--source-root",
            "operator-source",
            "--requirements-lock",
            "operator-lock",
            "--prepare-only",
        ]
    )
    captured = {}

    def prepare(stage, **kwargs):
        captured.update(stage=stage, **kwargs)

    monkeypatch.setattr(cli, "prepare_geographic_ner_staging", prepare)
    assert cli._prepare_from_args(args)
    assert captured == dict(
        stage=Path("stage"),
        source=Path("input"),
        contract=Path("contract"),
        source_root=Path("operator-source"),
        requirements_lock=Path("operator-lock"),
    )
