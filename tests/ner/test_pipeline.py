"""Geographic NER preserves source identities and resumes verified batches."""

import importlib
import json
import re
import sqlite3
from contextlib import nullcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

LANGUAGE_SELECTION = "Select unique, explicit language codes"
THRESHOLD_RANGE = "Threshold must be finite and between zero and one"
PINNED_MODEL = "Only the pinned geographic NER model and label are supported"
IDENTITY_REQUIRED = "Sentence and document identities cannot be empty"
ENTITY_FIELD_REQUIRED = "Entity output is missing a required field"
ENTITY_SURFACE = "Entity label or surface does not match the original sentence"
ENTITY_SCORE = "Invalid entity score"
COUNT_MISMATCH = "Model output count does not match input count"
ARTIFACT_INVALID = "Checkpoint artifact is invalid"
ARTIFACT_ORDER_INVALID = "Checkpoint artifact order is invalid"
ARTIFACT_ROWS_INVALID = "Checkpoint artifact row count is invalid"
ARTIFACTS_LIST = "Checkpoint artifacts must be a list"
STATUS_INVALID = "Checkpoint status is invalid"
SOURCE_CHANGED = "Source changed during inference"


def raises_exactly(message, kind=ValueError):
    """Pin an operator-facing diagnostic so message-only mutations cannot survive."""
    return pytest.raises(kind, match=rf"^{re.escape(message)}$")


@pytest.fixture
def api():
    return importlib.import_module("osm_polygon_wikidata_only.ner.pipeline")


def source(tmp_path, texts=("Paris", "Paris", "北京", "nothing")):
    path = tmp_path / "input.parquet"
    rows = [
        dict(
            sentence_id=str(i),
            document_id=f"doc-{i}",
            project="wikipedia",
            language="en",
            text=text,
            segmentation_status="split",
        )
        for i, text in enumerate(texts)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class Extractor:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def predict(self, texts):
        self.calls.append(list(texts))
        if self.fail_on == len(self.calls):
            raise RuntimeError("interrupted inference")
        return [
            [dict(text=t, label="named geographic location", start=0, end=len(t), score=0.9)]
            if t != "nothing"
            else []
            for t in texts
        ]


def outputs(directory):
    return [
        row
        for p in sorted(directory.glob("batch-*.parquet"))
        for row in pq.read_table(p).to_pylist()
    ]


def test_deduplicates_inference_but_preserves_every_source_identity(api, tmp_path):
    p = source(tmp_path)
    extractor = Extractor()
    contract = api.Contract(languages=("en",))
    receipt = api.run_shard(p, tmp_path / "out", extractor, contract, batch_size=2)
    rows = outputs(tmp_path / "out")
    assert extractor.calls == [["Paris"], ["北京", "nothing"]]
    assert [r["sentence_id"] for r in rows] == ["0", "1", "2", "3"]
    assert rows[2]["entities"][0]["end"] == 2
    assert rows[3]["entities"] == []
    assert rows[3]["status"] == "ok"
    assert rows[0]["validation_status"] == "pilot_unvalidated"
    assert rows[0]["contract_id"] == contract.identity
    assert receipt["processed_rows"] == receipt["source_rows"] == 4
    assert receipt["status"] == "completed"
    assert json.loads((tmp_path / "out" / "receipt.json").read_text())["status"] == "completed"


def test_default_batch_size_is_persisted(api, tmp_path):
    receipt = api.run_shard(
        source(tmp_path, ("Paris",)),
        tmp_path / "out",
        Extractor(),
        api.Contract(languages=("en",)),
    )
    assert receipt["batch_size"] == 256


def test_receipt_persists_the_contract_payload(api, tmp_path):
    contract = api.Contract(languages=("en",))
    receipt = api._receipt(source(tmp_path, ("Paris",)), contract, 1)

    assert receipt["contract"]["model_id"] == api.MODEL_ID
    assert receipt["contract"]["languages"] == ["en"]


def test_single_row_batches_are_valid(api, tmp_path):
    receipt = api.run_shard(
        source(tmp_path, ("Paris",)),
        tmp_path / "out",
        Extractor(),
        api.Contract(languages=("en",)),
        batch_size=1,
    )
    assert receipt["status"] == "completed"


def test_interrupted_run_resumes_without_repeating_completed_batches(api, tmp_path):
    p = source(tmp_path, ("Paris", "Rome", "Paris", "Tokyo"))
    out = tmp_path / "out"
    contract = api.Contract(languages=("en",))
    with raises_exactly("interrupted inference", RuntimeError):
        api.run_shard(p, out, Extractor(fail_on=2), contract, batch_size=2)
    first = (out / "batch-000000.parquet").read_bytes()
    extractor = Extractor()
    receipt = api.run_shard(p, out, extractor, contract, batch_size=2)
    assert extractor.calls == [["Tokyo"]]
    assert (out / "batch-000000.parquet").read_bytes() == first
    assert len(outputs(out)) == 4
    assert receipt["status"] == "completed"


def test_routing_preserves_skipped_and_empty_rows(api, tmp_path):
    p = source(tmp_path, ("Paris", "Paris", "Paris", " \n"))
    rows = pq.read_table(p).to_pylist()
    rows[0]["segmentation_status"] = "unsupported_language"
    rows[1]["language"] = "xx"
    pq.write_table(pa.Table.from_pylist(rows), p)
    extractor = Extractor()
    api.run_shard(p, tmp_path / "out", extractor, api.Contract(languages=("en",)))
    result = outputs(tmp_path / "out")
    assert [r["status"] for r in result] == [
        "skipped_unsplit",
        "skipped_language",
        "ok",
        "empty_text",
    ]
    assert extractor.calls == [["Paris"]]


def test_empty_entity_lists_still_write_the_declared_output_schema(api, tmp_path):
    p = source(tmp_path, ("Paris", "Rome"))
    rows = pq.read_table(p).to_pylist()
    for row in rows:
        row["segmentation_status"] = "unsupported_language"
    pq.write_table(pa.Table.from_pylist(rows), p)

    api.run_shard(p, tmp_path / "out", Extractor(), api.Contract(languages=("en",)))

    assert pq.read_schema(tmp_path / "out" / "batch-000000.parquet") == api.OUTPUT_SCHEMA


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("source", "Checkpoint identity changed: source_sha256"),
        ("contract", "Checkpoint identity changed: contract_id"),
        ("batch_size", "Checkpoint identity changed: batch_size"),
        ("artifact", "Checkpoint artifact hash mismatch"),
    ],
)
def test_resume_rejects_changed_inputs_or_corrupt_results(api, tmp_path, change, message):
    p = source(tmp_path)
    out = tmp_path / "out"
    contract = api.Contract(languages=("en",))
    api.run_shard(p, out, Extractor(), contract, batch_size=2)
    size = 2
    if change == "source":
        p = source(tmp_path, ("London",))
    elif change == "contract":
        contract = api.Contract(languages=("fr",))
    elif change == "batch_size":
        size = 3
    else:
        (out / "batch-000000.parquet").write_bytes(b"corrupt")
    with raises_exactly(message):
        api.run_shard(p, out, Extractor(), contract, batch_size=size)


@pytest.mark.parametrize(
    ("entities", "message"),
    [
        ([[dict(text="Paris", label="person", start=0, end=5, score=0.9)]], ENTITY_SURFACE),
        (
            [[dict(text="Rome", label="named geographic location", start=0, end=5, score=0.9)]],
            ENTITY_SURFACE,
        ),
        (
            [
                [
                    dict(
                        text="Paris",
                        label="named geographic location",
                        start=0,
                        end=5,
                        score=float("nan"),
                    )
                ]
            ],
            ENTITY_SCORE,
        ),
        ([], COUNT_MISMATCH),
    ],
)
def test_invalid_model_output_never_becomes_a_checkpoint(api, tmp_path, entities, message):
    class BadExtractor:
        def predict(self, texts):
            return entities

    with raises_exactly(message):
        api.run_shard(
            source(tmp_path, ("Paris",)),
            tmp_path / "out",
            BadExtractor(),
            api.Contract(languages=("en",)),
        )
    assert not list((tmp_path / "out").glob("batch-*.parquet"))


def test_deadline_leaves_a_resumable_receipt(api, tmp_path):
    p = source(tmp_path)
    out = tmp_path / "out"
    extractor = Extractor()
    receipt = api.run_shard(
        p, out, extractor, api.Contract(languages=("en",)), deadline=1.0, clock=lambda: 2.0
    )
    assert receipt["status"] == "paused"
    assert receipt["processed_rows"] == 0
    assert extractor.calls == []
    assert json.loads((out / "receipt.json").read_text()) == receipt


def test_deadline_boundary_does_not_start_a_batch(api, tmp_path):
    p = source(tmp_path)
    extractor = Extractor()
    receipt = api.run_shard(
        p,
        tmp_path / "out",
        extractor,
        api.Contract(languages=("en",)),
        deadline=1.0,
        clock=lambda: 1.0,
    )
    assert receipt["status"] == "paused"
    assert receipt["processed_rows"] == 0
    assert extractor.calls == []


def test_completed_batch_is_checkpointed_before_deadline_pause(api, tmp_path):
    p = source(tmp_path)
    out = tmp_path / "out"
    clock_values = iter((0.0, 2.0))
    extractor = Extractor()
    receipt = api.run_shard(
        p,
        out,
        extractor,
        api.Contract(languages=("en",)),
        batch_size=2,
        deadline=1.0,
        clock=lambda: next(clock_values),
    )
    assert receipt["status"] == "paused"
    assert receipt["processed_rows"] == 2
    assert extractor.calls == [["Paris"]]
    assert (out / "batch-000000.parquet").is_file()


def test_source_change_during_inference_is_rejected(api, tmp_path):
    p = source(tmp_path, ("Paris", "Rome", "Tokyo", "London"))
    out = tmp_path / "out"

    class MutatingExtractor:
        def __init__(self):
            self.changed = False

        def predict(self, texts):
            result = [
                [
                    dict(
                        text=text,
                        label="named geographic location",
                        start=0,
                        end=len(text),
                        score=0.9,
                    )
                ]
                for text in texts
            ]
            if not self.changed:
                source(tmp_path, ("Berlin", "Madrid", "Tokyo", "London"))
                self.changed = True
            return result

    with raises_exactly(SOURCE_CHANGED):
        api.run_shard(p, out, MutatingExtractor(), api.Contract(languages=("en",)), batch_size=2)


def test_source_change_during_inference_does_not_checkpoint_the_batch(api, tmp_path):
    p = source(tmp_path, ("Paris", "Rome", "Tokyo", "London"))
    out = tmp_path / "out"

    class MutatingExtractor:
        def predict(self, texts):
            rows = pq.read_table(p).to_pylist()
            rows[0]["text"] = "Berlin"
            pq.write_table(pa.Table.from_pylist(rows), p)
            return [
                [
                    dict(
                        text=text,
                        label="named geographic location",
                        start=0,
                        end=len(text),
                        score=0.9,
                    )
                ]
                for text in texts
            ]

    with raises_exactly(SOURCE_CHANGED):
        api.run_shard(p, out, MutatingExtractor(), api.Contract(languages=("en",)), batch_size=2)

    assert not list(out.glob("batch-*.parquet"))
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["artifacts"] == []
    assert receipt["processed_rows"] == 0


def test_source_guard_hashes_the_source_only_at_run_boundaries(api, tmp_path, monkeypatch):
    p = source(tmp_path)
    observed = []
    original_hash = api.sha256_file

    def counted_hash(path):
        observed.append(path)
        return original_hash(path)

    monkeypatch.setattr(api, "sha256_file", counted_hash)
    api.run_shard(p, tmp_path / "out", Extractor(), api.Contract(languages=("en",)), batch_size=1)

    assert sum(path == p for path in observed) == 2


def test_pipeline_uses_exact_checkpoint_and_runtime_paths(api, tmp_path, monkeypatch):
    lock_names = []
    receipt_names = []
    database_names = []
    real_write_json = api.atomic_write_json
    real_connect = api.sqlite3.connect

    def record_lock(path):
        lock_names.append(path.name)
        return nullcontext()

    def record_write_json(path, value):
        receipt_names.append(path.name)
        return real_write_json(path, value)

    def record_connect(path, *args, **kwargs):
        database_names.append(Path(path).name)
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(api, "exclusive_run_lock", record_lock)
    monkeypatch.setattr(api, "atomic_write_json", record_write_json)
    monkeypatch.setattr(api.sqlite3, "connect", record_connect)

    api.run_shard(
        source(tmp_path, ("Paris",)),
        tmp_path / "out",
        Extractor(),
        api.Contract(languages=("en",)),
        batch_size=1,
    )

    assert lock_names == ["run.lock"]
    assert receipt_names and set(receipt_names) == {"receipt.json"}
    assert database_names == ["predictions.sqlite3"]


def test_resume_reads_only_the_exact_receipt_name(api, monkeypatch):
    expected = {
        "contract_id": "contract",
        "source_sha256": "source",
        "source_rows": 0,
        "batch_size": 1,
        "contract": {"languages": ["en"]},
    }
    previous = {**expected, "status": "paused", "processed_rows": 0, "artifacts": []}
    names = []

    class ReceiptPath:
        def exists(self):
            return True

        def read_text(self):
            return json.dumps(previous)

    class Directory:
        def __truediv__(self, name):
            names.append(name)
            return ReceiptPath()

    monkeypatch.setattr(api, "_verify_artifacts", lambda directory, receipt: None)
    assert api._resume(Directory(), expected) == previous
    assert names == ["receipt.json"]


def test_execute_requests_the_declared_input_columns(api, monkeypatch):
    observed = []

    class ParquetFile:
        def iter_batches(self, **kwargs):
            observed.append(kwargs)
            return iter(())

    monkeypatch.setattr(api.pq, "ParquetFile", lambda source: ParquetFile())
    monkeypatch.setattr(api, "_source_fingerprint", lambda source: ())
    monkeypatch.setattr(api, "_assert_source_unchanged", lambda source, digest: None)
    monkeypatch.setattr(api, "atomic_write_json", lambda path, receipt: None)
    receipt = {
        "artifacts": [],
        "batch_size": 2,
        "processed_rows": 0,
        "source_rows": 0,
        "source_sha256": "source",
        "status": "paused",
    }
    db = sqlite3.connect(":memory:")
    try:
        api._execute(
            Path("input.parquet"),
            Path("out"),
            Extractor(),
            object(),
            receipt,
            db,
            deadline=1.0,
            clock=lambda: 0.0,
        )
    finally:
        db.close()

    assert observed == [{"batch_size": 2, "columns": api.INPUT_COLUMNS}]


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_contract_accepts_threshold_endpoints(api, threshold):
    assert api.Contract(languages=("en",), threshold=threshold).threshold == threshold


def test_validated_contract_status_is_accepted(api):
    assert api.Contract(languages=("en",), validation_status="validated").validation_status == (
        "validated"
    )


def test_wikivoyage_rows_are_valid_source_rows(api):
    row = {
        "sentence_id": "sentence",
        "document_id": "document",
        "project": "wikivoyage",
        "language": "en",
        "text": "Paris",
        "segmentation_status": "split",
    }
    api._validate_row(row)


def test_json_digest_serialization_is_canonical(api):
    assert api._json({"z": "é", "a": 1}) == '{"a":1,"z":"é"}'
    with raises_exactly("Out of range float values are not JSON compliant: nan"):
        api._json(float("nan"))


def test_empty_source_completes_with_explicit_zero_counts(api, tmp_path):
    p = source(tmp_path)
    pq.write_table(pq.read_table(p).slice(0, 0), p)
    receipt = api.run_shard(p, tmp_path / "out", Extractor(), api.Contract(languages=("en",)))
    assert receipt["status"] == "completed"
    assert receipt["source_rows"] == receipt["processed_rows"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(languages=()), LANGUAGE_SELECTION),
        (dict(languages=("en", "en")), LANGUAGE_SELECTION),
        (dict(languages=("en",), threshold=float("nan")), THRESHOLD_RANGE),
        (dict(languages=("en",), threshold=float("inf")), THRESHOLD_RANGE),
        (dict(languages=("en",), threshold=-float("inf")), THRESHOLD_RANGE),
        (dict(languages=("en",), threshold=True), THRESHOLD_RANGE),
        (dict(languages=("en",), threshold="0.5"), THRESHOLD_RANGE),
        (dict(languages=("en",), threshold=1.1), THRESHOLD_RANGE),
    ],
)
def test_invalid_contract_is_rejected(api, kwargs, message):
    with raises_exactly(message):
        api.Contract(**kwargs)


@pytest.mark.parametrize(
    ("languages", "message"),
    [
        (("EN",), "Invalid language code"),
        (("a" * 17,), "Invalid language code"),
        (["en"], LANGUAGE_SELECTION),
    ],
)
def test_invalid_language_selection_is_rejected(api, languages, message):
    with raises_exactly(message):
        api.Contract(languages=languages)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "other/model", PINNED_MODEL),
        ("model_revision", "other-revision", PINNED_MODEL),
        ("label", "person", PINNED_MODEL),
        ("validation_status", "unknown", "Unknown language validation status"),
    ],
)
def test_contract_rejects_unpinned_model_or_status(api, field, value, message):
    kwargs = {"languages": ("en",), field: value}
    with raises_exactly(message):
        api.Contract(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("text", None, "Sentence input fields must be strings"),
        ("sentence_id", "", IDENTITY_REQUIRED),
        ("document_id", "", IDENTITY_REQUIRED),
        ("project", "other", "Unknown source project"),
    ],
)
def test_invalid_sentence_row_is_rejected(api, field, value, message):
    row = {
        "sentence_id": "sentence",
        "document_id": "document",
        "project": "wikipedia",
        "language": "en",
        "text": "Paris",
        "segmentation_status": "split",
    }
    row[field] = value
    with raises_exactly(message):
        api._validate_row(row)


@pytest.mark.parametrize(
    ("entity", "message"),
    [
        (None, ENTITY_FIELD_REQUIRED),
        (
            {"text": "Paris", "label": "named geographic location", "start": 0, "end": 5},
            ENTITY_FIELD_REQUIRED,
        ),
        (
            {
                "text": "Paris",
                "label": "named geographic location",
                "start": 0.0,
                "end": 5,
                "score": 0.9,
            },
            "Entity offsets must be integers",
        ),
        (
            {
                "text": "Paris",
                "label": "named geographic location",
                "start": 0,
                "end": 5,
                "score": 1.1,
            },
            ENTITY_SCORE,
        ),
        (
            {
                "text": 4,
                "label": "named geographic location",
                "start": 0,
                "end": 5,
                "score": 0.9,
            },
            ENTITY_SURFACE,
        ),
    ],
)
def test_pipeline_entity_shape_and_value_guards(api, entity, message):
    with raises_exactly(message):
        api._validate_entity("Paris", entity)


@pytest.mark.parametrize("score", [0, 1])
def test_pipeline_entity_score_endpoints_are_preserved(api, score):
    entity = {
        "text": "Paris",
        "label": "named geographic location",
        "start": 0,
        "end": 5,
        "score": score,
    }
    assert api._validate_entity("Paris", entity)["score"] == score


def test_valid_pipeline_entity_preserves_its_offsets_and_surface(api):
    entity = {
        "text": "Paris",
        "label": "named geographic location",
        "start": 3,
        "end": 8,
        "score": 0.9,
    }

    assert api._validate_entity("in Paris now", entity) == entity


def test_pipeline_rejects_zero_length_entity_spans(api):
    entity = {
        "text": "",
        "label": "named geographic location",
        "start": 2,
        "end": 2,
        "score": 0.9,
    }
    with raises_exactly("Entity offsets are outside the original sentence"):
        api._validate_entity("Paris", entity)


def test_corrupt_inference_cache_entry_is_rejected(api):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE predictions (identity TEXT PRIMARY KEY, entities TEXT NOT NULL)")
    db.execute("INSERT INTO predictions VALUES (?, ?)", ("key", "{}"))
    with raises_exactly("Inference cache entry is not an entity list"):
        api._cached(db, "key")
    db.close()


def test_output_row_requires_a_cached_prediction(api):
    input_row = {
        "sentence_id": "0",
        "document_id": "doc-0",
        "project": "wikipedia",
        "language": "en",
        "text": "Paris",
        "segmentation_status": "split",
    }
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE predictions (identity TEXT PRIMARY KEY, entities TEXT NOT NULL)")
    with raises_exactly("Missing inference cache entry"):
        api._output_row(input_row, db, api.Contract(languages=("en",)))
    db.close()


def test_non_list_model_entities_are_rejected(api, tmp_path):
    class BadExtractor:
        def predict(self, texts):
            return [None]

    with raises_exactly("Model output entities must be lists"):
        api.run_shard(
            source(tmp_path, ("Paris",)),
            tmp_path / "out",
            BadExtractor(),
            api.Contract(languages=("en",)),
        )


def test_none_model_response_is_rejected_as_a_count_error(api, tmp_path):
    class BadExtractor:
        def predict(self, texts):
            return None

    with raises_exactly(COUNT_MISMATCH):
        api.run_shard(
            source(tmp_path, ("Paris",)),
            tmp_path / "out",
            BadExtractor(),
            api.Contract(languages=("en",)),
        )


@pytest.mark.parametrize("score", [True, "0.9", None, float("inf")])
def test_invalid_pipeline_score_is_rejected(api, tmp_path, score):
    entity = dict(text="Paris", label="named geographic location", start=0, end=5, score=score)

    class BadExtractor:
        def predict(self, texts):
            return [[entity]]

    with raises_exactly(ENTITY_SCORE):
        api.run_shard(
            source(tmp_path, ("Paris",)),
            tmp_path / "out",
            BadExtractor(),
            api.Contract(languages=("en",)),
        )


@pytest.mark.parametrize("field", ["text", "label", "start", "end", "score"])
def test_missing_pipeline_entity_field_is_rejected(api, tmp_path, field):
    entity = dict(text="Paris", label="named geographic location", start=0, end=5, score=0.9)
    del entity[field]

    class BadExtractor:
        def predict(self, texts):
            return [[entity]]

    with raises_exactly(ENTITY_FIELD_REQUIRED):
        api.run_shard(
            source(tmp_path, ("Paris",)),
            tmp_path / "out",
            BadExtractor(),
            api.Contract(languages=("en",)),
        )


def test_resume_rejects_hash_valid_artifact_with_wrong_schema(api, tmp_path):
    out = tmp_path / "out"
    api.run_shard(source(tmp_path), out, Extractor(), api.Contract(languages=("en",)))
    artifact = out / "batch-000000.parquet"
    pq.write_table(pa.table({"unexpected": ["x", "y", "z", "w"]}), artifact)
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["artifacts"][0]["sha256"] = api.sha256_file(artifact)
    (out / "receipt.json").write_text(json.dumps(receipt))

    with raises_exactly("Checkpoint artifact schema mismatch"):
        api.run_shard(source(tmp_path), out, Extractor(), api.Contract(languages=("en",)))


def test_resume_rejects_non_object_receipt(api, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "receipt.json").write_text("[]")
    with raises_exactly("Checkpoint must be an object"):
        api.run_shard(source(tmp_path), out, Extractor(), api.Contract(languages=("en",)))


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        (
            {
                "status": "running",
                "processed_rows": 0,
                "source_rows": 0,
                "batch_size": 1,
                "artifacts": [],
            },
            STATUS_INVALID,
        ),
        (
            {
                "status": "paused",
                "processed_rows": "0",
                "source_rows": 0,
                "batch_size": 1,
                "artifacts": [],
            },
            "Checkpoint processed rows must be an integer",
        ),
        (
            {
                "status": "paused",
                "processed_rows": 2,
                "source_rows": 1,
                "batch_size": 1,
                "artifacts": [],
            },
            "Checkpoint row counts are invalid",
        ),
        (
            {
                "status": "completed",
                "processed_rows": 0,
                "source_rows": 1,
                "batch_size": 1,
                "artifacts": [],
            },
            "Completed checkpoint row accounting is incomplete",
        ),
        (
            {
                "status": "paused",
                "processed_rows": 0,
                "source_rows": 0,
                "batch_size": 1,
                "artifacts": None,
            },
            ARTIFACTS_LIST,
        ),
    ],
)
def test_checkpoint_state_rejects_invalid_row_accounting(api, receipt, message):
    with raises_exactly(message):
        api._checkpoint_state(receipt)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("processed_rows", "0", "Checkpoint processed rows must be an integer"),
        ("processed_rows", -1, "Checkpoint processed rows cannot be negative"),
        ("source_rows", "0", "Checkpoint source rows must be an integer"),
        ("source_rows", -1, "Checkpoint source rows cannot be negative"),
        ("batch_size", "1", "Checkpoint batch size must be an integer"),
        ("batch_size", -1, "Checkpoint batch size cannot be negative"),
    ],
)
def test_checkpoint_count_diagnostics_name_the_offending_field(api, field, value, message):
    receipt = {
        "status": "paused",
        "processed_rows": 0,
        "source_rows": 0,
        "batch_size": 1,
        "artifacts": [],
    }
    receipt[field] = value
    with raises_exactly(message):
        api._checkpoint_state(receipt)


def test_checkpoint_state_accepts_a_valid_single_row_batch(api):
    assert api._checkpoint_state(
        {
            "status": "paused",
            "processed_rows": 0,
            "source_rows": 1,
            "batch_size": 1,
            "artifacts": [],
        }
    ) == ([], 0, 1, 1)


def test_checkpoint_state_accepts_a_completed_single_row_batch(api):
    assert api._checkpoint_state(
        {
            "status": "completed",
            "processed_rows": 1,
            "source_rows": 1,
            "batch_size": 1,
            "artifacts": [{"path": "batch-000000.parquet", "rows": 1}],
        }
    )[1:] == (1, 1, 1)


def test_resume_verifies_row_accounting_across_all_artifacts(api, tmp_path):
    out = tmp_path / "out"
    receipt = api.run_shard(
        source(tmp_path), out, Extractor(), api.Contract(languages=("en",)), batch_size=2
    )
    api._verify_artifacts(out, receipt)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (None, ARTIFACT_INVALID),
        ({}, ARTIFACT_ORDER_INVALID),
        ({"path": "batch-000001.parquet"}, ARTIFACT_ORDER_INVALID),
    ],
)
def test_checkpoint_artifact_name_is_strict(api, artifact, message):
    with raises_exactly(message):
        api._artifact_name(artifact, 0)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (None, ARTIFACT_INVALID),
        ({}, ARTIFACT_ROWS_INVALID),
        ({"rows": "1"}, ARTIFACT_ROWS_INVALID),
        ({"rows": -1}, ARTIFACT_ROWS_INVALID),
        ({"rows": 0}, ARTIFACT_ROWS_INVALID),
    ],
)
def test_checkpoint_artifact_rows_are_strict(api, artifact, message):
    with raises_exactly(message):
        api._artifact_rows(artifact)


def test_checkpoint_artifact_accepts_a_single_row(api):
    assert api._artifact_rows({"rows": 1}) == 1


def test_checkpoint_rejects_artifacts_past_source_end(api):
    with raises_exactly("Checkpoint contains too many artifacts"):
        api._expected_batch_rows(1, 1, 1)


def test_checkpoint_batch_size_uses_multiplication_for_the_batch_index(api):
    assert api._expected_batch_rows(5, 4, 1) == 1


def test_run_shard_requires_positive_batch_size(api, tmp_path):
    with raises_exactly("Batch size must be positive"):
        api.run_shard(
            source(tmp_path, ("Paris",)),
            tmp_path / "out",
            Extractor(),
            api.Contract(languages=("en",)),
            batch_size=0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "running", STATUS_INVALID),
        ("processed_rows", -1, "Checkpoint processed rows cannot be negative"),
        ("artifacts", None, ARTIFACTS_LIST),
    ],
)
def test_resume_rejects_invalid_receipt_state(api, tmp_path, field, value, message):
    out = tmp_path / "out"
    api.run_shard(source(tmp_path), out, Extractor(), api.Contract(languages=("en",)))
    receipt = json.loads((out / "receipt.json").read_text())
    receipt[field] = value
    (out / "receipt.json").write_text(json.dumps(receipt))

    with raises_exactly(message):
        api.run_shard(source(tmp_path), out, Extractor(), api.Contract(languages=("en",)))


def test_source_hash_guard_at_run_boundaries_rejects_a_changed_source(api, tmp_path):
    with raises_exactly(SOURCE_CHANGED):
        api._assert_source_unchanged(source(tmp_path), "0" * 64)


def test_checkpoint_rejects_a_zero_batch_size(api):
    with raises_exactly("Checkpoint row counts are invalid"):
        api._checkpoint_state(
            {
                "status": "paused",
                "processed_rows": 0,
                "source_rows": 0,
                "batch_size": 0,
                "artifacts": [],
            }
        )


@pytest.mark.parametrize("artifact", [None, [], "batch-000000.parquet"])
def test_artifact_mapping_requires_a_mapping(api, artifact):
    with raises_exactly(ARTIFACT_INVALID):
        api._artifact_mapping(artifact)


def test_verify_artifact_file_reports_a_missing_artifact(api, tmp_path):
    with raises_exactly("Checkpoint artifact is missing"):
        api._verify_artifact_file(tmp_path / "batch-000000.parquet", {}, 1)


def test_verify_artifact_rows_reports_a_row_count_mismatch(api, tmp_path):
    out = tmp_path / "out"
    api.run_shard(source(tmp_path), out, Extractor(), api.Contract(languages=("en",)))
    with raises_exactly("Checkpoint artifact row count mismatch"):
        api._verify_artifact_rows(out / "batch-000000.parquet", 99)


def test_verify_artifacts_rejects_a_batch_row_count_that_contradicts_the_batch_size(api, tmp_path):
    out = tmp_path / "out"
    receipt = api.run_shard(
        source(tmp_path), out, Extractor(), api.Contract(languages=("en",)), batch_size=2
    )
    receipt["artifacts"][0]["rows"] = 1
    with raises_exactly("Checkpoint artifact batch row count mismatch"):
        api._verify_artifacts(out, receipt)


def test_verify_artifacts_rejects_artifact_rows_that_do_not_sum_to_processed_rows(api, tmp_path):
    out = tmp_path / "out"
    receipt = api.run_shard(
        source(tmp_path), out, Extractor(), api.Contract(languages=("en",)), batch_size=2
    )
    receipt["status"] = "paused"
    receipt["processed_rows"] = 2
    with raises_exactly("Checkpoint row accounting mismatch"):
        api._verify_artifacts(out, receipt)
