"""Otter contract tests without installing torch or downloading a checkpoint."""

import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def extractor_type() -> Any:
    try:
        module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    except ModuleNotFoundError:
        pytest.fail("Otter adapter has not been implemented")
    return module.OtterLocationExtractor


@pytest.mark.parametrize(
    "function,args,message",
    [
        ("_entity_mapping", (None,), "Entity output must be a mapping"),
        ("_require_entity_fields", ({},), "Entity output is missing a required field"),
        (
            "_validate_entity_offsets",
            ({"start": True, "end": 1}, "x"),
            "Entity offsets must be integers within the source window",
        ),
        (
            "_validate_entity_offsets",
            ({"start": 0, "end": 2}, "x"),
            "Entity offsets are outside the source window",
        ),
        (
            "_validate_entity_surface",
            ({"label": "other"}, "x", 0, 1),
            "Entity label does not match the geographic label",
        ),
        (
            "_validate_entity_surface",
            ({"label": "named geographic location", "text": "y"}, "x", 0, 1),
            "Entity text does not match the source window slice",
        ),
        (
            "_validate_entity_score",
            ({"score": True},),
            "Entity score must be finite and between zero and one",
        ),
        (
            "_validate_entity_score",
            ({"score": -1},),
            "Entity score must be finite and between zero and one",
        ),
        (
            "_validate_predictions",
            ([], [(0, 0, "x")]),
            "Model output count does not match input count",
        ),
        ("_validate_prediction_entities", ({}, "x"), "Model output entities must be lists"),
        ("_validate_batch_size", (0,), "batch_size must be positive"),
        ("_validate_threshold", (True,), "threshold must be finite and between zero and one"),
        ("_validate_threshold", (-1,), "threshold must be between zero and one"),
    ],
)
def test_model_contract_errors_preserve_exact_diagnostics(function, args, message):
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
        getattr(module, function)(*args)


def test_missing_local_snapshot_error_includes_its_path(tmp_path):
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    path = tmp_path / "missing"
    with pytest.raises(
        FileNotFoundError,
        match=rf"^{re.escape(f'Local Otter snapshot directory not found: {path}')}$",
    ):
        module._load_model(path)


@pytest.mark.parametrize("start,end,shift", [(0, 0, 0), (0, 1, -1), (0, 1, 2)])
def test_merge_rejects_invalid_translated_spans(start, end, shift):
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    entity = dict(text="x", label="named geographic location", start=start, end=end, score=0.5)
    with pytest.raises(ValueError, match=r"^Entity offsets are outside the original sentence$"):
        module._merge({}, [entity], "x", shift)


def test_merge_keeps_the_first_entity_when_scores_tie():
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    entity = dict(text="x", label="named geographic location", start=0, end=1, score=0.5)
    results = {}
    module._merge(results, [entity], "x", 0)
    original = results[(0, 1, "named geographic location")]
    module._merge(results, [entity], "x", 0)
    assert results[(0, 1, "named geographic location")] is original


class Tokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs.get("truncation") is False
        prefix = "[LABEL] named geographic location [SEP] "
        shift = len(prefix) if text.startswith(prefix) else 0
        offsets = [(i, i + 1) for i in range(shift, len(text))]
        if shift:
            offsets = [(0, shift)] * 4 + offsets
        if kwargs.get("add_special_tokens", True):
            offsets = [(0, 0), *offsets, (0, 0)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_seq_length=16, max_span_length=3)
        self.token_encoder = SimpleNamespace(config=SimpleNamespace(reference_compile=None))
        self._tokenizer: Any = None
        self.calls: list[list[str]] = []
        self.events: list[str] = []
        self.surfaces = ["東京", "é", "abc"]

    def eval(self) -> "Model":
        self.events.append("eval")
        return self

    def to(self, device: str) -> "Model":
        self.events.append(device)
        return self

    @staticmethod
    def build_prompt(labels: list[str]) -> str:
        assert labels == ["named geographic location"]
        return "[LABEL] named geographic location [SEP] "

    def predict(self, texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        assert kwargs["labels"] == ["named geographic location"]
        assert kwargs["max_seq_length"] == self.config.max_seq_length
        assert len(texts) <= kwargs["batch_size"]
        assert self.events == ["eval", "cuda"]
        self.calls.append(texts)
        results = []
        for text in texts:
            encoded = self._tokenizer(self.build_prompt(kwargs["labels"]) + text, truncation=False)
            assert len(encoded["input_ids"]) <= self.config.max_seq_length
            entities = []
            for surface in self.surfaces:
                for start in range(len(text)):
                    score = 0.6 if start == 0 else 0.9
                    if text.startswith(surface, start) and score >= kwargs["threshold"]:
                        entities.append(
                            dict(
                                text=surface,
                                label=kwargs["labels"][0],
                                start=start,
                                end=start + len(surface),
                                score=score,
                            )
                        )
            results.append(entities)
        return results


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    model = Model()
    loads: list[tuple[str, Any, dict[str, Any]]] = []
    tokenizer = Tokenizer()

    def load_model(path: Any, **kwargs: Any) -> Model:
        loads.append(("model", path, kwargs))
        return model

    def load_tokenizer(path: Any, **kwargs: Any) -> Tokenizer:
        loads.append(("tokenizer", path, kwargs))
        return tokenizer

    cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        ),
    )
    return SimpleNamespace(model=model, loads=loads, cuda=cuda, tokenizer=tokenizer)


def test_lazy_local_cuda_load_and_reuse(boundary: SimpleNamespace, tmp_path: Path) -> None:
    extractor = extractor_type()(tmp_path)
    assert boundary.loads == []
    assert extractor.predict(["東京", "", "  ", "nothing"]) == [
        [dict(text="東京", label="named geographic location", start=0, end=2, score=0.6)],
        [],
        [],
        [],
    ]
    extractor.predict(["é"])
    assert [entry[0] for entry in boundary.loads].count("model") == 1
    assert [entry[0] for entry in boundary.loads].count("tokenizer") == 1
    for _, path, kwargs in boundary.loads:
        assert Path(path) == tmp_path
        assert kwargs["trust_remote_code"] is True
        assert kwargs["local_files_only"] is True


def test_no_cuda_fails_before_loading(boundary: SimpleNamespace, tmp_path: Path) -> None:
    boundary.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match=r"^Otter requires an available CUDA GPU$"):
        extractor_type()(tmp_path).predict(["Tokyo"])
    assert boundary.loads == []


def test_windows_cover_unicode_tail_and_boundary_spans(
    boundary: SimpleNamespace, tmp_path: Path
) -> None:
    text = "😀e\u0301---abc--東京-----é   "
    result = extractor_type()(tmp_path, batch_size=2).predict([text, "東京"])
    expected = []
    for surface in ["abc", "東京", "é"]:
        start = text.index(surface)
        expected.append(
            dict(
                text=surface,
                label="named geographic location",
                start=start,
                end=start + len(surface),
                score=0.9,
            )
        )
    assert result[0] == expected
    windows = [window for batch in boundary.model.calls for window in batch]
    assert windows[-2].endswith("   ")
    assert all(len(batch) <= 2 for batch in boundary.model.calls)


def test_exactly_fitting_window_does_not_enter_splitting_path() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")

    class RecordingTokenizer(Tokenizer):
        def __init__(self) -> None:
            self.special_tokens: list[bool] = []

        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            self.special_tokens.append(kwargs["add_special_tokens"])
            return super().__call__(text, **kwargs)

    tokenizer = RecordingTokenizer()
    model = Model()
    model._tokenizer = tokenizer
    text = "😀e\u0301abc東京  "

    assert list(module._windows(model, text)) == [(0, text)]
    assert tokenizer.special_tokens == [True]


def test_windows_budget_excludes_prompt_tokens_before_slicing() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    prefix = Model.build_prompt(["named geographic location"])

    class MergedContentTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["truncation"] is False
            if text.startswith(prefix):
                content = text[len(prefix) :]
                offsets = [(0, len(prefix))] * 4 + [
                    (len(prefix) + index, len(prefix) + index + 1) for index in range(len(content))
                ]
                token_count = 8 if len(content) in {4, 5} else 4 + len(content)
                return {"input_ids": list(range(token_count)), "offset_mapping": offsets}
            offsets = [(index, index + 1) for index in range(len(text))]
            return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    model = SimpleNamespace(
        _tokenizer=MergedContentTokenizer(),
        config=SimpleNamespace(max_seq_length=8, max_span_length=1),
        build_prompt=lambda labels: prefix,
    )

    assert list(module._windows(model, "abcdef")) == [(0, "abcd"), (3, "def")]


def test_windows_starts_slicing_at_zero_token_index() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    prefix = Model.build_prompt(["named geographic location"])

    class MergedContentTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["truncation"] is False
            if text.startswith(prefix):
                content = text[len(prefix) :]
                offsets = [(0, len(prefix))] * 4 + [
                    (len(prefix) + index, len(prefix) + index + 1) for index in range(len(content))
                ]
                token_count = 6 if len(content) in {2, 3} else 4 + len(content)
                return {"input_ids": list(range(token_count)), "offset_mapping": offsets}
            offsets = [(index, index + 1) for index in range(len(text))]
            return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    model = SimpleNamespace(
        _tokenizer=MergedContentTokenizer(),
        config=SimpleNamespace(max_seq_length=6, max_span_length=1),
        build_prompt=lambda labels: prefix,
    )

    assert list(module._windows(model, "abcde")) == [
        (0, "ab"),
        (1, "bc"),
        (2, "cd"),
        (3, "de"),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"threshold": True},
        {"threshold": "0.5"},
    ],
)
def test_invalid_options(kwargs: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        extractor_type()(tmp_path, **kwargs)


def test_threshold_zero_is_an_inclusive_boundary(boundary: SimpleNamespace, tmp_path: Path) -> None:
    assert extractor_type()(tmp_path, threshold=0).predict(["é"]) == [
        [dict(text="é", label="named geographic location", start=0, end=1, score=0.6)]
    ]


def test_default_batch_size_keeps_sixteen_jobs_per_model_call(
    boundary: SimpleNamespace, tmp_path: Path
) -> None:
    extractor_type()(tmp_path).predict(["nothing"] * 17)

    assert [len(batch) for batch in boundary.model.calls] == [16, 1]


def test_empty_input_does_not_load(boundary: SimpleNamespace, tmp_path: Path) -> None:
    assert extractor_type()(tmp_path).predict([]) == []
    assert boundary.loads == []


def test_encoding_always_requests_character_offsets() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    calls: list[dict[str, Any]] = []

    def tokenizer(text: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {}

    module._encoding(tokenizer, "Tokyo", special=False)
    assert calls == [
        {"add_special_tokens": False, "truncation": False, "return_offsets_mapping": True}
    ]


def test_fit_end_rejects_an_empty_candidate_window() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")

    with pytest.raises(ValueError, match=r"^No text token fits the checkpoint window budget$"):
        module._fit_end(
            lambda text, **kwargs: {"input_ids": [0], "offset_mapping": []},
            "p",
            "Tokyo",
            [(0, 1)],
            1,
            1,
            0,
            1,
        )


def test_fit_end_accepts_a_candidate_at_the_token_limit() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")

    def tokenizer(text: str, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": [0, 1, 2], "offset_mapping": []}

    assert module._fit_end(tokenizer, "p", "Tokyo", [(0, 1), (1, 2)], 0, 2, 0, 3) == (2, 5)


def test_fit_end_reduces_one_token_at_a_time() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")

    def tokenizer(text: str, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": list(range(len(text))), "offset_mapping": []}

    assert module._fit_end(tokenizer, "", "abcd", [(0, 1), (1, 2), (2, 3)], 0, 3, 0, 2) == (
        2,
        2,
    )


def test_threshold_passed_through(boundary: SimpleNamespace, tmp_path: Path) -> None:
    assert extractor_type()(tmp_path, threshold=1.0).predict(["東京"]) == [[]]


def test_import_and_construction_without_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    module = importlib.reload(importlib.import_module("osm_polygon_wikidata_only.ner.otter"))
    extractor = module.OtterLocationExtractor(tmp_path)
    with pytest.raises(ImportError):
        extractor.predict(["Tokyo"])


def test_missing_directory_is_not_a_hub_id(boundary: SimpleNamespace, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extractor_type()(tmp_path / "missing").predict(["Tokyo"])
    assert boundary.loads == []


def test_impossible_window_fails_explicitly(boundary: SimpleNamespace, tmp_path: Path) -> None:
    boundary.model.config.max_seq_length = 8
    with pytest.raises(ValueError, match=r"span|window|budget"):
        extractor_type()(tmp_path).predict(["x" * 20])


def test_slow_tokenizer_rejected(boundary: SimpleNamespace, tmp_path: Path) -> None:
    boundary.tokenizer.is_fast = False
    with pytest.raises(
        ValueError, match=r"^Otter requires a fast tokenizer with character offsets$"
    ):
        extractor_type()(tmp_path).predict(["Tokyo"])


class PrefixMergeTokenizer(Tokenizer):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        encoded = super().__call__(text, **kwargs)
        if text.startswith("[LABEL]"):
            offsets = encoded["offset_mapping"]
            content = offsets[5:-1]
            merged = [
                (part[0][0], part[-1][1])
                for part in [content[i : i + 2] for i in range(0, len(content), 2)]
            ]
            encoded["offset_mapping"] = offsets[:5] + merged + offsets[-1:]
            encoded["input_ids"] = list(range(len(encoded["offset_mapping"])))
        return encoded


def test_overlap_uses_tokens_as_seen_with_label_prefix(
    boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer = PrefixMergeTokenizer()
    monkeypatch.setattr(
        sys.modules["transformers"].AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    boundary.model.surfaces = ["abcdef"]
    text = "xxxxxabcdef" + "x" * 20
    assert extractor_type()(tmp_path).predict([text]) == [
        [dict(text="abcdef", label="named geographic location", start=5, end=11, score=0.9)]
    ]


class ExpandedTokenizer(Tokenizer):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        encoded = super().__call__(text, **kwargs)
        offsets = encoded["offset_mapping"]
        # Byte fallback can emit multiple tokens for the same Python character.
        expanded = [
            offset
            for offset in offsets
            for _ in range(2 if text[offset[0] : offset[1]] == "😀" else 1)
        ]
        if text.startswith("[LABEL]"):
            expanded = [*expanded[:5], expanded[5], *expanded[5:]]
        return {"input_ids": list(range(len(expanded))), "offset_mapping": expanded}


def test_retokenized_windows_shrink_without_losing_unicode(
    boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys.modules["transformers"].AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: ExpandedTokenizer(),
    )
    text = "😀😀abc😀😀abc😀😀東京  "
    result = extractor_type()(tmp_path).predict([text])[0]
    assert [(e["text"], e["start"], e["end"]) for e in result] == [
        ("abc", 2, 5),
        ("abc", 7, 10),
        ("東京", 12, 14),
    ]
    windows = [window for batch in boundary.model.calls for window in batch]
    assert windows[-1].endswith("  ")


@pytest.mark.parametrize("position", range(1, 24))
def test_every_maximum_width_boundary_span_is_seen(
    boundary: SimpleNamespace, tmp_path: Path, position: int
) -> None:
    text = "x" * position + "abc" + "x" * 25
    entities = extractor_type()(tmp_path).predict([text])[0]
    assert len(entities) == 1
    assert (entities[0]["start"], entities[0]["end"]) == (position, position + 3)


def test_model_batch_length_mismatch_is_not_silently_dropped(
    boundary: SimpleNamespace, tmp_path: Path
) -> None:
    boundary.model.predict = lambda *args, **kwargs: []
    with pytest.raises(ValueError, match=r"^Model output count does not match input count$"):
        extractor_type()(tmp_path).predict(["Tokyo"])


@pytest.mark.parametrize("response", [None, [None]])
def test_malformed_prediction_containers_are_rejected_as_model_errors(
    boundary: SimpleNamespace, tmp_path: Path, response: object
) -> None:
    boundary.model.predict = lambda *args, **kwargs: response
    with pytest.raises(ValueError, match=r"Model output|Entity output"):
        extractor_type()(tmp_path).predict(["Tokyo"])


def test_loading_failure_propagates(
    boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("missing local tokenizer")

    monkeypatch.setattr(sys.modules["transformers"].AutoTokenizer, "from_pretrained", fail)
    with pytest.raises(OSError, match="missing local tokenizer"):
        extractor_type()(tmp_path).predict(["Tokyo"])
    assert boundary.loads == []


def test_prefix_leaves_no_room_for_one_token(boundary: SimpleNamespace, tmp_path: Path) -> None:
    boundary.model.config.max_seq_length = 6
    with pytest.raises(ValueError, match="window budget"):
        extractor_type()(tmp_path).predict(["Tokyo"])


class MissingOffsetsTokenizer(Tokenizer):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": [], "offset_mapping": []}
        return super().__call__(text, **kwargs)


def test_missing_text_offsets_cannot_silently_drop_long_text(
    boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys.modules["transformers"].AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: MissingOffsetsTokenizer(),
    )
    with pytest.raises(ValueError, match="window budget"):
        extractor_type()(tmp_path).predict(["x" * 20])


def test_next_shift_requires_more_tokens_than_the_overlap() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")

    class SingleTokenTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            return {"input_ids": [0, 1], "offset_mapping": [(0, 1), (1, 2)]}

    with pytest.raises(ValueError, match=r"^Window budget cannot preserve maximum span overlap$"):
        module._next_shift(SingleTokenTokenizer(), "p", "x", 0, 1)


def test_nonadvancing_unicode_offsets_fail_instead_of_looping() -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")

    class NonAdvancingTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "input_ids": [0, 1, 2],
                "offset_mapping": [(0, 1), (1, 2), (1, 2)],
            }

    with pytest.raises(
        ValueError, match=r"^Window budget cannot preserve overlap and make progress$"
    ):
        module._next_shift(NonAdvancingTokenizer(), "p", "x", 0, 1)


@pytest.mark.parametrize(
    "patch, field",
    [
        ({"label": "person"}, "label"),
        ({"label": None}, "label"),
        ({"start": -1}, "offset"),
        ({"start": False}, "offset"),
        ({"start": 0.0}, "offset"),
        ({"end": True}, "offset"),
        ({"end": 2.0}, "offset"),
        ({"end": 0}, "offset"),
        ({"start": 2, "end": 1}, "offset"),
        ({"end": 3}, "offset"),
        ({"text": "Tokyo"}, "text"),
        ({"score": float("nan")}, "score"),
        ({"score": float("inf")}, "score"),
        ({"score": -float("inf")}, "score"),
        ({"score": -0.01}, "score"),
        ({"score": 1.01}, "score"),
        ({"score": "0.9"}, "score"),
        ({"score": True}, "score"),
        ({"score": None}, "score"),
    ],
)
def test_invalid_raw_entity_is_rejected(
    boundary: SimpleNamespace, tmp_path: Path, patch: dict[str, Any], field: str
) -> None:
    entity = dict(text="東京", label="named geographic location", start=0, end=2, score=0.9)
    entity.update(patch)
    boundary.model.predict = lambda *args, **kwargs: [[entity]]
    with pytest.raises(ValueError, match=field):
        extractor_type()(tmp_path).predict(["東京"])


@pytest.mark.parametrize("field", ["label", "start", "end", "text", "score"])
def test_missing_raw_entity_field_is_rejected(
    boundary: SimpleNamespace, tmp_path: Path, field: str
) -> None:
    entity = dict(text="東京", label="named geographic location", start=0, end=2, score=0.9)
    del entity[field]
    boundary.model.predict = lambda *args, **kwargs: [[entity]]
    with pytest.raises(ValueError):
        extractor_type()(tmp_path).predict(["東京"])


@pytest.mark.parametrize("score", [0, 0.0, 1, 1.0])
def test_valid_raw_entity_preserves_unicode_and_score_endpoints(
    boundary: SimpleNamespace, tmp_path: Path, score: float
) -> None:
    entity = dict(
        text="e\u0301東京", label="named geographic location", start=1, end=5, score=score
    )
    boundary.model.predict = lambda *args, **kwargs: [[entity]]
    assert extractor_type()(tmp_path).predict(["😀e\u0301東京"]) == [[entity]]


def test_entity_inside_source_but_outside_shifted_window_is_rejected(
    boundary: SimpleNamespace, tmp_path: Path
) -> None:
    calls = 0

    def predict(texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [[]]
        return [
            [dict(text="x" * 11, label="named geographic location", start=0, end=11, score=0.9)]
        ]

    boundary.model.predict = predict
    with pytest.raises(ValueError, match="offset"):
        extractor_type()(tmp_path, batch_size=1).predict(["x" * 40])
    assert calls == 2


def test_malformed_batch_response_is_rejected_before_any_span_is_rewritten(
    boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = importlib.import_module("osm_polygon_wikidata_only.ner.otter")
    merged: list[object] = []
    original_merge = module._merge

    def record_merge(*args: Any, **kwargs: Any) -> None:
        merged.append((args, kwargs))
        original_merge(*args, **kwargs)

    monkeypatch.setattr(module, "_merge", record_merge)

    def predict(texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        return [
            [
                dict(
                    text=texts[0],
                    label="named geographic location",
                    start=0,
                    end=len(texts[0]),
                    score=0.9,
                )
            ],
            [
                dict(
                    text="not-the-source-slice",
                    label="named geographic location",
                    start=0,
                    end=3,
                    score=0.9,
                )
            ],
        ]

    boundary.model.predict = predict
    with pytest.raises(ValueError, match="source window"):
        extractor_type()(tmp_path, batch_size=2).predict(["Tokyo", "Paris"])
    assert merged == []


@pytest.mark.parametrize("initial", [None, True, False])
def test_modernbert_auto_compile_disabled_before_inference(
    boundary: SimpleNamespace, tmp_path: Path, initial: bool | None
) -> None:
    boundary.model.token_encoder.config.reference_compile = initial
    original_predict = boundary.model.predict

    def predict(texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        assert boundary.model.token_encoder.config.reference_compile is False
        return original_predict(texts, **kwargs)

    boundary.model.predict = predict
    assert extractor_type()(tmp_path).predict(["東京"]) == [
        [dict(text="東京", label="named geographic location", start=0, end=2, score=0.6)]
    ]
