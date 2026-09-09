"""Offline tests for the diagnostic WikiNEuRal adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osm_polygon_wikidata_only.ner.wikineural import (
    WikiNeuralLocationExtractor,
    _advance_span,
    _span,
    _spans_for_text,
    _tagged_tokens,
    _validate_threshold,
)


class _Tokenizer:
    is_fast = True

    def __call__(self, texts, **kwargs):
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        assert kwargs["return_tensors"] == "pt"
        return {
            "input_ids": _Tensor([[101, 10, 11, 102] for _ in texts]),
            "attention_mask": _Tensor([[1, 1, 1, 1] for _ in texts]),
            "offset_mapping": [[(0, 0), (0, 5), (6, 12), (0, 0)] for _ in texts],
        }


class _Model:
    config = SimpleNamespace(id2label={0: "O", 1: "B-LOC", 2: "I-LOC"})

    def eval(self):
        return self

    def to(self, device):
        assert device == "cuda"
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(
            logits=_Tensor(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                    ]
                    for _ in kwargs["input_ids"]
                ]
            )
        )


class _Tensor(list):
    def to(self, device):
        assert device == "cuda"
        return self

    def __getitem__(self, index):
        value = super().__getitem__(index)
        return _Tensor(value) if isinstance(value, list) else value


def test_adapter_extracts_only_location_bio_spans(monkeypatch, tmp_path) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        inference_mode=lambda: _Context(),
        softmax=lambda values, dim: _softmax(values, dim),
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: _Tokenizer()),
        AutoModelForTokenClassification=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _Model()
        ),
    )

    def import_runtime(name):
        return torch if name == "torch" else transformers

    monkeypatch.setattr("osm_polygon_wikidata_only.ner.wikineural.import_module", import_runtime)

    result = WikiNeuralLocationExtractor(tmp_path).predict(["Paris London"])

    assert result == [
        [
            {
                "text": "Paris London",
                "label": "named geographic location",
                "start": 0,
                "end": 12,
                "score": 1.0,
            }
        ]
    ]


def test_adapter_applies_the_confidence_threshold(monkeypatch, tmp_path) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        inference_mode=lambda: _Context(),
        softmax=lambda values, dim: _softmax(values, dim),
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: _Tokenizer()),
        AutoModelForTokenClassification=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _LowScoreModel()
        ),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.ner.wikineural.import_module",
        lambda name: torch if name == "torch" else transformers,
    )

    assert WikiNeuralLocationExtractor(tmp_path, threshold=0.5).predict(["Paris London"]) == [[]]


def test_adapter_rejects_an_invalid_threshold(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"^threshold must be between zero and one$"):
        WikiNeuralLocationExtractor(tmp_path, threshold=1.1)


def test_constructor_preserves_defaults_and_rejects_invalid_batch_sizes(tmp_path) -> None:
    extractor = WikiNeuralLocationExtractor(tmp_path)

    assert extractor.batch_size == 16
    assert extractor.model_directory == tmp_path
    assert extractor._torch is None
    assert extractor._tokenizer is None

    assert WikiNeuralLocationExtractor(tmp_path, batch_size=1).batch_size == 1
    for invalid in (0, -1, 1.0, True, "16"):
        with pytest.raises(ValueError, match=r"^batch_size must be positive$"):
            WikiNeuralLocationExtractor(tmp_path, batch_size=invalid)


def test_load_uses_local_runtime_and_model_directory(monkeypatch, tmp_path) -> None:
    calls = []

    class Tokenizer(_Tokenizer):
        @classmethod
        def from_pretrained(cls, directory, **options):
            calls.append(("tokenizer", directory, options))
            return cls()

    class Model(_Model):
        @classmethod
        def from_pretrained(cls, directory, **options):
            calls.append(("model", directory, options))
            return cls()

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Tokenizer.from_pretrained),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=Model.from_pretrained),
    )

    def import_runtime(name):
        calls.append(("import", name))
        return torch if name == "torch" else transformers

    monkeypatch.setattr("osm_polygon_wikidata_only.ner.wikineural.import_module", import_runtime)

    WikiNeuralLocationExtractor(tmp_path)._load()

    assert calls == [
        ("import", "torch"),
        ("import", "transformers"),
        ("tokenizer", tmp_path, {"local_files_only": True}),
        ("model", tmp_path, {"local_files_only": True}),
    ]


def test_predict_batches_all_texts_and_uses_torch_softmax_dimension(monkeypatch, tmp_path) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        inference_mode=lambda: _Context(),
        softmax=_softmax,
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: _Tokenizer()),
        AutoModelForTokenClassification=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _Model()
        ),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.ner.wikineural.import_module",
        lambda name: torch if name == "torch" else transformers,
    )

    result = WikiNeuralLocationExtractor(tmp_path, batch_size=2).predict(
        ["Paris London", "Paris London", "Paris London"]
    )

    assert len(result) == 3


def test_tagged_tokens_requires_aligned_model_outputs() -> None:
    with pytest.raises(ValueError):
        _tagged_tokens([(0, 1), (1, 2)], [[1.0, 0.0]], {0: "O", 1: "B-LOC"}, 0.5)


def test_tagged_tokens_keeps_boundary_score_and_explicit_outside_tag() -> None:
    assert _tagged_tokens([(0, 1)], [[0.0, 0.5]], {0: "O", 1: "B-LOC"}, 0.5) == [
        ("B-LOC", 0, 1, 0.5)
    ]
    assert _tagged_tokens([(0, 1)], [[0.4, 0.3]], {0: "O", 1: "B-LOC"}, 0.5) == [("O", 0, 1, 0.4)]


def test_spans_flush_previous_entity_and_use_last_token_end() -> None:
    labels = {0: "O", 1: "B-LOC", 2: "I-LOC"}
    scores = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    offsets = [(0, 1), (1, 2), (2, 5)]

    assert _spans_for_text("ABCDE", offsets, scores, labels, 0.5) == [
        {"text": "A", "label": "named geographic location", "start": 0, "end": 1, "score": 1.0},
        {"text": "CDE", "label": "named geographic location", "start": 2, "end": 5, "score": 1.0},
    ]


def test_advance_span_does_not_start_an_entity_for_outside_tag() -> None:
    assert _advance_span([], "Paris", [], "O", 0, 5, 1.0) == []


@pytest.mark.parametrize(
    ("score", "should_raise"),
    [(0.0, False), (1.0, False), (1.5, True)],
)
def test_span_enforces_finite_probability_range(score, should_raise) -> None:
    def operation():
        return _span("Paris", [(0, 5, score)])

    if should_raise:
        with pytest.raises(ValueError, match=r"^WikiNEuRal produced an invalid entity score$"):
            operation()
    else:
        assert operation()["score"] == score


@pytest.mark.parametrize("value", ["0.5", object()])
def test_threshold_rejects_non_numeric_values(value) -> None:
    with pytest.raises(ValueError, match=r"^threshold must be between zero and one$"):
        _validate_threshold(value)


def test_threshold_accepts_both_inclusive_boundaries() -> None:
    assert _validate_threshold(0) == 0.0
    assert _validate_threshold(1) == 1.0


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _LowScoreModel(_Model):
    def __call__(self, **kwargs):
        return SimpleNamespace(
            logits=_Tensor(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [0.6, 0.4, 0.0],
                        [0.6, 0.0, 0.4],
                        [1.0, 0.0, 0.0],
                    ]
                    for _ in kwargs["input_ids"]
                ]
            )
        )


def _softmax(values, dim):
    assert dim == -1
    return values
