"""CUDA adapter for the pinned WikiNEuRal multilingual NER model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.ner.pipeline import LABEL


class WikiNeuralLocationExtractor:
    """Extract LOC spans from WikiNEuRal and expose the project geographic label."""

    def __init__(
        self, model_directory: Path, *, batch_size: int = 16, threshold: float = 0.5
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.threshold = _validate_threshold(threshold)
        self.model_directory = Path(model_directory)
        self.batch_size = batch_size
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch = import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("WikiNEuRal requires an available CUDA GPU")
        transformers = import_module("transformers")
        options = {"local_files_only": True}
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_directory, **options)
        if not tokenizer.is_fast:
            raise ValueError("WikiNEuRal requires a fast tokenizer with character offsets")
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.model_directory, **options
        )
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model.eval().to("cuda")

    def predict(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if not texts:
            return []
        self._load()
        return self._predict_all(texts)

    def _predict_all(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        results: list[list[dict[str, Any]]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            results.extend(
                _predict_batch(self._torch, self._tokenizer, self._model, batch, self.threshold)
            )
        return results


def _predict_batch(
    torch: Any, tokenizer: Any, model: Any, texts: Sequence[str], threshold: float
) -> list[list[dict[str, Any]]]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    device_inputs = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**device_inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    labels = model.config.id2label
    return [
        _spans_for_text(text, offsets[index], probabilities[index], labels, threshold)
        for index, text in enumerate(texts)
    ]


def _spans_for_text(
    text: str,
    offsets: Sequence[Sequence[int]],
    scores: Any,
    labels: Mapping[int, str],
    threshold: float,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: list[tuple[int, int, float]] = []
    for tag, start, end, score in _tagged_tokens(offsets, scores, labels, threshold):
        current = _advance_span(spans, text, current, tag, start, end, score)
    if current:
        spans.append(_span(text, current))
    return spans


def _tagged_tokens(
    offsets: Sequence[Sequence[int]],
    scores: Any,
    labels: Mapping[int, str],
    threshold: float,
) -> list[tuple[str, int, int, float]]:
    tagged: list[tuple[str, int, int, float]] = []
    for offset, score_row in zip(offsets, scores, strict=True):
        start, end = int(offset[0]), int(offset[1])
        if start == end:
            continue
        label_id = int(_argmax(score_row))
        score = float(score_row[label_id])
        tag = str(labels[label_id]) if score >= threshold else "O"
        tagged.append((tag, start, end, score))
    return tagged


def _advance_span(
    spans: list[dict[str, Any]],
    text: str,
    current: list[tuple[int, int, float]],
    tag: str,
    start: int,
    end: int,
    score: float,
) -> list[tuple[int, int, float]]:
    if tag == "I-LOC" and current:
        current.append((start, end, score))
        return current
    if current:
        spans.append(_span(text, current))
    return [(start, end, score)] if tag == "B-LOC" else []


def _span(text: str, tokens: Sequence[tuple[int, int, float]]) -> dict[str, Any]:
    start, end = tokens[0][0], tokens[-1][1]
    score = sum(value for _, _, value in tokens) / len(tokens)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("WikiNEuRal produced an invalid entity score")
    return {"text": text[start:end], "label": LABEL, "start": start, "end": end, "score": score}


def _argmax(values: Any) -> int:
    return max(range(len(values)), key=lambda index: float(values[index]))


def _validate_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("threshold must be between zero and one")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    return threshold
