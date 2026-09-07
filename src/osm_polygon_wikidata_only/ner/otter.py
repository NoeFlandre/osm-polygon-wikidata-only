"""Lazy CUDA adapter for an audited local whoisjones/otter-cross-mmbert snapshot.

API audited at 8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d. The caller owns
snapshot provenance/code auditing; this module neither downloads nor updates it.
"""

import math
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from importlib import import_module
from itertools import batched
from numbers import Real
from pathlib import Path
from typing import Any, cast

_LABEL = "named geographic location"


def _load_model(directory: Path) -> Any:
    if not directory.is_dir():
        raise FileNotFoundError(f"Local Otter snapshot directory not found: {directory}")
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Otter requires an available CUDA GPU")
    transformers = import_module("transformers")
    options = {"trust_remote_code": True, "local_files_only": True}
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(directory), **options)
    if not tokenizer.is_fast:
        raise ValueError("Otter requires a fast tokenizer with character offsets")
    model = transformers.AutoModel.from_pretrained(str(directory), **options)
    model.token_encoder.config.reference_compile = False
    # Upstream's lazy property otherwise permits a remote base-tokenizer fallback.
    model._tokenizer = tokenizer
    return model.eval().to("cuda")


def _encoding(tokenizer: Any, text: str, *, special: bool = True) -> Any:
    return tokenizer(
        text, add_special_tokens=special, truncation=False, return_offsets_mapping=True
    )


def _fit_end(
    tokenizer: Any,
    prefix: str,
    text: str,
    offsets: list[tuple[int, int]],
    start: int,
    stop: int,
    char_start: int,
    limit: int,
) -> tuple[int, int]:
    """Recheck sliced text: prefix and slice boundaries can change tokenization."""
    while stop > start:
        char_end = len(text) if stop == len(offsets) else offsets[stop - 1][1]
        if len(_encoding(tokenizer, prefix + text[char_start:char_end])["input_ids"]) <= limit:
            return stop, char_end
        stop -= 1
    raise ValueError("No text token fits the checkpoint window budget")


def _windows(model: Any, text: str) -> Iterator[tuple[int, str]]:
    tokenizer = model._tokenizer
    prefix = model.build_prompt([_LABEL])
    limit = model.config.max_seq_length
    if len(_encoding(tokenizer, prefix + text)["input_ids"]) <= limit:
        yield 0, text
        return
    offsets = _encoding(tokenizer, text, special=False)["offset_mapping"]
    token_ends = [end for _, end in offsets]
    budget = limit - len(_encoding(tokenizer, prefix)["input_ids"])
    # One full maximum span overlaps, including a spare boundary token.
    overlap = model.config.max_span_length
    start, char_start = 0, 0
    while True:
        stop, char_end = _fit_end(
            tokenizer,
            prefix,
            text,
            offsets,
            start,
            min(start + budget, len(offsets)),
            char_start,
            limit,
        )
        yield char_start, text[char_start:char_end]
        if stop == len(offsets):
            return
        char_start = _next_shift(tokenizer, prefix, text[char_start:char_end], char_start, overlap)
        start = bisect_right(token_ends, char_start)


def _next_shift(tokenizer: Any, prefix: str, window: str, shift: int, overlap: int) -> int:
    offsets = _encoding(tokenizer, prefix + window)["offset_mapping"]
    text_offsets = [(start, end) for start, end in offsets if end > len(prefix)]
    if len(text_offsets) <= overlap:
        raise ValueError("Window budget cannot preserve maximum span overlap")
    next_shift = shift + max(0, text_offsets[-overlap][0] - len(prefix))
    if next_shift <= shift:
        raise ValueError("Window budget cannot preserve overlap and make progress")
    return next_shift


def _merge(
    entities: dict[tuple[int, int, str], dict[str, Any]],
    predicted: list[dict[str, Any]],
    text: str,
    shift: int,
) -> None:
    for raw_entity in predicted:
        start, end = raw_entity["start"] + shift, raw_entity["end"] + shift
        if not 0 <= start < end <= len(text):
            raise ValueError("Entity offsets are outside the original sentence")
        key = (start, end, raw_entity["label"])
        score = raw_entity["score"]
        if key not in entities or score > entities[key]["score"]:
            entities[key] = {**raw_entity, "start": start, "end": end}


def _entity_mapping(entity: object) -> Mapping[str, Any]:
    if not isinstance(entity, Mapping):
        raise ValueError("Entity output must be a mapping")
    return cast(Mapping[str, Any], entity)


def _require_entity_fields(entity: Mapping[str, Any]) -> None:
    if any(field not in entity for field in ("text", "label", "start", "end", "score")):
        raise ValueError("Entity output is missing a required field")


def _validate_entity_offsets(entity: Mapping[str, Any], window: str) -> tuple[int, int]:
    start, end = entity["start"], entity["end"]
    if type(start) is not int or type(end) is not int:
        raise ValueError("Entity offsets must be integers within the source window")
    if not 0 <= start < end <= len(window):
        raise ValueError("Entity offsets are outside the source window")
    return start, end


def _validate_entity_surface(entity: Mapping[str, Any], window: str, start: int, end: int) -> None:
    if entity["label"] != _LABEL:
        raise ValueError("Entity label does not match the geographic label")
    if not isinstance(entity["text"], str) or entity["text"] != window[start:end]:
        raise ValueError("Entity text does not match the source window slice")


def _validate_entity_score(entity: Mapping[str, Any]) -> float:
    score = entity["score"]
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("Entity score must be finite and between zero and one")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Entity score must be finite and between zero and one")
    return score


def _validate_entity(entity: object, window: str) -> dict[str, Any]:
    """Validate one model result against the exact text window sent to Otter."""
    entity = _entity_mapping(entity)
    _require_entity_fields(entity)
    start, end = _validate_entity_offsets(entity, window)
    _validate_entity_surface(entity, window, start, end)
    score = _validate_entity_score(entity)
    return {
        "text": entity["text"],
        "label": entity["label"],
        "start": start,
        "end": end,
        "score": score,
    }


def _validate_predictions(
    predictions: object, jobs: Sequence[tuple[int, int, str]]
) -> list[list[dict[str, Any]]]:
    if not isinstance(predictions, list) or len(predictions) != len(jobs):
        raise ValueError("Model output count does not match input count")
    return [
        _validate_prediction_entities(predicted, window)
        for (_, _, window), predicted in zip(jobs, predictions, strict=True)
    ]


def _validate_prediction_entities(predicted: object, window: str) -> list[dict[str, Any]]:
    if not isinstance(predicted, list):
        raise ValueError("Model output entities must be lists")
    return [_validate_entity(entity, window) for entity in predicted]


def _validate_batch_size(batch_size: object) -> int:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    return batch_size


def _validate_threshold(threshold: object) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("threshold must be finite and between zero and one")
    value = float(threshold)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be between zero and one")
    return value


class OtterLocationExtractor:
    """Extract only named geographic locations; initialize GPU state on first use."""

    def __init__(
        self, model_directory: Path, *, batch_size: int = 16, threshold: float = 0.5
    ) -> None:
        self.model_directory = model_directory
        self.batch_size = _validate_batch_size(batch_size)
        self.threshold = _validate_threshold(threshold)
        self._model: Any = None

    def _jobs(self, texts: Sequence[str]) -> Iterator[tuple[int, int, str]]:
        for index, text in enumerate(texts):
            if text.strip():
                for shift, window in _windows(self._model, text):
                    yield index, shift, window

    def predict(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if not texts:
            return []
        if self._model is None:
            self._model = _load_model(self.model_directory)
        return [
            sorted(result.values(), key=lambda e: (e["start"], e["end"]))
            for result in self._predict_batches(texts)
        ]

    def _predict_batches(
        self, texts: Sequence[str]
    ) -> list[dict[tuple[int, int, str], dict[str, Any]]]:
        entities: list[dict[tuple[int, int, str], dict[str, Any]]] = [{} for _ in texts]
        for batch in batched(self._jobs(texts), self.batch_size):
            jobs = list(batch)
            predictions = self._model.predict(
                [job[2] for job in jobs],
                labels=[_LABEL],
                threshold=self.threshold,
                batch_size=self.batch_size,
                max_seq_length=self._model.config.max_seq_length,
            )
            validated = _validate_predictions(predictions, jobs)
            for (index, shift, _), predicted in zip(jobs, validated, strict=True):
                _merge(entities[index], predicted, texts[index], shift)
        return entities
