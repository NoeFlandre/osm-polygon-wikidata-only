"""Small, CPU-only adapter around the pinned SaT-3l-sm model."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.v2.sentence_logic import SAT_MODEL_ID

DEFAULT_SAT_MODEL_REVISION = "137da05"
_COREML_PROVIDER = "CoreMLExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


class SaT3lSegmenter:
    """Load ``segment-any-text/sat-3l-sm`` lazily and expose a testable API."""

    model_id = SAT_MODEL_ID

    def __init__(
        self,
        *,
        cache_dir: Path,
        revision: str,
        inference_batch_size: int = 16,
        ort_providers: Sequence[str] | None = None,
    ) -> None:
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        try:
            wtpsplit = importlib.import_module("wtpsplit")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Sentence splitting requires the optional dependency; "
                "run `uv sync --extra sentence-splitting` first"
            ) from exc
        model_class = getattr(wtpsplit, "SaT", None)
        if model_class is None:
            raise RuntimeError("The installed wtpsplit package does not expose SaT")

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.revision = revision
        self.inference_batch_size = inference_batch_size
        self.ort_providers = tuple(_resolve_ort_providers(ort_providers))
        self.version = str(getattr(wtpsplit, "__version__", "unknown"))
        self._model = model_class(
            self.model_id,
            hub_prefix=None,
            ort_providers=list(self.ort_providers),
            from_pretrained_kwargs={
                "cache_dir": str(cache_dir),
                "revision": revision,
            },
        )

    def split(self, texts: Sequence[str], *, language: str) -> list[list[str]]:
        """Return lossless SaT pieces for one exact language code."""
        del language  # Routing is enforced by split_sections before this adapter is called.
        values = list(texts)
        if not values:
            return []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="split_on_input_newlines=False will lead to newlines in the output",
            )
            result: Any = self._model.split(
                values,
                strip_whitespace=False,
                split_on_input_newlines=False,
                batch_size=self.inference_batch_size,
                outer_batch_size=len(values),
            )
        return [list(pieces) for pieces in result]


def _resolve_ort_providers(override: Sequence[str] | None) -> list[str]:
    """Select CoreML on Apple Silicon and retain a portable CPU fallback."""
    if override is not None:
        providers = list(override)
        if not providers:
            raise ValueError("ort_providers must not be empty")
        return providers
    try:
        onnxruntime = importlib.import_module("onnxruntime")
        available = set(onnxruntime.get_available_providers())
    except (ImportError, AttributeError):
        available = set()
    if _COREML_PROVIDER in available:
        return [
            _COREML_PROVIDER,
            *([_CPU_PROVIDER] if _CPU_PROVIDER in available else []),
        ]
    return [_CPU_PROVIDER]


__all__ = ["DEFAULT_SAT_MODEL_REVISION", "SaT3lSegmenter"]
