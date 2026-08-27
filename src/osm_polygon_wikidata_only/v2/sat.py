"""Small adapter around the pinned SaT-3l-sm model."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.v2.sentence_logic import SAT_MODEL_ID

DEFAULT_SAT_MODEL_REVISION = "137da05"
_COREML_PROVIDER = "CoreMLExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


class SaT3lSegmenter:
    """Load ``segment-any-text/sat-3l-sm`` lazily and expose a testable API."""

    model_id = SAT_MODEL_ID
    supports_mixed_languages = True

    def __init__(
        self,
        *,
        cache_dir: Path,
        revision: str,
        inference_batch_size: int = 16,
        ort_providers: Sequence[str] | None = None,
        require_gpu: bool = False,
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
        self.ort_providers = tuple(_resolve_ort_providers(ort_providers, require_gpu=require_gpu))
        self.version = str(getattr(wtpsplit, "__version__", "unknown"))
        model = model_class(
            self.model_id,
            hub_prefix=None,
            ort_providers=list(self.ort_providers),
            from_pretrained_kwargs={
                "cache_dir": str(cache_dir),
                "revision": revision,
            },
        )
        if require_gpu:
            self.ort_providers = _effective_ort_providers(model)
            if _CUDA_PROVIDER not in self.ort_providers:
                raise RuntimeError(
                    "Grid5000 sentence splitting requires an active CUDAExecutionProvider"
                )
        self._model = model

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


def _resolve_ort_providers(
    override: Sequence[str] | None,
    *,
    require_gpu: bool = False,
) -> list[str]:
    """Select providers while optionally requiring CUDA for a GPU job."""
    if override is not None:
        providers = list(override)
        if not providers:
            raise ValueError("ort_providers must not be empty")
    elif require_gpu:
        providers = _gpu_ort_providers()
    else:
        providers = _default_ort_providers()
    if require_gpu and _CUDA_PROVIDER not in providers:
        raise RuntimeError("Grid5000 sentence splitting requires CUDAExecutionProvider")
    return providers


def _default_ort_providers() -> list[str]:
    """Select CoreML on Apple Silicon and retain a portable CPU fallback."""
    available = _available_ort_providers()
    if _COREML_PROVIDER in available:
        return [
            _COREML_PROVIDER,
            *([_CPU_PROVIDER] if _CPU_PROVIDER in available else []),
        ]
    return [_CPU_PROVIDER]


def _gpu_ort_providers() -> list[str]:
    """Prefer CUDA and retain CPU fallback when the GPU runtime exposes it."""
    available = _available_ort_providers()
    return [
        *([_CUDA_PROVIDER] if _CUDA_PROVIDER in available else []),
        *([_CPU_PROVIDER] if _CPU_PROVIDER in available else []),
    ]


def _effective_ort_providers(model: Any) -> tuple[str, ...]:
    """Return the providers actually active in a constructed SaT session."""
    wrapper = getattr(model, "model", None)
    session = getattr(wrapper, "ort_session", None)
    get_providers = getattr(session, "get_providers", None)
    if not callable(get_providers):
        raise RuntimeError("Could not verify the active CUDAExecutionProvider")
    providers = tuple(str(provider) for provider in get_providers())
    if not providers:
        raise RuntimeError("Could not verify the active CUDAExecutionProvider")
    return providers


def _available_ort_providers() -> set[str]:
    """Return the providers exposed by the installed ONNX Runtime build."""
    try:
        onnxruntime = importlib.import_module("onnxruntime")
        return set(onnxruntime.get_available_providers())
    except (ImportError, AttributeError):
        return set()


__all__ = ["DEFAULT_SAT_MODEL_REVISION", "SaT3lSegmenter"]
