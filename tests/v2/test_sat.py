from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from osm_polygon_wikidata_only.v2.sat import SaT3lSegmenter


class _FakeSaT:
    init_args: tuple[object, ...] | None = None
    init_kwargs: dict[str, object] | None = None
    split_args: tuple[object, ...] | None = None
    split_kwargs: dict[str, object] | None = None
    effective_providers = ("CUDAExecutionProvider", "CPUExecutionProvider")

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).init_args = args
        type(self).init_kwargs = kwargs
        self.model = _FakeORTWrapper(type(self).effective_providers)

    def split(self, *args: object, **kwargs: object) -> list[list[str]]:
        type(self).split_args = args
        type(self).split_kwargs = kwargs
        return [["First. ", "Second."]]


class _FakeORTWrapper:
    def __init__(self, providers: tuple[str, ...]) -> None:
        self.ort_session = self
        self._providers = providers

    def get_providers(self) -> list[str]:
        return list(self._providers)


def _fake_wtpsplit() -> ModuleType:
    module = ModuleType("wtpsplit")
    module.__version__ = "2.2.1"
    module.SaT = _FakeSaT  # type: ignore[attr-defined]
    return module


def test_sat_3l_segmenter_uses_explicit_cpu_and_pinned_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())

    segmenter = SaT3lSegmenter(
        cache_dir=tmp_path,
        revision="model-revision",
        inference_batch_size=7,
        ort_providers=("CPUExecutionProvider",),
    )
    result = segmenter.split(["First. Second."], language="en")

    assert segmenter.model_id == "segment-any-text/sat-3l-sm"
    assert segmenter.version == "2.2.1"
    assert segmenter.revision == "model-revision"
    assert _FakeSaT.init_args == ("segment-any-text/sat-3l-sm",)
    assert _FakeSaT.init_kwargs == {
        "hub_prefix": None,
        "ort_providers": ["CPUExecutionProvider"],
        "from_pretrained_kwargs": {
            "cache_dir": str(tmp_path),
            "revision": "model-revision",
        },
    }
    assert _FakeSaT.split_args == (["First. Second."],)
    assert _FakeSaT.split_kwargs == {
        "strip_whitespace": False,
        "split_on_input_newlines": False,
        "batch_size": 7,
        "outer_batch_size": 1,
    }
    assert result == [["First. ", "Second."]]


def test_sat_3l_segmenter_requires_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wtpsplit", None)

    with pytest.raises(RuntimeError, match="uv sync --extra sentence-splitting"):
        SaT3lSegmenter(cache_dir=tmp_path, revision="model-revision")


def test_sat_3l_segmenter_prefers_coreml_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())
    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    SaT3lSegmenter(cache_dir=tmp_path, revision="model-revision")

    assert _FakeSaT.init_kwargs is not None
    assert _FakeSaT.init_kwargs["ort_providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_gpu_mode_requires_cuda_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())
    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        SaT3lSegmenter(
            cache_dir=tmp_path,
            revision="model-revision",
            require_gpu=True,
        )


def test_gpu_mode_passes_cuda_before_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())
    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    segmenter = SaT3lSegmenter(
        cache_dir=tmp_path,
        revision="model-revision",
        require_gpu=True,
    )

    assert segmenter.ort_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert _FakeSaT.init_kwargs is not None
    assert _FakeSaT.init_kwargs["ort_providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_gpu_mode_rejects_a_cpu_fallback_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())
    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(_FakeSaT, "effective_providers", ("CPUExecutionProvider",))

    with pytest.raises(RuntimeError, match="active CUDAExecutionProvider"):
        SaT3lSegmenter(
            cache_dir=tmp_path,
            revision="model-revision",
            require_gpu=True,
        )
