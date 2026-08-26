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

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).init_args = args
        type(self).init_kwargs = kwargs

    def split(self, *args: object, **kwargs: object) -> list[list[str]]:
        type(self).split_args = args
        type(self).split_kwargs = kwargs
        return [["First. ", "Second."]]


def _fake_wtpsplit() -> ModuleType:
    module = ModuleType("wtpsplit")
    module.__version__ = "2.2.1"
    module.SaT = _FakeSaT  # type: ignore[attr-defined]
    return module


def test_sat_3l_segmenter_uses_cpu_and_pinned_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wtpsplit", _fake_wtpsplit())

    segmenter = SaT3lSegmenter(
        cache_dir=tmp_path,
        revision="model-revision",
        inference_batch_size=7,
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
