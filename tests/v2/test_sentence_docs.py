from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def test_sentence_documentation_freezes_routing_scope_and_unsupported_policy() -> None:
    documentation = (REPOSITORY / "docs/sentence-splitting.md").read_text(encoding="utf-8")

    supported = (
        "af am ar az be bg bn ca ceb cs cy da de el en eo es et eu fa fi fr fy ga gd gl gu ha "
        "he hi hu hy id ig is it ja jv ka kk km kn ko ku ky la lt lv mg mk ml mn mr ms mt my "
        "ne nl no pa pl ps pt ro ru si sk sl sq sr sv ta te tg th tr uk ur uz vi xh yi yo zh zu"
    )
    assert supported in documentation
    assert "segment-any-text/sat-3l-sm" in documentation
    assert "Only these exact language codes are sent to SaT" in documentation
    assert "one unsplit row" in documentation
    assert "segmentation_status=unsupported_language" in documentation
    assert "zh-hans" in documentation
    assert "sentence_splitting.json" in documentation
    assert "uv sync --extra sentence-splitting" in documentation


def test_grid5000_documentation_freezes_gpu_controller_contract() -> None:
    documentation = (REPOSITORY / "docs/grid5000-sentence-splitting.md").read_text(encoding="utf-8")

    for required in (
        "scripts/grid5000_sentence_controller.py",
        "host=1/gpu=1",
        "0:30",
        "usagepolicycheck -t",
        "HF token",
        "CUDA",
        "grid5000_sentence_run.json",
        "one unsplit row",
        "every successful GPU job",
        "resumable",
    ):
        assert required in documentation
