import pytest

from scripts.quality import scope_manifest
from scripts.quality.scope_manifest import NER_SCOPE, QualityScope, render_scope_paths


def test_render_scope_paths_produces_shell_safe_space_separated_paths() -> None:
    assert render_scope_paths(NER_SCOPE, "source") == " ".join(NER_SCOPE.source_paths)
    assert render_scope_paths(NER_SCOPE, "test") == " ".join(NER_SCOPE.test_paths)


def test_main_validates_the_manifest_before_rendering(monkeypatch) -> None:
    monkeypatch.setattr(
        scope_manifest,
        "SCOPES",
        (QualityScope("broken", ("src/missing.py",), ("tests/missing.py",)),),
    )

    with pytest.raises(ValueError, match="does not exist"):
        scope_manifest.main(("--scope", "broken", "--kind", "source"))
