"""Direct behaviour of HF token resolution and repository authorization."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from osm_polygon_wikidata_only.hf._uploader import token as token_module
from osm_polygon_wikidata_only.hf._uploader.authorization import verify_repo_authorization
from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
from osm_polygon_wikidata_only.hf._uploader.token import resolve_hf_token, verify_hf_token


@pytest.fixture
def no_saved_token(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "hf-home" / "token"))


def test_explicit_token_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert resolve_hf_token("hf_explicit") == "hf_explicit"


def test_environment_token_is_used_without_explicit_value(
    no_saved_token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_env_override")
    assert resolve_hf_token(None) == "hf_env_override"


def test_missing_token_resolves_to_none(no_saved_token: None) -> None:
    assert resolve_hf_token(None) is None
    assert resolve_hf_token("") is None


def test_failing_token_backend_means_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def broken() -> str:
        raise RuntimeError("keyring exploded")

    monkeypatch.setattr(huggingface_hub, "get_token", broken)
    assert resolve_hf_token(None) is None


def test_non_string_backend_token_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: 123)
    assert resolve_hf_token(None) is None


def test_verify_returns_none_without_token(no_saved_token: None) -> None:
    def never(_token: str) -> dict[str, str]:
        raise AssertionError("whoami must not be called without a token")

    assert verify_hf_token(None, _whoami=never) is None


def test_verify_returns_verified_username() -> None:
    seen: list[str] = []

    def whoami(token: str) -> dict[str, str]:
        seen.append(token)
        return {"name": "alice"}

    assert verify_hf_token("hf_x", _whoami=whoami) == "alice"
    assert seen == ["hf_x"]


@pytest.mark.parametrize("info", [{}, {"name": ""}, "not-a-mapping"])
def test_verify_reports_unknown_for_nameless_identity(info: object) -> None:
    assert verify_hf_token("hf_x", _whoami=lambda _token: info) == "unknown"


def test_invalid_token_raises_actionable_upload_error() -> None:
    def rejected(_token: str) -> dict[str, str]:
        raise PermissionError("401 Invalid credentials")

    with pytest.raises(UploadError, match="rejected HF_TOKEN: 401 Invalid credentials") as info:
        verify_hf_token("hf_bad", _whoami=rejected)
    assert isinstance(info.value.__cause__, PermissionError)
    assert "settings/tokens" in str(info.value)


def test_default_whoami_client_uses_hf_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    class FakeApi:
        def __init__(self, token: str) -> None:
            self.token = token

        def whoami(self) -> dict[str, str]:
            return {"name": f"user-for-{self.token}"}

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert token_module._whoami_client(None)("hf_t") == {"name": "user-for-hf_t"}


def test_authorization_accepts_matching_namespace() -> None:
    assert verify_repo_authorization("hf_x", "alice/data", _verify=lambda _t: "alice") == "alice"


def test_authorization_rejects_repo_id_without_namespace() -> None:
    with pytest.raises(UploadError, match="namespace/name"):
        verify_repo_authorization("hf_x", "data", _verify=lambda _t: "alice")


def test_authorization_requires_a_token() -> None:
    with pytest.raises(UploadError, match="No Hugging Face token available"):
        verify_repo_authorization(None, "alice/data", _verify=lambda _t: None)


def test_authorization_propagates_token_rejection() -> None:
    def rejected(_token: str | None) -> str:
        raise UploadError("rejected")

    with pytest.raises(UploadError, match=r"^rejected$"):
        verify_repo_authorization("hf_x", "alice/data", _verify=rejected)


def test_authorization_rejects_foreign_namespace() -> None:
    with pytest.raises(UploadError, match=r"authenticates as 'bob'.*'alice' namespace"):
        verify_repo_authorization("hf_x", "alice/data", _verify=lambda _t: "bob")


def test_authorization_defaults_to_live_token_verification(
    no_saved_token: None,
) -> None:
    with pytest.raises(UploadError, match="No Hugging Face token available"):
        verify_repo_authorization(None, "alice/data")


def test_hf_token_loader_handles_token_and_backend_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = ModuleType("huggingface_hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    setattr(hub, "get_token", lambda: "token")
    assert token_module._load_hf_token() == "token"
    setattr(hub, "get_token", lambda: "")
    assert token_module._load_hf_token() is None

    def broken_backend() -> str:
        raise RuntimeError("cache unavailable")

    setattr(hub, "get_token", broken_backend)
    assert token_module._load_hf_token() is None
