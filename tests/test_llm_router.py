"""Router testlari — Claude/Gemini/DeepSeek imkoniyat farqini to'g'ri boshqarishi shart."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from app.config import get_settings
from app.llm import claude as C
from app.llm import router as R
from app.llm.base import Capability, LLMError
from app.llm.deepseek import DeepSeekProvider
from app.llm.gemini import GeminiProvider

_CLAUDE_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CODE_BIN",
    "CLAUDE_CODE_AUTO",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """Har test uchun sozlama keshini va provider instansiyalarini tozalaydi.

    Claude kredensiali testlarda **yo'q** deb boshlanadi — mashinadagi haqiqiy
    profil/env natijaga ta'sir qilmasligi uchun. Kerak bo'lsa `with_claude()`.
    """
    for key in (
        "LLM_PROVIDER",
        "LLM_FALLBACK_PROVIDER",
        "LLM_TASK_ROUTE",
        "LLM_TASK_SEARCH",
        "LLM_TASK_TOOLS",
        "LLM_TASK_DEEP",
        "LLM_TASK_EMBED",
        "LLM_TASK_IMAGE",
        *_CLAUDE_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    # bo'sh config dir → SDK profil topmaydi
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-anthropic-profile"))
    # mavjud bo'lmagan CLI → claude_code aniqlanmaydi (mashinadagi haqiqiy `claude` chetda)
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(tmp_path / "no-claude"))
    get_settings.cache_clear()
    R.reset()
    yield
    get_settings.cache_clear()
    R.reset()


def use(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    R.reset()


def with_claude(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    """Claude Code tokeni bor holatni simulyatsiya qiladi."""
    use(monkeypatch, CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-test", **env)  # noqa: S106


def fake_claude_cli(tmp_path, *, logged_in: bool = True) -> str:
    """`claude auth status` ga JSON qaytaradigan soxta CLI skripti."""
    script = tmp_path / "claude"
    payload = json.dumps({"loggedIn": logged_in, "authMethod": "claude.ai"})
    script.write_text(f"#!/bin/sh\nif [ \"$1\" = auth ]; then echo '{payload}'; fi\n")
    script.chmod(0o755)
    return str(script)


def with_claude_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path, *, logged_in: bool = True, **env: str
) -> None:
    use(monkeypatch, CLAUDE_CODE_BIN=fake_claude_cli(tmp_path, logged_in=logged_in), **env)


# ─── imkoniyat matritsasi ────────────────────────────────────────────────────


def test_deepseek_has_no_embedding_or_image() -> None:
    caps = DeepSeekProvider().capabilities("deepseek-chat")
    assert Capability.EMBED not in caps
    assert Capability.IMAGE_GEN not in caps
    assert Capability.LONG_CONTEXT not in caps


def test_deepseek_reasoner_has_no_tools() -> None:
    """R1 avlodda function calling yo'q — agent sikli unda ishlamaydi."""
    p = DeepSeekProvider()
    assert Capability.TOOLS in p.capabilities("deepseek-chat")
    assert Capability.TOOLS not in p.capabilities("deepseek-reasoner")


def test_gemini_is_fully_capable() -> None:
    g = GeminiProvider()
    chat = g.capabilities("gemini-2.5-pro")
    assert {Capability.CHAT, Capability.TOOLS, Capability.JSON_MODE} <= chat
    assert Capability.LONG_CONTEXT in chat
    assert Capability.EMBED in g.capabilities("gemini-embedding-001")
    assert Capability.IMAGE_GEN in g.capabilities("gemini-2.5-flash-image")


# ─── marshrutizatsiya ────────────────────────────────────────────────────────


def test_claude_has_no_embedding_or_image() -> None:
    caps = C.ClaudeProvider(auth=C.ClaudeAuth("api_key", "t", "k")).capabilities("claude-opus-5")
    assert {Capability.CHAT, Capability.TOOLS, Capability.JSON_MODE} <= caps
    assert Capability.EMBED not in caps
    assert Capability.IMAGE_GEN not in caps


# ─── auto: Claude bor bo'lsa u, bo'lmasa Gemini ──────────────────────────────


def test_auto_without_claude_creds_is_gemini_everywhere() -> None:
    assert R.auto_provider() == "gemini"
    for task in R.Task:
        provider, model = R.resolve(task)
        assert provider.name == "gemini"
        assert model


def test_auto_with_claude_code_token_prefers_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    with_claude(monkeypatch)
    assert R.auto_provider() == "claude"
    for task in (R.Task.ROUTE, R.Task.SEARCH, R.Task.TOOLS, R.Task.DEEP):
        provider, model = R.resolve(task)
        assert provider.name == "claude", task
        assert model.startswith("claude")
    # kredensial secret'i log/repr'ga chiqmasin
    auth = C.detect_claude_auth()
    assert auth is not None and auth.source == "CLAUDE_CODE_OAUTH_TOKEN"
    assert "sk-ant" not in repr(auth)


def test_auto_claude_still_uses_gemini_for_embed_and_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude'da embedding/rasm yo'q — auto rejimda ham Gemini'ga tushishi kerak."""
    with_claude(monkeypatch)
    provider, model = R.resolve(R.Task.EMBED)
    assert (provider.name, model) == ("gemini", "gemini-embedding-001")
    provider, _ = R.resolve(R.Task.IMAGE)
    assert provider.name == "gemini"


# ─── auto: claude_code (CLI login) ───────────────────────────────────────────


def test_auto_prefers_claude_code_when_cli_logged_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    with_claude_cli(monkeypatch, tmp_path)
    assert R.auto_provider() == "claude_code"
    for task in (R.Task.ROUTE, R.Task.SEARCH, R.Task.DEEP):
        provider, model = R.resolve(task)
        assert provider.name == "claude_code", task
        assert model.startswith("claude")


def test_auto_claude_code_tools_and_embed_fall_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """CLI provider'da function calling yo'q — TOOLS/EMBED/IMAGE Gemini'ga tushadi."""
    with_claude_cli(monkeypatch, tmp_path)
    for task in (R.Task.TOOLS, R.Task.EMBED, R.Task.IMAGE):
        assert R.resolve(task)[0].name == "gemini", task


def test_auto_api_token_beats_cli(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    with_claude_cli(monkeypatch, tmp_path, CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-test")  # noqa: S106
    assert R.auto_provider() == "claude"


def test_auto_cli_not_logged_in_is_gemini(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    with_claude_cli(monkeypatch, tmp_path, logged_in=False)
    assert R.auto_provider() == "gemini"


def test_auto_cli_disabled_by_flag(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    with_claude_cli(monkeypatch, tmp_path, CLAUDE_CODE_AUTO="false")
    assert R.auto_provider() == "gemini"


def test_explicit_claude_code_without_cli_fails_loudly_on_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(monkeypatch, LLM_PROVIDER="claude_code")
    provider, model = R.resolve(R.Task.DEEP)
    assert provider.name == "claude_code"
    import asyncio

    from app.llm.base import Msg

    with pytest.raises(LLMError, match="CLI topilmadi"):
        asyncio.run(provider.chat([Msg.user("hi")], model=model))


def test_explicit_gemini_ignores_claude_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eski xatti-harakat saqlanadi: LLM_PROVIDER=gemini aniq berilsa Claude tanlanmaydi."""
    with_claude(monkeypatch, LLM_PROVIDER="gemini")
    for task in R.Task:
        assert R.resolve(task)[0].name == "gemini"


def test_explicit_claude_without_creds_fails_loudly_on_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use(monkeypatch, LLM_PROVIDER="claude")
    provider, model = R.resolve(R.Task.DEEP)
    assert provider.name == "claude"
    import asyncio

    from app.llm.base import Msg

    with pytest.raises(LLMError, match="kredensial topilmadi"):
        asyncio.run(provider.chat([Msg.user("hi")], model=model))


def test_api_key_beats_oauth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    with_claude(monkeypatch, ANTHROPIC_API_KEY="sk-ant-api-test")
    auth = C.detect_claude_auth()
    assert auth is not None and auth.kind == "api_key"


def test_unknown_llm_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, LLM_PROVIDER="openai")
    with pytest.raises(LLMError, match="LLM_PROVIDER noma'lum"):
        R.resolve(R.Task.DEEP)


def test_deepseek_as_default_still_uses_gemini_for_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_PROVIDER=deepseek bo'lsa ham embedding Gemini'ga tushishi kerak."""
    use(monkeypatch, LLM_PROVIDER="deepseek")

    provider, model = R.resolve(R.Task.TOOLS)
    assert (provider.name, model) == ("deepseek", "deepseek-chat")

    provider, model = R.resolve(R.Task.EMBED)
    assert provider.name == "gemini"
    assert model == "gemini-embedding-001"

    provider, _ = R.resolve(R.Task.IMAGE)
    assert provider.name == "gemini"


def test_per_task_override(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, LLM_TASK_DEEP="deepseek:deepseek-reasoner")
    provider, model = R.resolve(R.Task.DEEP)
    assert (provider.name, model) == ("deepseek", "deepseek-reasoner")
    # qolgani o'zgarmaydi
    assert R.resolve(R.Task.TOOLS)[0].name == "gemini"


def test_provider_only_spec_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, LLM_TASK_SEARCH="deepseek")
    provider, model = R.resolve(R.Task.SEARCH)
    assert (provider.name, model) == ("deepseek", "deepseek-chat")


def test_reasoner_for_tools_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reasoner'da function calling yo'q — jimgina sinmasdan Gemini'ga o'tsin."""
    use(monkeypatch, LLM_TASK_TOOLS="deepseek:deepseek-reasoner")
    provider, model = R.resolve(R.Task.TOOLS)
    assert provider.name == "gemini"
    assert Capability.TOOLS in provider.capabilities(model)


def test_unknown_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, LLM_TASK_DEEP="openai:gpt-4")
    with pytest.raises(LLMError, match="noma'lum provider"):
        R.resolve(R.Task.DEEP)


def test_no_capable_provider_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback ham qila olmasa — aniq xato, jimgina ishlamay qolish emas."""
    use(monkeypatch, LLM_PROVIDER="deepseek", LLM_FALLBACK_PROVIDER="deepseek")
    with pytest.raises(LLMError, match="vazifasini bajaradigan provider yo'q"):
        R.resolve(R.Task.EMBED)


def test_every_task_has_requirements() -> None:
    assert set(R.TASK_REQUIREMENTS) == set(R.Task)
