"""Router testlari — Gemini/DeepSeek imkoniyat farqini to'g'ri boshqarishi shart."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.config import get_settings
from app.llm import router as R
from app.llm.base import Capability, LLMError
from app.llm.deepseek import DeepSeekProvider
from app.llm.gemini import GeminiProvider


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Har test uchun sozlama keshini va provider instansiyalarini tozalaydi."""
    for key in (
        "LLM_PROVIDER",
        "LLM_FALLBACK_PROVIDER",
        "LLM_TASK_ROUTE",
        "LLM_TASK_SEARCH",
        "LLM_TASK_TOOLS",
        "LLM_TASK_DEEP",
        "LLM_TASK_EMBED",
        "LLM_TASK_IMAGE",
    ):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    R._instances.clear()
    yield
    get_settings.cache_clear()
    R._instances.clear()


def use(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    R._instances.clear()


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


def test_default_is_gemini_everywhere() -> None:
    for task in R.Task:
        provider, model = R.resolve(task)
        assert provider.name == "gemini"
        assert model


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
