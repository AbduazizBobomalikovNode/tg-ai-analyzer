"""Gemini: vazifa bo'yicha fikrlash darajasi va Flex tier (docs/cost-research.md, C8/C2)."""

from __future__ import annotations

from typing import Any

import pytest
from google.genai import types

from app.config import Settings, get_settings
from app.llm import ChatResult, Msg, Usage
from app.llm import router as R
from app.llm.gemini import thinking_config


class _CapturingProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    async def chat(self, messages: list[Msg], **kw: Any) -> ChatResult:
        self.kwargs.append(kw)
        return ChatResult(
            text="ok",
            tool_calls=[],
            model=kw["model"],
            provider="gemini",
            usage=Usage(tokens_in=10, tokens_out=2),
            service_tier=str(kw.get("service_tier") or ""),
        )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> _CapturingProvider:
    prov = _CapturingProvider()
    monkeypatch.setattr(R, "resolve", lambda task: (prov, "gemini-3.5-flash-lite"))
    return prov


# ─── fikrlash darajasi ───────────────────────────────────────────────────────


def test_default_levels_per_task() -> None:
    assert R.thinking_level(R.Task.ROUTE) == "minimal"
    assert R.thinking_level(R.Task.SEARCH) == "low"
    assert R.thinking_level(R.Task.TOOLS) == "low"
    assert R.thinking_level(R.Task.DEEP) == "low"
    assert R.thinking_level(R.Task.EMBED) is None


def test_level_is_configurable_per_task_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_THINKING_LEVEL_DEEP", "high")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL_ROUTE", "")
    s = Settings()  # type: ignore[call-arg]
    assert s.gemini_thinking_level_deep == "high"
    assert s.gemini_thinking_level_route == ""
    monkeypatch.setenv("GEMINI_THINKING_LEVEL_TOOLS", "extreme")
    with pytest.raises(ValueError):
        Settings()  # type: ignore[call-arg]


def test_thinking_config_uses_the_task_level() -> None:
    cfg = thinking_config("gemini-3.5-flash-lite", budget=1024, level="minimal")
    assert cfg is not None and cfg.thinking_level == types.ThinkingLevel.MINIMAL
    assert cfg.include_thoughts is False
    high = thinking_config("gemini-3.5-flash", budget=1024, level="high")
    assert high is not None and high.thinking_level == types.ThinkingLevel.HIGH


def test_minimal_falls_back_to_low_where_the_model_does_not_support_it() -> None:
    for model in ("gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-flash-latest"):
        cfg = thinking_config(model, budget=1024, level="minimal")
        assert cfg is not None and cfg.thinking_level == types.ThinkingLevel.LOW, model


def test_empty_level_or_zero_budget_sends_nothing() -> None:
    assert thinking_config("gemini-3.5-flash", budget=1024, level="") is None
    assert thinking_config("gemini-3.5-flash", budget=0, level="high") is None
    # `level=None` — eski yo'l (byudjetdan)
    old = thinking_config("gemini-3.5-flash", budget=1024)
    assert old is not None and old.thinking_level == types.ThinkingLevel.LOW
    # 2.5 avlod byudjet bilan, darajaga qaramaydi
    b = thinking_config("gemini-2.5-flash", budget=512, level="minimal")
    assert b is not None and b.thinking_budget == 512


async def test_facade_passes_the_task_level_to_the_provider(
    captured: _CapturingProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "gemini_thinking_level_deep", "high")
    await R.LLM().chat(R.Task.ROUTE, [Msg.user("x")])
    await R.LLM().chat(R.Task.DEEP, [Msg.user("x")])
    assert [k["thinking_level"] for k in captured.kwargs] == ["minimal", "high"]


async def test_gemini_adapter_sends_the_level_in_the_request() -> None:
    from types import SimpleNamespace

    from app.llm.gemini import GeminiProvider

    seen: list[Any] = []

    class _Models:
        async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            seen.append(config)
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[types.Part(text="ok")]),
                        finish_reason="STOP",
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            )

    p = GeminiProvider(api_key="test-key")
    p._client = SimpleNamespace(aio=SimpleNamespace(models=_Models()))
    await p.chat([Msg.user("x")], model="gemini-3.5-flash-lite", thinking_level="minimal")
    assert seen[0].thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


# ─── Flex tier (fon ishlari) ─────────────────────────────────────────────────


def test_background_tier_follows_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert R.background_tier() == "flex"  # default: LLM_FLEX_BACKGROUND=true
    monkeypatch.setattr(get_settings(), "llm_flex_background", False)
    assert R.background_tier() == "standard"


async def test_facade_sends_flex_only_when_asked(captured: _CapturingProvider) -> None:
    await R.LLM().chat(R.Task.SEARCH, [Msg.user("x")])
    await R.LLM().chat(R.Task.ROUTE, [Msg.user("x")], tier="flex")
    assert [k["service_tier"] for k in captured.kwargs] == [None, "flex"]


def test_installed_sdk_supports_flex() -> None:
    from app.llm.gemini import flex_supported

    assert flex_supported() is True


def _gemini_with(responses: list[Any]) -> tuple[Any, list[Any]]:
    from types import SimpleNamespace

    from app.llm.gemini import GeminiProvider

    seen: list[Any] = []

    class _Models:
        async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            seen.append(config)
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    p = GeminiProvider(api_key="test-key")
    p.max_retries = 1
    p._client = SimpleNamespace(aio=SimpleNamespace(models=_Models()))
    return p, seen


def _ok() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[types.Part(text="ok")]), finish_reason="STOP"
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )


async def test_gemini_adapter_requests_flex_and_marks_the_result() -> None:
    p, seen = _gemini_with([_ok()])
    res = await p.chat([Msg.user("x")], model="gemini-3.5-flash-lite", service_tier="flex")
    assert seen[0].service_tier == types.ServiceTier.FLEX
    assert res.service_tier == "flex"
    std, seen2 = _gemini_with([_ok()])
    res2 = await std.chat([Msg.user("x")], model="gemini-3.5-flash-lite")
    assert seen2[0].service_tier is None and res2.service_tier == ""


async def test_flex_failure_falls_back_to_standard_once() -> None:
    class _Busy(Exception):
        code = 503

    p, seen = _gemini_with([_Busy("flex capacity"), _ok()])
    res = await p.chat([Msg.user("x")], model="gemini-3.5-flash-lite", service_tier="flex")
    assert [c.service_tier for c in seen] == [types.ServiceTier.FLEX, None]
    assert res.service_tier == "standard" and res.text == "ok"


class _TierLLM:
    def __init__(self, text: str = '{"relevance":5,"usefulness":5,"grounded":5,"note":"ok"}'):
        self.tiers: list[str] = []
        self.text = text

    async def chat(self, task: Any, messages: list[Any], **kw: Any) -> ChatResult:
        self.tiers.append(kw.get("tier", "standard"))
        return ChatResult(
            text=self.text,
            tool_calls=[],
            model="gemini-3.5-flash-lite",
            provider="gemini",
            usage=Usage(tokens_in=100, tokens_out=10),
            service_tier=kw.get("tier", ""),
        )


async def test_judge_runs_on_flex_and_is_priced_as_flex() -> None:
    from app.services import evaluation as E

    llm = _TierLLM()
    j = await E.judge("savol", "javob", had_context=True, llm=llm)  # type: ignore[arg-type]
    assert llm.tiers == ["flex"]
    assert j is not None and j.tier == "flex"


async def test_week_node_and_digest_use_the_background_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    from app.services import digests as D

    llm = _TierLLM(text="digest")
    await D._summarize(
        ["a: salom"],
        title="t",
        day=datetime(2026, 9, 20, tzinfo=UTC),
        llm=llm,  # type: ignore[arg-type]
        tier=R.background_tier(),
    )
    # Savol paytidagi (interaktiv) digest — standart
    await D._summarize(
        ["a: salom"],
        title="t",
        day=datetime(2026, 9, 20, tzinfo=UTC),
        llm=llm,  # type: ignore[arg-type]
    )
    assert llm.tiers == ["flex", "standard"]


def test_background_callers_pass_the_background_tier() -> None:
    """Fon yo'llari tier'ni `background_tier()` dan oladi (grep-qo'riqchi)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "app"
    for rel, n in (
        ("services/nodes.py", 1),
        ("services/briefing.py", 2),
        ("services/evaluation.py", 1),
        ("worker/tasks.py", 1),
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert text.count("tier=background_tier()") == n, rel
    # Interaktiv yo'llar tier bermaydi
    for rel in ("services/agent.py", "services/deep_search.py", "services/chat_service.py"):
        assert "background_tier" not in (root / rel).read_text(encoding="utf-8"), rel
