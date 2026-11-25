"""LLM hisobi: thinking va kesh tokenlari (docs/cost-research.md, D1-band).

Nima qo'riqlanadi:

  * Gemini adapter `thoughts_token_count` / `cached_content_token_count` ni
    o'qiydi va `ChatResult.tokens_thinking` / `tokens_cached` sifatida beradi;
  * `estimate_cost`: thinking **output** narxida, keshlangan input — kesh
    narxida (jadvalda bo'lsa), Flex — yarmi;
  * log maskalash token **sonlarini** yashirmaydi, sirlarni esa yashiradi;
  * `limits.daily_usage` thinking tokenlarini ham sanaydi (haqiqiy Postgres).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.llm import Msg
from app.llm import pricing as P

# ─── pricing ─────────────────────────────────────────────────────────────────


def test_thinking_tokens_are_priced_at_output_rate() -> None:
    base = P.estimate_cost("gemini", "gemini-3.5-flash", 1_000_000, 0)
    with_thinking = P.estimate_cost(
        "gemini", "gemini-3.5-flash", 1_000_000, 0, tokens_thinking=1_000_000
    )
    assert base == pytest.approx(1.50)
    assert with_thinking == pytest.approx(1.50 + 9.00)


def test_cached_tokens_use_the_cached_input_price() -> None:
    # 3.5-flash: input $1.50, kesh $0.15 (ai.google.dev pricing, 2026-09-25)
    cost = P.estimate_cost("gemini", "gemini-3.5-flash", 1_000_000, 0, tokens_cached=600_000)
    assert cost == pytest.approx(0.4 * 1.50 + 0.6 * 0.15)
    pro = P.estimate_cost("gemini", "gemini-3.1-pro-preview", 1_000_000, 0, tokens_cached=1_000_000)
    assert pro == pytest.approx(0.20)
    # Kesh tokeni input'dan oshib ketsa ham manfiy narx chiqmaydi
    clipped = P.estimate_cost("gemini", "gemini-3.8-flash", 100, 0, tokens_cached=10_000)
    assert clipped is not None and clipped >= 0


def test_cached_tokens_without_a_cached_price_fall_back_to_full_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 3.5-flash-lite: "Not available" — to'liq narx, bir marta log
    calls: list[dict[str, Any]] = []

    class _Log:
        def info(self, event: str, **kw: Any) -> None:
            calls.append({"event": event, **kw})

    monkeypatch.setattr(P, "_no_cached_warned", set())
    monkeypatch.setattr("app.logging.get_logger", lambda *_a, **_k: _Log())
    full = P.estimate_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, 0)
    cached = P.estimate_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, 0, tokens_cached=500_000)
    P.estimate_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, 0, tokens_cached=500_000)
    assert cached == full
    assert [c["event"] for c in calls] == ["pricing.no_cached_price"]


def test_flex_tier_halves_gemini_cost_only() -> None:
    std = P.estimate_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, 1_000_000)
    flex = P.estimate_cost("gemini", "gemini-3.5-flash-lite", 1_000_000, 1_000_000, tier="flex")
    assert std is not None and flex == pytest.approx(std / 2)
    # Boshqa provider'da tier e'tiborsiz
    claude = P.estimate_cost("claude", "claude-haiku-4-5", 1_000_000, 0, tier="flex")
    assert claude == pytest.approx(1.0)


def test_result_cost_reads_usage_and_tier() -> None:
    from app.llm import ChatResult, Usage

    res = ChatResult(
        text="",
        tool_calls=[],
        model="gemini-3.5-flash",
        provider="gemini",
        usage=Usage(tokens_in=1_000_000, tokens_out=0, cached_tokens=0, thinking_tokens=100_000),
        service_tier="flex",
    )
    assert res.tokens_thinking == 100_000
    assert res.tokens_cached == 0
    assert P.result_cost(res) == pytest.approx((1.50 + 0.9) / 2)


# ─── Gemini adapter ──────────────────────────────────────────────────────────


class _FakeModels:
    def __init__(self, resp: Any) -> None:
        self.resp = resp

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        return self.resp


async def test_gemini_adapter_reads_thinking_and_cached_tokens() -> None:
    from google.genai import types

    from app.llm.gemini import GeminiProvider

    resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[types.Part(text="javob")]), finish_reason="STOP"
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=5000,
            candidates_token_count=120,
            cached_content_token_count=4096,
            thoughts_token_count=333,
        ),
    )
    provider = GeminiProvider(api_key="test-key")
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=_FakeModels(resp)))
    res = await provider.chat([Msg.user("salom")], model="gemini-3.5-flash")
    assert res.usage.tokens_in == 5000
    assert res.usage.tokens_out == 120  # ko'rinadigan chiqish — thinking alohida
    assert res.tokens_thinking == 333
    assert res.tokens_cached == 4096


# ─── log maskalash ───────────────────────────────────────────────────────────


def test_token_counts_are_not_masked_but_secrets_still_are() -> None:
    from app.logging import MASK, _redact

    out = _redact(
        None,
        "info",
        {
            "event": "llm.chat",
            "tokens_in": 1200,
            "tokens_out": 80,
            "tokens_thinking": 40,
            "tokens_cached": 1024,
            "token": "abc",
            "bot_token": "123:xyz",
            "refresh_token": "r",
            "tokens": 5,  # aniq ro'yxatda yo'q — avvalgidek maskalanadi
        },
    )
    assert out["tokens_in"] == 1200
    assert out["tokens_out"] == 80
    assert out["tokens_thinking"] == 40
    assert out["tokens_cached"] == 1024
    assert out["token"] == MASK
    assert out["bot_token"] == MASK
    assert out["refresh_token"] == MASK
    assert out["tokens"] == MASK


# ─── byudjet: thinking ham sanaladi (haqiqiy Postgres) ───────────────────────


async def test_daily_usage_counts_thinking_tokens(pg_engine: Any) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from test_db_integration import _fixture_chat

    from app.services import limits as LIM

    account_id, _chat_id = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        user_id = (
            await conn.execute(
                text("SELECT user_id FROM accounts WHERE id = :a"), {"a": account_id}
            )
        ).scalar_one()
        maker = async_sessionmaker(pg_engine, expire_on_commit=False)
        async with maker() as session:
            before = await LIM.daily_usage(session, int(user_id))
        conv_id = (
            await conn.execute(
                text(
                    "INSERT INTO conversations (user_id, account_id, title) "
                    "VALUES (:u, :a, 'thinking') RETURNING id"
                ),
                {"u": user_id, "a": account_id},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO conversation_messages "
                "(conversation_id, role, content, tokens_in, tokens_out, "
                " tokens_thinking, tokens_cached) VALUES "
                "(:c, 'assistant', 'a', 10, 5, 1000, 8), "
                "(:c, 'assistant', 'b', 1, 1, NULL, NULL)"
            ),
            {"c": conv_id},
        )
    async with maker() as session:
        after = await LIM.daily_usage(session, int(user_id))
    assert after.tokens - before.tokens == 10 + 5 + 1000 + 1 + 1
