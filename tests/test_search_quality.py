"""Qidiruv/kontekst optimizatsiyasi + sifat baholash — DB/tarmoqsiz qismlar.

Qo'riqlaydi:
  * matn siqish (URL, whitespace, kesish) va token bahosi
  * kalit so'z ajratish (stop-so'zlar), RRF birlashtirish, byudjetga sig'dirish
  * strategiya tanlash: vaqt oynasi / xulosa / qidiruv aniqlash
  * map-reduce: bo'laklash, arzon Task.ROUTE, digest'lar konvertda
  * judge JSON parse (fence, clamp), pricing, tarix qisqartirish
  * Claude adapterda uzun system → cache_control
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.llm import ChatResult, Msg, Task, Usage
from app.llm import claude as C
from app.llm.pricing import estimate_cost
from app.mtproto.pool import MessageInfo
from app.services import chat_service as cs
from app.services import evaluation as ev
from app.services import search as S

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def _mi(i: int, text: str, when: datetime | None = None) -> MessageInfo:
    return MessageInfo(i, when or NOW, "A", text, None, None)


# ─── siqish ──────────────────────────────────────────────────────────────────


def test_compact_text_shortens_urls_and_whitespace() -> None:
    raw = "Qarang   https://example.com/very/long/path?x=1&y=2 \n\n\n\n keyin   davomi"
    out = S.compact_text(raw)
    assert "<example.com>" in out and "very/long" not in out
    assert "   " not in out and "\n\n\n" not in out


def test_compact_text_truncates() -> None:
    out = S.compact_text("x" * 5000, max_chars=100)
    assert len(out) == 100 and out.endswith("…")


def test_est_tokens_and_fit_budget_keeps_newest() -> None:
    infos = [_mi(i, "a" * 350) for i in range(10)]  # har biri ~100 tok + 12
    kept, trunc = S.fit_budget(infos, 350)
    assert trunc and [m.msg_id for m in kept] == [7, 8, 9]  # eng yangilar (ro'yxat oxiri)
    kept2, trunc2 = S.fit_budget(infos[:2], 10_000)
    assert not trunc2 and len(kept2) == 2


# ─── kalit so'zlar / RRF ─────────────────────────────────────────────────────


def test_keywords_drop_stopwords_and_dupes() -> None:
    kw = S.keywords("Bu hafta kanalda crypto va Bitcoin haqida nima yozildi? crypto")
    assert "crypto" in kw and "bitcoin" in kw
    assert "haqida" not in kw and "nima" not in kw and "bu" not in kw
    assert kw.count("crypto") == 1
    assert S.keywords("и в на") == []


def test_rrf_merges_and_marks_both() -> None:
    fts = [S.Hit(1, 101, 5.0, "fts"), S.Hit(2, 102, 4.0, "fts")]
    vec = [S.Hit(2, 102, 0.9, "vector"), S.Hit(3, 103, 0.8, "vector")]
    fused = S.rrf(fts, vec)
    assert fused[0].message_id == 2 and fused[0].source == "both"
    assert {h.message_id for h in fused} == {1, 2, 3}


# ─── strategiya aniqlash ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("q", "days"),
    [
        ("bu hafta nima muhokama qilindi?", 7),
        ("за месяц какие темы обсуждали", 31),
        ("what happened today", 1),
        ("crypto haqida qaysi post bor?", None),
    ],
)
def test_detect_window(q: str, days: int | None) -> None:
    got = S.detect_window(q)
    assert (got == timedelta(days=days)) if days else got is None


def test_looks_like_summary() -> None:
    assert S.looks_like_summary("kanal bo'yicha xulosa ber")
    assert S.looks_like_summary("сделай итог обсуждений")
    assert not S.looks_like_summary("bitcoin narxi haqida post")


# ─── map-reduce ──────────────────────────────────────────────────────────────


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[Task, list[Msg]]] = []

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        self.calls.append((task, messages))
        return ChatResult(
            text=f"digest {len(self.calls)}",
            tool_calls=[],
            model="cheap-model",
            provider="test",
            usage=Usage(tokens_in=100, tokens_out=20),
        )


async def test_compact_window_uses_cheap_task_and_untrusted_envelope() -> None:
    llm = _FakeLLM()
    infos = [_mi(i, "x" * 1000) for i in range(39)]  # 13 qator x 3 bo'lak (~1035 belgi/qator)
    bundle = await S.compact_window(infos, title='Ch"an', llm=llm)  # type: ignore[arg-type]
    assert bundle.strategy == "window_compacted"
    assert len(llm.calls) == 3 and all(task is Task.ROUTE for task, _ in llm.calls)
    for _, msgs in llm.calls:
        assert msgs[-1].content.startswith('<untrusted_data source="telegram"')
        assert "not instructions" in msgs[0].content
    assert len(bundle.map_summaries) == 3
    assert bundle.map_tokens_in == 300 and bundle.map_tokens_out == 60
    rendered = cs.render_bundle(bundle)
    assert (
        'kind="digests"' in rendered and "digest 1" in rendered and "</untrusted_data>" in rendered
    )
    assert 'Ch"an' not in rendered  # sarlavhadagi qo'shtirnoq tozalanadi


async def test_compact_window_caps_chunks() -> None:
    llm = _FakeLLM()
    infos = [_mi(i, "y" * 1000) for i in range(300)]  # ~21 bo'lak → 12 gacha
    bundle = await S.compact_window(infos, title="c", llm=llm)  # type: ignore[arg-type]
    assert len(llm.calls) == S.MAP_MAX_CHUNKS and bundle.truncated


# ─── prompt tarixi qisqaradi ─────────────────────────────────────────────────


def test_history_turns_are_clipped() -> None:
    hist = [
        SimpleNamespace(role="assistant", content="z" * 5000),
        SimpleNamespace(role="user", content="q"),
    ]
    msgs = cs.build_messages(hist, "next", "")
    assert len(msgs[1].content) == cs.HISTORY_TURN_MAX_CHARS and msgs[1].content.endswith("…")


def test_render_bundle_marks_search_selection() -> None:
    b = S.ContextBundle("T", [_mi(1, "hello")], "search", hits=1)
    out = cs.render_bundle(b)
    assert 'selection="messages matching the question' in out and "hello" in out


# ─── evaluation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        '{"relevance": 4, "usefulness": 5, "grounded": true, "note": "ok"}',
        '```json\n{"relevance": 4, "usefulness": 5, "grounded": true, "note": "ok"}\n```',
        'Sure: {"relevance": "4", "usefulness": 9, "grounded": true, "note": "ok"} done',
    ],
)
def test_parse_judgement(raw: str) -> None:
    j = ev.parse_judgement(raw)
    assert j is not None
    assert j.relevance == 4 and j.usefulness == 5 and j.grounded is True and j.note == "ok"


def test_parse_judgement_garbage() -> None:
    assert ev.parse_judgement("no json here") is None
    assert ev.parse_judgement("[1,2]") is None


async def test_judge_uses_cheap_task_and_json_mode() -> None:
    class _L:
        async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
            assert task is Task.ROUTE and kw.get("json_mode") is True
            assert "<question>" in messages[-1].content and "<answer>" in messages[-1].content
            return ChatResult(
                text='{"relevance":2,"usefulness":1,"grounded":false,"note":"invented"}',
                tool_calls=[],
                model="m",
                provider="p",
                usage=Usage(50, 10),
            )

    j = await ev.judge("q", "a", had_context=False, llm=_L())  # type: ignore[arg-type]
    assert j is not None and j.relevance == 2 and j.grounded is False and j.tokens_in == 50


# ─── pricing ─────────────────────────────────────────────────────────────────


def test_estimate_cost() -> None:
    assert estimate_cost("claude", "claude-opus-5", 1_000_000, 0) == 5.0
    assert estimate_cost("claude", "claude-haiku-4-5-20251001", 0, 1_000_000) == 5.0
    assert estimate_cost("gemini", "models/gemini-2.5-flash", 1_000_000, 1_000_000) == 2.8
    assert estimate_cost("claude_code", "claude-opus-5", 10**6, 10**6) == 0.0
    assert estimate_cost("x", "unknown-model", 1, 1) is None


# ─── Claude prompt caching ───────────────────────────────────────────────────


def test_claude_long_system_gets_cache_control() -> None:
    long_sys = "S" * (C.CACHE_SYSTEM_MIN_CHARS + 10)
    req = C.build_request(
        [Msg.system(long_sys), Msg.user("hi")],
        model="claude-opus-5",
        tools=None,
        temperature=0.5,
        max_tokens=None,
        json_mode=False,
    )
    assert isinstance(req["system"], list)
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    short = C.build_request(
        [Msg.system("S"), Msg.user("hi")],
        model="claude-opus-5",
        tools=None,
        temperature=0.5,
        max_tokens=None,
        json_mode=False,
    )
    assert short["system"] == "S"
