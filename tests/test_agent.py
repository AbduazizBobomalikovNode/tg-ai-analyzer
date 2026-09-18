"""Agent sikli, tool registry va rejim tanlash — DB/tarmoqsiz.

Qo'riqlaydi:
  * registry'da faqat o'qish tool'lari (delete/send/edit yo'q — invariant 1)
  * har natija <untrusted_data> konvertida; xato istisno emas, matn
  * sikl: tool chaqiruvi → natija → yakuniy javob; iteratsiya limiti; natija byudjeti;
    limitdan ortiq chaqiruvlar rad etiladi; audit (agent_actions) yoziladi
  * choose_mode heuristikasi
  * prompt'lar: system'da o'zgaruvchan sana yo'q (kesh), untrusted qoidasi bor
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.llm import ChatResult, Msg, Task, ToolCall, Usage
from app.services import agent as A
from app.services import chat_service as cs
from app.services import prompts as P
from app.services import tools as T

# ─── registry ────────────────────────────────────────────────────────────────


def test_registry_is_read_only_and_well_formed() -> None:
    names = list(T.READ_TOOLS)
    assert len(names) == len(set(names))
    for bad in ("delete", "send", "edit", "pin", "forward", "logout", "write"):
        assert not any(bad in n for n in names), names
    for spec in T.tool_specs():
        assert spec.name in T.READ_TOOLS
        assert spec.description and len(spec.description) > 20
        assert spec.parameters["type"] == "object"
        assert spec.parameters.get("additionalProperties") is False
        for prop in spec.parameters.get("properties", {}).values():
            assert "type" in prop


def test_envelope_and_clip() -> None:
    env = T.envelope("search_messages", T.ToolResult("hello", ok=True))
    assert env.startswith('<untrusted_data source="tool:search_messages" status="ok">')
    assert env.endswith("</untrusted_data>")
    assert 'status="error"' in T.envelope("x", T.ToolResult("e", ok=False))
    clipped = T._clip("x" * 20_000)
    assert len(clipped) < 9_100 and clipped.endswith("(truncated for budget)")


async def test_run_tool_unknown_and_exception_are_text() -> None:
    ctx = T.ToolContext(session=object(), account_id=1)  # type: ignore[arg-type]
    r = await T.run_tool("nope", ctx, {})
    assert not r.ok and "unknown tool" in r.text

    async def boom(ctx: Any, args: Any) -> T.ToolResult:
        raise RuntimeError("db down")

    T.READ_TOOLS["__boom__"] = T.Tool(T.tool_specs()[0], boom)
    try:
        r = await T.run_tool("__boom__", ctx, {})
        assert not r.ok and "RuntimeError" in r.text
    finally:
        del T.READ_TOOLS["__boom__"]


# ─── agent loop ──────────────────────────────────────────────────────────────


class _Session:
    def __init__(self, writable: bool = False, chats: list[Any] | None = None) -> None:
        self.added: list[Any] = []
        self.writable = writable
        self.chats = chats or []  # runtime_note ichidagi chat indeksi uchun

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = len(self.added)

    async def flush(self) -> None: ...

    async def execute(self, *a: Any, **k: Any) -> Any:
        # `writable_chats_exist` → .first(); `chat_index` → .scalars().all()
        return SimpleNamespace(
            first=lambda: (1,) if self.writable else None,
            scalars=lambda: SimpleNamespace(all=lambda: list(self.chats)),
        )


class _LLM:
    """Ssenariy: birinchi chaqiruvda tool so'raydi, keyin javob beradi."""

    def __init__(self, script: list[ChatResult]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        self.calls.append({"task": task, "messages": list(messages), "tools": kw.get("tools")})
        return self.script.pop(0)


def _res(
    text: str = "",
    calls: list[ToolCall] | None = None,
    *,
    signature: bytes | None = None,
) -> ChatResult:
    return ChatResult(
        text=text,
        tool_calls=calls or [],
        model="m",
        provider="p",
        usage=Usage(100, 20),
        finish_reason="tool_use" if calls else "end_turn",
        signature=signature,
    )


@pytest.fixture
def fake_tool(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def fake_run(name: str, ctx: Any, args: dict[str, Any]) -> T.ToolResult:
        seen.append({"name": name, "args": args})
        if name == "get_chat_stats":
            return T.ToolResult('{"posts": 12}', meta={"chat": "C", "posts": 12})
        return T.ToolResult("x" * 4000, meta={"chat": "C", "n": 10})

    monkeypatch.setattr(A.T, "run_tool", fake_run)
    return seen


async def test_agent_tool_then_answer(fake_tool: list[dict[str, Any]]) -> None:
    llm = _LLM(
        [
            _res(calls=[ToolCall("c1", "get_chat_stats", {"days": 7})]),
            _res("12 ta post."),
        ]
    )
    db = _Session()
    out = await A.run_agent(
        db,  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="bu hafta nechta post?",
        history=[],
        pinned_chat=SimpleNamespace(id=9, title="Chan"),  # type: ignore[arg-type]
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
    )
    assert out.text == "12 ta post." and out.iterations == 2
    assert out.tokens_in == 200 and out.tokens_out == 40
    assert [c["tool"] for c in out.tool_calls] == ["get_chat_stats"]
    assert fake_tool[0]["args"] == {"days": 7}
    # birinchi chaqiruvda tool'lar berilgan, system prompt agent varianti
    first = llm.calls[0]
    assert first["task"] is Task.TOOLS and first["tools"]
    assert first["messages"][0].content == P.AGENT_SYSTEM_PROMPT
    assert "Chan" in first["messages"][-1].content  # pinned chat runtime note'da
    # ikkinchi chaqiruvda tool natijasi untrusted konvertda
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg.role.value == "tool" and tool_msg.tool_call_id == "c1"
    assert tool_msg.content.startswith('<untrusted_data source="tool:get_chat_stats"')
    # audit: AgentRun + AgentAction
    kinds = [type(o).__name__ for o in db.added]
    assert kinds.count("AgentRun") == 1 and kinds.count("AgentAction") == 1
    action = next(o for o in db.added if type(o).__name__ == "AgentAction")
    assert action.tool == "get_chat_stats" and action.status == "executed"


async def test_agent_forces_final_answer_after_max_iterations(
    fake_tool: list[dict[str, Any]],
) -> None:
    call = ToolCall("c", "get_recent_messages", {})
    llm = _LLM([_res(calls=[call]), _res(calls=[call]), _res("final")])
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="en",
        llm=llm,  # type: ignore[arg-type]
        max_iterations=2,
    )
    assert out.text == "final"
    # oxirgi (2-) iteratsiyada tool'lar berilmaydi
    assert llm.calls[1]["tools"] is None
    assert len(fake_tool) == 1  # 2-iteratsiyada tool bo'lmagani uchun 1 marta


async def test_agent_caps_calls_per_turn_and_result_budget(
    fake_tool: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    monkeypatch.setenv("AGENT_TOOL_RESULT_TOKENS", "1500")
    get_settings.cache_clear()
    try:
        # Har biri boshqa argument bilan — aks holda takror-qopqog'i (`seen_calls`)
        # ularni bajarilishidan oldin to'sadi va bu test qopqoqni o'lchamaydi.
        calls = [ToolCall(f"c{i}", "get_recent_messages", {"limit": 10 + i}) for i in range(6)]
        llm = _LLM([_res(calls=calls), _res("done")])
        out = await A.run_agent(
            _Session(),  # type: ignore[arg-type]
            user_id=1,
            account_id=5,
            question="q",
            history=[],
            pinned_chat=None,
            locale="en",
            llm=llm,  # type: ignore[arg-type]
        )
    finally:
        get_settings.cache_clear()
    # Ikkita qopqoq birga ishlaydi: turnda ≤4, bitta tool uchun ≤3 → 3 bajariladi
    assert len(fake_tool) == A.MAX_CALLS_PER_TOOL
    msgs = llm.calls[1]["messages"]
    tool_msgs = [m for m in msgs if m.role.value == "tool"]
    assert len(tool_msgs) == 6  # bajarilmaganlarga ham javob (error) bor
    assert any("too many tool calls" in m.content for m in tool_msgs)
    assert any("already called" in m.content for m in tool_msgs)  # tool qopqog'i
    assert any("result budget exhausted" in m.content for m in tool_msgs)
    assert out.result_tokens <= 1500 + 400


async def test_agent_history_round_trips_thought_signature(
    fake_tool: list[dict[str, Any]],
) -> None:
    """Gemini 3 imzosi tarixga o'tishi shart — usiz 2-iteratsiya 400 bilan yiqiladi.

    Jonli xato (2026-09-17): `Function call is missing a thought_signature in
    functionCall parts` → har tool yugurishi yiqilib direct rejimga tushardi.
    """
    llm = _LLM(
        [
            _res(
                "olaman",
                calls=[ToolCall("c1", "get_chat_stats", {"days": 7}, signature=b"sig-1")],
                signature=b"sig-text",
            ),
            _res("12 ta post."),
        ]
    )
    await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
    )
    msgs = llm.calls[1]["messages"]
    assistant = next(m for m in msgs if m.role.value == "assistant")
    assert assistant.tool_calls[0].signature == b"sig-1"
    assert assistant.signature == b"sig-text"


async def test_agent_blocks_identical_repeat_calls(fake_tool: list[dict[str, Any]]) -> None:
    """Aynan bir xil (tool, args) ikkinchi marta bajarilmaydi — natija o'zgarmaydi."""
    same = {"chat": "C", "query": "narx"}
    llm = _LLM(
        [
            _res(calls=[ToolCall("a", "search_messages", same)]),
            _res(calls=[ToolCall("b", "search_messages", dict(same))]),  # aynan o'sha
            _res("done"),
        ]
    )
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="en",
        llm=llm,  # type: ignore[arg-type]
    )
    assert len(fake_tool) == 1  # ikkinchisi DB'ga/MTProto'ga umuman bormadi
    tool_msgs = [m for m in llm.calls[2]["messages"] if m.role.value == "tool"]
    assert any("identical call already made" in m.content for m in tool_msgs)
    assert out.text == "done"


# ─── rejim tanlash ───────────────────────────────────────────────────────────


def _ctx(strategy: str = "auto") -> cs.ContextSpec:
    return cs.ContextSpec(account_id=1, peer_id=2, limit=50, strategy=strategy)


@pytest.mark.parametrize(
    ("mode", "text", "synced", "ctx", "expected"),
    [
        ("agent", "salom", False, None, "agent"),
        ("direct", "kanal statistikasi", True, None, "direct"),
        ("auto", "salom!", True, None, "direct"),
        ("auto", "ok", True, None, "direct"),
        ("auto", "bu hafta eng ko'p ko'rilgan post qaysi?", True, None, "agent"),
        ("auto", "bu hafta eng ko'p ko'rilgan post qaysi?", False, None, "direct"),
        ("auto", "crypto haqida postlar", True, _ctx("search"), "direct"),
        ("auto", "crypto haqida postlar", True, _ctx("auto"), "agent"),
    ],
)
def test_choose_mode(mode: str, text: str, synced: bool, ctx: Any, expected: str) -> None:
    assert cs.choose_mode(mode, text=text, has_synced_chats=synced, context=ctx) == expected


# ─── prompt'lar ──────────────────────────────────────────────────────────────


def test_prompts_are_stable_and_safe() -> None:
    for p in (P.CHAT_SYSTEM_PROMPT, P.AGENT_SYSTEM_PROMPT, P.MAP_DIGEST_PROMPT, P.JUDGE_PROMPT):
        assert "untrusted" in p.lower()
        assert "2026" not in p  # sana system'da emas — kesh buzilmasin
    assert P.CHAT_SYSTEM_PROMPT.startswith(P._CORE) and P.AGENT_SYSTEM_PROMPT.startswith(P._CORE)
    for tool in T.READ_TOOLS:
        assert tool in P.AGENT_SYSTEM_PROMPT or tool == "list_chats"
    note = P.runtime_note(now_iso="2026-08-18 12:00 UTC", locale="uz", pinned_chat="X")
    assert "2026-08-18" in note and "X" in note
    assert cs.SYSTEM_PROMPT == P.CHAT_SYSTEM_PROMPT


# ─── yugurish byudjetlari (round 2: conv-20260917-144833) ────────────────────
# Jonli o'lchovda iteratsiya shifti yolg'iz o'zi yetmadi: model har raundda
# 4 tadan chaqirib, bitta savolga 6-11 bajarilgan chaqiruv ($0.11-$0.40)
# sarfladi. Shuning uchun **yugurish bo'yicha** ikkita qopqoq.


def _priced(text: str = "", calls: list[ToolCall] | None = None) -> ChatResult:
    """Narxi ma'lum model bilan javob (`pricing.PRICES` da bor) — narx qopqog'i uchun."""
    return ChatResult(
        text=text,
        tool_calls=calls or [],
        model="claude-opus-5",
        provider="claude",
        usage=Usage(100, 20),
        finish_reason="tool_use" if calls else "end_turn",
    )


@pytest.fixture
def env_settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """`.env` emas, jarayon muhiti: qopqoq qiymatlarini test uchun o'zgartiradi."""
    from app.config import get_settings

    def apply(**env: str) -> None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield apply
    get_settings.cache_clear()


async def test_agent_stops_at_the_run_call_budget(
    fake_tool: list[dict[str, Any]], env_settings: Any
) -> None:
    """2 ta bajarilgan chaqiruvdan keyin tool'lar o'chadi va javob so'raladi."""
    env_settings(AGENT_MAX_EXECUTED_CALLS="2")
    first = [ToolCall(f"a{i}", "get_recent_messages", {"limit": 10 + i}) for i in range(2)]
    later = [ToolCall("b", "search_messages", {"query": "x"})]
    llm = _LLM([_res(calls=first), _res(calls=later), _res("javob")])
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
        max_iterations=5,
    )
    assert len(fake_tool) == 2 and out.calls_executed == 2
    assert out.stop_reason == "call_budget" and out.text == "javob"
    # Byudjet tugagach keyingi chaqiruvda tool'lar berilmaydi + "javob ber" turn'i
    assert llm.calls[1]["tools"] is None
    assert any(m.content == A.FINAL_NUDGE for m in llm.calls[1]["messages"])


async def test_agent_blocks_calls_over_the_budget_inside_one_turn(
    fake_tool: list[dict[str, Any]], env_settings: Any
) -> None:
    """Bitta turn ichida ham qopqoq ishlaydi — qolganlari bajarilmaydi."""
    env_settings(AGENT_MAX_EXECUTED_CALLS="1")
    calls = [ToolCall(f"c{i}", "get_recent_messages", {"limit": 10 + i}) for i in range(3)]
    llm = _LLM([_res(calls=calls), _res("javob")])
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
    )
    assert len(fake_tool) == 1 and out.calls_executed == 1
    assert len(out.tool_calls) == 3  # to'silganlar ham `calls_log` da ko'rinadi
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role.value == "tool"]
    assert sum("budget of 1 tool calls" in m.content for m in tool_msgs) == 2


async def test_agent_stops_when_the_estimated_cost_cap_is_passed(
    fake_tool: list[dict[str, Any]], env_settings: Any
) -> None:
    """Narx qopqog'i: 100/20 token Opus'da ≈ $0.001 — 0.0005 dan oshadi."""
    env_settings(AGENT_MAX_COST_USD="0.0005", AGENT_MAX_EXECUTED_CALLS="20")
    llm = _LLM(
        [
            _priced(calls=[ToolCall("a", "get_recent_messages", {"limit": 1})]),
            _priced("javob"),
        ]
    )
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
    )
    assert out.stop_reason == "cost_budget" and (out.cost_usd or 0) > 0.0005
    assert llm.calls[1]["tools"] is None
    assert len(fake_tool) == 1


async def test_agent_records_answered_stop_reason(fake_tool: list[dict[str, Any]]) -> None:
    llm = _LLM([_res("darhol javob")])
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="salom",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
    )
    assert out.stop_reason == "answered" and out.calls_executed == 0


# ─── bir xil so'zlarni serverda takrorlash ───────────────────────────────────


@pytest.fixture
def search_tool(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """`search_messages` topilma soni bilan; server qidiruvlari ham qayd etiladi."""
    seen: list[dict[str, Any]] = []

    async def fake_run(name: str, ctx: Any, args: dict[str, Any]) -> T.ToolResult:
        seen.append({"name": name, "args": args})
        if name == "search_messages":
            return T.ToolResult("topildi", meta={"chat": "C", "hits": 3})
        return T.ToolResult("server natijasi", meta={"hits": 1})

    monkeypatch.setattr(A.T, "run_tool", fake_run)
    return seen


async def test_agent_blocks_server_search_repeating_a_successful_local_query(
    search_tool: list[dict[str, Any]],
) -> None:
    llm = _LLM(
        [
            _res(calls=[ToolCall("a", "search_messages", {"chat": "C", "query": "kommunal"})]),
            # aynan o'sha so'zlar, faqat registr boshqa — serverga chiqarilmaydi
            _res(calls=[ToolCall("b", "search_telegram", {"chat": "C", "query": "KOMMUNAL"})]),
            _res("javob"),
        ]
    )
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
    )
    assert [c["name"] for c in search_tool] == ["search_messages"]
    assert out.calls_executed == 1
    tool_msgs = [m for m in llm.calls[2]["messages"] if m.role.value == "tool"]
    assert any("local search already found 3 hits" in m.content for m in tool_msgs)


async def test_agent_allows_server_search_when_local_found_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lokal 0 bergan bo'lsa server yo'li ochiq: u yerda haqiqatan yangi ma'lumot bor."""
    seen: list[str] = []

    async def fake_run(name: str, ctx: Any, args: dict[str, Any]) -> T.ToolResult:
        seen.append(name)
        return T.ToolResult("topilmadi", meta={"hits": 0})

    monkeypatch.setattr(A.T, "run_tool", fake_run)
    llm = _LLM(
        [
            _res(calls=[ToolCall("a", "search_messages", {"chat": "C", "query": "webhook"})]),
            _res(calls=[ToolCall("b", "search_telegram", {"chat": "C", "query": "webhook"})]),
            _res("javob"),
        ]
    )
    await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
    )
    assert seen == ["search_messages", "search_telegram"]


def test_fold_query_ignores_case_and_punctuation() -> None:
    assert A.fold_query("Kommunal  to'lov!") == A.fold_query("kommunal to lov")
    assert A.fold_query(None) == ""


def test_prompt_states_the_numeric_budget_rules() -> None:
    p = P.AGENT_SYSTEM_PROMPT
    low = p.lower()
    assert "after at most 4 executed tool calls" in low
    assert "never repeat a local query verbatim" in low
    assert "never construct t.me urls yourself" in low
    assert "the answer IS `list_chats`" in p
    assert "cannot exist in Telegram data" in p


# ─── suhbat bo'yicha tool natijalari keshi (`services/tool_cache.py`) ────────
#
# Eval `econ-01`: bir xil savol qayta berilganda agent aynan o'sha qidiruvni
# qaytadan bajarardi. Natija o'zgarmaydi, DB/MTProto ishi va token esa
# ikkinchi marta to'lanadi. `seen_calls` bu yerda yordam bermaydi — u bitta
# yugurish ichidagi takrorni to'sadi, bu esa turn'lar orasida.


class _FakeRedis:
    """Xotiradagi Redis o'rnini bosuvchi (TTL yozib boriladi, lekin kutilmaydi)."""

    def __init__(self, store: dict[str, str] | None = None) -> None:
        self.store = store if store is not None else {}
        self.ttls: dict[str, int] = {}
        self.closed = 0

    async def get(self, key: str) -> bytes | None:
        raw = self.store.get(key)
        return raw.encode("utf-8") if raw is not None else None

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    from app.services import tool_cache as TC

    redis = _FakeRedis()

    async def _redis() -> Any:
        return redis

    monkeypatch.setattr(TC, "_redis", _redis)
    return redis


def test_tool_cache_key_folds_arguments_and_scopes_to_the_conversation() -> None:
    from app.services import tool_cache as TC

    a = TC.cache_key(7, "search_messages", {"query": "To'lov  MUDDATI", "days": 30})
    b = TC.cache_key(7, "search_messages", {"days": 30, "query": "to'lov muddati"})
    assert a == b, "registr/bo'shliq/tartib farqi kesh promahiga aylanmasin"
    assert a != TC.cache_key(8, "search_messages", {"query": "to'lov muddati", "days": 30})
    assert a != TC.cache_key(7, "search_telegram", {"query": "to'lov muddati", "days": 30})
    assert a.startswith(TC.KEY_PREFIX + "7:")
    # Suhbat yo'q → kesh yo'q (xotira/maxfiylik chegarasi).
    assert TC.cache_key(None, "search_messages", {"query": "x"}) == ""


async def test_tool_cache_round_trip_and_silent_failure(fake_redis: _FakeRedis) -> None:
    from app.services import tool_cache as TC

    key = TC.cache_key(1, "search_messages", {"query": "q"})
    assert await TC.get(key) is None
    await TC.put(key, "natija")
    assert await TC.get(key) == "natija"
    assert fake_redis.ttls[key] == TC.TTL_S
    # Juda uzun natija yozilmaydi
    long_key = TC.cache_key(1, "search_messages", {"query": "long"})
    await TC.put(long_key, "x" * (TC.MAX_VALUE_CHARS + 1))
    assert await TC.get(long_key) is None


async def test_tool_cache_is_a_miss_when_redis_is_down(
    monkeypatch: pytest.MonkeyPatch, fake_tool: list[dict[str, Any]]
) -> None:
    from app.services import tool_cache as TC

    async def boom() -> Any:
        raise RuntimeError("no redis")

    monkeypatch.setattr(TC, "_redis", boom)
    key = TC.cache_key(1, "search_messages", {"query": "q"})
    await TC.put(key, "natija")  # jim o'tadi
    assert await TC.get(key) is None
    # ...va agent baribir ishlaydi (tool haqiqatan bajariladi)
    llm = _LLM([_res(calls=[ToolCall("c1", "get_chat_stats", {"days": 7})]), _res("javob")])
    out = await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
        conversation_id=1,
    )
    assert out.text == "javob" and len(fake_tool) == 1 and out.cache_hits == 0


async def _run_once(
    llm: Any, fake_tool: list[dict[str, Any]], *, conversation_id: int | None
) -> Any:
    return await A.run_agent(
        _Session(),  # type: ignore[arg-type]
        user_id=1,
        account_id=5,
        question="to'lov muddati qachon?",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,  # type: ignore[arg-type]
        conversation_id=conversation_id,
    )


def _two_turns() -> Any:
    return _LLM([_res(calls=[ToolCall("c1", "search_messages", {"query": "to'lov"})]), _res("ok")])


async def test_repeated_question_reuses_the_cached_tool_result(
    fake_redis: _FakeRedis, fake_tool: list[dict[str, Any]]
) -> None:
    first = await _run_once(_two_turns(), fake_tool, conversation_id=42)
    assert first.cache_hits == 0 and len(fake_tool) == 1

    second = await _run_once(_two_turns(), fake_tool, conversation_id=42)
    assert len(fake_tool) == 1, "ikkinchi yugurish tool'ni qayta bajardi"
    assert second.cache_hits == 1
    assert second.tool_calls[0]["cached"] is True
    # Byudjet baribir sanaladi — kesh cheksiz sikl ochmaydi.
    assert second.calls_executed == 1


async def test_cache_result_is_marked_for_the_model(
    fake_redis: _FakeRedis, fake_tool: list[dict[str, Any]]
) -> None:
    from app.services import tool_cache as TC

    await _run_once(_two_turns(), fake_tool, conversation_id=42)
    llm = _two_turns()
    await _run_once(llm, fake_tool, conversation_id=42)
    tool_msg = llm.calls[1]["messages"][-1]
    assert TC.CACHED_NOTE in tool_msg.content
    assert tool_msg.content.startswith('<untrusted_data source="tool:search_messages"')


async def test_cache_does_not_leak_between_conversations(
    fake_redis: _FakeRedis, fake_tool: list[dict[str, Any]]
) -> None:
    await _run_once(_two_turns(), fake_tool, conversation_id=42)
    out = await _run_once(_two_turns(), fake_tool, conversation_id=43)
    assert len(fake_tool) == 2 and out.cache_hits == 0


async def test_cache_is_off_without_a_conversation_id(
    fake_redis: _FakeRedis, fake_tool: list[dict[str, Any]]
) -> None:
    await _run_once(_two_turns(), fake_tool, conversation_id=None)
    await _run_once(_two_turns(), fake_tool, conversation_id=None)
    assert len(fake_tool) == 2
    assert fake_redis.store == {}


def test_agent_prompt_tells_the_model_to_answer_repeats_from_history() -> None:
    assert "repeats a question you have already answered" in P.AGENT_SYSTEM_PROMPT
    assert "(cached)" in P.AGENT_SYSTEM_PROMPT
    assert "2026" not in P.AGENT_SYSTEM_PROMPT  # system kesh prefiksi barqaror
