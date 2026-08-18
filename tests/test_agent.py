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
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = len(self.added)

    async def flush(self) -> None: ...


class _LLM:
    """Ssenariy: birinchi chaqiruvda tool so'raydi, keyin javob beradi."""

    def __init__(self, script: list[ChatResult]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        self.calls.append({"task": task, "messages": list(messages), "tools": kw.get("tools")})
        return self.script.pop(0)


def _res(text: str = "", calls: list[ToolCall] | None = None) -> ChatResult:
    return ChatResult(
        text=text,
        tool_calls=calls or [],
        model="m",
        provider="p",
        usage=Usage(100, 20),
        finish_reason="tool_use" if calls else "end_turn",
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
        calls = [ToolCall(f"c{i}", "get_recent_messages", {"limit": 10}) for i in range(6)]
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
    assert len(fake_tool) == A.MAX_CALLS_PER_TURN  # 6 so'raldi, 4 bajarildi
    msgs = llm.calls[1]["messages"]
    tool_msgs = [m for m in msgs if m.role.value == "tool"]
    assert len(tool_msgs) == 6  # bajarilmaganlarga ham javob (error) bor
    assert any("too many tool calls" in m.content for m in tool_msgs)
    assert any("result budget exhausted" in m.content for m in tool_msgs)
    assert out.result_tokens <= 1500 + 400


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
