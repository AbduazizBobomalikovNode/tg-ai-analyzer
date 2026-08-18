"""6-bosqich: yozish amallari — taklif/tasdiq/bajarish, DB va Telegram'siz.

Qo'riqlaydi (CLAUDE.md invariant 2 va 4):
  * agent yozish tool'i hech qachon to'g'ridan-to'g'ri yubormaydi — `proposed`
  * boshqaruv boti / 777000 / @replies → `blocked` (istisno emas, audit yozuvi)
  * read_only chat → rad; matn limiti; schedule validatsiyasi
  * autonomous → darhol bajariladi (audit bilan); akkauntga faqat bitta autonomous chat
  * execute: bajarish oldidan ham guard; rate limit; TTL; MTProto mapping (allowlist metodlar)
  * agent write spec'larni faqat yozuvchi chat bo'lsa beradi
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import get_settings
from app.llm import ChatResult, Msg, Task, ToolCall, Usage
from app.mtproto.guard import PeerBlocked
from app.services import actions as ACT
from app.services import agent as A
from app.services import tools as T
from app.services import write_tools as W


def _chat(**kw: Any) -> Any:
    base = dict(
        id=11,
        account_id=5,
        tg_peer_id=1001,
        title="Kanal",
        username="kanal",
        type="channel",
        write_mode="write_with_confirm",
        is_writable=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Session:
    """add/flush/get/execute — minimal soxta AsyncSession."""

    def __init__(
        self, objects: dict[tuple[str, int], Any] | None = None, execute_first: Any = None
    ) -> None:
        self.added: list[Any] = []
        self.objects = objects or {}
        self.execute_first = execute_first
        self._seq = 100

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            self._seq += 1
            obj.id = self._seq

    async def flush(self) -> None: ...

    async def get(self, model: Any, key: Any) -> Any:
        return self.objects.get((model.__name__, int(key)))

    async def execute(self, *a: Any, **k: Any) -> Any:
        first = self.execute_first
        return SimpleNamespace(
            first=lambda: first,
            scalar_one=lambda: first[0] if first else 0,
            scalar_one_or_none=lambda: first,
            scalars=lambda: SimpleNamespace(all=lambda: [], first=lambda: None),
        )


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> T.ToolContext:
    chats = {
        "kanal": _chat(),
        "ro": _chat(id=12, title="RO", write_mode="read_only"),
        "bot": _chat(id=13, title="Bot", tg_peer_id=get_settings().control_bot_id),
    }

    async def resolve(ctx: Any, args: dict[str, Any]) -> Any:
        key = str(args.get("chat") or "kanal").lower()
        return chats.get(key, chats["kanal"] if key in ("", "11") else None)

    monkeypatch.setattr(W, "_resolve_chat", resolve)
    return T.ToolContext(session=_Session(), account_id=5, pinned_chat_id=11)  # type: ignore[arg-type]


# ─── build_proposal ──────────────────────────────────────────────────────────


async def test_build_send_proposal(ctx: T.ToolContext) -> None:
    p = await W.build_proposal(ctx, "send_message", {"text": "  Salom dunyo  ", "reply_to": "5"})
    assert p.tool == "send_message" and p.target_peer_id == 1001 and p.chat_id == 11
    assert p.args["text"] == "Salom dunyo" and p.args["reply_to"] == 5 and p.args["peer_id"] == 1001
    assert p.preview["chat"] == "Kanal"


async def test_build_proposal_rejects_read_only_and_bad_args(ctx: T.ToolContext) -> None:
    with pytest.raises(W.ProposalError, match="read_only"):
        await W.build_proposal(ctx, "send_message", {"chat": "ro", "text": "x"})
    with pytest.raises(W.ProposalError, match="empty_text"):
        await W.build_proposal(ctx, "send_message", {"text": "  "})
    with pytest.raises(W.ProposalError, match="text_too_long"):
        await W.build_proposal(ctx, "send_message", {"text": "x" * 5000})
    with pytest.raises(W.ProposalError, match="missing_arg"):
        await W.build_proposal(ctx, "pin_message", {})
    with pytest.raises(W.ProposalError, match="bad_arg"):
        await W.build_proposal(ctx, "send_message", {"text": "x", "schedule_at": "yesterday"})
    with pytest.raises(W.ProposalError, match="unknown_tool"):
        await W.build_proposal(ctx, "delete_message", {"message_id": 1})


async def test_build_proposal_schedule_future_ok(ctx: T.ToolContext) -> None:
    when = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    p = await W.build_proposal(ctx, "send_message", {"text": "x", "schedule_at": when})
    assert p.args["schedule_at"] == when


async def test_build_proposal_blocks_control_bot(ctx: T.ToolContext) -> None:
    with pytest.raises(PeerBlocked):
        await W.build_proposal(ctx, "send_message", {"chat": "bot", "text": "hi"})


# ─── propose_or_execute ──────────────────────────────────────────────────────


async def test_propose_creates_proposed_action_not_sent(ctx: T.ToolContext) -> None:
    db = _Session(objects={("Chat", 11): _chat()})
    res = await W.propose_or_execute(db, ctx, run_id=7, tool="send_message", args={"text": "Post"})  # type: ignore[arg-type]
    assert res.ok and res.meta.get("proposed") and "waiting for the user's confirmation" in res.text
    (action,) = db.added
    assert type(action).__name__ == "AgentAction"
    assert action.status == "proposed" and action.tool == "send_message"
    assert action.args["text"] == "Post" and action.target_peer_id == 1001
    assert res.meta["action_id"] == action.id


async def test_propose_blocked_writes_audit_row(ctx: T.ToolContext) -> None:
    db = _Session()
    res = await W.propose_or_execute(
        db, ctx, run_id=7, tool="send_message", args={"chat": "bot", "text": "x"}
    )  # type: ignore[arg-type]
    assert not res.ok and "blocked" in res.text
    (action,) = db.added
    assert (
        action.status == "blocked"
        and action.block_reason
        and action.target_peer_id == get_settings().control_bot_id
    )


async def test_propose_autonomous_executes(
    ctx: T.ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed: list[Any] = []

    async def fake_exec(session: Any, action: Any, *, actor: str) -> int:
        executed.append((action.tool, actor))
        return 555

    monkeypatch.setattr(ACT, "execute_action", fake_exec)
    db = _Session(objects={("Chat", 11): _chat(write_mode="autonomous")})
    res = await W.propose_or_execute(db, ctx, run_id=7, tool="send_message", args={"text": "x"})  # type: ignore[arg-type]
    assert res.ok and res.meta.get("executed") and "555" in res.text
    assert executed == [("send_message", "autonomous")]


# ─── execute_action ──────────────────────────────────────────────────────────


def _action(**kw: Any) -> Any:
    base = dict(
        id=1,
        run_id=7,
        tool="send_message",
        status="proposed",
        args={"chat_id": 11, "peer_id": 1001, "text": "hi", "chat_title": "Kanal"},
        target_peer_id=1001,
        result_msg_id=None,
        error=None,
        block_reason=None,
        confirmed_at=None,
        created_at=datetime.now(UTC),
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def exec_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    env: dict[str, Any] = {"mtproto": [], "rate_ok": True}

    async def fake_run(account_id: int, tool: str, args: dict[str, Any]) -> int:
        env["mtproto"].append((account_id, tool, dict(args)))
        return 999

    async def fake_rate(session: Any, account_id: int) -> bool:
        return env["rate_ok"]

    monkeypatch.setattr(ACT, "_run_mtproto", fake_run)
    monkeypatch.setattr(ACT, "_rate_ok", fake_rate)
    env["db"] = _Session(
        objects={
            ("AgentRun", 7): SimpleNamespace(id=7, account_id=5, user_id=1),
            ("Chat", 11): _chat(),
        }
    )
    return env


async def test_execute_action_success(exec_env: dict[str, Any]) -> None:
    a = _action()
    rid = await ACT.execute_action(exec_env["db"], a, actor="user:1")
    assert rid == 999 and a.status == "executed" and a.result_msg_id == 999 and a.confirmed_at
    assert exec_env["mtproto"] == [(5, "send_message", a.args)]


async def test_execute_action_guard_at_execution(exec_env: dict[str, Any]) -> None:
    a = _action(args={"chat_id": 11, "peer_id": get_settings().control_bot_id, "text": "x"})
    with pytest.raises(ACT.ActionError, match="blocked"):
        await ACT.execute_action(exec_env["db"], a, actor="user:1")
    assert a.status == "failed" and "blocked" in (a.error or "") and exec_env["mtproto"] == []


async def test_execute_action_rate_limited(exec_env: dict[str, Any]) -> None:
    exec_env["rate_ok"] = False
    a = _action()
    with pytest.raises(ACT.ActionError, match="rate_limited"):
        await ACT.execute_action(exec_env["db"], a, actor="user:1")
    assert a.status == "failed" and exec_env["mtproto"] == []


async def test_execute_action_read_only_chat(exec_env: dict[str, Any]) -> None:
    exec_env["db"].objects[("Chat", 11)] = _chat(write_mode="read_only")
    a = _action()
    with pytest.raises(ACT.ActionError, match="read_only"):
        await ACT.execute_action(exec_env["db"], a, actor="user:1")
    assert exec_env["mtproto"] == []


async def test_confirm_expired_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _action(created_at=datetime.now(UTC) - timedelta(days=3))

    async def owned(session: Any, user_id: int, action_id: int) -> Any:
        return a

    monkeypatch.setattr(ACT, "_owned", owned)
    with pytest.raises(ACT.ActionError, match="expired"):
        await ACT.confirm_action(_Session(), 1, 1)  # type: ignore[arg-type]
    assert a.status == "rejected" and a.error == "expired"


async def test_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _action()

    async def owned(session: Any, user_id: int, action_id: int) -> Any:
        return a

    monkeypatch.setattr(ACT, "_owned", owned)
    v = await ACT.reject_action(_Session(), 1, 1)  # type: ignore[arg-type]
    assert v.status == "rejected" and a.confirmed_at
    with pytest.raises(ACT.ActionError, match="wrong_status"):
        await ACT.reject_action(_Session(), 1, 1)  # type: ignore[arg-type]


# ─── MTProto mapping ─────────────────────────────────────────────────────────


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []

    async def send_message(self, peer: Any, text: str, **kw: Any) -> Any:
        self.calls.append(("send_message", peer, {"text": text, **kw}))
        return SimpleNamespace(id=42)

    async def edit_message(self, peer: Any, mid: int, text: str) -> Any:
        self.calls.append(("edit_message", peer, {"mid": mid, "text": text}))
        return SimpleNamespace(id=mid)

    async def pin_message(self, peer: Any, mid: int, *, notify: bool) -> Any:
        self.calls.append(("pin_message", peer, {"mid": mid, "notify": notify}))

    async def forward_messages(self, peer: Any, mid: int, src: Any, *, drop_author: bool) -> Any:
        self.calls.append(
            ("forward_messages", peer, {"mid": mid, "src": src, "drop_author": drop_author})
        )
        return [SimpleNamespace(id=77)]


async def test_run_mtproto_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()

    async def get(account_id: int) -> Any:
        return client

    async def input_peer(account_id: int, peer_id: int) -> Any:
        return f"peer:{peer_id}"

    monkeypatch.setattr(ACT.pool, "get", get)
    monkeypatch.setattr(ACT.pool, "input_peer", input_peer)

    assert (
        await ACT._run_mtproto(5, "send_message", {"peer_id": 1, "text": "hi", "reply_to": 3}) == 42
    )
    assert (
        await ACT._run_mtproto(5, "edit_message", {"peer_id": 1, "message_id": 9, "text": "n"}) == 9
    )
    assert (
        await ACT._run_mtproto(5, "pin_message", {"peer_id": 1, "message_id": 9, "silent": True})
        == 9
    )
    assert (
        await ACT._run_mtproto(
            5,
            "forward_message",
            {"peer_id": 1, "from_peer_id": 2, "message_id": 4, "drop_author": True},
        )
        == 77
    )
    names = [c[0] for c in client.calls]
    assert names == ["send_message", "edit_message", "pin_message", "forward_messages"]
    assert client.calls[0][2]["reply_to"] == 3
    assert client.calls[2][2]["notify"] is False  # silent → notify=False
    assert client.calls[3][2]["drop_author"] is True
    with pytest.raises(ACT.ActionError, match="unknown_tool"):
        await ACT._run_mtproto(5, "delete_message", {"peer_id": 1})


# ─── write mode ──────────────────────────────────────────────────────────────


async def test_autonomous_only_one_chat_per_account() -> None:
    chat = _chat(write_mode="write_with_confirm")
    db = _Session(
        objects={("Chat", 11): chat, ("Account", 5): SimpleNamespace(id=5, user_id=1)},
        execute_first=(12,),
    )
    with pytest.raises(ACT.ActionError, match="autonomous_limit"):
        await ACT.set_chat_write_mode(db, user_id=1, chat_id=11, mode="autonomous")  # type: ignore[arg-type]
    db2 = _Session(
        objects={("Chat", 11): chat, ("Account", 5): SimpleNamespace(id=5, user_id=1)},
        execute_first=None,
    )
    out = await ACT.set_chat_write_mode(db2, user_id=1, chat_id=11, mode="autonomous")  # type: ignore[arg-type]
    assert out.write_mode == "autonomous"
    with pytest.raises(ACT.ActionError, match="bad_mode"):
        await ACT.set_chat_write_mode(db2, user_id=1, chat_id=11, mode="yolo")  # type: ignore[arg-type]
    with pytest.raises(ACT.ActionError, match="not_found"):
        await ACT.set_chat_write_mode(db2, user_id=2, chat_id=11, mode="read_only")  # type: ignore[arg-type]


# ─── agent: write spec'lar faqat yozuvchi chat bo'lsa ────────────────────────


class _LLM:
    def __init__(self, script: list[ChatResult]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        self.calls.append({"tools": kw.get("tools"), "messages": list(messages)})
        return self.script.pop(0)


def _res(text: str = "", calls: list[ToolCall] | None = None) -> ChatResult:
    return ChatResult(text=text, tool_calls=calls or [], model="m", provider="p", usage=Usage(1, 1))


class _AgentSession(_Session):
    def __init__(self, writable: bool) -> None:
        super().__init__(execute_first=(1,) if writable else None)


async def test_agent_offers_write_tools_only_when_writable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_propose(
        session: Any, ctx: Any, *, run_id: int, tool: str, args: dict[str, Any]
    ) -> T.ToolResult:
        return T.ToolResult("proposed action #9", meta={"action_id": 9, "proposed": True})

    monkeypatch.setattr(A.W, "propose_or_execute", fake_propose)

    llm = _LLM(
        [_res(calls=[ToolCall("c1", "send_message", {"text": "hi"})]), _res("Taklif qilindi")]
    )
    out = await A.run_agent(
        _AgentSession(True),
        user_id=1,
        account_id=5,
        question="postni yubor",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm,
    )  # type: ignore[arg-type]
    names = [t.name for t in llm.calls[0]["tools"]]
    assert "send_message" in names and "search_messages" in names
    assert out.action_ids == [9] and out.tool_calls[0]["action_id"] == 9

    llm2 = _LLM([_res("javob")])
    await A.run_agent(
        _AgentSession(False),
        user_id=1,
        account_id=5,
        question="q",
        history=[],
        pinned_chat=None,
        locale="uz",
        llm=llm2,
    )  # type: ignore[arg-type]
    assert "send_message" not in [t.name for t in llm2.calls[0]["tools"]]
