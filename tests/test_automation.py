"""`create_alert` / `set_briefing` — tasdiqli amallar (DB va Telegram'siz).

Qo'riqlaydi (CLAUDE.md invariant 4):
  * ko'nikma faqat `agent_actions(proposed)` yozadi — qoida/sozlama **yaratilmaydi**;
  * yozish niyati foydalanuvchi matnida bo'lmasa — hech narsa taklif qilinmaydi;
  * `confirm_action` → `execute_action` → `automation.execute` → qoida yaratiladi;
  * tasdiqsiz (`autonomous` yo'li, `confirmed=False`) bajarilmaydi;
  * boshqa akkauntga burilgan taklif bajarilmaydi;
  * `WRITE_TOOL_NAMES` o'zgarmagan (avtomatlashtirish alohida ro'yxatda);
  * Actions kartasi uchun `preview.text` — muzlatilgan "nima yaratiladi" matni.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from app.db.models import Account, AgentAction, AgentRun, AlertRule
from app.services import actions as ACT
from app.services import automation as AU
from app.services import deep_search as DS
from app.services import skills_ext as SX
from app.services import write_tools as W


class _Session:
    """add/flush/get/execute — minimal soxta AsyncSession."""

    def __init__(self, objects: dict[tuple[str, int], Any] | None = None) -> None:
        self.added: list[Any] = []
        self.objects = objects or {}
        self._seq = 100
        self.executed: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            self._seq += 1
            obj.id = self._seq

    async def flush(self) -> None: ...

    async def get(self, model: Any, key: Any) -> Any:
        return self.objects.get((model.__name__, int(key)))

    async def execute(self, stmt: Any, *a: Any, **k: Any) -> Any:
        self.executed.append(stmt)
        return SimpleNamespace(
            scalar=lambda: 0,
            scalar_one=lambda: 0,
            first=lambda: None,
            scalars=lambda: SimpleNamespace(all=lambda: [], first=lambda: None),
        )


class _Factory:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _Factory:
        return self

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *a: Any) -> None:
        return None


def _chat(**kw: Any) -> Any:
    base = dict(id=11, account_id=5, title="Ijara <b>chat</b>", tg_peer_id=1001)
    base.update(kw)
    return SimpleNamespace(**base)


def _objects() -> dict[tuple[str, int], Any]:
    return {
        ("Chat", 11): _chat(),
        ("Chat", 12): _chat(id=12, account_id=99, title="Begona"),
        ("Account", 5): SimpleNamespace(
            id=5, user_id=1, tz="Asia/Tashkent", briefing_hour=8, briefing_enabled=False
        ),
        ("AgentRun", 7): SimpleNamespace(id=7, account_id=5, user_id=1),
    }


def _ctx(question: str, session: _Session, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "raw_question": question,
        "question": question,
        "locale": "uz",
        "account_id": 5,
        "user_id": 1,
        "run_id": None,
        "action_ids": [],
        "selected": [],
        "chats_by_id": {11: _chat()},
        "plan": DS.DeepPlan(),
        "tz": "Asia/Tashkent",
        "facts": {},
        "claims": [],
        "session_factory": _Factory(session),
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _actions(session: _Session) -> list[AgentAction]:
    return [o for o in session.added if isinstance(o, AgentAction)]


# ─── tahlil (sof) ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "kind", "pattern"),
    [
        ("'ijara' so'zi chiqsa xabar ber", "keyword", "ijara"),
        ("«to'lov» degan so'z uchrasa ogohlantir", "keyword", "to'lov"),
        ("kripto haqida yozishsa ogohlantir", "semantic", "kripto"),
        ("agar kimdir ijara haqida yozsa xabar ber", "semantic", "ijara"),
        ("сообщи, если напишут про аренду квартир", "semantic", "аренду квартир"),
        ("notify me when someone writes about crypto", "semantic", "crypto"),
    ],
)
def test_parse_alert_request(text: str, kind: str, pattern: str) -> None:
    assert AU.parse_alert_request(text) == (kind, pattern)


def test_parse_alert_request_none() -> None:
    assert AU.parse_alert_request("xabar ber") is None


@pytest.mark.parametrize(
    ("text", "enabled", "hour"),
    [
        ("har kuni ertalab 8 da briefing yubor", True, 8),
        ("briefingni o'chir", False, None),
        ("briefing kechqurun 9 da kelsin", True, 21),
        ("присылай брифинг в 7", True, 7),
        ("stop sending me the briefing", False, None),
        ("briefing yoq ertalab", True, 8),
    ],
)
def test_parse_briefing_request(text: str, enabled: bool, hour: int | None) -> None:
    assert AU.parse_briefing_request(text) == (enabled, hour)


# ─── taklif (ko'nikma) ───────────────────────────────────────────────────────


async def test_create_alert_skill_only_proposes() -> None:
    s = _Session(_objects())
    ctx = _ctx("'ijara' so'zi chiqsa xabar ber", s)
    res = await SX._create_alert(ctx, {"chat_id": 11})
    [a] = _actions(s)
    assert a.tool == "create_alert" and a.status == "proposed" and a.target_peer_id is None
    assert a.args["kind"] == "keyword" and a.args["pattern"] == "ijara"
    assert a.args["chat_id"] == 11 and a.args["account_id"] == 5
    assert a.args["cooldown_minutes"] == AU.ALERT_DEFAULT_COOLDOWN
    assert "ijara" in a.args["summary"] and "<b>" not in a.args["summary"]
    assert not any(isinstance(o, AlertRule) for o in s.added)  # yaratilmagan!
    assert res.stop and f"#{a.id}" in res.answer and ctx.action_ids == [a.id]
    assert res.summary == {"proposed": True, "tool": "create_alert"}
    assert any(isinstance(o, AgentRun) for o in s.added)


async def test_create_alert_refuses_without_intent_or_pattern() -> None:
    s = _Session(_objects())
    res = await SX._create_alert(_ctx("ijara haqida nima yozishdi?", s), {"pattern": "ijara"})
    assert res.stop and res.summary["reason"] == "no_intent" and s.added == []
    res = await SX._create_alert(_ctx("xabar ber", s), {})
    assert res.summary["reason"] == "no_pattern" and s.added == []


async def test_create_alert_semantic_from_planner_and_foreign_chat() -> None:
    s = _Session(_objects())
    res = await SX._create_alert(_ctx("buni kuzatib xabar ber", s), {"pattern": "kripto"})
    [a] = _actions(s)
    assert a.args["kind"] == "semantic" and a.args["chat_id"] is None and res.stop
    s2 = _Session(_objects())
    res = await SX._create_alert(_ctx("'ijara' chiqsa xabar ber", s2), {"chat_id": 12})
    assert res.summary["reason"] == "unknown_chat" and _actions(s2) == []


async def test_set_briefing_skill_proposes() -> None:
    s = _Session(_objects())
    res = await SX._set_briefing(_ctx("har kuni ertalab 8 da briefing yubor", s), {})
    [a] = _actions(s)
    assert a.tool == "set_briefing" and a.args == {
        "account_id": 5,
        "enabled": True,
        "hour": 8,
        "summary": a.args["summary"],
    }
    assert "08:00" in a.args["summary"] and "Asia/Tashkent" in a.args["summary"]
    assert res.stop
    s2 = _Session(_objects())
    await SX._set_briefing(_ctx("briefingni o'chir", s2), {})
    [b] = _actions(s2)
    assert b.args["enabled"] is False and b.args["hour"] is None


# ─── tasdiq → bajarish ───────────────────────────────────────────────────────


def _action(tool: str, args: dict[str, Any], **kw: Any) -> Any:
    base = dict(
        id=1,
        run_id=7,
        tool=tool,
        status="proposed",
        args=args,
        target_peer_id=None,
        result_msg_id=None,
        error=None,
        block_reason=None,
        confirmed_at=None,
        created_at=datetime.now(UTC),
    )
    base.update(kw)
    return SimpleNamespace(**base)


_ALERT_ARGS = {
    "account_id": 5,
    "kind": "keyword",
    "pattern": "ijara",
    "chat_id": 11,
    "chat_title": "Ijara chat",
    "cooldown_minutes": 30,
    "threshold": 0.8,
    "label": "ijara",
    "summary": "…",
}


async def test_confirm_executes_create_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _action("create_alert", dict(_ALERT_ARGS))
    s = _Session(_objects())

    async def owned(session: Any, user_id: int, action_id: int) -> Any:
        return a

    async def claim(session: Any, action: Any) -> bool:
        action.status = "confirmed"
        return True

    async def no_mtproto(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("automation must never reach MTProto")

    monkeypatch.setattr(ACT, "_owned", owned)
    monkeypatch.setattr(ACT, "_claim_proposed", claim)
    monkeypatch.setattr(ACT, "_run_mtproto", no_mtproto)
    view = await ACT.confirm_action(s, 1, 1)  # type: ignore[arg-type]
    [rule] = [o for o in s.added if isinstance(o, AlertRule)]
    assert rule.account_id == 5 and rule.chat_id == 11 and rule.pattern == "ijara"
    assert rule.kind == "keyword" and rule.enabled and rule.cooldown_minutes == 30
    assert view.status == "executed" and a.result_msg_id == rule.id
    assert view.preview["text"] == "…" and view.preview["automation"] == "create_alert"


async def test_automation_never_runs_without_confirmation() -> None:
    a = _action("create_alert", dict(_ALERT_ARGS))
    s = _Session(_objects())
    with pytest.raises(ACT.ActionError, match="needs_confirm"):
        await ACT.execute_action(s, a, actor="autonomous")  # type: ignore[arg-type]
    assert a.status == "failed" and not any(isinstance(o, AlertRule) for o in s.added)


async def test_execute_refuses_foreign_account_and_bad_values() -> None:
    s = _Session(_objects())
    a = _action("create_alert", {**_ALERT_ARGS, "account_id": 99})
    with pytest.raises(ACT.ActionError, match="not_found"):
        await ACT.execute_action(s, a, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    b = _action("create_alert", {**_ALERT_ARGS, "chat_id": 12})
    with pytest.raises(ACT.ActionError, match="not_found"):
        await ACT.execute_action(s, b, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    c = _action("create_alert", {**_ALERT_ARGS, "pattern": ""})
    with pytest.raises(ACT.ActionError, match="empty_pattern"):
        await ACT.execute_action(s, c, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    assert not any(isinstance(o, AlertRule) for o in s.added)
    # boshqa foydalanuvchining akkaunti (run.user_id ≠ account.user_id)
    s.objects[("Account", 5)].user_id = 2
    d = _action("create_alert", dict(_ALERT_ARGS))
    with pytest.raises(ACT.ActionError, match="not_found"):
        await ACT.execute_action(s, d, actor="user:1", confirmed=True)  # type: ignore[arg-type]


async def test_execute_set_briefing_updates_account() -> None:
    s = _Session(_objects())
    a = _action("set_briefing", {"account_id": 5, "enabled": True, "hour": 7, "summary": "x"})
    rid = await ACT.execute_action(s, a, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    acc = s.objects[("Account", 5)]
    assert rid is None and a.status == "executed"
    assert acc.briefing_enabled is True and acc.briefing_hour == 7
    assert not hasattr(acc, "briefing_chats")  # chat tanlovi tegilmadi
    off = _action("set_briefing", {"account_id": 5, "enabled": False, "hour": None})
    await ACT.execute_action(s, off, actor="user:1", confirmed=True)  # type: ignore[arg-type]
    assert acc.briefing_enabled is False and acc.briefing_hour == 7


async def test_execute_set_briefing_rejects_bad_hour() -> None:
    s = _Session(_objects())
    a = _action("set_briefing", {"account_id": 5, "enabled": True, "hour": 30})
    with pytest.raises(ACT.ActionError, match="bad_hour"):
        await ACT.execute_action(s, a, actor="user:1", confirmed=True)  # type: ignore[arg-type]


# ─── registrlar ──────────────────────────────────────────────────────────────


def test_write_tool_names_unchanged_and_automation_separate() -> None:
    assert (
        frozenset({"send_message", "edit_message", "forward_message", "pin_message"})
        == W.WRITE_TOOL_NAMES
    )
    assert W.AUTOMATION_TOOL_NAMES == ACT.AUTOMATION_TOOLS
    assert not (W.AUTOMATION_TOOL_NAMES & W.WRITE_TOOL_NAMES)
    assert all("delete" not in t and "remove" not in t for t in W.AUTOMATION_TOOL_NAMES)


async def test_propose_automation_rejects_unknown_tool() -> None:
    with pytest.raises(W.ProposalError):
        await W.propose_automation(_Session(), run_id=1, tool="delete_message", args={})  # type: ignore[arg-type]


async def test_actions_for_runs_includes_automation_cards() -> None:
    rows = [
        AgentAction(id=1, run_id=7, tool="create_alert", args={"summary": "s"}, status="proposed"),
        AgentAction(id=2, run_id=7, tool="search_messages", args={}, status="executed"),
    ]

    class S(_Session):
        async def execute(self, stmt: Any, *a: Any, **k: Any) -> Any:
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

    got = await ACT.actions_for_runs(S(), [7])  # type: ignore[arg-type]
    assert [c["tool"] for c in got[7]] == ["create_alert"]
    assert got[7][0]["preview"]["text"] == "s"


def test_account_model_has_briefing_columns() -> None:
    cols = set(Account.__table__.columns.keys())
    assert {"briefing_enabled", "briefing_hour", "briefing_chats"} <= cols
