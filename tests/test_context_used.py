"""`chat_service._send_via_agent` — `context_used` metadata to'liqligi.

DB/tarmoqsiz: `app.services.agent.run_agent` mock qilinadi, `session` faqat
`add`/`flush` ni duck-type qiladi (`tests/test_agent.py::_Session` uslubida).
Maqsad — `AgentOutcome` maydonlari `context_used` ga to'g'ri ko'chirilishini
tekshirish, xususan repeat-guard (`agent.py`: keshdan kelgan qidiruvdan keyin
server qidiruvi to'silgani) `context_used["repeat_guard"]` da ko'rinishi shart
— aks holda eval/UI buni ko'rmaydi.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import agent as A
from app.services import chat_service as cs


class _Session:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = len(self.added) + 100

    async def flush(self) -> None: ...


def _outcome(**kw: Any) -> A.AgentOutcome:
    base: dict[str, Any] = dict(
        text="javob",
        model="m",
        provider="p",
        tokens_in=10,
        tokens_out=5,
        iterations=1,
    )
    base.update(kw)
    return A.AgentOutcome(**base)


async def _run(monkeypatch: pytest.MonkeyPatch, outcome: A.AgentOutcome) -> cs.Reply:
    async def fake_run_agent(*a: Any, **kw: Any) -> A.AgentOutcome:
        return outcome

    monkeypatch.setattr(A, "run_agent", fake_run_agent)
    session = _Session()
    conversation = SimpleNamespace(id=1, title="", updated_at=None)
    user_row = SimpleNamespace(id=1, context=None)
    return await cs._send_via_agent(
        session,  # type: ignore[arg-type]
        user_id=1,
        conversation=conversation,  # type: ignore[arg-type]
        user_row=user_row,  # type: ignore[arg-type]
        text="salom",
        account_id=1,
        pinned=None,
        history=[],
        locale="uz",
        started=time.monotonic(),
        llm=None,
    )


async def test_context_used_surfaces_repeat_guard_when_agent_blocked_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = await _run(monkeypatch, _outcome(repeat_guard=True))
    assert reply.context_used is not None
    assert reply.context_used["repeat_guard"] is True


async def test_context_used_repeat_guard_false_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = await _run(monkeypatch, _outcome())
    assert reply.context_used is not None
    assert reply.context_used["repeat_guard"] is False
