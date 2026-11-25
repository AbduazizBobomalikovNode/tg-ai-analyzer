"""Foydalanuvchi xotirasi (migratsiya 0026, `services/user_memory.py`).

Qo'riqlanadigan narsalar:

  * ajratish naqshlari (uz/ru/en) — alias, afzallik, fakt; bir martalik
    so'rov ("shuni qisqa yoz") afzallik bo'lmaydi;
  * telefon, kod, parol, karta **hech qachon** saqlanmaydi (naqsh ham, LLM ham);
  * LLM faqat eslatma so'zi bo'lsa va naqsh topmasa — **bitta** `Task.ROUTE`;
    LLM qiymati foydalanuvchi matnida bo'lishi shart;
  * planner kirishida `user_memory` bloki, alias → chat (kodda) → reja va
    `conv_state.resolved_entities`; afzallik sintezga; "Eslab qoldim" qatori;
  * global kalit o'chiq → hech narsa ishlatilmaydi va saqlanmaydi;
  * API: login + CSRF, validatsiya, begona yozuv 404.

Real LLM yo'q; Postgres bo'lsa — haqiqiy yozish/dedup/qopqoq.
"""

# Naqshlar uch tilda — mahsulot matni (9-18 soat oralig'i en dash bilan ham).
# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_orchestrator import (  # type: ignore[import-not-found]
    FakeLLM,
    _hit,
    _plan,
    run,
    use_multi,
    wired,
)

from app.llm import Task
from app.llm.base import ChatResult, Usage
from app.services import conv_state as CS
from app.services import orchestrator as OR
from app.services import user_memory as UM
from app.services.prompts import MEMORY_EXTRACT_PROMPT
from app.services.prompts_deep import DEEP_SYNTH_PROMPT
from app.web import main as web_main
from app.web import security as sec

__all__ = ["wired"]

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
H = {"X-Requested-With": "fetch"}


def _kinds(items: list[UM.Candidate]) -> list[tuple[str, str, str]]:
    return [(c.kind, c.key, c.value) for c in items]


# ─── ajratish naqshlari ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Eslab qol: Ishxona deganda Ish chat guruhini nazarda tutaman",
            ("alias", "Ishxona", "Ish chat"),
        ),
        (
            "Bundan keyin «Uydagilar» deganda Oila chatni nazarda tutaman",
            ("alias", "Uydagilar", "Oila chat"),
        ),
        (
            "Запомни: когда я говорю Работа, я имею в виду группу Ish chat",
            ("alias", "Работа", "Ish chat"),
        ),
        ("Remember: when I say Work, I mean the Ish chat group", ("alias", "Work", "Ish chat")),
        ("Meni Aziz aka deb chaqir", ("alias", "self", "Aziz aka")),
        ("call me Boss", ("alias", "self", "Boss")),
        ("javobni ruscha ber", ("preference", "language", "ru")),
        ("har doim inglizcha javob ber", ("preference", "language", "en")),
        ("отвечай всегда по-русски", ("preference", "language", "ru")),
        ("qisqa yoz", ("preference", "length", "short")),
        ("bundan keyin batafsil yoz", ("preference", "length", "detailed")),
        ("har doim jadval qilib ber", ("preference", "format", "table")),
        ("men Samarqanddaman", ("fact", "location", "Samarqand")),
        ("I live in Tashkent", ("fact", "location", "Tashkent")),
        ("ish vaqtim 9–18", ("fact", "work_hours", "9–18")),
        ("ish vaqtim 9 dan 18 gacha", ("fact", "work_hours", "9–18")),
    ],
)
def test_rules_extract_aliases_preferences_and_facts(
    text: str, expected: tuple[str, str, str]
) -> None:
    got = _kinds(UM.extract_rules(text))
    assert expected in got, got


@pytest.mark.parametrize(
    "text",
    [
        "Otabek kecha nima dedi?",
        "shuni qisqa yoz",  # oldingi javob — bir martalik
        "bugun nima bo'ldi? qisqa yoz",  # savol turn'i — shu javob uchun
        "buni ruscha yoz",
        "jadval qilib ber",  # "har doim"siz — shu javob uchun format
        "men bugun charchadim",
        "",
    ],
)
def test_rules_ignore_one_off_requests_and_plain_questions(text: str) -> None:
    assert UM.extract_rules(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Eslab qol: telefonim +998 90 123 45 67",
        "Eslab qol, Otabek deganda +998901234567 raqamini nazarda tutaman",
        "unutma: kodim 12345",
        "Eslab qol: parolim qwerty, men Samarqanddaman",
        "Запомни мой пароль: hunter2",
        "remember my card number 4111 1111 1111 1111",
        "eslab qol: tasdiqlash kodi 55821",
        "Ishxona deganda sk-abcdefghijklmnopqrstuvwxyz0123 ni nazarda tutaman",
        "remember my email aziz@example.com",
    ],
)
def test_phone_code_password_are_never_saved(text: str) -> None:
    assert UM.is_sensitive(text)
    assert UM.extract_rules(text) == []


async def test_sensitive_turn_gets_no_llm_call_either() -> None:
    calls: list[Any] = []

    async def chat(*a: Any, **kw: Any) -> Any:
        calls.append(a)
        raise AssertionError("LLM chaqirilmasligi kerak")

    assert await UM.extract("eslab qol: kodim 4821, bundan keyin qisqa yoz", chat) == []
    assert calls == []


def test_values_are_checked_too() -> None:
    assert not UM.valid(UM.Candidate("fact", "phone", "+998 90 123 45 67"))
    assert not UM.valid(UM.Candidate("preference", "language", "de"))
    assert not UM.valid(UM.Candidate("preference", "mood", "happy"))
    assert not UM.valid(UM.Candidate("kind?", "a", "b"))
    assert UM.valid(UM.Candidate("fact", "location", "Samarqand"))


# ─── LLM faqat eslatma so'zi bilan ───────────────────────────────────────────


class _CountLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: Any, messages: list[Any], **kw: Any) -> ChatResult:
        self.calls.append(
            {"task": task, "system": messages[0].content, "user": messages[-1].content}
        )
        return ChatResult(
            text=self.reply,
            tool_calls=[],
            model="fake",
            provider="fake",
            usage=Usage(tokens_in=5, tokens_out=5),
        )


async def test_llm_runs_only_on_a_memory_cue_and_only_when_rules_found_nothing() -> None:
    reply = '{"items": [{"kind": "fact", "key": "job", "value": "dasturchi"}]}'
    llm = _CountLLM(reply)
    # Eslatma so'zi yo'q — hech qanday chaqiruv.
    assert await UM.extract("men dasturchiman, kecha nima bo'ldi?", llm.chat) == []
    assert llm.calls == []
    # Eslatma so'zi bor, lekin naqsh topdi — LLM kerak emas.
    got = await UM.extract("Eslab qol: men Samarqanddaman", llm.chat)
    assert _kinds(got) == [("fact", "location", "Samarqand")] and llm.calls == []
    # Eslatma so'zi bor, naqsh yo'q — **bitta** arzon chaqiruv.
    got = await UM.extract("eslab qol, men dasturchi bo'lib ishlayman", llm.chat)
    assert _kinds(got) == [("fact", "job", "dasturchi")]
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["task"] == Task.ROUTE and call["system"] == MEMORY_EXTRACT_PROMPT
    assert call["user"].startswith("<user_turn>")


async def test_llm_items_must_come_from_the_users_words_and_hold_no_secrets() -> None:
    reply = (
        '{"items": ['
        '{"kind": "fact", "key": "phone", "value": "+998901234567"},'
        '{"kind": "alias", "key": "Ofis", "value": "Bosh ofis guruhi"},'
        '{"kind": "fact", "key": "City!", "value": "Buxoro"},'
        '{"kind": "preference", "key": "language", "value": "ru"}'
        "]}"
    )
    llm = _CountLLM(reply)
    got = await UM.extract("eslab qol, men asli Buxorodanman, ruschani yaxshi bilaman", llm.chat)
    assert _kinds(got) == [("fact", "city", "Buxoro"), ("preference", "language", "ru")]
    assert UM.parse_llm_items("not json", "x") == []
    assert UM.parse_llm_items('{"items": "x"}', "x") == []


# ─── planner / sintez bloki ──────────────────────────────────────────────────


def _item(i: int, kind: str, key: str, value: str) -> UM.MemoryItem:
    return UM.MemoryItem(id=i, kind=kind, key=key, value=value, enabled=True)


MEM = UM.Memory(
    enabled=True,
    items=[
        _item(1, "alias", "Ishxona", "Ish chat"),
        _item(2, "preference", "language", "ru"),
        _item(3, "preference", "length", "short"),
        _item(4, "fact", "location", "Samarqand"),
        _item(5, "alias", "self", "Aziz"),
    ],
)


def test_block_lists_memory_and_the_code_resolved_alias(wired: dict[str, Any]) -> None:
    chats = wired["chats"]  # Honadoshlar(1), Ish chat(2), Arxiv(3)
    aliases = UM.alias_matches("Ishxonada bugun nima gaplar?", MEM.items, chats)
    assert [(it.key, int(h.chat_id)) for it, h in aliases] == [("Ishxona", 2)]
    assert UM.alias_matches("Honadoshlar nima dedi?", MEM.items, chats) == []
    run_mem = UM.RunMemory(memory=MEM, aliases=aliases)
    block = UM.render_block(run_mem)
    assert block.startswith('<untrusted_data kind="user_memory" count="5">')
    assert 'alias "Ishxona" = "Ish chat" → chat "Ish chat" [id:2]' in block
    assert "preference language = ru" in block and "fact location = Samarqand" in block
    assert 'alias self = "Aziz"' in block and "memory_saved" not in block
    note = UM.synth_note(run_mem)
    assert "answer language: Russian" in note and "length: short" in note
    assert "location=Samarqand" in note and 'address the user as "Aziz"' in note
    assert UM.render_block(UM.RunMemory()) == ""
    assert "rule 16" in block and "<user_memory>" in OR.ORCH_PLAN_PROMPT


def _mem_store(monkeypatch: pytest.MonkeyPatch, memory: UM.Memory) -> dict[str, Any]:
    box: dict[str, Any] = {"saved": [], "memory": memory}

    async def load(session: Any, user_id: int) -> UM.Memory:
        m = box["memory"]
        return UM.Memory(enabled=m.enabled, items=list(m.items))

    async def save(session: Any, user_id: int, cands: Any, **kw: Any) -> list[UM.MemoryItem]:
        box["saved"].append(list(cands))
        return [_item(100 + i, c.kind, c.key, c.value) for i, c in enumerate(cands)]

    monkeypatch.setattr(UM, "load", load)
    monkeypatch.setattr(UM, "save", save)
    return box


async def test_alias_resolves_the_chat_and_feeds_the_conversation_state(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    _mem_store(monkeypatch, MEM)
    box: dict[str, Any] = {}

    async def save_state(_f: Any, conversation_id: int, state: CS.ConvState) -> bool:
        box["state"] = state
        return True

    monkeypatch.setattr(CS, "save", save_state)
    use_multi(monkeypatch, lambda q: [_hit(i, chat_id=2) for i in range(1, 6)])
    llm = FakeLLM(
        plan=_plan(server_scope="none", steps=[{"skill": "search_local"}]),
        rerank='{"ranking":[0,1,2]}',
        synth="Ответ #c2m101.",
    )
    out = await run("Ishxonada ijara haqida nima deyishdi?", llm, mode="fast")
    assert out.plan["chats"] == [2]
    assert any(n.startswith("memory: alias") for n in out.plan["plan_notes"])
    planner = llm.user_turn(OR.ORCH_PLAN_PROMPT)
    assert '<untrusted_data kind="user_memory"' in planner
    assert "[id:2] (named in this question" in planner
    synth = llm.user_turn(DEEP_SYNTH_PROMPT)
    assert "answer language: Russian" in synth and "locale: ru" in synth
    assert box["state"].resolved_entities["Ishxona"]["chat_id"] == 2
    assert "Eslab qoldim" not in out.answer  # shu turn hech narsa saqlamadi


async def test_saved_items_get_one_short_line_in_the_answer(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    store = _mem_store(monkeypatch, UM.Memory(enabled=True))
    llm = FakeLLM(
        plan=_plan(intent_follow_up="chitchat", steps=[{"skill": "answer_direct"}]),
        direct="Albatta!",
    )
    out = await run("Eslab qol: men Samarqanddaman", llm, mode="fast")
    assert _kinds(store["saved"][0]) == [("fact", "location", "Samarqand")]
    assert out.answer.startswith("Albatta!")
    last = out.answer.strip().splitlines()[-1]
    assert last == "_Eslab qoldim: shahar: Samarqand (Sozlamalar → Xotira)_"
    assert "memory_saved: this turn saved 1 item(s)" in llm.user_turn(OR.ORCH_PLAN_PROMPT)


async def test_memory_off_means_nothing_used_and_nothing_saved(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    store = _mem_store(monkeypatch, UM.Memory(enabled=False))

    async def no_extract(*a: Any, **kw: Any) -> Any:
        raise AssertionError("xotira o'chiq — ajratish yo'q")

    monkeypatch.setattr(UM, "extract", no_extract)
    use_multi(monkeypatch, lambda q: [_hit(i, chat_id=2) for i in range(1, 6)])
    llm = FakeLLM(
        plan=_plan(server_scope="none", steps=[{"skill": "search_local"}]),
        rerank='{"ranking":[0,1,2]}',
    )
    out = await run("Eslab qol: Ishxona deganda Ish chat guruhini nazarda tutaman", llm)
    assert store["saved"] == []
    assert "user_memory" not in llm.user_turn(OR.ORCH_PLAN_PROMPT)
    assert "Eslab qoldim" not in out.answer
    assert llm.count(MEMORY_EXTRACT_PROMPT) == 0


async def test_prepare_survives_a_broken_db() -> None:
    class _Broken:
        async def __aenter__(self) -> Any:
            raise ConnectionError("db down")

        async def __aexit__(self, *exc: Any) -> None:
            return None

    out = await UM.prepare(
        lambda: _Broken(), None, user_id=7, conversation_id=1, question="eslab qol: qisqa yoz"
    )
    assert not out.enabled and out.saved == []


def test_saved_note_is_localized_and_short() -> None:
    items = [_item(1, "preference", "language", "ru"), _item(2, "alias", "Ishxona", "Ish chat")]
    assert UM.saved_note(items, "uz") == (
        "_Eslab qoldim: javob tili — ruscha; «Ishxona» = Ish chat (Sozlamalar → Xotira)_"
    )
    assert UM.saved_note(items, "ru").startswith("_Запомнил: язык ответа — русский")
    assert UM.saved_note([], "uz") == ""


# ─── API ─────────────────────────────────────────────────────────────────────


class _Scope:
    def __init__(self, db: Any) -> None:
        self.db = db

    async def __aenter__(self) -> Any:
        return self.db

    async def __aexit__(self, *exc: Any) -> None:
        return None


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, dict[str, Any]]]:
    from app.web.routers import settings as R

    now = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)
    state: dict[str, Any] = {
        "enabled": True,
        "items": {
            1: UM.MemoryItem(1, "alias", "Ishxona", "Ish chat", 42, now, None, True),
            2: UM.MemoryItem(2, "preference", "language", "ru", None, now, None, True),
        },
        "owner": {1: 7, 2: 7, 3: 8},
    }

    async def get_enabled(db: Any, user_id: int) -> bool:
        return bool(state["enabled"])

    async def list_all(db: Any, user_id: int) -> list[UM.MemoryItem]:
        return [it for i, it in state["items"].items() if state["owner"].get(i) == user_id]

    async def set_enabled(db: Any, user_id: int, enabled: bool) -> bool:
        state["enabled"] = enabled
        return True

    async def set_item_enabled(
        db: Any, user_id: int, item_id: int, enabled: bool
    ) -> UM.MemoryItem | None:
        if state["owner"].get(item_id) != user_id or item_id not in state["items"]:
            return None
        state["items"][item_id].enabled = enabled
        return state["items"][item_id]

    async def delete_item(db: Any, user_id: int, item_id: int) -> bool:
        if state["owner"].get(item_id) != user_id or item_id not in state["items"]:
            return False
        del state["items"][item_id]
        return True

    async def delete_all(db: Any, user_id: int) -> int:
        mine = [i for i in state["items"] if state["owner"].get(i) == user_id]
        for i in mine:
            del state["items"][i]
        return len(mine)

    monkeypatch.setattr(R, "session_scope", lambda: _Scope(SimpleNamespace()))
    for name, fn in {
        "get_enabled": get_enabled,
        "list_all": list_all,
        "set_enabled": set_enabled,
        "set_item_enabled": set_item_enabled,
        "delete_item": delete_item,
        "delete_all": delete_all,
    }.items():
        monkeypatch.setattr(R.UM, name, fn)
    with TestClient(web_main.app) as c:
        c.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
        yield c, state


def test_memory_api_requires_login_and_csrf(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, state = api
    assert c.delete("/api/settings/memory").status_code == 403
    assert c.patch("/api/settings/memory/1", json={"enabled": False}).status_code == 403
    assert c.delete("/api/settings/memory/1").status_code == 403
    assert c.put("/api/settings/memory/toggle", json={"enabled": False}).status_code == 403
    c.cookies.clear()
    assert c.get("/api/settings/memory").status_code == 401
    assert c.delete("/api/settings/memory", headers=H).status_code == 401
    assert (
        c.put("/api/settings/memory/toggle", json={"enabled": False}, headers=H).status_code == 401
    )
    assert len(state["items"]) == 2 and state["enabled"] is True


def test_memory_api_lists_toggles_and_deletes(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, _state = api
    body = c.get("/api/settings/memory").json()
    assert body["enabled"] is True and body["max"] == UM.MAX_ITEMS
    first = {it["id"]: it for it in body["items"]}[1]
    assert first == {
        "id": 1,
        "kind": "alias",
        "key": "Ishxona",
        "value": "Ish chat",
        "source_conversation_id": 42,
        "created_at": "2026-09-28T10:00:00+00:00",
        "last_used_at": None,
        "enabled": True,
    }
    r = c.patch("/api/settings/memory/1", json={"enabled": False}, headers=H)
    assert r.status_code == 200 and r.json()["item"]["enabled"] is False
    r = c.put("/api/settings/memory/toggle", json={"enabled": False}, headers=H)
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert c.delete("/api/settings/memory/2", headers=H).json() == {"ok": True, "id": 2}
    r = c.delete("/api/settings/memory", headers=H)
    assert r.status_code == 200 and r.json()["removed"] == 1 and r.json()["items"] == []


def test_memory_api_validation_and_foreign_items(api: tuple[TestClient, dict[str, Any]]) -> None:
    c, state = api
    assert c.patch("/api/settings/memory/1", json={}, headers=H).status_code == 422
    assert (
        c.patch("/api/settings/memory/1", json={"enabled": "maybe"}, headers=H).status_code == 422
    )
    assert c.patch("/api/settings/memory/abc", json={"enabled": True}, headers=H).status_code == 422
    assert c.put("/api/settings/memory/toggle", json={}, headers=H).status_code == 422
    # Begona foydalanuvchining yozuvi (owner 8) — 404, o'zgarmaydi.
    state["items"][3] = UM.MemoryItem(3, "fact", "location", "Buxoro")
    r = c.patch("/api/settings/memory/3", json={"enabled": False}, headers=H)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "web.memory.err_not_found"
    assert c.delete("/api/settings/memory/3", headers=H).status_code == 404
    assert state["items"][3].enabled is True


# ─── haqiqiy Postgres ────────────────────────────────────────────────────────


async def test_memory_roundtrip_on_postgres(pg_engine: Any) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from test_db_integration import _fixture_chat  # type: ignore[import-not-found]

    account_id, _ = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        uid = int(
            (
                await conn.execute(
                    text("SELECT user_id FROM accounts WHERE id = :a"), {"a": account_id}
                )
            ).scalar_one()
        )
        await conn.execute(text("DELETE FROM user_memories WHERE user_id = :u"), {"u": uid})
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    cands = [
        UM.Candidate("alias", "Ishxona", "Ish chat"),
        UM.Candidate("preference", "language", "ru"),
    ]
    async with maker() as s:
        saved = await UM.save(s, uid, cands, conversation_id=None)
        await s.commit()
    assert [it.key for it in saved] == ["Ishxona", "language"]
    async with maker() as s:
        # Bir xil qiymat — hech narsa; fold qilingan kalit bo'yicha dedup.
        same = await UM.save(
            s, uid, [UM.Candidate("alias", "ISHXONA", "Ish chat")], conversation_id=None
        )
        assert same == []
        again = await UM.save(
            s, uid, [UM.Candidate("preference", "language", "en")], conversation_id=None
        )
        await s.commit()
    assert [(it.key, it.value) for it in again] == [("language", "en")]
    async with maker() as s:
        mem = await UM.load(s, uid)
        assert mem.enabled and mem.pref("language") == "en" and len(mem.items) == 2
        alias = next(it for it in mem.items if it.kind == "alias")
        assert (await UM.set_item_enabled(s, uid, alias.id, False)).enabled is False  # type: ignore[union-attr]
        assert await UM.set_item_enabled(s, uid + 999_999, alias.id, True) is None
        await UM.touch(s, uid, [alias.id])
        await s.commit()
    async with maker() as s:
        mem = await UM.load(s, uid)
        assert [it.kind for it in mem.items] == ["preference"]  # o'chirib qo'yilgani yo'q
        assert len(await UM.list_all(s, uid)) == 2
        assert await UM.set_enabled(s, uid, False)
        await s.commit()
    async with maker() as s:
        assert (await UM.load(s, uid)).enabled is False
        assert await UM.get_enabled(s, uid) is False
        await UM.set_enabled(s, uid, True)
        # Qopqoq: 100 dan ortig'i yozilmaydi.
        many = [UM.Candidate("fact", f"f{i}", f"qiymat {i}") for i in range(120)]
        await UM.save(s, uid, many, conversation_id=None)
        await s.commit()
    async with maker() as s:
        assert len(await UM.list_all(s, uid)) == UM.MAX_ITEMS
        assert await UM.delete_all(s, uid) == UM.MAX_ITEMS
        await s.commit()
