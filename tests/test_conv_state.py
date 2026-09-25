"""Suhbat ishchi holati (`services/conv_state.py`, migratsiya 0024).

Qo'riqlanadigan narsalar:

  * "Otabek kecha nima dedi?" → holatda `Otabek → 3893`, oyna `yesterday`;
    "u yana nima yozdi?" → reja **3893** ga qaratiladi (planner "all" +
    `clarify` desa ham), oyna meros qoladi, qayta aniqlash yo'q;
  * `correction` — hech narsa meros qolmaydi, oyna/natija tozalanadi;
  * holatda xabar matni bo'lsa `_Taint` ushlaydi (saqlash ham, plan ham);
  * sof yordamchilar: til, referent, kalit, buzuq JSON;
  * haqiqiy Postgres'da `conversations.state` ustuni (bo'lsa).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from test_orchestrator import (  # type: ignore[import-not-found]
    INJECTION,
    FakeLLM,
    _chat,
    _hit,
    _plan,
    run,
    use_multi,
    wired,
)

from app.db.models import ChatType
from app.services import conv_state as CS
from app.services import deep_search as DS
from app.services import orchestrator as OR

__all__ = ["wired"]  # fixture — import qilingan, pytest shu nom bilan topadi


# ─── sof yordamchilar ────────────────────────────────────────────────────────


def test_language_referent_and_entity_key() -> None:
    assert CS.detect_language("Otabek nima dedi?") == "uz"
    assert CS.detect_language("Что он написал вчера?") == "ru"
    assert CS.detect_language("Ўтган ойда нима бўлди?") == "uz"
    assert CS.detect_language("what did he write?") == "en"
    assert CS.detect_language("ok", previous="ru") == "ru"
    for q in (
        "u yana nima yozdi?",
        "unga nima dedim?",
        "o'sha odam kim edi",
        "shu guruhda nima bo'ldi",
        "что он писал?",
        "what did he say",
        "in that chat",
    ):
        assert CS.has_referent(q), q
    for q in ("bu oyda nechta xabar yozdim", "Otabek nima dedi", "ijara narxi"):
        assert not CS.has_referent(q), q
    assert CS.entity_key("Otabekning xabarlari qani", "Otabek Karimov") == "Otabek"
    assert CS.entity_key("IT guruhida kim bor", "Qandaydir chat") == "Qandaydir chat"


def test_from_json_is_defensive() -> None:
    st = CS.from_json(
        {
            "resolved_entities": {"Ali": {"chat_id": "12", "title": "Ali <b>"}, "x": "bad"},
            "selected_chats": [1, "2", -3, True],
            "last_window": {"since": None, "until": "2026-09-25T00:00:00+00:00", "kind": "today"},
            "last_result": {
                "skill": "stats",
                "evidence_refs": [{"chat_id": 1, "tg_msg_id": 5}, {"chat_id": "x"}],
                "facts": {"total": 12, "label": "text", "flag": True},
            },
            "language": "fr",
        }
    )
    assert list(st.resolved_entities) == ["Ali"] and st.resolved_entities["Ali"]["chat_id"] == 12
    assert "<" not in st.resolved_entities["Ali"]["title"]  # sarlavha teg yasay olmaydi
    assert st.selected_chats == [1, 2]
    assert st.last_result["evidence_refs"] == [{"chat_id": 1, "tg_msg_id": 5}]
    assert st.last_result["facts"] == {"total": 12}
    assert st.language == "" and CS.from_json("junk").empty
    block = CS.render_block(st)
    assert block.startswith('<untrusted_data kind="conversation_state">')
    assert '"evidence_refs": 1' in block  # havolalar sanoq bo'lib, ro'yxat emas


def test_taint_catches_message_text_in_state() -> None:
    taint = DS._Taint()
    taint.add_hits([_hit(1, text=f"Maxfiy: {INJECTION}")])
    clean = CS.ConvState(topic="Otabek kecha nima dedi?")
    taint.assert_state("plan", CS.render_block(clean))
    dirty = CS.ConvState(topic=f"Maxfiy: {INJECTION}")
    assert taint.state_leak(CS.render_block(dirty))
    with pytest.raises(DS.DeepSearchError):
        taint.assert_state("plan", CS.render_block(dirty))


# ─── uchdan-uchgacha: referent, oyna merosi, tuzatish ────────────────────────


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]) -> dict[str, Any]:
    otabek = _chat(3893, "Otabek Karimov")
    otabek.type = ChatType.PRIVATE
    wired["chats"].append(otabek)
    box: dict[str, Any] = {"state": CS.ConvState(), "saves": 0}

    async def load(_f: Any, conversation_id: int) -> CS.ConvState:
        return CS.from_json(box["state"].to_json())

    async def save(_f: Any, conversation_id: int, state: CS.ConvState) -> bool:
        box["state"] = state
        box["saves"] += 1
        return True

    monkeypatch.setattr(CS, "load", load)
    monkeypatch.setattr(CS, "save", save)
    use_multi(monkeypatch, lambda q: [_hit(i, chat_id=3893) for i in range(1, 6)])
    return box


async def test_referent_resolves_from_state_and_window_carries_over(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any], stored: dict[str, Any]
) -> None:
    # 1-turn: nom savolda, planner chatni topdi.
    llm1 = FakeLLM(
        plan=_plan(
            standalone_question="Otabek kecha nima dedi?",
            target_chats=[3893],
            server_scope="none",
            steps=[{"skill": "search_local"}],
        ),
        rerank='{"ranking":[0,1,2]}',
        synth="Otabek ijara haqida yozgan #c3893m101.",
    )
    first = await run("Otabek kecha nima dedi?", llm1, mode="fast", context_mode="agent")
    st = stored["state"]
    assert st.resolved_entities["Otabek"]["chat_id"] == 3893 and st.last_entity == "Otabek"
    assert st.last_window and st.last_window["kind"] == "yesterday"
    assert st.language == "uz" and st.last_result["skill"] == "search_local"
    assert st.last_result["evidence_refs"]
    assert '"Otabek Karimov" [id:3893] private' in llm1.user_turn(OR.ORCH_PLAN_PROMPT)

    # 2-turn: "u" — planner hech narsa bilmasa ham (all + clarify) kod holatdan oladi.
    wired["history"] = [
        DS.Turn(role="user", text="Otabek kecha nima dedi?"),
        DS.Turn(role="assistant", text="Otabek ijara haqida yozgan.", plan=first.plan),
    ]
    llm2 = FakeLLM(
        plan=_plan(
            standalone_question="U yana nima yozdi?",
            target_chats="all",
            server_scope="none",
            steps=[{"skill": "clarify", "args": {"question": "Kim?"}}],
        ),
        rerank='{"ranking":[0,1]}',
        synth="Yana to'lov haqida yozgan #c3893m102.",
    )
    second = await run("u yana nima yozdi?", llm2, mode="fast", context_mode="agent")
    assert second.plan["chats"] == [3893] and second.plan["target_chats"] == [3893]
    assert [s["skill"] for s in second.plan["steps"]] == ["search_local"]
    assert second.plan["since"] == first.plan["since"] and second.plan["window_from_question"]
    notes = " | ".join(second.plan["plan_notes"])
    assert "referent → Otabek [id:3893]" in notes and "window carried over" in notes
    plan_turn = llm2.user_turn(OR.ORCH_PLAN_PROMPT)
    assert 'kind="conversation_state"' in plan_turn and '"last_entity": "Otabek"' in plan_turn
    assert stored["state"].last_entity == "Otabek" and stored["saves"] == 2

    # 3-turn: tuzatish — hech narsa meros qolmaydi, oyna/natija tozalanadi.
    wired["history"] += [
        DS.Turn(role="user", text="u yana nima yozdi?"),
        DS.Turn(role="assistant", text="Yana to'lov haqida.", plan=second.plan),
    ]
    llm3 = FakeLLM(
        plan=_plan(
            standalone_question="Ijara narxi haqida kim yozgan?",
            intent_follow_up="correction",
            target_chats="all",
            server_scope="none",
            steps=[{"skill": "search_local"}],
        ),
        rerank='{"ranking":[0]}',
    )
    third = await run("yo'q, u emas — ijara narxi haqida kim yozgan?", llm3, mode="fast")
    assert third.plan["chats"] is None and third.plan["since"] is None
    assert "correction — nothing carried over" in " | ".join(third.plan["plan_notes"])
    assert stored["state"].last_window is None


async def test_new_window_in_the_follow_up_wins_over_the_state(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any], stored: dict[str, Any]
) -> None:
    now = datetime.now(UTC)
    stored["state"] = CS.ConvState(
        resolved_entities={"Otabek": {"chat_id": 3893, "title": "Otabek Karimov", "kind": ""}},
        last_entity="Otabek",
        last_window={
            "since": (now - timedelta(days=30)).isoformat(),
            "until": (now - timedelta(days=29)).isoformat(),
            "kind": "date",
            "tz": "Asia/Tashkent",
        },
    )
    wired["history"] = [DS.Turn(role="user", text="Otabek nima dedi?")]
    llm = FakeLLM(plan=_plan(server_scope="none", steps=[{"skill": "search_local"}]))
    out = await run("u bugun nima yozdi?", llm, mode="fast")
    assert out.plan["chats"] == [3893] and out.plan["window_kind"] == "today"


async def test_reuse_prior_reads_the_state_refs(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any], stored: dict[str, Any]
) -> None:
    stored["state"] = CS.ConvState(
        last_result={"skill": "stats", "evidence_refs": [{"chat_id": 1, "tg_msg_id": 101}]}
    )
    seen: list[Any] = []

    async def refs_to_hits(session: Any, refs: Any, **kw: Any) -> list[Any]:
        seen.append(list(refs))
        return [_hit(1)]

    monkeypatch.setattr(DS, "_refs_to_hits", refs_to_hits)
    from app.services import skills as SK

    ctx = SK.RunState(
        session_factory=lambda: None,  # type: ignore[arg-type,return-value]
        account_id=1,
        user_id=7,
        conversation_id=10,
        question="q",
        raw_question="q",
        locale="uz",
        mode="fast",
        confirm=True,
        deep=False,
        meter=DS._Meter(FakeLLM()),  # type: ignore[arg-type]
        taint=DS._Taint(),
        events=None,
        history=[],
        tz="Asia/Tashkent",
        chats_by_id={},
        plan=DS.DeepPlan(),
        conv=stored["state"],
    )

    class _S:
        async def __aenter__(self) -> Any:
            return object()

        async def __aexit__(self, *exc: Any) -> None:
            return None

    ctx.session_factory = lambda: _S()  # type: ignore[assignment,return-value]
    res = await SK.SKILLS["reuse_prior"].run(ctx, {})
    assert seen == [[{"chat_id": 1, "tg_msg_id": 101}]] and res.summary == {"hits": 1}


# ─── haqiqiy Postgres: ustun + load/save ─────────────────────────────────────


async def test_state_column_roundtrip(pg_engine: Any) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)

    class _Scope:
        async def __aenter__(self) -> Any:
            self.s = maker()
            return self.s

        async def __aexit__(self, *exc: Any) -> None:
            if exc[0] is None:
                await self.s.commit()
            await self.s.close()

    async with pg_engine.begin() as conn:
        uid = (
            await conn.execute(
                text(
                    "INSERT INTO users (tg_user_id, username) VALUES (900000441, 'st') RETURNING id"
                )
            )
        ).scalar_one()
        cid = (
            await conn.execute(
                text("INSERT INTO conversations (user_id, title) VALUES (:u, 't') RETURNING id"),
                {"u": uid},
            )
        ).scalar_one()
        raw = (
            await conn.execute(text("SELECT state FROM conversations WHERE id = :c"), {"c": cid})
        ).scalar_one()
    try:
        assert raw == {}  # sukut — eski suhbatlar bo'sh holat bilan
        st = CS.ConvState(
            resolved_entities={"Otabek": {"chat_id": 3893, "title": "Otabek", "kind": "private"}},
            last_entity="Otabek",
            language="uz",
        )
        assert await CS.save(lambda: _Scope(), int(cid), st)
        back = await CS.load(lambda: _Scope(), int(cid))
        assert back.resolved_entities == st.resolved_entities and back.last_entity == "Otabek"
        assert (await CS.load(lambda: _Scope(), 10**12)).empty
    finally:
        async with pg_engine.begin() as conn:
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})


# ─── API: faqat o'qish, faqat egasi ──────────────────────────────────────────


def test_state_api_is_read_only_and_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from app.services import chat_service as cs
    from app.web import main as web_main
    from app.web import security as sec
    from app.web.routers import chat as chat_router

    class _Scope:
        async def __aenter__(self) -> Any:
            return object()

        async def __aexit__(self, *exc: Any) -> None:
            return None

    async def fake_conv(db: Any, user_id: int, conversation_id: int) -> Any:
        if user_id != 7:
            raise cs.ChatError("not_found")
        return SimpleNamespace(
            id=conversation_id,
            state={
                "resolved_entities": {"Otabek": {"chat_id": 3893, "title": "Otabek"}},
                "last_entity": "Otabek",
            },
        )

    monkeypatch.setattr(chat_router, "session_scope", lambda: _Scope())
    monkeypatch.setattr(chat_router.cs, "get_conversation", fake_conv)
    with TestClient(web_main.app) as client:
        assert client.get("/api/conversations/1/state").status_code == 401
        client.cookies.set(sec.SESSION_COOKIE, sec.issue_session(7))
        r = client.get("/api/conversations/1/state")
        assert r.status_code == 200
        body = r.json()
        assert body["conversation_id"] == 1 and body["state"]["last_entity"] == "Otabek"
        assert body["state"]["resolved_entities"]["Otabek"]["chat_id"] == 3893
        # Yozish yo'li yo'q.
        assert client.put("/api/conversations/1/state", json={}).status_code in (403, 405)
        client.cookies.set(sec.SESSION_COOKIE, sec.issue_session(8))
        assert client.get("/api/conversations/1/state").status_code == 404
