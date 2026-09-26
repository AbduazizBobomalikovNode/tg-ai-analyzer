"""Orkestrator bo'shliqlari (2026-09-26 jonli eval'dan): 1, 3-8-bandlar.

  1. `answer_direct` — fikrlash zaxirasi + `minimal`, kesilsa **bitta** davom (daily-37);
  3. media savollari — `window`/`search_local` `media` argumenti, `[photo]` belgisi,
     nomsiz jo'natuvchi uchun "nomi yo'q" izohi (uni-16);
  4. dalilda yo'q sitata javobdan **olib tashlanadi**, qamrovda izoh (uni-16 `#c307m…`);
  5. bo'sh natija — `EMPTY_RESULT_PROMPT`, bog'liq bo'lmagan nomzodlarsiz (uni-18);
  6. nom to'qnashuvi — `clarify`, holatdagi yechim esa hamma qadamda (+ `send_result`);
  7. sanoq sarlavhasi va qamrov "oraliq" — akkaunt mintaqasida (uni-12);
  8. `meta_cost` — "bu javob qancha turdi" DB'dan, LLM'siz (uni-19).

2-band (tuzatishdan keyingi follow-up) — `test_conv_state.py`, 9-band — `test_evals.py`.
Real LLM/DB yo'q.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from test_orchestrator import (  # type: ignore[import-not-found]
    FakeLLM,
    _chat,
    _hit,
    _plan,
    run,
    use_multi,
    wired,
)

from app.db.models import ChatType
from app.llm.base import ChatResult, Usage
from app.services import agent as AG
from app.services import conv_state as CS
from app.services import deep_search as DS
from app.services import orchestrator as OR
from app.services import scale_search as SC
from app.services import skills as SK
from app.services import skills_ext as SX
from app.services import window_search as WS
from app.services.prompts_deep import (
    DEEP_SYNTH_PROMPT,
    EMPTY_RESULT_PROMPT,
)

__all__ = ["wired"]  # fixture — import qilingan, pytest shu nom bilan topadi

TZ = "Asia/Tashkent"


# ─── 1. answer_direct: fikrlash zaxirasi + bitta davom ───────────────────────


class _SeqLLM:
    """Ketma-ket javoblar (matn, finish_reason) — chaqiruv kwarg'lari yoziladi."""

    def __init__(self, *replies: tuple[str, str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: Any, messages: list[Any], **kw: Any) -> ChatResult:
        self.calls.append({"task": task, "messages": messages, **kw})
        text, finish = self.replies.pop(0)
        return ChatResult(
            text=text,
            tool_calls=[],
            model="gemini-3.5-flash",
            provider="gemini",
            usage=Usage(tokens_in=10, tokens_out=5),
            finish_reason=finish,
        )


async def test_direct_answer_reserves_thinking_and_continues_once() -> None:
    llm = _SeqLLM(
        ("Men sizning arxivingiz yordamchisiman:\n* **", "FinishReason.MAX_TOKENS"),
        ("Qidiruv** — mavzu bo'yicha.", "FinishReason.STOP"),
    )
    text, res = await DS.answer_direct(
        DS._Meter(llm),  # type: ignore[arg-type]
        "Sen nimalar qila olasan?",
        intent="meta",
    )
    assert DS.DIRECT_CALL_MAX_TOKENS == DS.DIRECT_MAX_TOKENS + DS.SYNTH_THINKING_RESERVE
    assert len(llm.calls) == 2
    for call in llm.calls:
        assert call["max_tokens"] == DS.DIRECT_CALL_MAX_TOKENS
        assert call["thinking"] == "minimal"
    assert llm.calls[1]["messages"][-1].content == DS.SYNTH_CONTINUE_NUDGE
    assert text.endswith("Qidiruv** — mavzu bo'yicha.") and text.startswith("Men sizning")
    assert res is not None and res.finish_reason == "FinishReason.STOP"


async def test_complete_direct_answer_is_one_call() -> None:
    llm = _SeqLLM(("Salom! Nima qidiray?", "STOP"))
    text, _res = await DS.answer_direct(
        DS._Meter(llm),  # type: ignore[arg-type]
        "salom",
        intent="greeting",
    )
    assert text == "Salom! Nima qidiray?" and len(llm.calls) == 1
    assert llm.calls[0]["thinking"] == "minimal"


# ─── 8. meta_cost ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "bu javob qancha turdi?",
        "qancha xarajat bo'ldi?",
        "shu javob necha pul turdi",
        "Сколько стоил этот ответ?",
        "how much did this answer cost?",
    ],
)
def test_cost_questions_hit_the_code_gate(question: str) -> None:
    assert DS.direct_intent(question) == "meta_cost", question


@pytest.mark.parametrize(
    "question",
    [
        "ijara narxi qancha?",
        "bu oyda nechta xabar yozdim?",
        "qancha pul sarfladim ijara uchun",
        "Otabek qancha turdi deb yozdi?",
    ],
)
def test_ordinary_questions_are_not_cost_questions(question: str) -> None:
    assert DS.direct_intent(question) != "meta_cost", question


def test_cost_text_is_rendered_by_code_in_three_languages() -> None:
    last = {
        "tokens_in": 12_000,
        "tokens_out": 800,
        "tokens_thinking": 300,
        "tokens_cached": 4_000,
        "cost_usd": 0.0262,
        "model": "gemini-3.5-flash",
    }
    totals = {"n": 2, "tokens": 25_000, "cost_usd": 0.0471}
    uz = SK.render_cost_text(last, totals, locale="uz")
    assert "12000 kirish + 800 chiqish" in uz and "$0.0262" in uz and "$0.0471" in uz
    assert "fikrlash: 300" in uz and "LLM chaqirmadi" in uz
    assert "$0.0262" in SK.render_cost_text(last, totals, locale="ru")
    assert "estimated cost" in SK.render_cost_text(last, totals, locale="en")
    assert "hali javob yo'q" in SK.render_cost_text(None, {"n": 0}, locale="uz")
    unknown = SK.render_cost_text({**last, "cost_usd": None}, {"n": 1}, locale="uz")
    assert "hisoblanmagan" in unknown


async def test_meta_cost_reads_the_db_and_calls_no_llm(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    seen: list[Any] = []

    async def fake_cost(ctx: SK.RunState) -> str:
        seen.append(ctx.conversation_id)
        return "Oldingi javob: 100 kirish + 20 chiqish token, taxminiy narx **$0.0100**."

    monkeypatch.setattr(SK, "meta_cost_text", fake_cost)
    llm = FakeLLM()
    out = await run("bu javob qancha turdi?", llm, mode="fast", context_mode="agent")
    assert seen == [10] and out.direct and out.intent == "meta_cost"
    assert "$0.0100" in out.answer
    assert llm.prompts == []  # planner ham, LLM ham chaqirilmadi
    assert [s["skill"] for s in out.plan["steps"]] == ["answer_direct"]


async def test_meta_cost_text_sql_shape() -> None:
    rows = [
        (1500, 200, 50, 0, 0.0123, "gemini-3.5-flash"),  # oxirgi assistant qatori
        (3, 9000, 0.04),  # jami
    ]

    class _Res:
        def __init__(self, row: Any) -> None:
            self.row = row

        def first(self) -> Any:
            return self.row

        def one(self) -> Any:
            return self.row

    class _Session:
        async def execute(self, stmt: Any) -> Any:
            return _Res(rows.pop(0))

    class _Scope:
        async def __aenter__(self) -> Any:
            return _Session()

        async def __aexit__(self, *exc: Any) -> None:
            return None

    ctx = SimpleNamespace(session_factory=lambda: _Scope(), conversation_id=5, locale="uz")
    text = await SK.meta_cost_text(ctx)  # type: ignore[arg-type]
    assert "1500 kirish + 200 chiqish" in text and "$0.0123" in text
    assert "3 ta javob, 9000 token" in text and "$0.0400" in text


# ─── 3. media ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("bu hafta kim menga rasm yuborgan?", "photo"),
        ("kecha kim video tashladi?", "video"),
        ("кто присылал мне фото на этой неделе?", "photo"),
        ("who sent me files today?", "document"),
        ("ovozli xabar yuborganlar kim", "voice"),
        ("rasm haqida nima deyishdi?", ""),  # mavzu savoli — filtr emas
        ("bugun kim yozdi?", ""),
    ],
)
def test_media_kind_is_detected_by_code(question: str, kind: str) -> None:
    assert WS.detect_media(question) == kind


def test_media_filter_maps_to_the_stored_media_types() -> None:
    assert WS.normalize_media("image") == "photo" and WS.normalize_media("junk") == ""
    # video / ovozli / fayl — sxemada hammasi `document` (MessageMediaDocument).
    assert WS.MEDIA_VALUES["video"] == WS.MEDIA_VALUES["voice"] == ("document",)
    spec = WS.WindowSpec(
        account_id=1, chats=[1], since=None, until=datetime.now(UTC), media="photo"
    )
    sql = str(WS._base(spec).compile(compile_kwargs={"literal_binds": True}))
    assert "media_type IN ('photo')" in sql
    anymedia = WS.WindowSpec(
        account_id=1, chats=[1], since=None, until=datetime.now(UTC), media="any"
    )
    sql_any = str(WS._base(anymedia).compile(compile_kwargs={"literal_binds": True}))
    assert "media_type IS NOT NULL" in sql_any and "webpage" in sql_any
    assert "media filter" in WS.render_window_stats(WS.WindowCount(total=1), spec=spec)
    hits = [_hit(1), _hit(2), _hit(3)]
    hits[0].media_type, hits[1].media_type, hits[2].media_type = "photo", "document", "webpage"
    assert [h.message_id for h in SK.media_filter(hits, "photo")] == [1]
    assert [h.message_id for h in SK.media_filter(hits, "any")] == [1, 2]


def test_evidence_line_marks_the_media_type() -> None:
    h = _hit(4, text="")
    h.media_type = "photo"
    h.sender = "user:777"
    [ev] = DS.to_evidence([h], [])
    line = DS.evidence_line(ev)
    assert "[photo]" in line and "user:777" in line


async def test_window_media_question_filters_and_forbids_guessing_names(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    seen: list[dict[str, Any]] = []

    async def run_window(_f: Any, **kw: Any) -> SC.ScaleOutcome:
        seen.append(kw)
        plan = kw["plan"]
        until = datetime(2026, 9, 26, 8, 0, tzinfo=UTC)
        spec = WS.WindowSpec(
            account_id=1,
            chats=None,
            since=until - timedelta(days=5),
            until=until,
            direction=plan.direction,
            media=kw.get("media", ""),
        )
        hits = [_hit(1, text=""), _hit(2, text="kulgili")]
        for h in hits:
            h.media_type, h.sender = "photo", "user:777"
        out = SC.ScaleOutcome(stats=SC.ScanStats())
        out.window, out.evidence, out.pool = spec, hits, hits
        out.window_count = WS.WindowCount(total=2, chats=1, first_at=spec.since)
        out.window_stats = WS.render_window_stats(
            out.window_count, titles={1: "Honadoshlar"}, evidence=hits, spec=spec
        )
        return out

    monkeypatch.setattr(WS, "run_window", run_window)
    llm = FakeLLM(
        plan=_plan(
            server_scope="none",
            steps=[{"skill": "window", "args": {"media": "photo"}}],
        ),
        synth="user:777 (nomi yo'q) 2 ta rasm yuborgan #c1m101.",
    )
    out = await run("bu hafta kim menga rasm yuborgan?", llm, mode="fast", context_mode="agent")
    assert seen and seen[0]["media"] == "photo" and seen[0]["plan"].direction == "in"
    synth = llm.user_turn(DEEP_SYNTH_PROMPT)
    assert "[photo]" in synth and "media filter" in synth and "nomi yo'q" in synth
    assert out.coverage.window_media == "photo"
    assert "[photo]" in DS.render_coverage(out.coverage)


# ─── 4. yaroqsiz sitata javobdan olib tashlanadi ─────────────────────────────


def _ev(token: str) -> DS.DeepEvidence:
    return DS.DeepEvidence(
        token=token, tg_msg_id=1, chat_id=1, chat_title="x", date=None, sender="", text=""
    )


def test_invalid_citations_are_stripped_with_their_links() -> None:
    evidence = [_ev("c3923m211766"), _ev("c3807m211637")]
    answer = (
        "* **Fari Xesh** — rasm yuborgan `#c3923m211766`, `#c307m211637`.\n"
        "* **Otabek** — PUBG haqida (#c307m211671) [#c3807m211637](https://t.me/c/3807/211637).\n"
        "* Yana biri #c307m211637 (https://t.me/c/307/211637) ham bor."
    )
    text, removed = DS.strip_invalid_citations(answer, evidence)
    assert removed == 3
    assert "c307m" not in text and "t.me/c/307/" not in text
    assert "`#c3923m211766`" in text and "[#c3807m211637](https://t.me/c/3807/211637)" in text
    assert "()" not in text and ", ." not in text and ",." not in text
    same, zero = DS.strip_invalid_citations("Hammasi to'g'ri #c3923m211766.", evidence)
    assert zero == 0 and same == "Hammasi to'g'ri #c3923m211766."


async def test_orchestrator_removes_invalid_citations_and_says_so(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    use_multi(monkeypatch, lambda q: [_hit(i) for i in range(1, 6)])
    llm = FakeLLM(
        plan=_plan(server_scope="none", steps=[{"skill": "search_local"}]),
        rerank='{"ranking":[0,1,2,3,4]}',
        synth="Ijara oshgan #c1m101, lekin #c307m211637 da boshqacha.",
    )
    out = await run("ijara narxi haqida kim nima yozgan?", llm, mode="fast")
    assert "#c307m211637" not in out.answer and "#c1m101" in out.answer
    assert out.coverage.invalid_citations_removed == 1 and out.invalid_citations == 1
    assert "1 ta noto'g'ri sitata olib tashlandi" in out.answer


# ─── 5. bo'sh natija ─────────────────────────────────────────────────────────


async def test_nothing_relevant_gets_the_honest_empty_answer(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    use_multi(monkeypatch, lambda q: [_hit(i) for i in range(1, 6)])
    llm = FakeLLM(
        plan=_plan(
            standalone_question="kvant pingvinlar festivali haqida kim nima yozgan?",
            subqueries=["kvant pingvinlar festivali"],
            server_scope="none",
            steps=[{"skill": "search_local"}],
        ),
        rerank='{"ranking":[]}',  # reranker: hech biri mos emas
    )
    llm.answers[EMPTY_RESULT_PROMPT] = (
        "«Kvant pingvinlar festivali» haqida hech narsa topilmadi. Boshqa so'z bilan so'rang."
    )
    out = await run("kvant pingvinlar festivali haqida kim nima yozgan?", llm, mode="fast")
    assert llm.count(OR.ORCH_REPLAN_PROMPT) == 1  # replan urinib ko'rildi
    assert llm.count(DEEP_SYNTH_PROMPT) == 0 and llm.count(EMPTY_RESULT_PROMPT) == 1
    turn = llm.user_turn(EMPTY_RESULT_PROMPT)
    assert "search summary" in turn and "judged relevant: 0" in turn
    assert "suggestion hints" in turn and "server search" in turn
    assert '"kvant pingvinlar festivali"' in turn
    assert "Ijara narxi" not in turn  # bog'liq bo'lmagan nomzodlar berilmaydi
    assert out.answer.startswith("«Kvant pingvinlar festivali» haqida hech narsa topilmadi")
    assert wired["persisted"]["context_meta"]["empty_result"] is True


async def test_relevant_hits_keep_the_normal_synthesis(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    use_multi(monkeypatch, lambda q: [_hit(i) for i in range(1, 6)])
    llm = FakeLLM(
        plan=_plan(server_scope="none", steps=[{"skill": "search_local"}]),
        rerank='{"ranking":[0,1,2]}',
    )
    await run("ijara narxi haqida kim nima yozgan?", llm, mode="fast")
    assert llm.count(EMPTY_RESULT_PROMPT) == 0 and llm.count(DEEP_SYNTH_PROMPT) == 1
    assert wired["persisted"]["context_meta"]["empty_result"] is False


def test_empty_strategy_never_replaces_counts_or_digests() -> None:
    ctx = SimpleNamespace(used={"stats"}, claims=[], facts={}, relevant_hits=None, ranked=[])
    assert OR.empty_result(ctx) is False  # type: ignore[arg-type]
    ctx.used = {"search_local"}
    assert OR.empty_result(ctx) is True  # type: ignore[arg-type]
    ctx.claims = ["### scan claims"]
    assert OR.empty_result(ctx) is False  # type: ignore[arg-type]


# ─── 6. nom to'qnashuvi ──────────────────────────────────────────────────────


def _private(cid: int, title: str) -> Any:
    c = _chat(cid, title)
    c.type = ChatType.PRIVATE
    return c


OTABEKS = [
    _private(110, "Otabek Kursdosh"),
    _private(115, "Otabek Xonadosh"),
    _private(194, "Otabek Saidakhmadovich"),
]


def test_name_collision_needs_an_equal_tie_of_private_chats() -> None:
    hints = AG.chat_hints("Otabek nima dedi?", OTABEKS)
    key, tied = CS.name_collision(hints, "Otabek nima dedi?") or ("", [])
    assert key == "Otabek" and sorted(h.chat_id for h in tied) == [110, 115, 194]
    exact = [_private(3893, "Otabek"), _private(3853, "Otabek Kursdosh")]
    q = "Otabek nima dedi?"
    assert CS.name_collision(AG.chat_hints(q, exact), q) is None  # "Otabek" aniq
    q2 = "Otabek Kursdoshga yubor"
    assert CS.name_collision(AG.chat_hints(q2, exact), q2) is None  # to'liq nom aniqroq
    assert OR.means_all_names("hamma Otabeklar nima dedi?", "Otabek")
    assert not OR.means_all_names("Otabek nima dedi?", "Otabek")


async def test_colliding_name_is_clarified_then_remembered(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    box: dict[str, Any] = {"state": CS.ConvState()}

    async def load(_f: Any, conversation_id: int) -> CS.ConvState:
        return CS.from_json(box["state"].to_json())

    async def save(_f: Any, conversation_id: int, state: CS.ConvState) -> bool:
        box["state"] = state
        return True

    monkeypatch.setattr(CS, "load", load)
    monkeypatch.setattr(CS, "save", save)
    wired["chats"].extend(OTABEKS)
    use_multi(monkeypatch, lambda q: [_hit(i, chat_id=110) for i in range(1, 6)])

    # 1-turn: uchta "Otabek" — planner qidirmoqchi, kod so'raydi.
    llm1 = FakeLLM(plan=_plan(target_chats=[115], steps=[{"skill": "search_local"}]))
    first = await run("Otabek nima dedi?", llm1, mode="fast", context_mode="agent")
    assert [s["skill"] for s in first.plan["steps"]] == ["clarify"]
    assert "Otabek Kursdosh" in first.answer and "Otabek Xonadosh" in first.answer
    assert llm1.count(DEEP_SYNTH_PROMPT) == 0
    assert box["state"].pending_name == {"key": "Otabek", "chat_ids": [110, 115, 194]}

    # 2-turn: javob "Kursdosh" — reja bitta chatga → holatda Otabek → 110.
    wired["history"] = [
        DS.Turn(role="user", text="Otabek nima dedi?"),
        DS.Turn(role="assistant", text=first.answer, plan=first.plan),
    ]
    llm2 = FakeLLM(
        plan=_plan(target_chats=[110], server_scope="none", steps=[{"skill": "search_local"}]),
        rerank='{"ranking":[0,1,2]}',
    )
    await run("Kursdosh", llm2, mode="fast", context_mode="agent")
    st = box["state"]
    assert st.resolved_entities["Otabek"]["chat_id"] == 110 and st.last_entity == "Otabek"
    assert st.pending_name == {}

    # 3-turn: "Otabek" yana — to'qnashuv bor, lekin holat biladi: so'ramaydi, 110.
    llm3 = FakeLLM(
        plan=_plan(target_chats=[194], server_scope="none", steps=[{"skill": "search_local"}]),
        rerank='{"ranking":[0,1,2]}',
    )
    third = await run("Otabek kecha nima yozdi?", llm3, mode="fast", context_mode="agent")
    assert [s["skill"] for s in third.plan["steps"]] == ["search_local"]
    assert third.plan["chats"] == [110]
    assert "name 'Otabek' resolved → [id:110]" in " | ".join(third.plan["plan_notes"])


async def test_all_of_them_is_not_clarified(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    wired["chats"].extend(OTABEKS)
    use_multi(monkeypatch, lambda q: [_hit(i, chat_id=110) for i in range(1, 6)])
    llm = FakeLLM(
        plan=_plan(server_scope="none", steps=[{"skill": "search_local"}]),
        rerank='{"ranking":[0,1,2]}',
    )
    out = await run("hamma Otabeklar nima dedi?", llm, mode="fast")
    assert [s["skill"] for s in out.plan["steps"]] == ["search_local"]


def test_state_resolution_rewrites_targeted_steps() -> None:
    conv = CS.ConvState(
        resolved_entities={
            "Otabek": {"chat_id": 110, "title": "Otabek Kursdosh", "kind": "private"}
        },
        last_entity="Otabek",
    )
    hints = AG.chat_hints("Otabekka shu xulosani yubor", OTABEKS)
    plan = DS.DeepPlan(chats=[115])
    steps = [
        OR.Step("chat_digest", {}),
        OR.Step("send_result", {"chat_id": 115}),
    ]
    notes: list[str] = []
    collision = CS.name_collision(hints, "Otabekka shu xulosani yubor")
    assert collision is not None
    fixed, target, pending = OR.resolve_collision(
        collision,
        plan=plan,
        steps=steps,
        target=[115],
        conv=conv,
        known_chats={110, 115, 194},
        question="Otabekka shu xulosani yubor",
        notes=notes,
    )
    assert plan.chats == [110] and target == [110] and pending == {}
    assert fixed[1].args == {"chat_id": 110}


def test_send_result_target_prefers_the_state_resolution() -> None:
    chats = {c.id: c for c in OTABEKS}
    conv = CS.ConvState(
        resolved_entities={
            "Otabek": {"chat_id": 110, "title": "Otabek Kursdosh", "kind": "private"}
        },
        last_entity="Otabek",
    )
    ctx = SimpleNamespace(conv=conv, chats_by_id=chats, raw_question="Otabekka yubor", locale="uz")
    assert SX.send_target(ctx, {"chat": "Otabek"}) == ("110", "")  # type: ignore[arg-type]
    # Planner boshqa "Otabek" ni tanlagan — holat yutadi.
    assert SX.send_target(ctx, {"chat_id": 115}) == ("110", "")  # type: ignore[arg-type]
    assert SX.send_target(ctx, {"chat": "unga"}) == ("110", "")  # type: ignore[arg-type]
    # Holatsiz — uchta teng "Otabek": taklif yo'q, so'raladi.
    ctx.conv = None
    target, ask = SX.send_target(ctx, {"chat": "Otabek"})  # type: ignore[arg-type]
    assert target is None and "Otabek Xonadosh" in ask and "qaysi biriga" in ask


# ─── 7. sarlavha va "oraliq" — akkaunt mintaqasida ───────────────────────────


def test_stats_header_and_coverage_range_use_the_account_tz() -> None:
    zi = ZoneInfo(TZ)
    since = datetime(2026, 9, 20, tzinfo=zi).astimezone(UTC)  # 19:00 UTC, 19-sentyabr
    until = datetime(2026, 9, 26, 13, 0, tzinfo=zi).astimezone(UTC)
    spec = WS.WindowSpec(account_id=1, chats=None, since=since, until=until, tz=TZ)
    stats = WS.WindowStats(direction="out", per_chat=[WS.ChatStat(chat_id=1, n=5)])
    block = WS.render_window_stats(
        WS.WindowCount(total=5, chats=1, first_at=since, last_at=until),
        titles={1: "A"},
        stats=stats,
        spec=spec,
    )
    assert "period: 2026-09-20 → 2026-09-26 (Asia/Tashkent)" in block
    assert "2026-09-19" not in block
    cov = DS.build_coverage(
        scanned_hits=[_hit(3)],  # namuna dalil sanasi — "oraliq" ga ta'sir qilmaydi
        messages_scanned=5,
        chats_total=3,
        searched_chats=[],
        evidence=[],
        clusters=[],
        rounds=1,
        truncated=False,
        marginal_stop=False,
        window={"since": since, "until": until, "tz": TZ, "stats": True, "total": 5},
    )
    rendered = DS.render_coverage(cov)
    assert "oraliq: 2026-09-20 … 2026-09-26" in rendered
    # "O'tgan hafta" — eksklyuziv yuqori chegara yarim tunda: yakshanba ko'rsatiladi.
    week = WS.WindowSpec(
        account_id=1,
        chats=None,
        since=datetime(2026, 9, 14, tzinfo=zi).astimezone(UTC),
        until=datetime(2026, 9, 21, tzinfo=zi).astimezone(UTC),
        tz=TZ,
    )
    assert "period: 2026-09-14 → 2026-09-20" in WS.render_window_stats(
        WS.WindowCount(total=1), spec=week
    )


def test_planner_sees_a_collision_line_only_when_code_finds_one() -> None:
    """Jonli daily-45: aniq "Otabek" chati bor edi, planner 14-qoidani o'zicha
    qo'llab `clarify` qildi. Endi to'qnashuvni **kod** aytadi, yo'q bo'lsa jim."""
    q = "Otabek nima dedi?"
    line = OR.name_collision_line(AG.chat_hints(q, OTABEKS), q, None)
    assert line.startswith("name_collision: 'Otabek' matches 3 private chats")
    exact = [_private(3893, "Otabek"), _private(3853, "Otabek Kursdosh")]
    q45 = "Otabekka 'ertaga 10 da' deb yoz."
    assert OR.name_collision_line(AG.chat_hints(q45, exact), q45, None) == ""
    resolved = CS.ConvState(
        resolved_entities={"Otabek": {"chat_id": 110, "title": "", "kind": "private"}}
    )
    assert OR.name_collision_line(AG.chat_hints(q, OTABEKS), q, resolved) == ""
    prompt = OR.build_orch_prompt(
        q,
        [],
        now=datetime.now(UTC),
        history=[],
        selected=[],
        mode="fast",
        tz=TZ,
        collision_line=line,
    )
    assert "name_collision:" in prompt
