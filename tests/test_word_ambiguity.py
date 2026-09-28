"""So'z / chat nomi ikki ma'noliligi — jonli daily-40 "Oyga qancha xabar yozdim?".

"Oy" — vaqt birligi ham, akkauntdagi chat nomi ham bo'lishi mumkin. Qaror
**kodda** (`conv_state.word_chat_ambiguity`):

  * chat "Oy" bor, tuzilma hech narsa demaydi ("Oyga" — kelishik qo'shimchasi
    o'qilishni hal qilmaydi) → bitta `clarify`, ikkala variant bilan;
  * qo'shtirnoq / "… chati" / tanlangan chat → o'sha chat; "bu oy", "o'tgan
    oyda", "oyiga" → oddiy so'z;
  * "Oy" chati yo'q → aniqlovchisiz "Oyga" = shu oy → `stats` (joriy oy).

Real LLM/DB yo'q.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from test_orchestrator import (  # type: ignore[import-not-found]
    FakeLLM,
    _plan,
    fake_window,
    run,
    wired,
)

from app.db.models import Chat, ChatType
from app.services import conv_state as CS
from app.services import deep_search as DS
from app.services import orchestrator as OR
from app.services import timewin as TW
from app.services.prompts_deep import DEEP_SYNTH_PROMPT

__all__ = ["wired"]

TZ = TW.DEFAULT_TZ
Q = "Oyga qancha xabar yozdim?"


def _named(cid: int, title: str, *, kind: ChatType = ChatType.PRIVATE, username: str = "") -> Chat:
    return Chat(
        id=cid,
        account_id=1,
        tg_peer_id=5000 + cid,
        type=kind,
        title=title,
        username=username or None,
        synced_total=300,
        synced_min_id=1,
    )


OY = _named(77, "Oy")


# ─── detektor ────────────────────────────────────────────────────────────────


def test_case_suffix_does_not_decide_the_reading() -> None:
    amb = CS.word_chat_ambiguity(Q, [OY])
    assert amb is not None
    assert (amb.token, amb.word, amb.chat_id, amb.reading) == ("Oyga", "oy", 77, "ask")
    assert amb.chat_option("uz") == "«Oy» chati"
    assert amb.word_option("uz") == "shu oy (vaqt)"


def test_no_chat_named_like_the_word_means_no_ambiguity() -> None:
    assert CS.word_chat_ambiguity(Q, []) is None
    assert CS.word_chat_ambiguity(Q, [_named(5, "Oy yangiliklari")]) is None  # qism moslik
    assert CS.word_chat_ambiguity("Otabek nima dedi?", [OY]) is None
    # Umumiy so'z bo'lmagan nom — `chat_hints` ishi, bu yerniki emas.
    assert CS.word_chat_ambiguity("Otabekka yozdimmi?", [_named(6, "Otabek")]) is None


@pytest.mark.parametrize(
    ("question", "reading"),
    [
        ("«Oy» chatiga qancha xabar yozdim?", "chat"),
        ('"Oy"ga qancha xabar yozdim?', "chat"),
        ("Oy chatida nima gaplar?", "chat"),
        ("guruh Oy da nima bo'ldi?", "chat"),
        ("bu oyda qancha xabar yozdim?", "word"),
        ("o'tgan oyda qancha xabar yozdim?", "word"),
        ("oyiga qancha xabar yozaman?", "word"),
        ("3 oyda qancha yozdim?", "word"),
        ("Oyda qancha xabar yozdim?", "ask"),
    ],
)
def test_structure_decides_when_it_can(question: str, reading: str) -> None:
    amb = CS.word_chat_ambiguity(question, [OY])
    assert amb is not None and amb.reading == reading, (question, amb)


def test_selected_chat_or_state_resolution_means_the_chat() -> None:
    assert CS.word_chat_ambiguity(Q, [OY], selected=[77]).reading == "chat"  # type: ignore[union-attr]
    st = CS.ConvState(resolved_entities={"Oy": {"chat_id": 77, "title": "Oy", "kind": "private"}})
    assert CS.word_chat_ambiguity(Q, [OY], state=st).reading == "chat"  # type: ignore[union-attr]


def test_username_and_other_common_words_count() -> None:
    ish = _named(8, "Ish", kind=ChatType.SUPERGROUP)
    amb = CS.word_chat_ambiguity("Ishda kim ko'p yozdi?", [ish])
    assert amb is not None and amb.word == "ish" and amb.reading == "ask"
    assert amb.word_option("uz") == "oddiy so'z «ish»"
    dom = _named(9, "Семья", kind=ChatType.GROUP, username="uy")
    amb2 = CS.word_chat_ambiguity("uyda kim yozdi?", [dom])
    assert amb2 is not None and amb2.chat_id == 9


def test_bare_period_word_is_the_current_period() -> None:
    now = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)
    assert OR.bare_period(Q, tz=TZ, now=now) == "this_month"
    assert OR.bare_period("Haftada nechta xabar yozdim?", tz=TZ, now=now) == "this_week"
    assert OR.bare_period("Yilda qancha xabar yozdim?", tz=TZ, now=now) == "this_year"
    # Aniqlovchi bor — boshqa ma'no yoki `timewin` o'zi hal qiladi.
    for q in ("har oyga qancha yozaman?", "bir oyda qancha yozdim?", "bu oyda qancha yozdim?"):
        assert OR.bare_period(q, tz=TZ, now=now) is None, q
    assert OR.bare_period("Oyga nima dedim?", tz=TZ, now=now) is None  # sanoq emas


def test_planner_sees_the_ambiguity_line_and_no_bare_window() -> None:
    amb = CS.word_chat_ambiguity(Q, [OY])
    line = OR.ambiguity_fact(amb)
    assert line.startswith("ambiguous_name: 'Oyga' → chat \"Oy\" [id:77]")
    assert "rule 15" in line
    now = datetime.now(UTC)
    prompt = OR.build_orch_prompt(
        Q, [], now=now, history=[], selected=[], mode="fast", tz=TZ,
        ambiguity_line=line, bare_ok=False,
    )  # fmt: skip
    assert "ambiguous_name:" in prompt
    assert "time_window: none named in the question" in prompt
    plain = OR.build_orch_prompt(Q, [], now=now, history=[], selected=[], mode="fast", tz=TZ)
    assert "time_window: this_month" in plain
    assert OR.ambiguity_fact(None) == ""
    word = CS.word_chat_ambiguity("bu oyda qancha xabar yozdim?", [OY])
    assert "resolved by code → the ordinary word" in OR.ambiguity_fact(word)
    assert "15. A word that is both" in OR.ORCH_PLAN_PROMPT
    assert "`ambiguous_name:`" in OR.ORCH_PLAN_PROMPT


# ─── uchdan-uchgacha ─────────────────────────────────────────────────────────


async def test_daily40_with_an_oy_chat_asks_one_question_with_both_readings(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    wired["chats"].append(OY)
    seen = fake_window(monkeypatch, total=40, days_back=27)
    # Planner o'zicha "stats" desa ham — kod bitta savol beradi.
    llm = FakeLLM(plan=_plan(retrieval_kind="window", steps=[{"skill": "stats"}]))
    out = await run(Q, llm, mode="fast", context_mode="agent")
    assert [s["skill"] for s in out.plan["steps"]] == ["clarify"]
    assert out.direct and out.intent == "clarify"
    assert "«Oy» chati" in out.answer and "shu oy (vaqt)" in out.answer
    assert out.answer.count("?") == 1
    assert not seen  # sanoq yugurmadi
    assert llm.count(DEEP_SYNTH_PROMPT) == 0
    assert "ambiguous_name: 'Oyga'" in llm.user_turn(OR.ORCH_PLAN_PROMPT)
    assert any("is a word and a chat name" in n for n in out.plan["plan_notes"])


async def test_daily40_without_an_oy_chat_counts_this_month(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    seen = fake_window(monkeypatch, total=40, days_back=27)
    llm = FakeLLM(
        plan=_plan(retrieval_kind="window", steps=[{"skill": "stats"}]),
        synth="Shu oyda 40 ta xabar yozgansiz.",
    )
    out = await run(Q, llm, mode="fast", context_mode="agent")
    assert [s["skill"] for s in out.plan["steps"]] == ["stats"]
    assert seen, "stats yugurishi kerak"
    plan = seen[0]
    expect = TW.period_window("this_month", tz=TZ)
    assert expect is not None
    assert plan.window_from_question and plan.window_kind == "this_month"
    assert plan.since == expect.since and plan.direction == "out"
    assert "time_window: this_month" in llm.user_turn(OR.ORCH_PLAN_PROMPT)
    assert "ambiguous_name" not in llm.user_turn(OR.ORCH_PLAN_PROMPT)


async def test_clarify_reply_with_the_chat_option_targets_the_chat(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    wired["chats"].append(OY)
    fake_window(monkeypatch, total=12, days_back=27)
    wired["history"] = [
        DS.Turn(role="user", text=Q),
        DS.Turn(role="assistant", text="«Oyga» deganda nimani nazarda tutyapsiz?"),
    ]
    llm = FakeLLM(
        plan=_plan(
            standalone_question="«Oy» chatiga qancha xabar yozdim?",
            retrieval_kind="window",
            steps=[{"skill": "stats"}],
        ),
        synth="«Oy» chatiga 12 ta xabar yozgansiz.",
    )
    out = await run("«Oy» chati", llm, mode="fast", context_mode="agent")
    assert out.plan["chats"] == [77]
    assert [s["skill"] for s in out.plan["steps"]] != ["clarify"]
    assert '"Oy" [id:77]' in llm.user_turn(OR.ORCH_PLAN_PROMPT)  # maslahat qatori


async def test_a_time_phrase_is_the_word_even_with_a_chat_of_that_name(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """Chat "Bugun" bor, lekin "bugun qancha yozdim" — `timewin` iborasi, so'ramaydi."""
    wired["chats"].append(_named(78, "Bugun"))
    seen = fake_window(monkeypatch, total=5, days_back=1)
    llm = FakeLLM(plan=_plan(retrieval_kind="window", steps=[{"skill": "stats"}]), synth="5 ta.")
    out = await run("Bugun qancha xabar yozdim?", llm, mode="fast", context_mode="agent")
    assert [s["skill"] for s in out.plan["steps"]] == ["stats"] and seen
    assert "resolved by code → the ordinary word" in llm.user_turn(OR.ORCH_PLAN_PROMPT)
