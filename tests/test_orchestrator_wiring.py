"""Orkestrator ↔ `skills_ext` ulagichlari (docs/skills.md, "ma'lum chegaralar").

* G1/G2 (qidiruvsiz / qayta ishlatish turn'i) avtomatlashtirish qadamini
  **tashlamaydi**: "shu xulosani guruhga tashla" `reformat` bo'lsa ham
  `send_result` qoladi (oldidan `reuse_prior`), eslatma yolg'iz qoladi;
* `format_output` bloki model ko'chirmasa ham javobga **kod** qo'shadi
  (`output_formats.ensure_blocks`);
* ko'nikma izohlari (`FORMAT_NOTE`) sintez promptiga tushadi.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from test_orchestrator import (  # type: ignore[import-not-found]
    FakeLLM,
    _plan,
    fake_window,
    run,
    wired,
)

from app.services import deep_search as DS
from app.services import orchestrator as OR
from app.services import output_formats as OF
from app.services import skills as SK
from app.services.prompts_deep import DEEP_SYNTH_PROMPT

__all__ = ["wired"]


def _dp(**kw: Any) -> DS.DeepPlan:
    return DS.DeepPlan(standalone_question="q", **kw)


def test_automation_steps_survive_direct_and_reuse_guards() -> None:
    assert OR.AUTOMATION_SKILLS <= set(SK.SKILLS), "skills_ext ro'yxatdan o'tmagan"
    steps, notes = OR.apply_guards(
        [OR.Step("send_result", {"chat_id": 2})],
        plan=_dp(needs_retrieval=False, intent_follow_up="reformat"),
        mode="fast",
        question="shu xulosani Ish chatga tashla",
    )
    assert [s.skill for s in steps] == ["reuse_prior", "send_result"] and notes
    steps, _ = OR.apply_guards(
        [OR.Step("remind", {"text": "kommunal"}), OR.Step("search_local")],
        plan=_dp(intent_follow_up="chitchat"),
        mode="fast",
        question="ertaga 10 da eslat: kommunal to'lash",
    )
    assert [s.skill for s in steps] == ["remind"]
    # Avtomatlashtirishsiz — eski qoida o'zgarmagan.
    steps, _ = OR.apply_guards(
        [OR.Step("search_local")], plan=_dp(needs_retrieval=False), mode="fast", question="q"
    )
    assert [s.skill for s in steps] == ["reuse_prior"]


async def test_format_block_is_appended_by_code_and_note_reaches_synthesis(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    fake_window(monkeypatch, total=250, days_back=25)
    block = "| chat | n |\n|---|---:|\n| Honadoshlar | 230 |"

    async def fake_format(ctx: SK.RunState, args: dict[str, Any]) -> SK.SkillResult:
        ctx.claims.append(OF.wrap_block(block))
        return SK.SkillResult(notes=[OF.FORMAT_NOTE], summary={"rows": 1})

    monkeypatch.setitem(
        SK.SKILLS, "format_output", replace(SK.SKILLS["format_output"], run=fake_format)
    )
    llm = FakeLLM(
        plan=_plan(
            retrieval_kind="window",
            steps=[{"skill": "stats"}, {"skill": "format_output", "args": {"format": "table"}}],
        ),
        synth="Eng faol chat — Honadoshlar (230).",  # model jadvalni ko'chirmadi
    )
    result = await run("bu oyda qaysi chatda ko'p yozdim, jadval qil", llm, mode="fast")
    assert block in result.answer
    synth = llm.user_turn(DEEP_SYNTH_PROMPT)
    system = next(s for s, _ in llm.prompts if s == DEEP_SYNTH_PROMPT)
    assert OF.FORMAT_NOTE in synth or OF.FORMAT_NOTE in system
