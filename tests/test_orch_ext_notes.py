"""`skills_ext` ko'nikmalari (`trends` / `extract`) orkestratorda: izohlar sintezga, reyk.

docs/skills.md 4-cheklov: ko'nikma `SkillResult.notes` i (`FORMAT_NOTE` va o'z
izohlari) `format_output` nikidek sintez promptiga tushishi kerak; UI reyki
(`deep.js::SKILL_STAGE`) ularni `stats` / `compute` bosqichida chizadi.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_orchestrator import (  # type: ignore[import-not-found]
    FakeLLM,
    _hit,
    _plan,
    run,
    wired,
)

from app.services import compute as CMP
from app.services import deep_search as DS
from app.services import output_formats as OF
from app.services import skills as SK
from app.services.prompts_deep import DEEP_SYNTH_PROMPT

__all__ = ["wired"]

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("skill", "stage"), [("trends", "stats"), ("extract", "compute")])
async def test_ext_skill_notes_reach_the_synthesis_and_the_rail(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any], skill: str, stage: str
) -> None:
    assert skill in SK.SKILLS, "skills_ext ro'yxatga olinmagan"
    assert SK.SKILLS[skill].rail[0] == stage

    async def fake(ctx: SK.RunState, args: dict[str, Any]) -> SK.SkillResult:
        hits = [_hit(1), _hit(2)]
        ctx.add_candidates(hits)
        ctx.add_evidence(hits)
        ctx.claims.append(OF.wrap_block(f"| a | b |\n|---|---|\n| {skill} | 1 |"))
        return SK.SkillResult(
            evidence=hits,
            facts=[CMP.Fact(f"{skill}_n", 1, source=skill)],
            notes=[OF.FORMAT_NOTE, f"{skill}: own note for the writer"],
            summary={"n": 1},
        )

    monkeypatch.setitem(SK.SKILLS, skill, replace(SK.SKILLS[skill], run=fake))
    llm = FakeLLM(plan=_plan(steps=[{"skill": skill}]), synth="Javob.")
    events: list[DS.DeepEvent] = []
    await run("postlar dinamikasi va havolalar", llm, mode="fast", events=events)
    synth = llm.user_turn(DEEP_SYNTH_PROMPT)
    assert f"{skill}: own note for the writer" in synth
    assert OF.FORMAT_NOTE[:40] in synth
    stages = {e.data.get("stage") for e in events if e.kind == "step" and e.data["skill"] == skill}
    assert stages == {stage}
    js = (ROOT / "src/app/web/static/pages/ask/deep.js").read_text(encoding="utf-8")
    assert f'{skill}: "{stage}"' in js
