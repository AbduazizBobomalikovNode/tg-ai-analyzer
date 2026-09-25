"""Orkestrator v1 — "qachon nima qilish" ketma-ketligi (deep/exhaustive kirishi).

Tanqid (2026-09-25): deep quvuri **tor** edi — rejim bir shakl tanlardi
(fan-out yoki korpus kaskadi) va savolning o'zi nima talab qilishini
so'ramasdi. Jonli misol: tanlangan chat + "bu chatda nima bor edi!"
`exhaustive` da 54 chat x 521 hafta skan rejasini tuzib tasdiq so'radi,
holbuki javob bitta chatning dayjest keshida turardi.

Endi:

    kod darvozasi (`deep_search.direct_intent`) — salom/meta: planner ham yo'q
      ↓
    PLAN (bitta `Task.SEARCH` chaqiruvi) — eski reja maydonlari + QADAMLAR:
        katalogdan (`services/skills.py`) eng arzon yetarli yo'l, ≤4 qadam
      ↓ kod qo'riqchilari (guard'lar) — asosiy mantiq emas, to'siq:
        direct/reuse, window/stats, rejim qopqog'i, discover, bitta chatga
        skan yo'q, bo'sh reja → deterministik sukut
      ↓
    EXECUTOR — qadamlar tartibda; har biri faktlar (raqamlar, kodda) va
      dalil qo'shadi; `needs_server` qo'riqchisi; ≤1 REPLAN (≤2 qadam)
      ↓
    SINTEZ (eski `deep_search.synthesize`, niyat bo'yicha byudjet) → qamrov →
    `_persist` (yagona yozuvchi)

Invariantlar o'zgarmagan: plan/replan promptiga xabar matni tushmaydi
(`_Taint.assert_clean`, karantin ko'nikmalar chiqishi faqat **sanoq** bo'lib
uzatiladi), dalillar bitta `<untrusted_data kind="evidence">` konvertida,
narx qopqoqlari, baho + tasdiq qoidalari, kunlik byudjet birinchi LLM
chaqirig'idan oldin. Qaytish yo'li: `ORCHESTRATOR=false` → eski
`deep_search._run_pipeline`.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Chat, ChatType
from app.llm import LLMError, Msg, Task
from app.logging import get_logger
from app.observability import DEEP_CANDIDATES, DEEP_COST, DEEP_LATENCY, DEEP_ROUNDS, DEEP_RUNS
from app.services import compute as CMP
from app.services import conv_state as CS
from app.services import deep_search as DS
from app.services import limits as LIM
from app.services import search as S
from app.services import skills as SK
from app.services import timewin as TW
from app.services import untrusted as U
from app.services.prompts_deep import (
    DEEP_PLAN_PROMPT,
    DIGEST_SYNTH_NOTE,
    FACTS_SYNTH_NOTE,
    ORCH_PLAN_ADDENDUM,
    ORCH_REPLAN_PROMPT,
    STATS_SYNTH_NOTE,
    WINDOW_SYNTH_NOTE,
)

log = get_logger(__name__)

# ─── byudjetlar (kodda) ──────────────────────────────────────────────────────
MAX_STEPS = 4
MAX_REPLAN_STEPS = 2
#: Replan faqat qidiruv shundan kam dalil bersa (yoki umuman bermasa) chaqiriladi.
REPLAN_MIN_HITS = 3
#: Planner `Task.SEARCH` da; Gemini'da fikrlash tokeni shu byudjetdan yeyiladi.
ORCH_PLAN_MAX_TOKENS = 2_500
ORCH_PLAN_THINKING = "minimal"
REPLAN_MAX_TOKENS = 600
MAX_SELECTED = 20
ANSWER_KINDS = ("digest", "list", "number", "table", "text")
OUTPUT_BUDGETS = ("short", "normal", "detailed")
#: Sintez `Task.SEARCH` da yetarli bo'lgan ko'nikmalar (raqam/dayjestni so'zga aylantirish).
CHEAP_SYNTH_SKILLS = frozenset({"stats", "compute", "chat_digest"})
#: Faqat qidiruvdan keyin ma'noli bo'lgan server qo'riqchisi shu ko'nikmalardan keyin.
SEARCH_SKILLS = frozenset({"search_local", "scan_corpus"})
REPLAN_SKILLS = frozenset(
    {"search_local", "search_server", "discover", "scan_corpus", "chat_digest", "stats", "window"}
)

#: System prompt — eski reja prompti + katalog (registrdan, kodda). Bayt
#: darajasida barqaror: registr o'zgarmaguncha matn o'zgarmaydi.
ORCH_PLAN_PROMPT = DEEP_PLAN_PROMPT + ORCH_PLAN_ADDENDUM.replace("{catalog}", SK.catalog_text())

#: Planner yiqilganda (yoki qadam bermaganda) — makro-ko'nikma qoplamaydigan
#: tool ishi: profil, a'zolar, reaksiyalar, izohlar, saqlangan xabarlar,
#: kalendar. Faqat **sukut** yo'lida ishlatiladi; planner tanlovini bosmaydi.
AGENT_WORK_RE = re.compile(
    r"\ba'zo\w*|\bazo(?:lar)?\w*|\bmembers?\b|участник\w*|подписчик\w*"
    r"|reaksiya\w*|reaction\w*|реакци\w*|\blayk\w*|\blike[sd]?\b"
    r"|\bizoh\w*|komment\w*|comment\w*|коммент\w*"
    r"|saqlangan\s+xabar\w*|saved\s+messages?|избранн\w*"
    r"|\bprofil\w*|\bprofile\b|профил\w*|\bbio\b|\bkalendar\w*|\bcalendar\b",
    re.I,
)


# ─── reja tuzilmasi ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class Step:
    skill: str
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""
    origin: str = "plan"  # plan | guard | replan | fallback

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "args": self.args, "why": self.why, "origin": self.origin}

    def key(self) -> str:
        return f"{self.skill}:{json.dumps(self.args, sort_keys=True, default=str)}"


@dataclass(slots=True)
class OrchPlan:
    plan: DS.DeepPlan
    steps: list[Step] = field(default_factory=list)
    target_chats: Any = "all"
    expected_answer_kind: str = "text"
    output_budget: str = "normal"
    notes: list[str] = field(default_factory=list)
    planner_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.plan.to_dict(),
            "orchestrator": True,
            "steps": [s.to_dict() for s in self.steps],
            "target_chats": self.target_chats,
            "expected_answer_kind": self.expected_answer_kind,
            "output_budget": self.output_budget,
            "plan_notes": list(self.notes),
        }


# ─── planner kirishi ─────────────────────────────────────────────────────────


def selected_lines(chats: list[Chat]) -> list[str]:
    return [
        f'"{c.title}" {c.type} msgs={int(c.synced_total or 0)} [id:{c.id}]'
        + ("" if (c.synced_total or 0) > 0 else " (not synced)")
        for c in chats
    ]


def window_fact(question: str, *, tz: str, now: datetime) -> str:
    """Savoldagi oyna — **kodda** (`timewin`), planner uchun fakt satri."""
    parsed = TW.parse_window(question, tz=tz, now=now)
    if parsed is None:
        return "time_window: none named in the question"
    return (
        f"time_window: {parsed.kind} {TW.format_window(parsed)} — parsed by code from the "
        "question; the system applies it on its own"
    )


def build_orch_prompt(
    question: str,
    chat_lines: list[str],
    *,
    now: datetime,
    history: list[DS.Turn],
    selected: list[str],
    mode: str,
    tz: str,
    state_block: str = "",
    hint_lines: list[str] | None = None,
) -> str:
    """Eski planner kirishi + tanlangan chatlar, vaqt oynasi fakti va rejim qopqog'i.

    Xabar **matni** yo'q. Tanlangan chat sarlavhalari ham begona ma'lumot —
    ular alohida konvertda, `neutralize` dan o'tgan. `state_block` — suhbat
    holati (`conv_state.render_block`, id/sarlavha/raqam/user matni),
    `hint_lines` — savoldagi nomga mos chatlar (`agent.chat_hints`, kodda).
    """
    base = DS.build_plan_prompt(question, chat_lines, now=now, history=history)
    parts = [base]
    if selected:
        body = "\n".join(U.neutralize(x) for x in selected)
        parts.append(
            f'<untrusted_data kind="selected_chats" count="{len(selected)}">\n{body}\n'
            "</untrusted_data>\n"
            "selected_chats: the user picked these chats as the context of this question."
        )
    else:
        parts.append("selected_chats: none")
    if hint_lines:
        body = "\n".join(U.neutralize(x) for x in hint_lines)
        parts.append(
            f'<untrusted_data kind="chat_hints" count="{len(hint_lines)}">\n{body}\n'
            "</untrusted_data>\n"
            "chat_hints: chats whose title matches a name in the question (matched by code)."
        )
    if state_block:
        parts.append(state_block)
    parts.append(window_fact(question, tz=tz, now=now))
    allowed = ", ".join(SK.allowed_skills(mode))
    parts.append(f"mode_ceiling: {mode} — allowed skills: {allowed}")
    return "\n".join(parts)


# ─── validatsiya + qo'riqchilar ──────────────────────────────────────────────


def _topical(question: str) -> bool:
    return bool(DS.TOPIC_MARK_RE.search(TW.fold(question or "")))


def parse_steps(
    raw: Any, *, mode: str, known_chats: set[int], question: str, origin: str = "plan"
) -> tuple[list[Step], list[str]]:
    """Model qadamlarini **qattiq** tekshiradi: nom, argument tipi, chat egaligi, qopqoq.

    Rejim qopqog'idan yuqori ko'nikma o'chirilmaydi, **pastga tushiriladi**
    (`scan_corpus` → `search_local`, `discover` → `search_server`) va izoh
    qoldiriladi — foydalanuvchi savoli javobsiz qolmasin.
    """
    notes: list[str] = []
    steps: list[Step] = []
    if not isinstance(raw, list):
        return [], (["steps: not a list"] if raw is not None else [])
    for item in raw[: MAX_STEPS * 2]:
        if not isinstance(item, dict):
            notes.append("steps: item is not an object")
            continue
        name = str(item.get("skill") or "").strip()
        skill = SK.SKILLS.get(name)
        if skill is None:
            notes.append(f"steps: unknown skill {name[:40]!r} dropped")
            continue
        if not skill.allowed(mode):
            down = {"scan_corpus": "search_local", "discover": "search_server"}.get(name)
            notes.append(f"steps: {name} is above the {mode} ceiling → {down or 'dropped'}")
            if down is None:
                continue
            skill = SK.SKILLS[down]
            name = down
        args = SK.coerce_args(skill, item.get("args"))
        for key in ("chat_id",):
            if key in args and args[key] not in known_chats:
                notes.append(f"steps: {name}.{key}={args[key]} is not an account chat")
                args.pop(key)
        if "chat_ids" in args:
            args["chat_ids"] = [c for c in args["chat_ids"] if c in known_chats]
            if not args["chat_ids"]:
                args.pop("chat_ids")
        if name == "write_propose" and not SK.WRITE_INTENT_RE.search(question or ""):
            notes.append("steps: write_propose without a write request in the question dropped")
            continue
        why = U.neutralize(str(item.get("why") or ""))[:160]
        steps.append(Step(name, args, why, origin))
    return steps, notes


def _dedupe(steps: list[Step]) -> list[Step]:
    seen: set[str] = set()
    out: list[Step] = []
    for s in steps:
        if s.key() in seen:
            continue
        seen.add(s.key())
        out.append(s)
    return out


def default_steps(plan: DS.DeepPlan, *, mode: str, question: str) -> list[Step]:
    """Planner qadam bermasa (yoki yiqilsa) — eski signallardan deterministik sukut."""
    if plan.intent_follow_up in DS.DIRECT_FOLLOW_UPS:
        return [Step("answer_direct", {"intent": plan.intent_follow_up}, "direct", "fallback")]
    if not plan.needs_retrieval:
        return [Step("reuse_prior", {}, "reuse previous evidence", "fallback")]
    if plan.retrieval_kind == "window" and plan.window_stats:
        out = [Step("stats", {}, "exact counts", "fallback")]
        if CMP.heuristic_items(question, _PROBE_FACTS):
            out.append(Step("compute", {}, "arithmetic on the counts", "fallback"))
        return out
    if plan.retrieval_kind == "window":
        return [Step("window", {}, "time window digest", "fallback")]
    if (
        plan.chats
        and len(plan.chats) <= SK.DIGEST_MAX_CHATS
        and not _topical(question)
        and (plan.intent == "summary")
    ):
        return [Step("chat_digest", {}, "overview of the named chat", "fallback")]
    if AGENT_WORK_RE.search(question or "") and "agent_tools" in SK.SKILLS:
        return [Step("agent_tools", {}, "tool work no macro-skill covers", "fallback")]
    out = [Step("search_local", {}, "topic search", "fallback")]
    if plan.server_scope == "discover" and SK.SKILLS["discover"].allowed(mode):
        out.append(Step("discover", {}, "public channels were asked about", "fallback"))
    return out


#: `heuristic_items` uchun "qaysi faktlar bo'ladi" sinovi — `stats` beradiganlari.
_PROBE_FACTS = {
    name: CMP.Fact(name, 1)
    for name in ("total", "mine", "theirs", "chats", "days", "weeks", "months")
}


def apply_guards(
    steps: list[Step], *, plan: DS.DeepPlan, mode: str, question: str
) -> tuple[list[Step], list[str]]:
    """Kod qo'riqchilari — planner **ustidan**, lekin faqat aniq holatlarda.

    Ular asosiy mantiq emas: har biri eski quvurda allaqachon kodda turgan
    qoidani (direct/reuse darvozasi, oyna darvozasi, discover, rejim
    qopqog'i) yoki jonli xatoni (bitta chatga korpus skani) to'sadi.
    """
    notes: list[str] = []
    # G1. qidiruvsiz turn (planner intenti): faqat `answer_direct`.
    if plan.intent_follow_up in DS.DIRECT_FOLLOW_UPS:
        if [s.skill for s in steps] != ["answer_direct"]:
            notes.append("guard: direct intent → answer_direct")
        return [Step("answer_direct", {"intent": plan.intent_follow_up}, "direct", "guard")], notes
    # G2. oldingi dalilni qayta ishlatish — faqat havola bor bo'lsa (`parse_plan`).
    if not plan.needs_retrieval:
        if [s.skill for s in steps] != ["reuse_prior"]:
            notes.append("guard: reuse turn → reuse_prior")
        return [Step("reuse_prior", {}, "reuse previous evidence", "guard")], notes
    steps = [
        Step("search_local", {}, s.why, "guard") if s.skill == "reuse_prior" else s for s in steps
    ]
    # Terminal ko'nikma yolg'iz turadi.
    terminal = [s for s in steps if SK.SKILLS[s.skill].terminal]
    if terminal:
        if len(steps) > 1:
            notes.append(f"guard: {terminal[0].skill} is a single step")
        return [terminal[0]], notes
    # G3. sanoq savoli (kod darvozasi `window_stats`) — `stats` birinchi.
    if plan.retrieval_kind == "window" and plan.window_stats:
        kept = [s for s in steps if s.skill in ("stats", "compute")]
        if not any(s.skill == "stats" for s in kept):
            kept.insert(0, Step("stats", {}, "exact counts (window stats gate)", "guard"))
            notes.append("guard: window stats → stats")
        dropped = [s.skill for s in steps if s.skill not in ("stats", "compute")]
        if dropped:
            notes.append(f"guard: counts need no {', '.join(sorted(set(dropped)))}")
        steps = kept
    # G4. mavzusiz oyna (kod darvozasi) — `window`.
    elif plan.retrieval_kind == "window":
        # Kod "ro'yxat, sanoq emas" dedi ("bugun menga kimlar yozgan, nima
        # maqsadda"): yolg'iz `stats` faqat raqam beradi, maqsadni emas —
        # u `window` ga almashtiriladi. `compute` bilan kelgan `stats` esa
        # arifmetika savoli ("o'rtacha kuniga") — o'sha qoladi.
        arithmetic = any(s.skill == "compute" for s in steps)
        allowed_here = ("window", "chat_digest", *(("stats", "compute") if arithmetic else ()))
        kept = [s for s in steps if s.skill in allowed_here]
        if not any(s.skill in ("window", "chat_digest", "stats") for s in kept):
            kept.insert(0, Step("window", {}, "time window (window gate)", "guard"))
            notes.append("guard: window question → window")
        steps = kept
    else:
        # `window` ko'nikmasi oynasiz — bitta chat bo'lsa dayjest, aks holda qidiruv.
        fixed: list[Step] = []
        for s in steps:
            if s.skill == "window" and plan.since is None and not plan.window_from_question:
                alt = (
                    "chat_digest"
                    if plan.chats and len(plan.chats) <= SK.DIGEST_MAX_CHATS
                    else "search_local"
                )
                notes.append(f"guard: window without a period → {alt}")
                fixed.append(Step(alt, {}, s.why, "guard"))
            else:
                fixed.append(s)
        steps = fixed
    # G5. bitta-uchta chatga korpus skani yo'q (jonli xato: 1 chat → 54x521 reja).
    fixed = []
    for s in steps:
        if s.skill == "scan_corpus" and plan.chats and len(plan.chats) <= SK.DIGEST_MAX_CHATS:
            alt = (
                "search_local"
                if _topical(question) or plan.intent in ("lookup", "person")
                else "chat_digest"
            )
            notes.append(f"guard: scan_corpus over {len(plan.chats)} chat(s) → {alt}")
            fixed.append(Step(alt, {}, s.why, "guard"))
        else:
            fixed.append(s)
    steps = fixed
    # G6. `compute` o'zi raqam topmaydi — manbasi bo'lmasa `stats` oldidan.
    for i, s in enumerate(steps):
        if s.skill != "compute":
            continue
        if not any(p.skill in ("stats", "chat_digest", "window") for p in steps[:i]):
            steps.insert(i, Step("stats", {}, "numbers for compute", "guard"))
            notes.append("guard: compute needs numbers → stats first")
        break
    # G7. "men a'zo bo'lmagan kanallar" (`DISCOVER_RE`) — discover qadami.
    if (
        plan.server_scope == "discover"
        and SK.SKILLS["discover"].allowed(mode)
        and not any(s.skill == "discover" for s in steps)
    ):
        steps.append(Step("discover", {}, "the question is about public channels", "guard"))
        notes.append("guard: discover question → discover")
    steps = _dedupe(steps)
    if not steps:
        steps = default_steps(plan, mode=mode, question=question)
        notes.append("guard: empty plan → default steps")
    if len(steps) > MAX_STEPS:
        notes.append(f"guard: {len(steps)} steps capped at {MAX_STEPS}")
        steps = steps[:MAX_STEPS]
    return steps, notes


def parse_orch_plan(
    raw: str,
    *,
    question: str,
    known_chats: list[int],
    selected: list[int],
    history: list[DS.Turn],
    mode: str,
    tz: str,
    now: datetime,
    conv: CS.ConvState | None = None,
) -> OrchPlan:
    """Planner JSON → `OrchPlan`. Eski maydonlar `deep_search.parse_plan` dan o'tadi.

    `conv` — suhbat holati: referent ("u", "o'sha guruh") oxirgi entity
    chatiga, follow-up'da oxirgi oyna meros (`conv_state.apply_to_plan`) —
    guard'lardan **oldin**, ular yangi chat/oyna bilan ishlasin.

    Shu tufayli barcha eski kod qarorlari (oyna savoldan, `retrieval_kind`
    darvozasi, `force_all_chats`, `DISCOVER_RE`, follow-up kengaytirish) o'z
    joyida qoladi — orkestrator ularni qayta yozmaydi.
    """
    data = DS._json_obj(raw)
    target = data.get("target_chats")
    sel_synced = [c for c in selected if c in set(known_chats)]
    if target == "selected" and sel_synced:
        data["chats"] = list(sel_synced)
    elif isinstance(target, list):
        data["chats"] = target
    elif target == "all":
        data["chats"] = "all"
    plan = DS.parse_plan(
        json.dumps(data, ensure_ascii=False, default=str),
        question=question,
        known_chats=known_chats,
        history=history,
        tz=tz,
        now=now,
    )
    notes: list[str] = []
    # 5-qoida: tanlangan chatlar — "hamma/barcha chatlar" deyilmagan bo'lsa.
    from app.services import scale_search as SC

    if sel_synced and not SC.force_all_chats(plan.standalone_question or question):
        if plan.chats != sorted(sel_synced):
            notes.append("guard: selected context chats")
        plan.chats = sorted(sel_synced)
        target = "selected"
    elif sel_synced:
        target = "all"
    applied = CS.Applied()
    if conv is not None:
        applied = CS.apply_to_plan(
            plan, conv, question=question, selected=sel_synced, known_chats=known_chats
        )
        notes.extend(applied.notes)
        if applied.chats:
            target = list(applied.chats)
        if applied.window:
            # Oyna endi ma'lum — oyna/sanoq darvozasi shunga qarab qayta hal qiladi.
            plan.retrieval_kind = DS.apply_retrieval_kind(
                plan, plan.retrieval_kind, question=question
            )
    steps, step_notes = parse_steps(
        data.get("steps"), mode=mode, known_chats=set(known_chats), question=question
    )
    notes.extend(step_notes)
    if applied.chats and [s.skill for s in steps] == ["clarify"]:
        # Holat "kim/qaysi chat" ni allaqachon biladi — so'rash shart emas.
        steps = []
        notes.append("state: referent resolved — clarify dropped")
    steps, guard_notes = apply_guards(
        steps, plan=plan, mode=mode, question=plan.standalone_question or question
    )
    notes.extend(guard_notes)
    kind = data.get("expected_answer_kind")
    budget = data.get("output_budget")
    return OrchPlan(
        plan=plan,
        steps=steps,
        target_chats=target if target in ("all", "selected") or isinstance(target, list) else "all",
        expected_answer_kind=kind if kind in ANSWER_KINDS else "text",
        output_budget=budget if budget in OUTPUT_BUDGETS else "normal",
        notes=notes,
    )


def plan_task() -> Task:
    """Planner vazifasi — `ORCHESTRATOR_PLAN_TASK` (`route|search`, sukut `search`)."""
    return Task.ROUTE if get_settings().orchestrator_plan_task == "route" else Task.SEARCH


async def plan_orchestrated(
    meter: DS._Meter,
    taint: DS._Taint,
    question: str,
    *,
    chat_lines: list[str],
    known_chats: list[int],
    selected: list[Chat],
    history: list[DS.Turn],
    mode: str,
    tz: str,
    now: datetime | None = None,
    conv: CS.ConvState | None = None,
    hint_lines: list[str] | None = None,
) -> OrchPlan:
    """**Bitta** planner chaqiruvi (`plan_task()`). Yiqilsa — kod sukuti bilan davom.

    `conv` — suhbat holati: promptga ixcham blok bo'lib tushadi (oldin
    `_Taint.assert_state` — xabar matni yo'qligi tekshiriladi) va
    `parse_orch_plan` da referent/oyna merosi kodda qo'llanadi.
    """
    taint.add_turns(history, question=question)
    when = now or datetime.now(UTC)
    sel_lines = selected_lines(selected)
    sel_ids = [c.id for c in selected]
    state_block = CS.render_block(conv) if conv is not None else ""
    if state_block:
        taint.assert_state("plan", state_block)

    def prompt_for(turns: list[DS.Turn]) -> str:
        return build_orch_prompt(
            question,
            chat_lines,
            now=when,
            history=turns,
            selected=sel_lines,
            mode=mode,
            tz=tz,
            state_block=state_block,
            hint_lines=hint_lines,
        )

    prompt = prompt_for(history)
    if history and taint.prior_leak(prompt):
        log.warning("orch.taint_prior_skipped", turns=len(history))
        history = []
        prompt = prompt_for(history)
    taint.assert_clean("plan", prompt)
    try:
        res = await meter.chat(
            plan_task(),
            [Msg.system(ORCH_PLAN_PROMPT), Msg.user(prompt)],
            json_mode=True,
            max_tokens=ORCH_PLAN_MAX_TOKENS,
            thinking=ORCH_PLAN_THINKING,
        )
        raw = res.text
        ok = bool(DS._json_obj(raw))
        if not ok:
            log.warning("orch.plan_invalid", task=plan_task().value, chars=len(raw or ""))
    except LLMError as exc:
        log.warning("orch.plan_failed", error=str(exc)[:200])
        raw, ok = "{}", False
    out = parse_orch_plan(
        raw,
        question=question,
        known_chats=known_chats,
        selected=sel_ids,
        history=history,
        mode=mode,
        tz=tz,
        now=when,
        conv=conv,
    )
    out.planner_ok = ok
    return out


# ─── replan ──────────────────────────────────────────────────────────────────


def build_replan_prompt(
    question: str, ran: list[dict[str, Any]], *, allowed: list[str], facts: list[CMP.Fact]
) -> str:
    """Faqat savol, bajarilgan qadamlar va **sanoqlar** — matn yo'q.

    `ran` elementlari `{skill, args, summary}`: `args` planner yozgan (u ham
    matn ko'rmagan), `summary` esa kodda hisoblangan raqamlar. Fakt
    belgilari (chat sarlavhasi bo'lishi mumkin) **uzatilmaydi** — faqat nom
    va qiymat.
    """
    lines = [
        f"- {r['skill']} {json.dumps(r.get('args') or {}, ensure_ascii=False)} -> "
        f"{json.dumps(r.get('summary') or {}, ensure_ascii=False, default=str)}"
        for r in ran
    ]
    fact_line = (
        ", ".join(f"{f.name}={CMP.fmt_number(f.value)}" for f in facts if f.numeric) or "(none)"
    )
    return (
        f"<question>{U.neutralize(question)}</question>\n"
        "steps that ran:\n" + "\n".join(lines) + "\n"
        f"facts: {fact_line}\n"
        f"allowed skills: {', '.join(allowed)}"
    )


def needs_replan(ctx: SK.RunState) -> bool:
    """Replan faqat qidiruv qadami ingichka natija berganda — kodda qaror."""
    if not ctx.used & (SEARCH_SKILLS | {"search_server", "discover"}):
        return False
    return len(ctx.ranked) < REPLAN_MIN_HITS


async def replan(
    ctx: SK.RunState, ran: list[dict[str, Any]], *, done_keys: set[str]
) -> tuple[list[Step], list[str]]:
    allowed = [
        n
        for n in SK.allowed_skills(ctx.mode)
        if n in REPLAN_SKILLS
        and not (SK.SKILLS[n].needs_confirm(ctx.mode) and not ctx.confirm and n == "scan_corpus")
    ]
    prompt = build_replan_prompt(ctx.question, ran, allowed=allowed, facts=list(ctx.facts.values()))
    ctx.taint.assert_clean("replan", prompt)
    try:
        res = await ctx.meter.chat(
            Task.ROUTE,
            [Msg.system(ORCH_REPLAN_PROMPT), Msg.user(prompt)],
            json_mode=True,
            max_tokens=REPLAN_MAX_TOKENS,
        )
    except LLMError as exc:
        log.warning("orch.replan_failed", error=str(exc)[:200])
        return [], ["replan: failed"]
    data = DS._json_obj(res.text)
    steps, notes = parse_steps(
        data.get("add_steps"),
        mode=ctx.mode,
        known_chats=set(ctx.chats_by_id),
        question=ctx.raw_question,
        origin="replan",
    )
    out = [
        s
        for s in steps
        if s.skill in allowed and s.key() not in done_keys and not SK.SKILLS[s.skill].terminal
    ]
    return out[:MAX_REPLAN_STEPS], notes


# ─── UI reyki ────────────────────────────────────────────────────────────────


def rail_for(steps: list[Step]) -> list[str]:
    """`stages` hodisasi: rejadan keyin **haqiqatan** yuguradigan reyk bosqichlari."""
    out = ["plan"]
    for s in steps:
        for key in SK.SKILLS[s.skill].rail:
            if key not in out:
                out.append(key)
    out.append("write")
    return out


# ─── asosiy sikl ─────────────────────────────────────────────────────────────

_STEPS_LABEL = {"uz": "Qadamlar", "ru": "Шаги", "en": "Steps"}
_DIGEST_LABEL = {
    "uz": "dayjest: {d} kunlik + {w} haftalik (keshdan{c}) · chatda {n} xabar",
    "ru": "дайджест: {d} дневных + {w} недельных (из кэша{c}) · в чате {n} сообщений",
    "en": "digest: {d} daily + {w} weekly (cached{c}) · {n} messages in the chat",
}
_COMPUTED = {
    "uz": ", {n} kun hisoblandi",
    "ru": ", {n} дн. рассчитано",
    "en": ", {n} day(s) computed",
}


def steps_footer(ctx: SK.RunState, steps: list[Step]) -> str:
    loc = ctx.locale if ctx.locale in _STEPS_LABEL else "uz"
    chain = " → ".join(s.skill for s in steps)
    out = f"\n\n{_STEPS_LABEL[loc]}: {chain}"
    if ctx.digest_meta:
        d = ctx.digest_meta
        computed = int(d.get("computed_days") or 0)
        out += "\n\n" + _DIGEST_LABEL[loc].format(
            d=d.get("day_digests", 0),
            w=d.get("week_digests", 0),
            c=_COMPUTED[loc].format(n=computed) if computed else "",
            n=d.get("total", 0),
        )
    return out


async def _load_selected(
    session_factory: DS.SessionFactory,
    *,
    account_id: int,
    chat_ids: list[int],
    synced: dict[int, Chat],
) -> list[Chat]:
    """Tanlangan chatlar — akkauntniki bo'lishi shart; sinxronlanmaganlari ham (sarlavha)."""
    ids = [int(c) for c in chat_ids if isinstance(c, int) and c > 0][:MAX_SELECTED]
    found = {c: synced[c] for c in ids if c in synced}
    missing = [c for c in ids if c not in found]
    if missing:
        async with session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(Chat).where(Chat.account_id == account_id, Chat.id.in_(missing))
                    )
                )
                .scalars()
                .all()
            )
        for c in rows:
            found[int(c.id)] = c
    return [found[c] for c in ids if c in found]


async def run_pipeline(
    session_factory: DS.SessionFactory,
    account_id: int,
    conversation_id: int,
    question: str,
    *,
    locale: str,
    deep: bool,
    events: DS.EventSink | None,
    meter: DS._Meter,
    started: float,
    state: dict[str, Any],
    mode: str = "deep",
    confirm: bool = True,
    chat_ids: list[int] | None = None,
    context_mode: str = "deep",
) -> DS.DeepResult:
    """Orkestrator yugurishi — `deep_search.run_deep_search` (hisob kafolati) ichidan."""
    question = (question or "").strip()
    if not question:
        raise DS.DeepSearchError("empty")
    taint = DS._Taint()
    s = get_settings()
    soft_wall = s.deep_soft_wall_s or DS.SOFT_WALL_S
    hard_wall = s.deep_hard_wall_s or DS.HARD_WALL_S
    if mode == "exhaustive":
        from app.services.scale_search import EXHAUSTIVE_WALL_S

        hard_wall = max(hard_wall, EXHAUSTIVE_WALL_S)

    def elapsed() -> float:
        return time.monotonic() - started

    # ── byudjet: **har qanday LLM chaqirig'idan oldin** ──────────────────────
    async with session_factory() as session:
        rctx = await DS.load_run_context(
            session, account_id=account_id, conversation_id=conversation_id
        )
        user_id = rctx.user_id
        state["user_id"] = user_id
        exceeded = await LIM.check_daily_budget(session, user_id)
        if exceeded:
            await DS._emit(events, "error", code="budget", detail=exceeded)
            DEEP_RUNS.labels("budget").inc()
            raise DS.DeepSearchError("budget", exceeded)
    chats_by_id = {c.id: c for c in rctx.chats}
    history = list(rctx.history)
    selected_rows = (
        await _load_selected(
            session_factory,
            account_id=account_id,
            chat_ids=list(chat_ids or []),
            synced=chats_by_id,
        )
        if chat_ids
        else []
    )
    selected = [c.id for c in selected_rows if c.id in chats_by_id]

    ctx = SK.RunState(
        session_factory=session_factory,
        account_id=account_id,
        user_id=user_id,
        conversation_id=conversation_id,
        question=question,
        raw_question=question,
        locale=locale,
        mode=mode,
        confirm=confirm,
        deep=deep,
        meter=meter,
        taint=taint,
        events=events,
        history=history,
        tz=rctx.tz,
        chats_by_id=chats_by_id,
        plan=DS.DeepPlan(subqueries=[question[:200]], standalone_question=question[:400]),
        selected=selected,
        started=started,
        soft_wall=soft_wall,
        hard_wall=hard_wall,
        now=time.monotonic,
        context_mode=context_mode if context_mode in ("deep", "agent") else "deep",
    )
    if len(selected_rows) > len(selected):
        ctx.notes.append("selected: some context chats are not synced yet")
    # Suhbat holati (migratsiya 0024) — referent/oyna merosi va keyingi turn uchun.
    ctx.conv = await CS.load(session_factory, conversation_id)

    # ── (a) kod darvozasi: salom / meta — planner ham chaqirilmaydi ──────────
    gate = DS.direct_intent(question, history)
    if gate:
        ctx.plan = DS.DeepPlan(
            subqueries=[],
            standalone_question=question[:400],
            intent_follow_up="chitchat" if gate == "greeting" else gate,
            needs_retrieval=False,
        )
        orch = OrchPlan(
            plan=ctx.plan,
            steps=[Step("answer_direct", {"intent": gate}, "code gate", "gate")],
            expected_answer_kind="text",
            output_budget="short",
        )
        await DS._emit(events, "stages", stages=["plan", "write"], mode=mode, direct=True)
        await DS._emit(events, "plan", **orch.to_dict())
        res = await SK.SKILLS["answer_direct"].run(ctx, {"intent": gate})
        return await _finish_terminal(ctx, orch, res, intent=gate, gate="code", state=state)

    await DS._emit(events, "stages", stages=["plan"], mode=mode)

    status = "ok"
    truncated = False
    ran: list[dict[str, Any]] = []
    orch = OrchPlan(plan=ctx.plan)
    try:
        async with asyncio.timeout(hard_wall):
            orch = await plan_orchestrated(
                meter,
                taint,
                question,
                chat_lines=rctx.chat_lines,
                known_chats=list(chats_by_id),
                selected=selected_rows,
                history=history,
                mode=mode,
                tz=rctx.tz,
                conv=ctx.conv,
                hint_lines=[hint_line(h) for h in ctx_hints(ctx)],
            )
            ctx.plan = orch.plan
            ctx.question = orch.plan.standalone_question or question
            steps = list(orch.steps)
            await DS._emit(events, "plan", **orch.to_dict())
            await DS._emit(
                events,
                "stages",
                stages=rail_for(steps),
                steps=[st.to_dict() for st in steps],
                mode=mode,
                direct=bool(steps and SK.SKILLS[steps[0].skill].terminal),
            )
            log.info(
                "orch.plan",
                conversation_id=conversation_id,
                mode=mode,
                steps=[st.skill for st in steps],
                selected=selected,
                notes=orch.notes[:6],
                planner_ok=orch.planner_ok,
                plan_task=plan_task().value,
            )
            done_keys: set[str] = set()
            for i, step in enumerate(steps):
                res = await _run_step(ctx, step, i, ran, done_keys)
                if res.needs_confirm:
                    return await _finish_confirm(ctx, orch, state=state)
                if res.stop:
                    intent = str(res.summary.get("intent") or step.skill)
                    if step.skill == "clarify":
                        intent = "clarify"
                    elif step.skill == "write_propose":
                        intent = "write"
                    return await _finish_terminal(
                        ctx, orch, res, intent=intent, gate="plan", state=state
                    )
            # ── yolg'iz `agent_tools`: uning javobi — yakuniy javob ─────────
            # Ikkinchi sintez faqat narx qo'shardi (eski `fast` yo'li ham
            # agent javobini to'g'ridan-to'g'ri berardi).
            if ctx.used == {"agent_tools"}:
                return await _finish_agent(ctx, orch, state=state)
            # ── server qo'riqchisi (`needs_server`, kodda) ──────────────────
            guard = await _server_guard(ctx)
            if guard is not None:
                orch.steps.append(guard)
                await DS._emit(
                    events,
                    "stages",
                    stages=rail_for(orch.steps),
                    steps=[st.to_dict() for st in orch.steps],
                    mode=mode,
                )
                await _run_step(ctx, guard, len(orch.steps) - 1, ran, done_keys)
            # ── ≤1 replan ────────────────────────────────────────────────────
            if needs_replan(ctx) and elapsed() < soft_wall:
                extra, rnotes = await replan(ctx, ran, done_keys=done_keys)
                orch.notes.extend(rnotes)
                if extra:
                    orch.steps.extend(extra)
                    await DS._emit(
                        events,
                        "stages",
                        stages=rail_for(orch.steps),
                        steps=[st.to_dict() for st in orch.steps],
                        mode=mode,
                    )
                    for step in extra:
                        res = await _run_step(ctx, step, len(ran), ran, done_keys)
                        if res.needs_confirm:  # pragma: no cover — replan'da tasdiqli skan yo'q
                            return await _finish_confirm(ctx, orch, state=state)
    except TimeoutError:
        truncated = True
        status = "timeout"
        log.warning("orch.hard_timeout", elapsed_s=round(elapsed(), 1))

    return await _finish_answer(ctx, orch, state=state, status=status, truncated=truncated)


async def _run_step(
    ctx: SK.RunState, step: Step, index: int, ran: list[dict[str, Any]], done_keys: set[str]
) -> SK.SkillResult:
    skill = SK.SKILLS[step.skill]
    stage = skill.rail[0] if skill.rail else "write"
    await DS._emit(
        ctx.events, "step", index=index, skill=step.skill, state="start", stage=stage, why=step.why
    )
    t0 = time.monotonic()
    res = await skill.run(ctx, dict(step.args))
    ctx.used.add(step.skill)
    done_keys.add(step.key())
    for f in res.facts:
        ctx.facts[f.name] = f
    ctx.notes.extend(res.notes)
    summary = dict(res.summary)
    ran.append({"skill": step.skill, "args": step.args, "summary": summary})
    log.info(
        "orch.step",
        skill=step.skill,
        origin=step.origin,
        ms=int((time.monotonic() - t0) * 1000),
        summary=summary,
        facts=[f.name for f in res.facts],
    )
    await DS._emit(
        ctx.events,
        "step",
        index=index,
        skill=step.skill,
        state="done",
        stage=stage,
        summary=summary,
        facts=[f.to_dict() for f in res.facts],
    )
    return res


async def _server_guard(ctx: SK.RunState) -> Step | None:
    """Eski `maybe_escalate` qoidasi — qidiruvdan keyin, kodda (`needs_server`)."""
    if not ctx.used & SEARCH_SKILLS or ctx.used & {"search_server", "discover"}:
        return None
    if ctx.plan.server_scope == "none" or SK.server_calls_left(ctx) <= 0:
        return None
    gaps = SK._gap_chats(ctx)
    earliest: datetime | None = None
    quick = (
        ctx.plan.server_scope in ("global", "discover")
        or bool(gaps)
        or (len(ctx.ranked) < int(get_settings().deep_server_min_local_hits))
    )
    if not quick and ctx.plan.intent in ("lookup", "person") and ctx.plan.since is not None:
        async with ctx.session_factory() as session:
            earliest = await DS.earliest_synced_at(
                session, ctx.plan.chats or sorted(ctx.chats_by_id)
            )
    need, reason = DS.needs_server(
        ctx.plan,
        local_hits=len(ctx.ranked),
        gap_chats=gaps,
        earliest=earliest,
        min_local_hits=int(get_settings().deep_server_min_local_hits),
    )
    if not need:
        return None
    if ctx.plan.server_scope == "discover" and SK.SKILLS["discover"].allowed(ctx.mode):
        return Step("discover", {}, f"server guard: {reason}", "guard")
    scope = "global" if ctx.plan.server_scope in ("global", "discover") else "gaps"
    return Step("search_server", {"scope": scope}, f"server guard: {reason}", "guard")


# ─── yakunlash ───────────────────────────────────────────────────────────────


def _usage(meter: DS._Meter) -> dict[str, Any]:
    return {
        "tokens_in": meter.tokens_in,
        "tokens_out": meter.tokens_out,
        "tokens_thinking": meter.tokens_thinking,
        "tokens_cached": meter.tokens_cached,
        "cost_usd": round(meter.cost_usd, 6) if meter.cost_usd else None,
    }


async def _actions(ctx: SK.RunState) -> list[dict[str, Any]]:
    if not ctx.run_id or not ctx.action_ids:
        return []
    try:
        from app.services import actions as ACT

        async with ctx.session_factory() as session:
            got = await ACT.actions_for_runs(session, [ctx.run_id])
        return list(got.get(ctx.run_id, []))
    except Exception as exc:  # amal kartasi javobdan muhim emas
        log.warning("orch.actions_failed", error=str(exc)[:200])
        return []


def ctx_hints(ctx: SK.RunState) -> list[Any]:
    """Savoldagi nom → chat (`agent.chat_hints`, LLM'siz) — yugurishda bir marta."""
    if ctx.hints is None:
        from app.services import agent as AG

        ctx.hints = AG.chat_hints(ctx.raw_question, list(ctx.chats_by_id.values()))
    return ctx.hints


def hint_line(h: Any) -> str:
    """Planner uchun maslahat satri (sarlavha begona — konvert ichida chiqadi)."""
    return f'"{h.title}" [id:{h.chat_id}] {h.kind} msgs={h.msgs} match={h.score:g}'


async def _save_state(
    ctx: SK.RunState,
    orch: OrchPlan,
    *,
    evidence_refs: list[dict[str, int]] | None = None,
    terminal: str = "",
) -> None:
    """Turn'dan keyingi suhbat holati — **kod** yozadi (`conv_state.update_after_turn`).

    Saqlashdan oldin `_Taint.state_leak`: holatga xabar matni yoki oldingi
    javob bo'lagi tushgan bo'lsa u saqlanmaydi (log) — keyingi planner
    promptiga begona matn o'tmasin. Xato javobni buzmaydi.
    """
    if ctx.conv is None:
        return
    try:
        new = CS.update_after_turn(
            ctx.conv,
            question=ctx.raw_question,
            plan=ctx.plan,
            steps=[s.skill for s in orch.steps],
            facts=list(ctx.facts.values()),
            evidence_refs=list(evidence_refs or []),
            hints=ctx_hints(ctx),
            chat_titles=ctx.chat_titles,
            tool_calls=list((ctx.agent_meta or {}).get("tool_calls") or []),
            selected=ctx.selected,
            answer_kind=orch.expected_answer_kind,
            terminal=terminal,
        )
        blob = json.dumps(new.to_json(), ensure_ascii=False, default=str)
        if ctx.taint.state_leak(U.neutralize(blob)) or ctx.taint.state_leak(blob):
            log.error("orch.state_taint", conversation_id=ctx.conversation_id)
            return
        ctx.conv = new
        await CS.save(ctx.session_factory, ctx.conversation_id, new)
    except Exception as exc:  # holat — yordamchi; javob allaqachon yozilgan
        log.warning("orch.state_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")


def _orch_meta(orch: OrchPlan, ctx: SK.RunState) -> dict[str, Any]:
    return {
        "steps": [s.to_dict() for s in orch.steps],
        "used": sorted(ctx.used),
        "target_chats": orch.target_chats,
        "expected_answer_kind": orch.expected_answer_kind,
        "output_budget": orch.output_budget,
        "notes": (orch.notes + ctx.notes)[:20],
        "planner_ok": orch.planner_ok,
    }


async def _finish_terminal(
    ctx: SK.RunState,
    orch: OrchPlan,
    res: SK.SkillResult,
    *,
    intent: str,
    gate: str,
    state: dict[str, Any],
) -> DS.DeepResult:
    """Qidiruvsiz javob (`answer_direct` / `clarify` / `write_propose`).

    Eski `finish_direct` bilan bir xil yozuv va hodisalar.
    """
    answer = res.answer
    state["answer"] = answer
    for start in range(0, len(answer), DS.TOKEN_CHUNK_CHARS):
        await DS._emit(ctx.events, "token", text=answer[start : start + DS.TOKEN_CHUNK_CHARS])
    usage = _usage(ctx.meter)
    result = DS.DeepResult(
        answer=answer,
        plan=orch.to_dict(),
        mode=ctx.mode,
        direct=True,
        intent=intent,
        model=res.result.model if res.result else "",
        provider=res.result.provider if res.result else "",
        latency_ms=int(ctx.elapsed() * 1000),
        conversation_id=ctx.conversation_id,
        **usage,
    )
    await DS._persist(
        ctx.session_factory,
        conversation_id=ctx.conversation_id,
        user_id=ctx.user_id,
        account_id=ctx.account_id,
        question=ctx.raw_question,
        result=result,
        context_meta={
            "mode": ctx.context_mode,
            "scale_mode": ctx.mode,
            "strategy": "direct",
            "source": "deep",
            "account_id": ctx.account_id,
            "direct": True,
            "intent": intent,
            "messages": 0,
            "candidates": 0,
            "coverage": result.coverage.to_dict(),
            "plan": result.plan,
            "orchestrator": _orch_meta(orch, ctx),
            "selected_chats": list(ctx.selected),
            "citations": [],
            "est_tokens": S.est_tokens(answer),
            "finish_reason": res.result.finish_reason if res.result else "",
            "evidence_refs": DS.prior_refs(ctx.history),
            "reused_evidence": False,
            **(agent_context(ctx) if ctx.context_mode == "agent" else {}),
        },
        run_id=ctx.run_id,
    )
    state["persisted"] = True
    await _save_state(ctx, orch, terminal=intent)
    state["status"] = "direct"
    DEEP_RUNS.labels("direct").inc()
    DEEP_LATENCY.observe(ctx.elapsed())
    if result.cost_usd:
        DEEP_COST.inc(result.cost_usd)
    log.info(
        "orch.direct",
        conversation_id=ctx.conversation_id,
        intent=intent,
        gate=gate,
        mode=ctx.mode,
        llm_calls=ctx.meter.calls,
    )
    await DS._emit(
        ctx.events,
        "done",
        direct=True,
        intent=intent,
        citations=[],
        invalid_citations=0,
        steps=[s.skill for s in orch.steps],
        actions=await _actions(ctx),
        model=result.model,
        provider=result.provider,
        latency_ms=result.latency_ms,
        assistant_message_id=result.assistant_message_id,
        conversation_id=ctx.conversation_id,
        **usage,
    )
    return result


_EMPTY_AGENT = "…"


def agent_context(ctx: SK.RunState) -> dict[str, Any]:
    """`agent` rejimi (chat API `fast`) uchun eski `context_used` maydonlari.

    UI (`messages.js::contextLine`) `mode == "agent"` da `tools` chiplarini
    chizadi, judge (`evaluation.run_flags`) `tool_calls` ga qaraydi — shu
    kalitlar o'z nomi bilan qoladi. Ko'nikma nomlari ham "tool" bo'lib
    ko'rinadi (`stats`, `chat_digest`, …), agent tool'lari esa ularning
    ichida.
    """
    meta = ctx.agent_meta or {}
    tools = [s for s in sorted(ctx.used) if s != "agent_tools"] + list(meta.get("tools") or [])
    out: dict[str, Any] = {
        "tools": tools,
        "tools_available": True,
        "iterations": int(meta.get("iterations") or len(ctx.used) or 1),
    }
    if meta:
        out.update(
            tool_calls=meta.get("tool_calls") or [],
            calls_executed=int(meta.get("calls_executed") or 0),
            stop_reason=meta.get("stop_reason") or "",
            repeat_guard=bool(meta.get("repeat_guard")),
            cache_hits=int(meta.get("cache_hits") or 0),
        )
    return out


async def _finish_agent(
    ctx: SK.RunState, orch: OrchPlan, *, state: dict[str, Any]
) -> DS.DeepResult:
    """Yolg'iz `agent_tools` qadami — agent javobi yakuniy javob (ikkinchi sintez yo'q).

    Yozuv o'sha yagona yozuvchida (`_persist`), `context.orchestrator` bilan.
    Qamrov da'vosi yo'q: agent nimani ko'rgani tool jurnalida (`tool_calls`).
    """
    meta = ctx.agent_meta or {}
    answer = ctx.agent_answer or _EMPTY_AGENT
    full_answer = f"{answer}{steps_footer(ctx, orch.steps)}"
    state["answer"] = full_answer
    for start in range(0, len(answer), DS.TOKEN_CHUNK_CHARS):
        await DS._emit(ctx.events, "token", text=answer[start : start + DS.TOKEN_CHUNK_CHARS])
    usage = _usage(ctx.meter)
    finish_reason = str(meta.get("finish_reason") or "")
    result = DS.DeepResult(
        answer=full_answer,
        plan=orch.to_dict(),
        mode=ctx.mode,
        intent="agent",
        model=str(meta.get("model") or ""),
        provider=str(meta.get("provider") or ""),
        latency_ms=int(ctx.elapsed() * 1000),
        conversation_id=ctx.conversation_id,
        **usage,
    )
    facts = list(ctx.facts.values())
    context_meta: dict[str, Any] = {
        "mode": ctx.context_mode,
        "scale_mode": ctx.mode,
        "strategy": "agent" if ctx.context_mode == "agent" else "deep",
        "source": "tools",
        "account_id": ctx.account_id,
        "intent": "agent",
        "messages": sum(
            int(c.get("hits") or c.get("n") or 0) for c in (meta.get("tool_calls") or [])
        )
        or None,
        "candidates": 0,
        "plan": result.plan,
        "orchestrator": _orch_meta(orch, ctx),
        "facts": [f.to_dict() for f in facts],
        "selected_chats": list(ctx.selected),
        "citations": [],
        "est_tokens": S.est_tokens(answer),
        "finish_reason": finish_reason,
        "truncated_answer": DS.is_length_finish(finish_reason),
        "evidence_refs": [],
        "reused_evidence": False,
        **agent_context(ctx),
    }
    await DS._persist(
        ctx.session_factory,
        conversation_id=ctx.conversation_id,
        user_id=ctx.user_id,
        account_id=ctx.account_id,
        question=ctx.raw_question,
        result=result,
        context_meta=context_meta,
        run_id=ctx.run_id,
    )
    state["persisted"] = True
    await _save_state(ctx, orch)
    state["status"] = "ok"
    DEEP_RUNS.labels("ok").inc()
    DEEP_LATENCY.observe(ctx.elapsed())
    if result.cost_usd:
        DEEP_COST.inc(result.cost_usd)
    log.info(
        "orch.agent_done",
        conversation_id=ctx.conversation_id,
        mode=ctx.mode,
        tools=meta.get("tools"),
        stop_reason=meta.get("stop_reason"),
        llm_calls=ctx.meter.calls,
        cost_usd=result.cost_usd,
    )
    await DS._emit(
        ctx.events,
        "done",
        citations=[],
        invalid_citations=0,
        steps=[s.skill for s in orch.steps],
        tools=meta.get("tools") or [],
        actions=await _actions(ctx),
        model=result.model,
        provider=result.provider,
        latency_ms=result.latency_ms,
        assistant_message_id=result.assistant_message_id,
        conversation_id=ctx.conversation_id,
        **usage,
    )
    return result


async def _finish_confirm(
    ctx: SK.RunState, orch: OrchPlan, *, state: dict[str, Any]
) -> DS.DeepResult:
    """Tasdiqsiz `exhaustive` skan — faqat baho (eski quvur bilan bir xil)."""
    await DS._emit(ctx.events, "done", needs_confirm=True, estimate=ctx.estimate_meta)
    state["status"] = "needs_confirm"
    state["needs_confirm"] = True
    log.info("orch.needs_confirm", steps=[s.skill for s in orch.steps], estimate=ctx.estimate_meta)
    await DS._persist_scan_only(
        ctx.session_factory,
        account_id=ctx.account_id,
        conversation_id=ctx.conversation_id,
        scan=ctx.scan_meta,
        elapsed_ms=int(ctx.elapsed() * 1000),
    )
    usage = _usage(ctx.meter)
    usage.pop("cost_usd")
    return DS.DeepResult(
        # Faqat oqimsiz API uchun (UI baho kartasini `done` dan chizadi, `token`
        # yuborilmaydi): javob bo'sh qolmasin — nima to'lanishi va nega
        # tasdiq kerakligi. Suhbatga **yozilmaydi**.
        answer=confirm_text(ctx.estimate_meta, ctx.locale),
        mode=ctx.mode,
        needs_confirm=True,
        estimate=ctx.estimate_meta,
        scan=ctx.scan_meta,
        plan=orch.to_dict(),
        conversation_id=ctx.conversation_id,
        latency_ms=int(ctx.elapsed() * 1000),
        **usage,
    )


_CONFIRM_TEXT = {
    "uz": (
        "Bu savol korpus skanini talab qiladi: ~{buckets} bo'lak, ~{msgs} xabar, taxminiy "
        "narx ${cost} (qopqoq ${ceiling}). Skan hali boshlanmadi — davom etish uchun "
        "tasdiqlang yoki savolni «Chuqur» rejimida bering."
    ),
    "ru": (
        "Этот вопрос требует скана корпуса: ~{buckets} блоков, ~{msgs} сообщений, оценка "
        "${cost} (потолок ${ceiling}). Скан ещё не запускался — подтвердите или задайте "
        "вопрос в режиме «Глубокий»."
    ),
    "en": (
        "This question needs a corpus scan: ~{buckets} buckets, ~{msgs} messages, estimated "
        "${cost} (ceiling ${ceiling}). Nothing has been scanned yet — confirm to run it, "
        "or ask in Deep mode."
    ),
}


def confirm_text(estimate: dict[str, Any], locale: str) -> str:
    """Tasdiq kutilayotgan yugurishning qisqa matni — raqamlar **bahodan** (kodda)."""
    tpl = _CONFIRM_TEXT.get(locale, _CONFIRM_TEXT["uz"])
    return tpl.format(
        buckets=int(estimate.get("buckets") or 0),
        msgs=int(estimate.get("est_messages") or 0),
        cost=f"{float(estimate.get('est_total_usd') or estimate.get('est_cost_usd') or 0):.2f}",
        ceiling=f"{float(estimate.get('ceiling_usd') or 0):.2f}",
    )


def facts_block(facts: list[CMP.Fact]) -> str:
    if not facts:
        return ""
    lines = [f.render() for f in facts]
    return "facts (computed by code — exact; use these numbers as given):\n" + "\n".join(lines)


def synth_task(used: set[str], deep: bool) -> Task | None:
    """Raqam/dayjestni so'zga aylantirish — `Task.SEARCH`; mavzu dalili — eski qoida."""
    if used and used <= CHEAP_SYNTH_SKILLS:
        return Task.SEARCH
    return None


def answer_budget(ctx: SK.RunState, orch: OrchPlan) -> DS.SynthBudget:
    digestish = bool(ctx.used & {"stats", "window", "chat_digest", "compute"})
    base = DS.synth_budget(ctx.raw_question, follow_up=ctx.plan.intent_follow_up, window=digestish)
    if base.kind == "detail":
        return base
    if orch.output_budget == "short" or orch.expected_answer_kind in ("number", "digest"):
        return DS.SynthBudget(DS.SYNTH_BUDGET_DIGEST, "digest", False)
    return base


async def _finish_answer(
    ctx: SK.RunState,
    orch: OrchPlan,
    *,
    state: dict[str, Any],
    status: str,
    truncated: bool,
) -> DS.DeepResult:
    plan = ctx.plan
    chat_rows = list(ctx.chats_by_id.values())
    from app.services import scale_search as SC

    cap = (
        DS.DIGEST_SYNTH_LINES
        if ctx.used <= CHEAP_SYNTH_SKILLS | {"window"}
        else SC.SYNTH_EVIDENCE_LINES
    )
    evidence = DS.to_evidence(
        ctx.ranked[:cap],
        ctx.clusters,
        link_prefixes={
            c.id: (c.link_prefix or "") for c in chat_rows if c.type != ChatType.PRIVATE
        },
        server_ids=ctx.server_out.server_ids,
        external_links=ctx.server_out.external_links,
    )
    ctx.taint.add(evidence)
    facts = list(ctx.facts.values())
    claims_block = "\n\n".join(x for x in (facts_block(facts), *ctx.claims) if x)
    notes: list[str] = []
    if "stats" in ctx.used:
        notes.append(STATS_SYNTH_NOTE)
    elif "window" in ctx.used:
        notes.append(WINDOW_SYNTH_NOTE)
    if "chat_digest" in ctx.used:
        notes.append(DIGEST_SYNTH_NOTE)
    if facts:
        notes.append(FACTS_SYNTH_NOTE)
    notes.append(
        f"expected_answer_kind: {orch.expected_answer_kind} (steps: "
        f"{' → '.join(s.skill for s in orch.steps)})"
    )
    budget = answer_budget(ctx, orch)
    answer, kept, res = await DS.synthesize(
        ctx.meter,
        ctx.raw_question,
        evidence,
        ctx.clusters,
        locale=ctx.locale,
        deep=ctx.deep,
        history=ctx.history,
        follow_up=plan.intent_follow_up,
        claims_block=claims_block,
        mode_note="\n".join(notes),
        outside_asked=DS.question_asks_outside(plan),
        task=synth_task(ctx.used, ctx.deep),
        budget=budget,
    )
    state["answer"] = answer
    if len(kept) < len(evidence):
        truncated = True
    finish_reason = res.finish_reason if res else ""
    truncated_answer = DS.is_length_finish(finish_reason)
    if truncated_answer:
        truncated = True
    citations, invalid = DS.validate_citations(answer, kept)
    shown_external = sum(1 for e in kept if e.external)
    ext_note = DS.external_claim_note(answer, shown_external, locale=ctx.locale)
    if ext_note:
        log.warning("deep.external_claim_unbacked", external_hits=ctx.server_out.external)
        answer = f"{answer}\n\n{ext_note}"
        state["answer"] = answer
    unbacked: list[float] = []
    if facts:
        unbacked = CMP.unbacked_numbers(answer, facts, extra_text=claims_block)
        if unbacked:
            log.warning("orch.unbacked_numbers", numbers=unbacked[:10])
    searched_chats = [
        ctx.chats_by_id[cid]
        for cid in sorted({h.chat_id for h in ctx.ranked if h.chat_id})
        if cid in ctx.chats_by_id
    ]
    if not searched_chats and ctx.candidates:
        searched_chats = (
            [ctx.chats_by_id[c] for c in plan.chats if c in ctx.chats_by_id]
            if plan.chats
            else chat_rows
        )
    scanned = len(ctx.candidates)
    if ctx.digest_meta:
        scanned = max(scanned, int(ctx.digest_meta.get("covered_messages") or 0) + len(ctx.ranked))
        digest_chats = [
            ctx.chats_by_id[c] for c in ctx.digest_meta.get("chats", []) if c in ctx.chats_by_id
        ]
        searched_chats = digest_chats or searched_chats
    scan_for_cov = ctx.scan_meta if ctx.used & {"scan_corpus", "stats", "window"} else None
    coverage = DS.build_coverage(
        scanned_hits=ctx.ranked,
        messages_scanned=scanned,
        chats_total=len(chat_rows),
        searched_chats=searched_chats,
        evidence=kept,
        clusters=ctx.clusters,
        rounds=1,
        truncated=truncated,
        marginal_stop=False,
        server=ctx.server_out,
        scan=scan_for_cov,
        window=ctx.window_meta or None,
    )
    for i in range(0, len(answer), DS.TOKEN_CHUNK_CHARS):
        await DS._emit(ctx.events, "token", text=answer[i : i + DS.TOKEN_CHUNK_CHARS])
    await DS._emit(ctx.events, "coverage", **coverage.to_dict())
    full_answer = (
        f"{answer}\n\n{DS.render_coverage(coverage, locale=ctx.locale)}"
        f"{steps_footer(ctx, orch.steps)}"
    )
    usage = _usage(ctx.meter)
    result = DS.DeepResult(
        answer=full_answer,
        citations=citations,
        invalid_citations=invalid,
        clusters=DS._cluster_payload(ctx.clusters),
        coverage=coverage,
        plan=orch.to_dict(),
        candidates=len(ctx.candidates),
        evidence_count=len(kept),
        model=res.model if res else "",
        provider=res.provider if res else "",
        latency_ms=int(ctx.elapsed() * 1000),
        conversation_id=ctx.conversation_id,
        mode=ctx.mode,
        estimate=ctx.estimate_meta,
        scan=ctx.scan_meta,
        **usage,
    )
    context_meta: dict[str, Any] = {
        "mode": ctx.context_mode,
        "scale_mode": ctx.mode,
        "scan": ctx.scan_meta,
        "estimate": ctx.estimate_meta,
        "retrieval_kind": plan.retrieval_kind,
        "inbox": plan.inbox,
        "direction": plan.direction,
        "window_stats": plan.window_stats,
        "strategy": "deep",
        "source": "deep",
        "account_id": ctx.account_id,
        "messages": len(kept),
        "est_tokens": S.est_tokens(answer),
        "finish_reason": finish_reason,
        "truncated_answer": truncated_answer,
        "answer_budget": {
            "kind": budget.kind,
            "max_tokens": budget.max_tokens,
            "call_max_tokens": budget.call_max_tokens,
        },
        "rounds": 1,
        "candidates": len(ctx.candidates),
        "clusters": result.clusters,
        "coverage": coverage.to_dict(),
        "citations": citations,
        "invalid_citations": invalid,
        "plan": result.plan,
        "orchestrator": _orch_meta(orch, ctx),
        "facts": [f.to_dict() for f in facts],
        "unbacked_numbers": unbacked[:10],
        "selected_chats": list(ctx.selected),
        "evidence_refs": DS.evidence_refs(kept),
        "reused_evidence": "reuse_prior" in ctx.used,
        **(agent_context(ctx) if ctx.context_mode == "agent" or ctx.agent_meta else {}),
    }
    scan_persist = ctx.scan_meta if ctx.used & {"scan_corpus", "window", "stats"} else None
    await DS._persist(
        ctx.session_factory,
        conversation_id=ctx.conversation_id,
        user_id=ctx.user_id,
        account_id=ctx.account_id,
        question=ctx.raw_question,
        result=result,
        context_meta=context_meta,
        scan=scan_persist,
        run_id=ctx.run_id,
    )
    state["persisted"] = True
    await _save_state(ctx, orch, evidence_refs=context_meta["evidence_refs"])
    DEEP_RUNS.labels(status).inc()
    DEEP_ROUNDS.observe(1)
    DEEP_CANDIDATES.observe(len(ctx.candidates))
    DEEP_LATENCY.observe(ctx.elapsed())
    if result.cost_usd:
        DEEP_COST.inc(result.cost_usd)
    log.info(
        "orch.done",
        conversation_id=ctx.conversation_id,
        steps=[s.skill for s in orch.steps],
        evidence=len(kept),
        citations=len(citations),
        invalid_citations=invalid,
        facts=len(facts),
        unbacked=len(unbacked),
        truncated=truncated,
        llm_calls=ctx.meter.calls,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
    await DS._emit(
        ctx.events,
        "done",
        citations=citations,
        invalid_citations=invalid,
        steps=[s.skill for s in orch.steps],
        facts=[f.to_dict() for f in facts],
        model=result.model,
        provider=result.provider,
        latency_ms=result.latency_ms,
        assistant_message_id=result.assistant_message_id,
        conversation_id=ctx.conversation_id,
        **usage,
    )
    return result


__all__ = [
    "MAX_STEPS",
    "ORCH_PLAN_PROMPT",
    "OrchPlan",
    "Step",
    "apply_guards",
    "build_orch_prompt",
    "build_replan_prompt",
    "default_steps",
    "facts_block",
    "needs_replan",
    "parse_orch_plan",
    "parse_steps",
    "plan_orchestrated",
    "rail_for",
    "replan",
    "run_pipeline",
    "synth_task",
]
