"""Orkestrator katalogining kengaytmasi — natija formati va avtomatlashtirish.

`skills.register_extra()` shu modulni import qilib `register(SKILLS)` ni
chaqiradi. Bu yerdagi besh ko'nikma:

* `format_output` — faktlardan jadval / ```chart / raqamli ro'yxat, **kodda**
  (`services/output_formats.py`). LLM yo'q, raqam o'ylab topilmaydi.
* `create_alert` / `set_briefing` / `remind` (+ `cancel_reminder`) — akkaunt
  holatini o'zgartiradigan amallar. Hech biri shu yerda **bajarilmaydi**:
  `write_tools.propose_automation` → `agent_actions(proposed)` → foydalanuvchi
  tasdig'i → `actions.execute_action` → `automation.execute`. `autonomous`
  chat rejimi bularga tegishli emas.
* `send_result` — "topib Y ga yubor": oldingi qadamlar (yoki oldingi javob)
  natijasini `write_tools.propose_or_execute("send_message")` orqali **bitta**
  taklifga aylantiradi; matn Actions sahifasida to'liq ko'rinadi,
  `assert_writable` / peer guard / `read_only` o'zgarmagan.

Taint: planner bu modul chiqishini ko'rmaydi — `summary` da faqat sanoq va
bayroqlar. `send_result` karantinda (sintez qilingan matnni o'qiydi).
Yozish niyati **foydalanuvchining o'z matnida** tekshiriladi (planner
xato qilsa ham taklif yaratilmaydi).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from app.db.models import AgentRun, ChatType, ConversationMessage
from app.logging import get_logger
from app.services import automation as AU
from app.services import output_formats as OF
from app.services import scheduled as SCH
from app.services import untrusted as U
from app.services.compute import Fact

if TYPE_CHECKING:
    from app.services.skills import RunState, Skill, SkillResult

# `skills` bu modulni **o'z importi paytida** (`register_extra`) chaqiradi, shuning
# uchun `app.services.skills` dan modul darajasida import qilinmaydi: aks holda
# `skills_ext` birinchi import qilinganda (testlar, worker) `register_extra`
# yarim yuklangan modulni ko'rib, hech narsa ro'yxatga olmasdi. `Skill` /
# `SkillResult` chaqiruv paytida olinadi (`_result`, `ext_skills`).

log = get_logger(__name__)


def _result(**kw: Any) -> SkillResult:
    from app.services.skills import SkillResult

    return SkillResult(**kw)


#: `send_result` — yuborish niyati (foydalanuvchi matnida).
SEND_INTENT_RE = re.compile(
    r"(yubor\w*|jo'nat\w*|tashla\w*|ulash\w*|joyla\w*|post\s*qil\w*|"
    r"отправ\w*|скин\w*|перешли|закин\w*|опубликуй\w*|"
    r"\bsend\b|\bpost\b|\bshare\b|forward)",
    re.I,
)
#: `remind` — eslatish niyati.
REMIND_INTENT_RE = re.compile(r"(eslat\w*|напомн\w*|напомин\w*|remind\w*)", re.I)
CANCEL_RE = re.compile(r"(bekor\s+qil\w*|o'chir\w*|отмен\w*|удали\w*|cancel\w*)", re.I)
ALL_RE = re.compile(r"\b(hammasi\w*|barcha\w*|hamma\w*|все\w*|всё|all)\b", re.I)

SEND_TEXT_MAX = 4000  # Telegram 4096 — zaxira bilan
SEND_EVIDENCE_LINES = 40


def _loc(ctx: RunState) -> str:
    loc = getattr(ctx, "locale", "uz")
    return loc if loc in ("uz", "ru", "en") else "uz"


_MSG = {
    "uz": {
        "proposed": "Taklif tayyor — hali hech narsa **o'zgarmadi**. Amallar sahifasida "
        "tasdiqlang yoki rad eting: #{id}\n\n{summary}",
        "no_intent": "So'rovingizda bunday amal buyrug'i ko'rinmadi — hech narsa taklif qilinmadi.",
        "alert_what": "Nimani kuzatay? Masalan: «'ijara' so'zi chiqsa xabar ber» yoki "
        "«kripto haqida yozishsa ogohlantir».",
        "remind_when": "Qachon eslatay? Masalan: «ertaga 10 da eslat: …» yoki «2 soatdan "
        "keyin eslat: …».",
        "remind_what": "Nimani eslatay? Masalan: «ertaga 10 da eslat: hisobotni yubor».",
        "no_reminders": "Kutayotgan eslatma yo'q.",
        "which_reminder": "Qaysi eslatmani bekor qilay? (hammasini — «hammasini bekor qil»)",
        "send_where": "Qaysi chatga yuboray? Chat nomini yoki @username ni yozing.",
        "send_which": "«{name}» nomli bir nechta chat bor — qaysi biriga yuboray?\n{options}",
        "send_nothing": "Yuboradigan natija yo'q — avval qidiruv yoki xulosa kerak.",
        "send_empty": "\n\n---\nNatija bo'sh — hech narsa yuborish taklif qilinmadi. Boshqa davr "
        "yoki chatni ko'rsating.",
        "send_proposed": "\n\n---\nYuborish taklifi #{id} → «{chat}»: xabar hali "
        "**yuborilmadi**. Amallar sahifasida matnni ko'rib, tasdiqlang.",
        "send_failed": "\n\n---\nYuborish taklifini tayyorlab bo'lmadi: {err}",
        "failed": "Amalni tayyorlab bo'lmadi: {err}",
    },
    "ru": {
        "proposed": "Предложение готово — пока **ничего не изменено**. Подтвердите или "
        "отклоните на странице действий: #{id}\n\n{summary}",
        "no_intent": "В запросе не видно такой команды — ничего не предложено.",
        "alert_what": "Что отслеживать? Например: «сообщи, если появится слово 'аренда'» или "
        "«предупреди, если напишут про крипту».",
        "remind_when": "Когда напомнить? Например: «завтра в 10 напомни: …» или «через 2 часа "
        "напомни: …».",
        "remind_what": "О чём напомнить? Например: «завтра в 10 напомни: отправить отчёт».",
        "no_reminders": "Нет ожидающих напоминаний.",
        "which_reminder": "Какое напоминание отменить? (все — «отмени все»)",
        "send_where": "В какой чат отправить? Напишите название или @username.",
        "send_which": "Нашлось несколько чатов «{name}» — в какой отправить?\n{options}",
        "send_nothing": "Нечего отправлять — сначала нужен поиск или вывод.",
        "send_empty": "\n\n---\nРезультат пуст — отправка не предложена. Укажите другой период "
        "или чат.",
        "send_proposed": "\n\n---\nПредложение отправки #{id} → «{chat}»: сообщение ещё "
        "**не отправлено**. Проверьте текст и подтвердите на странице действий.",
        "send_failed": "\n\n---\nНе удалось подготовить отправку: {err}",
        "failed": "Не удалось подготовить действие: {err}",
    },
    "en": {
        "proposed": "Proposal ready — nothing has **changed** yet. Confirm or reject it on the "
        "Actions page: #{id}\n\n{summary}",
        "no_intent": "Your message doesn't ask for this action — nothing was proposed.",
        "alert_what": "What should I watch for? E.g. “notify me if the word 'rent' appears” or "
        "“alert me when someone writes about crypto”.",
        "remind_when": "When should I remind you? E.g. “tomorrow at 10 remind me: …” or “in 2 "
        "hours remind me: …”.",
        "remind_what": "What should I remind you about? E.g. “tomorrow at 10 remind me: send "
        "the report”.",
        "no_reminders": "No pending reminders.",
        "which_reminder": "Which reminder should I cancel? (all of them — “cancel all”)",
        "send_where": "Which chat should I send it to? Give its title or @username.",
        "send_which": "There are several chats named “{name}” — which one should I send it "
        "to?\n{options}",
        "send_nothing": "Nothing to send yet — a search or summary is needed first.",
        "send_empty": "\n\n---\nThe result is empty — nothing was proposed for sending. Try "
        "another period or chat.",
        "send_proposed": "\n\n---\nSend proposal #{id} → “{chat}”: the message has **not** been "
        "sent. Review the text on the Actions page and confirm.",
        "send_failed": "\n\n---\nCould not prepare the send proposal: {err}",
        "failed": "Could not prepare the action: {err}",
    },
}


def _m(ctx: RunState, key: str, **kw: Any) -> str:
    return _MSG[_loc(ctx)][key].format(**kw)


def _stop(text: str, **summary: Any) -> SkillResult:
    return _result(answer=text, stop=True, summary=summary)


async def _ensure_run(ctx: RunState, session: Any) -> int:
    """`agent_actions.run_id` uchun audit qatori (`skills._write_propose` bilan bir xil)."""
    if ctx.run_id is None:
        run = AgentRun(
            user_id=ctx.user_id,
            account_id=ctx.account_id,
            chat_id=None,
            prompt=ctx.raw_question[:4000],
            model="",
        )
        session.add(run)
        await session.flush()
        ctx.run_id = int(run.id)
    return int(ctx.run_id)


async def _propose(ctx: RunState, draft: AU.Draft) -> SkillResult:
    from app.services import write_tools as W

    async with ctx.session_factory() as session:
        run_id = await _ensure_run(ctx, session)
        res = await W.propose_automation(session, run_id=run_id, tool=draft.tool, args=draft.args)
    action_id = int(res.meta.get("action_id") or 0)
    if action_id:
        ctx.action_ids.append(action_id)
    return _stop(
        _m(ctx, "proposed", id=action_id, summary=draft.summary),
        proposed=True,
        tool=draft.tool,
    )


def _single_chat(ctx: RunState, args: dict[str, Any]) -> int | None:
    cid = args.get("chat_id")
    if isinstance(cid, int) and cid in ctx.chats_by_id:
        return cid
    if len(ctx.selected) == 1:
        return ctx.selected[0]
    if ctx.plan.chats and len(ctx.plan.chats) == 1:
        return int(ctx.plan.chats[0])
    return None


# ─── 1. format_output ────────────────────────────────────────────────────────


async def _format_output(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    choice = OF.detect_format(
        ctx.raw_question,
        arg_format=str(args.get("format") or ""),
        arg_chart=str(args.get("chart") or ""),
    )
    if choice is None:
        choice = OF.FormatChoice(fmt="table")
    facts = list(ctx.facts.values())
    series_arg = str(args.get("series") or "")
    ranking = OF.ranking_rows(facts)
    title = U.neutralize(str(args.get("title") or "")).strip()[:80]
    # Hafta kuni / soat taqsimoti (`stats(by=…)`) reytingdan ustun: "haftaning qaysi
    # kuni … jadval" da blok ilgari chat reytingi bo'lib chiqardi.
    dist_kind, dist = OF.distribution_rows(
        facts,
        locale=_loc(ctx),
        prefer=series_arg if series_arg in ("weekday", "hour") else ctx.raw_question,
    )
    explicit_time = series_arg == "per_day" or bool(
        OF._TIME_SERIES_RE.search(OF._fold(ctx.raw_question))
    )
    want_dist = (
        bool(dist)
        and series_arg not in ("ranking", "per_day", "facts")
        and (series_arg in ("weekday", "hour") or not explicit_time)
    )
    want_time = not want_dist and (
        series_arg == "per_day"
        or (
            series_arg != "ranking"
            and (choice.time_series or (choice.fmt == "chart" and not ranking))
        )
    )
    block = ""
    new_facts: list[Fact] = []
    kind = ""
    rows = 0
    if want_dist:
        kind = dist_kind
        rows = len(dist)
        if choice.fmt == "chart":
            spec = OF.chart_spec(
                [label for label, _ in dist],
                [("messages", [v for _, v in dist])],
                kind="bar",
                title=title,
            )
            block = OF.render_chart(spec) if spec else ""
        elif choice.fmt == "list":
            block = OF.render_list(dist)
        else:
            block = OF.render_table([dist_kind, "messages"], dist, numeric_cols=(1,))
    if want_time:
        block, new_facts, rows = await _time_series_block(ctx, choice, title)
        kind = "series"
    if not block and ranking:
        kind = "ranking"
        rows = len(ranking)
        if choice.fmt == "chart":
            spec = OF.chart_spec(
                [name for name, _ in ranking],
                [("messages", [v for _, v in ranking])],
                kind="bar",
                title=title,
            )
            block = OF.render_chart(spec) if spec else ""
        elif choice.fmt == "list":
            block = OF.render_list(ranking)
        else:
            block = OF.render_table(
                ["#", "chat", "messages"],
                [(i, name, v) for i, (name, v) in enumerate(ranking, 1)],
                numeric_cols=(0, 2),
            )
    if not block:
        nums = OF.numeric_rows(facts)
        if nums:
            kind = "facts"
            rows = len(nums)
            if choice.fmt == "list":
                block = OF.render_list([(label or name, v) for name, label, v in nums])
            else:
                block = OF.render_table(
                    ["fact", "value", "note"],
                    [(name, v, label) for name, label, v in nums],
                    numeric_cols=(1,),
                )
    if not block:
        return _result(
            notes=["format_output: no numbers to format (run stats/compute first)"],
            summary={"format": choice.fmt, "rows": 0},
        )
    ctx.claims.append(OF.wrap_block(block))
    return _result(
        facts=new_facts,
        notes=[OF.FORMAT_NOTE],
        summary={"format": choice.fmt, "kind": kind, "rows": rows},
    )


async def _time_series_block(
    ctx: RunState, choice: OF.FormatChoice, title: str
) -> tuple[str, list[Fact], int]:
    """Oynadagi xabarlar sanog'i kun/hafta/oy bo'yicha — `count(*)` (kodda)."""
    wm = ctx.window_meta or {}
    since = wm.get("since") or ctx.plan.since
    until = wm.get("until") or ctx.plan.until
    tz = str(wm.get("tz") or ctx.plan.window_tz or ctx.tz)
    note_default = False
    if not isinstance(since, datetime):
        since, until = OF.default_window()
        note_default = True
    if not isinstance(until, datetime):
        until = datetime.now(UTC)
    chat_ids = list(ctx.plan.chats or ctx.selected or sorted(ctx.chats_by_id))
    if not chat_ids:
        return "", [], 0
    direction = str(wm.get("direction") or ctx.plan.direction or "")
    async with ctx.session_factory() as session:
        grain, counts = await OF.count_series(
            session, chat_ids=chat_ids, since=since, until=until, tz=tz, direction=direction
        )
    lo, hi = OF.local_bounds(since, until, tz)
    labels, values = OF.fill_series(counts, lo, hi, grain)
    if not labels:
        return "", [], 0
    total = sum(values)
    name = {"day": "messages per day", "week": "messages per week", "month": "messages per month"}
    if choice.fmt == "chart":
        spec = OF.chart_spec(
            labels, [(name[grain], values)], kind=choice.chart or "line", title=title
        )
        block = OF.render_chart(spec) if spec else ""
    elif choice.fmt == "list":
        block = OF.render_list(list(zip(labels, values, strict=True)))
    else:
        block = OF.render_table(
            ["period", "messages"], list(zip(labels, values, strict=True)), numeric_cols=(1,)
        )
    facts = [
        Fact(
            "series_total",
            total,
            label=f"messages in the charted range ({lo.isoformat()} … {hi.isoformat()}, {tz})",
            source="format_output",
        ),
        Fact("series_peak", max(values), label=f"largest {grain} count", source="format_output"),
    ]
    if note_default:
        ctx.notes.append(
            f"format_output: no period given, charted the last {OF.DEFAULT_SERIES_DAYS} days"
        )
    return block, facts, len(labels)


# ─── 2. create_alert ─────────────────────────────────────────────────────────


async def _create_alert(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    if not AU.ALERT_INTENT_RE.search(ctx.raw_question):
        return _stop(_m(ctx, "no_intent"), proposed=False, reason="no_intent")
    parsed = AU.parse_alert_request(ctx.raw_question)
    kind = str(args.get("kind") or "")
    pattern = str(args.get("pattern") or "").strip()
    if parsed is not None:
        # Qo'shtirnoq (kalit so'z) — foydalanuvchining aniq so'zi, planner ustidan.
        if parsed[0] == "keyword" or not pattern:
            kind, pattern = parsed
    if not pattern:
        return _stop(_m(ctx, "alert_what"), proposed=False, reason="no_pattern")
    cooldown = args.get("cooldown_minutes")
    try:
        async with ctx.session_factory() as session:
            draft = await AU.build_alert(
                session,
                account_id=ctx.account_id,
                kind=kind or "semantic",
                pattern=pattern,
                chat_id=args["chat_id"] if isinstance(args.get("chat_id"), int) else None,
                cooldown_minutes=int(cooldown) if isinstance(cooldown, int) else None,
                locale=_loc(ctx),
            )
    except AU.AutomationError as exc:
        return _stop(_m(ctx, "failed", err=exc.code), proposed=False, reason=exc.code)
    return await _propose(ctx, draft)


# ─── 3. set_briefing ─────────────────────────────────────────────────────────


async def _set_briefing(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    if not AU.BRIEFING_INTENT_RE.search(ctx.raw_question):
        return _stop(_m(ctx, "no_intent"), proposed=False, reason="no_intent")
    enabled, hour = AU.parse_briefing_request(ctx.raw_question)
    if hour is None and isinstance(args.get("hour"), int):
        hour = int(args["hour"])
    chat_ids = [c for c in (args.get("chat_ids") or []) if c in ctx.chats_by_id] or None
    try:
        async with ctx.session_factory() as session:
            draft = await AU.build_briefing(
                session,
                account_id=ctx.account_id,
                enabled=enabled,
                hour=hour,
                chat_ids=chat_ids,
                locale=_loc(ctx),
            )
    except AU.AutomationError as exc:
        return _stop(_m(ctx, "failed", err=exc.code), proposed=False, reason=exc.code)
    return await _propose(ctx, draft)


# ─── 4. remind / cancel_reminder ─────────────────────────────────────────────


async def _remind(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    if not REMIND_INTENT_RE.search(ctx.raw_question):
        return _stop(_m(ctx, "no_intent"), proposed=False, reason="no_intent")
    if CANCEL_RE.search(ctx.raw_question):
        return await _cancel_reminder(ctx, args)
    # Vaqt — **kodda**, foydalanuvchi matnidan (akkaunt tz). Planner `when`
    # faqat zaxira (ibora yoki ISO).
    due = SCH.parse_due(ctx.raw_question, tz=ctx.tz)
    when = str(args.get("when") or "")
    if due is None and when:
        due = SCH.parse_iso_due(when, tz=ctx.tz) or SCH.parse_due(when, tz=ctx.tz)
    if due is None:
        return _stop(_m(ctx, "remind_when"), proposed=False, reason="no_time")
    text = SCH.extract_text(ctx.raw_question)
    if (not text or ":" not in ctx.raw_question) and str(args.get("text") or "").strip():
        text = str(args["text"]).strip()
    if not text:
        return _stop(_m(ctx, "remind_what"), proposed=False, reason="no_text")
    try:
        draft = AU.build_reminder(
            account_id=ctx.account_id, due=due, text=text, tz=ctx.tz, locale=_loc(ctx)
        )
    except AU.AutomationError as exc:
        return _stop(_m(ctx, "failed", err=exc.code), proposed=False, reason=exc.code)
    return await _propose(ctx, draft)


async def _cancel_reminder(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    if not (REMIND_INTENT_RE.search(ctx.raw_question) and CANCEL_RE.search(ctx.raw_question)):
        return _stop(_m(ctx, "no_intent"), proposed=False, reason="no_intent")
    async with ctx.session_factory() as session:
        pending = await SCH.pending_reminders(session, ctx.account_id)
    if not pending:
        return _stop(_m(ctx, "no_reminders"), proposed=False, reason="none_pending")
    wanted = {int(x) for x in (args.get("job_ids") or []) if isinstance(x, int)}
    if wanted:
        chosen = [j for j in pending if int(j.id) in wanted]
    elif len(pending) == 1 or ALL_RE.search(ctx.raw_question):
        chosen = list(pending)
    else:
        lines = [_m(ctx, "which_reminder")]
        for j in pending[:10]:
            payload = j.payload if isinstance(j.payload, dict) else {}
            when = SCH.format_local(j.run_at, ctx.tz) if j.run_at else "?"
            lines.append(f"- #{j.id} {when} — {U.neutralize(str(payload.get('text') or ''))[:80]}")
        return _stop("\n".join(lines), proposed=False, reason="ambiguous", pending=len(pending))
    if not chosen:
        return _stop(_m(ctx, "no_reminders"), proposed=False, reason="not_found")
    jobs = [
        (
            int(j.id),
            SCH.format_local(j.run_at, ctx.tz) if j.run_at else "?",
            str((j.payload or {}).get("text") or "") if isinstance(j.payload, dict) else "",
        )
        for j in chosen
    ]
    draft = AU.build_cancel(account_id=ctx.account_id, jobs=jobs, tz=ctx.tz, locale=_loc(ctx))
    return await _propose(ctx, draft)


# ─── 5. send_result ──────────────────────────────────────────────────────────


#: Joriy turn natijasining hajmini bildiradigan sanoq faktlari (`chat_digest`, `stats`).
EMPTY_COUNT_FACTS = ("chat_total", "total")


def turn_found_nothing(ctx: RunState) -> bool:
    """Joriy turn qidirdi, dalil yo'q va sanoq faktlari hammasi 0 — yuboradigan narsa yo'q.

    Sanoq fakti umuman bo'lmasa (masalan faqat oldingi javobni yuborish) — `False`:
    bu holatda matn manbasi oldingi javob.
    """
    if ctx.ranked:
        return False
    counts = [
        ctx.facts[name].value
        for name in EMPTY_COUNT_FACTS
        if name in ctx.facts and ctx.facts[name].numeric
    ]
    return bool(counts) and all(float(v or 0) == 0 for v in counts)


def clean_for_send(text: str) -> str:
    """Yuboriladigan matn: sitata tokenlari, qamrov/qadamlar izohi va chart fence'larsiz."""
    body = text or ""
    for marker in ("\n---\n**", "\n\n---\n"):
        i = body.find(marker)
        if i > 0:
            body = body[:i]
    body = re.sub(r"```chart\n.*?\n```", "", body, flags=re.S)
    body = re.sub(r"\s*\[?" + U.CITE_TOKEN_RE.pattern + r"\]?", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()[:SEND_TEXT_MAX]


def facts_text(ctx: RunState) -> str:
    """`text_from="facts"` — kodda hisoblangan faktlar + format bloklari (chart'siz)."""
    lines: list[str] = []
    for f in ctx.facts.values():
        if not f.numeric:
            continue
        label = U.neutralize(f.label or f.name)
        unit = f" {f.unit}" if f.unit else ""
        from app.services.compute import fmt_number

        lines.append(f"• {label}: {fmt_number(f.value)}{unit}")
    blocks = [b for b in OF.extract_blocks(ctx.claims) if not b.startswith("```chart")]
    return "\n\n".join(x for x in ("\n".join(lines), *blocks) if x).strip()


async def _prior_answer(ctx: RunState) -> str:
    """Suhbatning oxirgi assistant javobi (to'liq, `load_history` kesmasidan oldin)."""
    async with ctx.session_factory() as session:
        row = (
            await session.execute(
                select(ConversationMessage.content)
                .where(
                    ConversationMessage.conversation_id == ctx.conversation_id,
                    ConversationMessage.role == "assistant",
                )
                .order_by(ConversationMessage.id.desc())
                .limit(1)
            )
        ).first()
    return str(row[0] or "") if row else ""


async def _synth_current(ctx: RunState) -> str:
    """Joriy turn dalillari bo'yicha **bitta** sintez (javob ham, yuboriladigan matn ham)."""
    from app.services import deep_search as DS

    chats = list(ctx.chats_by_id.values())
    evidence = DS.to_evidence(
        ctx.ranked[:SEND_EVIDENCE_LINES],
        ctx.clusters,
        link_prefixes={c.id: (c.link_prefix or "") for c in chats if c.type != ChatType.PRIVATE},
        server_ids=ctx.server_out.server_ids,
        external_links=ctx.server_out.external_links,
    )
    ctx.taint.add(evidence)
    claims = "\n\n".join(ctx.claims)
    answer, _kept, _res = await DS.synthesize(
        ctx.meter,
        ctx.raw_question,
        evidence,
        ctx.clusters,
        locale=ctx.locale,
        deep=False,
        history=ctx.history,
        claims_block=claims,
        mode_note=(
            "send_result: the answer will be proposed as a Telegram message to another chat — "
            "write it as a clean, self-contained message (no citations needed)."
        ),
    )
    return answer


def send_target(ctx: RunState, args: dict[str, Any]) -> tuple[str | None, str]:
    """Yuborish manzili → `(chat id yoki nom, aniqlashtirish matni)`.

    Tartib: (1) holatdagi yechim (`conv_state.resolved_entities`) — "Otabekka
    yubor" suhbatda allaqachon aniqlangan "Otabek" ga boradi, planner boshqa
    "Otabek" chat id sini bergan bo'lsa ham; (2) planner bergan chat id;
    (3) nom bir nechta shaxsiy chatga **teng** mos kelsa — so'raladi (hech
    narsa taklif qilinmaydi); (4) aks holda nom `write_tools` ga o'tadi.
    """
    from app.services import agent as AG
    from app.services import conv_state as CS

    named = str(args.get("chat") or "").strip()[:120]
    cid = args.get("chat_id")
    conv = getattr(ctx, "conv", None)
    ent = CS.resolve_name(conv, named) if named else None
    if ent is None and isinstance(cid, int) and cid in ctx.chats_by_id:
        # Planner id bergan, lekin savoldagi nom holatda **boshqa** "shu nomli"
        # chatga bog'langan (bir nechta "Otabek": planner tasodifan boshqasini
        # tanlagan) — holat yutadi.
        asked = CS.resolve_entity(conv, ctx.raw_question)
        if asked is not None and int(asked[1]["chat_id"]) != cid:
            key_words = set(CS.name_words(asked[0]))
            if key_words and key_words <= set(CS.name_words(ctx.chats_by_id[cid].title)):
                ent = asked[1]
    if ent is not None and int(ent["chat_id"]) in ctx.chats_by_id:
        return str(int(ent["chat_id"])), ""
    if isinstance(cid, int) and cid in ctx.chats_by_id:
        return str(cid), ""
    if not named:
        return None, ""
    hints = AG.chat_hints(named, list(ctx.chats_by_id.values()))
    collision = CS.name_collision(hints, named)
    if collision is not None:
        key, tied = collision
        options = "\n".join(f"- {U.attr(str(h.title))}" for h in tied[:3])
        return None, _m(ctx, "send_which", name=U.attr(key, max_chars=40), options=options)
    return named, ""


async def _send_result(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Oldingi qadam natijasi → **bitta** `send_message` taklifi (tasdiq bilan)."""
    from app.services import tools as T
    from app.services import write_tools as W

    if not SEND_INTENT_RE.search(ctx.raw_question):
        return _stop(_m(ctx, "no_intent"), proposed=False, reason="no_intent")
    target, ask = send_target(ctx, args)
    if ask:
        return _stop(ask, proposed=False, reason="ambiguous_target")
    if target is None:
        return _stop(_m(ctx, "send_where"), proposed=False, reason="no_target")
    source = str(args.get("text_from") or "answer")
    shown = ""
    if source == "facts":
        text = facts_text(ctx)
        shown = text
    elif ctx.ranked or ctx.claims:
        shown = await _synth_current(ctx)
        text = clean_for_send(shown)
    else:
        text = clean_for_send(await _prior_answer(ctx))
    if not text and ctx.facts:
        text = facts_text(ctx)
    if not text:
        return _stop(_m(ctx, "send_nothing"), proposed=False, reason="empty")
    if turn_found_nothing(ctx):
        # Joriy turn qidirdi va hech narsa topmadi — "topilmadi" matnini uchinchi
        # odamga yuborish taklifi foydasiz (jonli: conv 385, taklif #518).
        head = clean_for_send(shown) if shown else ""
        return _result(
            answer=f"{head}{_m(ctx, 'send_empty')}".strip(),
            stop=True,
            summary={"proposed": False, "intent": "send_result", "reason": "empty_result"},
        )
    async with ctx.session_factory() as session:
        run_id = await _ensure_run(ctx, session)
        tctx = T.ToolContext(
            session=session, account_id=ctx.account_id, pinned_chat_id=None, user_id=ctx.user_id
        )
        res = await W.propose_or_execute(
            session, tctx, run_id=run_id, tool="send_message", args={"chat": target, "text": text}
        )
    action_id = int(res.meta.get("action_id") or 0)
    if action_id:
        ctx.action_ids.append(action_id)
    head = clean_for_send(shown) if shown else text
    if not res.ok:
        tail = _m(ctx, "send_failed", err=U.neutralize(res.text)[:200])
    elif res.meta.get("executed"):
        tail = "\n\n---\n" + U.neutralize(res.text)[:300]  # `autonomous` — foydalanuvchi yoqqan
    else:
        chat = ctx.chats_by_id.get(int(target)) if target.isdigit() else None
        tail = _m(
            ctx,
            "send_proposed",
            id=action_id,
            chat=U.attr(chat.title if chat else target),
        )
    return _result(
        answer=f"{head}{tail}",
        stop=True,
        summary={"proposed": bool(res.ok), "intent": "send_result", "source": source},
    )


# ─── registr ─────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def ext_skills() -> tuple[Skill, ...]:
    """Kengaytma ko'nikmalari (`register` va testlar). Kechiktirilgan: `Skill` klassi
    `skills` modulidan chaqiruv paytida olinadi (yuqoridagi izoh)."""
    from app.services.skills import MODE_RANK, Skill

    #: Barcha rejimlarda tasdiq — katalogda "asks for confirmation" bo'lib ko'rinadi.
    #: (Bu `agent_actions` tasdig'i; skan baho kartasi emas — executor uni faqat
    #: `res.needs_confirm` bilan ko'taradi, bu ko'nikmalar esa uni qo'ymaydi.)
    all_modes = frozenset(MODE_RANK)
    ext = (
        Skill(
            "format_output",
            "raqamlarni jadval / diagramma (```chart) / ro'yxat qilib chizish — kodda, faqat "
            "faktlardan. Misol: «har kuni nechta xabar yozilgan, grafik qil» → stats + "
            "format_output",
            "render numbers as a table / chart (```chart) / numbered list — by code, only from "
            "facts. Example: 'messages per day this week as a chart' → stats + format_output",
            {"format": "str", "chart": "str", "series": "str", "title": "str"},
            0.0,
            0.5,
            "fast",
            quarantined=False,
            run=_format_output,
            rail=("compute",),
        ),
        Skill(
            "create_alert",
            "alert qoidasi TAKLIFI (tasdiqdan keyin yaratiladi). Misol: «'ijara' so'zi chiqsa "
            "xabar ber» (kind=keyword, pattern=ijara)",
            "PROPOSE an alert rule (created only after confirmation). Example: 'notify me when "
            "someone writes about crypto' (kind=semantic, pattern=crypto)",
            {"kind": "str", "pattern": "str", "chat_id": "int", "cooldown_minutes": "int"},
            0.0,
            0.5,
            "fast",
            quarantined=False,
            run=_create_alert,
            terminal=True,
            confirm_modes=all_modes,
        ),
        Skill(
            "set_briefing",
            "kunlik briefing sozlamasi TAKLIFI: yoqish/o'chirish, soat, chatlar. Misol: «har "
            "kuni ertalab 8 da briefing yubor»",
            "PROPOSE daily briefing settings: on/off, hour, chats. Example: 'stop sending me "
            "the briefing'",
            {"hour": "int", "chat_ids": "list[int]"},
            0.0,
            0.5,
            "fast",
            quarantined=False,
            run=_set_briefing,
            terminal=True,
            confirm_modes=all_modes,
        ),
        Skill(
            "remind",
            "eslatma TAKLIFI (bot orqali, tasdiqdan keyin); bekor qilish ham shu. Misol: "
            "«ertaga 10 da eslat: hisobotni yubor», «eslatmani bekor qil»",
            "PROPOSE a reminder sent by the bot (after confirmation); also cancels. Example: "
            "'remind me in 2 hours: call Aziz'",
            {"when": "str", "text": "str", "job_ids": "list[int]"},
            0.0,
            0.5,
            "fast",
            quarantined=False,
            run=_remind,
            terminal=True,
            confirm_modes=all_modes,
        ),
        Skill(
            "send_result",
            "oxirgi qadam: topilgan natija/xulosani chatga yuborish TAKLIFI (bitta tasdiq). "
            "Misol: «IT ta'lim haqida topib, xulosani 'Jamoa' guruhiga tashla» → search_local + "
            "send_result(chat='Jamoa')",
            "last step: PROPOSE sending the found result/summary to a chat (one confirmation). "
            "Example: 'send this summary to the Team group' → send_result(chat='Team', "
            "text_from='answer')",
            {"chat_id": "int", "chat": "str", "text_from": "str"},
            0.01,
            8.0,
            "fast",
            quarantined=True,
            run=_send_result,
            confirm_modes=all_modes,
        ),
    )

    #: `remind` bilan bir xil yo'l — faqat bekor qilish (planner nomini aniq tanlasa).
    cancel = Skill(
        "cancel_reminder",
        "kutayotgan eslatmani bekor qilish TAKLIFI. Misol: «eslatmani bekor qil»",
        "PROPOSE cancelling pending reminders. Example: 'cancel all my reminders'",
        {"job_ids": "list[int]"},
        0.0,
        0.3,
        "fast",
        quarantined=False,
        run=_cancel_reminder,
        terminal=True,
        confirm_modes=all_modes,
    )
    return (*ext, cancel)


def register(registry: dict[str, Skill]) -> None:
    """`skills.register_extra` chaqiradi — yadro nomlarini almashtirmaydi."""
    for skill in ext_skills():
        if skill.name in registry:
            continue
        registry[skill.name] = skill


__all__ = [
    "SEND_INTENT_RE",
    "clean_for_send",
    "ext_skills",
    "facts_text",
    "register",
]
