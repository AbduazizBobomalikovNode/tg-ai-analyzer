"""Orkestrator ko'nikmalari katalogi — mavjud implementatsiyalar ustida yupqa o'ramlar.

Muammo (foydalanuvchi tanqidi, 2026-09-25): deep quvuri "bitta shakl" edi —
rejim `exhaustive` bo'lsa, "bu chatda nima bor edi!" ham 54 chat x 521 hafta
korpus skaniga aylanardi, holbuki javob bitta chatning dayjest keshida turardi.
Endi planner (`services/orchestrator.py`) **qaysi ko'nikma, qaysi tartibda**
kerakligini tanlaydi, bu modul esa har ko'nikmaning:

* `name`, bir qatorlik tavsifi (uz + en) va argumentlar sxemasi;
* `est_cost_usd` (ko'nikmaning **o'z** LLM xarajati, sintez alohida) va
  `est_latency_s` — planner katalogda ko'radi va "eng arzon yetarli yo'l"ni
  tanlaydi;
* `min_mode` — rejim qopqog'i (`fast` < `deep` < `exhaustive`), planner
  undan yuqorisini rejalashtira olmaydi, executor esa baribir tekshiradi;
* `needs_confirm(mode)` — baho kartasi (estimate + tasdiq) faqat shu
  ko'nikma so'raganda;
* `quarantined` — begona xabar matnini ko'radimi. Karantin ko'nikmaning
  chiqishi (dalil, dayjest matni) **hech qachon** plan/replan promptiga
  tushmaydi: executor faqat sanoqlarni uzatadi, `_Taint` esa buni tekshiradi;
* `run(ctx, args) -> SkillResult` — **mavjud** funksiyani chaqiradi
  (`deep_search.fan_out`, `window_search.run_window`,
  `scale_search.run_scale`, `deep_search.escalate`, `digests`, …). Hech
  bir qidiruv/skan mantiqi bu yerda qayta yozilmagan.

Ko'nikmalar umumiy holatni (`RunState`) boyitadi: dalil (`S.Hit`), faktlar
(`compute.Fact` — raqamlar **kodda**), da'volar bloki (konvertga) va qamrov
metama'lumoti. Sintez va qamrov — `orchestrator.finish`.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import func, select

from app.db.models import AgentRun, Chat, ChatDigest, Message, RetrievalNode
from app.llm import LLM, ChatResult, Msg, Task
from app.logging import get_logger
from app.services import cluster as C
from app.services import compute as CMP
from app.services import deep_search as DS
from app.services import search as S
from app.services import textnorm as TN
from app.services import timewin as TW
from app.services import untrusted as U
from app.services.compute import Fact

log = get_logger(__name__)

#: Rejimlar tartibi — ko'nikmaning `min_mode` i shu bilan solishtiriladi.
MODE_RANK = {"fast": 0, "deep": 1, "exhaustive": 2}

# ─── chat_digest byudjetlari (kodda) ─────────────────────────────────────────
DIGEST_MAX_CHATS = 3
DIGEST_DAY_ROWS = 7  # keshdan olinadigan eng yangi kun dayjestlari
DIGEST_WEEK_ROWS = 8  # hafta tugunlari (level 3) — uzoq tarix uchun
DIGEST_TEXT_CHARS = 450
DIGEST_COMPUTE_DAYS = 5  # kesh bo'sh bo'lsa ko'pi bilan shuncha kun hisoblanadi
DIGEST_SAMPLE_MSGS = 12  # sitata uchun eng yangi xabarlar
WINDOW_DEFAULT_DAYS = 7  # oynasiz `window` → oxirgi 7 kun (izoh bilan)

#: Foydalanuvchining **o'z** matnida yozish niyati — `write_propose` shu bo'lmasa
#: rejadan tashlanadi (planner begona matnni ko'rmaydi, lekin baribir).
WRITE_INTENT_RE = re.compile(
    r"\b(yoz(ib)?\s*(qo'y|ber|yubor)|yubor|jo'nat|deb\s+yoz|xabar\s+yoz|"
    r"напиши|отправ\w*|перешли|закрепи|send|write\s+to|forward|pin)\b",
    re.I,
)


# ─── natija va holat ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class SkillResult:
    """Bitta qadamning natijasi. Hammasi ixtiyoriy — ko'nikma nima topsa shu."""

    evidence: list[S.Hit] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Qamrov bo'laklari (`window` / `scan` / `digest` …) — `finish` yig'adi.
    coverage_delta: dict[str, Any] = field(default_factory=dict)
    #: Terminal ko'nikma (`answer_direct` / `clarify` / `write_propose`) javobi.
    answer: str = ""
    stop: bool = False
    needs_confirm: bool = False
    #: Replan uchun **faqat sanoqlar** (matn yo'q): `{"hits": 3, "chats": 1}`.
    summary: dict[str, Any] = field(default_factory=dict)
    result: ChatResult | None = None


@dataclass(slots=True)
class RunState:
    """Bitta yugurishning umumiy holati — ko'nikmalar shu yerga yozadi."""

    session_factory: DS.SessionFactory
    account_id: int
    user_id: int
    conversation_id: int
    question: str  # standalone (planner qayta yozgan) savol
    raw_question: str  # foydalanuvchining xom matni
    locale: str
    mode: str
    confirm: bool
    deep: bool
    meter: DS._Meter
    taint: DS._Taint
    events: DS.EventSink | None
    history: list[DS.Turn]
    tz: str
    chats_by_id: dict[int, Chat]
    plan: DS.DeepPlan
    selected: list[int] = field(default_factory=list)
    started: float = 0.0
    soft_wall: float = DS.SOFT_WALL_S
    hard_wall: float = DS.HARD_WALL_S
    now: Callable[[], float] = field(default=lambda: 0.0)
    # ── yig'ilgan natija ──
    ranked: list[S.Hit] = field(default_factory=list)  # sintezga, tartibda
    candidates: list[S.Hit] = field(default_factory=list)  # ko'rilgan hovuz
    clusters: list[C.Cluster] = field(default_factory=list)
    facts: dict[str, Fact] = field(default_factory=dict)
    claims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    server_out: DS.ServerOutcome = field(default_factory=DS.ServerOutcome)
    scan_meta: dict[str, Any] = field(default_factory=dict)
    estimate_meta: dict[str, Any] = field(default_factory=dict)
    window_meta: dict[str, Any] = field(default_factory=dict)
    digest_meta: dict[str, Any] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)  # bajarilgan ko'nikma nomlari
    run_id: int | None = None
    action_ids: list[int] = field(default_factory=list)
    #: `conversation_messages.context.mode` — `deep` (deep/exhaustive kirishi)
    #: yoki `agent` (chat API'ning `fast` yo'li; UI va eval shu qiymatni biladi).
    context_mode: str = "deep"
    #: `agent_tools` ko'nikmasi natijasi (javob matni + tool jurnali) — yolg'iz
    #: qadam bo'lsa `orchestrator` shu javobni ikkinchi sintezsiz beradi.
    agent_answer: str = ""
    agent_meta: dict[str, Any] = field(default_factory=dict)
    #: Suhbatning ishchi holati (`services/conv_state.py`) — faqat id, sarlavha,
    #: raqam va foydalanuvchi matni; xabar matni **yo'q**.
    conv: Any = None
    #: Savoldagi nomga mos chatlar (`agent.chat_hints`, kodda) — bir marta hisoblanadi.
    hints: list[Any] | None = None

    def elapsed(self) -> float:
        return self.now() - self.started

    def fan_budget(self) -> float:
        return max(
            DS.FAN_OUT_MIN_S,
            min(self.soft_wall, self.hard_wall - self.elapsed() - DS.FAN_OUT_RESERVE_S),
        )

    def deadline(self) -> float:
        return self.started + self.hard_wall - DS.FAN_OUT_RESERVE_S

    @property
    def chat_titles(self) -> dict[int, str]:
        return {c.id: c.title for c in self.chats_by_id.values()}

    def target_chats(self) -> list[int] | None:
        return list(self.plan.chats) if self.plan.chats else None

    def add_evidence(self, hits: Sequence[S.Hit], *, cap: int = 400) -> int:
        """Sintez ro'yxatiga **tartibni saqlab** qo'shadi (takrorlarsiz)."""
        seen = {h.message_id for h in self.ranked}
        added = 0
        for h in hits:
            if h.message_id in seen or len(self.ranked) >= cap:
                continue
            seen.add(h.message_id)
            self.ranked.append(h)
            added += 1
        return added

    def add_candidates(self, hits: Sequence[S.Hit]) -> None:
        seen = {h.message_id for h in self.candidates}
        self.candidates.extend(h for h in hits if h.message_id not in seen)


SkillFn = Callable[[RunState, dict[str, Any]], Awaitable[SkillResult]]


@dataclass(frozen=True, slots=True)
class Skill:
    """Katalog yozuvi. `run` — mavjud implementatsiya ustidagi o'ram."""

    name: str
    description_uz: str
    description_en: str
    args: dict[str, str]  # nom → "int" | "str" | "list[int]" | "list[str]" | "list[dict]"
    cost_usd: float  # tipik o'z xarajati (sintezsiz)
    est_latency_s: float
    min_mode: str
    quarantined: bool
    run: SkillFn
    terminal: bool = False
    #: UI reykidagi bosqich kalitlari (`static/pages/ask/deep.js::STAGE_ORDER`).
    rail: tuple[str, ...] = ()
    confirm_modes: frozenset[str] = frozenset()
    #: Rejim bo'yicha narx (`scan_corpus`: deep ≠ exhaustive) — bo'lmasa `cost_usd`.
    cost_by_mode: tuple[tuple[str, float], ...] = ()

    def est_cost_usd(self, args: dict[str, Any] | None = None, mode: str = "deep") -> float:
        by_mode = dict(self.cost_by_mode)
        return float(by_mode.get(mode, self.cost_usd))

    def needs_confirm(self, mode: str) -> bool:
        return mode in self.confirm_modes

    def allowed(self, mode: str) -> bool:
        return MODE_RANK.get(mode, 1) >= MODE_RANK[self.min_mode]


# ─── yordamchilar ────────────────────────────────────────────────────────────


def _int_list(value: Any, *, limit: int = 20) -> list[int]:
    if not isinstance(value, list):
        return []
    out = [int(x) for x in value if isinstance(x, int) and not isinstance(x, bool) and x > 0]
    return out[:limit]


def _str_list(value: Any, *, limit: int = 8, max_chars: int = 120) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for x in value:
        if isinstance(x, str) and x.strip():
            out.append(U.neutralize(x.strip())[:max_chars])
    return out[:limit]


def coerce_args(skill: Skill, raw: Any) -> dict[str, Any]:
    """Planner argumentlarini sxema bo'yicha **tiplaydi**; bilmagani tashlanadi."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for name, kind in skill.args.items():
        if name not in raw:
            continue
        v = raw[name]
        if kind == "int":
            if isinstance(v, int) and not isinstance(v, bool):
                out[name] = v
            elif isinstance(v, str) and v.strip().isdigit():
                out[name] = int(v.strip())
        elif kind == "str":
            if isinstance(v, str) and v.strip():
                out[name] = v.strip()[:400]
        elif kind == "list[int]":
            ids = _int_list(v)
            if ids:
                out[name] = ids
        elif kind == "list[str]":
            items = _str_list(v)
            if items:
                out[name] = items
        elif kind == "list[dict]":
            if isinstance(v, list):
                out[name] = [x for x in v if isinstance(x, dict)][: CMP.MAX_ITEMS]
    return out


def _days_between(since: datetime | None, until: datetime | None) -> float:
    """Oynadagi kalendar kunlari (bugun ham kiradi): `ceil`, 0.01 kun chetlanish bilan."""
    if since is None or until is None or until <= since:
        return 1.0
    raw = (until - since).total_seconds() / 86400.0
    whole = math.floor(raw)
    return float(max(1, whole + (1 if raw - whole > 0.01 else 0)))


def period_facts(
    since: datetime | None, until: datetime | None, *, source: str, tz: str = TW.DEFAULT_TZ
) -> list[Fact]:
    """`days` / `weeks` / `months` + davr satri — `compute` shular ustida ishlaydi."""
    days = _days_between(since, until)
    period = TW.format_window(
        TW.TimeWindow(since=since, until=until or datetime.now(UTC), kind="", tz=tz)
    )
    return [
        Fact("days", int(days), label=f"calendar days in the period {period}", source=source),
        Fact("weeks", round(days / 7.0, 2), label="days / 7", source=source),
        Fact("months", round(days / 30.44, 2), label="days / 30.44", source=source),
    ]


async def _absorb_server(ctx: RunState, out: DS.ServerOutcome) -> list[S.Hit]:
    """Server natijasini umumiy holatga qo'shadi (`_run_pipeline.maybe_escalate` bilan bir xil)."""
    srv = ctx.server_out
    srv.reason = srv.reason or out.reason
    srv.calls += out.calls
    srv.external += out.external
    srv.hits.extend(out.hits)
    srv.server_ids |= out.server_ids
    srv.external_links.update(out.external_links)
    for rung in out.rungs:
        if rung not in srv.rungs:
            srv.rungs.append(rung)
    for bucket in out.skipped:
        if bucket not in srv.skipped:
            srv.skipped.append(bucket)
    for note in out.notes:
        if note not in srv.notes:
            srv.notes.append(note)
    if not out.hits:
        return []
    async with ctx.session_factory() as session:
        hydrated = await DS._hydrate(session, out.hits, account_id=ctx.account_id)
    ctx.taint.add_hits(hydrated)
    return hydrated


def _gap_chats(ctx: RunState) -> list[Chat]:
    ids = ctx.plan.chats or sorted(ctx.chats_by_id)
    return [
        ctx.chats_by_id[c]
        for c in ids
        if c in ctx.chats_by_id and (ctx.chats_by_id[c].synced_min_id or 0) != 1
    ]


def server_calls_left(ctx: RunState) -> int:
    from app.config import get_settings

    s = get_settings()
    cap = int(s.deep_server_max_calls or 0) if s.deep_server_rungs_enabled else 0
    return cap - ctx.server_out.calls


async def run_escalation(ctx: RunState, *, scope: str, reason: str) -> SkillResult:
    """`deep_search.escalate` — rejadagi `server_scope` ni vaqtincha almashtirib."""
    left = server_calls_left(ctx)
    if left <= 0:
        return SkillResult(notes=["server: no calls left"], summary={"server_calls": 0})
    plan = replace(ctx.plan, server_scope=scope)
    out = await DS.escalate(
        ctx.session_factory,
        account_id=ctx.account_id,
        plan=plan,
        gap_chats=_gap_chats(ctx),
        chats_by_id=ctx.chats_by_id,
        events=ctx.events,
        max_calls=left,
    )
    if not ctx.server_out.reason:
        ctx.server_out.reason = reason
    hydrated = await _absorb_server(ctx, out)
    if hydrated:
        # Server kanali alohida RRF ro'yxati (og'irlik 0.9) — eski quvur bilan bir xil.
        merged = S.rrf(ctx.ranked, hydrated, weights=(1.0, DS.SERVER_RRF_WEIGHT))
        ctx.ranked = merged[: DS.MAX_CANDIDATES]
        ctx.add_candidates(hydrated)
    return SkillResult(
        evidence=hydrated,
        summary={
            "hits": len(hydrated),
            "server_calls": out.calls,
            "external": out.external,
            "skipped": len(out.skipped),
        },
    )


# ─── 1. answer_direct / clarify ──────────────────────────────────────────────


async def _answer_direct(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    intent = str(args.get("intent") or "")
    if intent not in DS.DIRECT_INTENTS:
        follow = ctx.plan.intent_follow_up
        intent = follow if follow in DS.DIRECT_FOLLOW_UPS else "chitchat"
    text, res = await DS.answer_direct(
        ctx.meter,
        ctx.raw_question,
        intent=intent,
        locale=ctx.locale,
        history=ctx.history,
        taint=ctx.taint,
    )
    return SkillResult(answer=text, stop=True, result=res, summary={"intent": intent})


_CLARIFY_DEFAULT = {
    "uz": "Qaysi chat, odam yoki davrni nazarda tutyapsiz?",
    "ru": "Какой чат, человека или период вы имеете в виду?",
    "en": "Which chat, person or period do you mean?",
}


async def _clarify(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Bitta aniqlashtiruvchi savol + 2-3 variant. LLM chaqirig'i **yo'q**.

    Matnni planner yozgan (u faqat foydalanuvchi matni va chat sarlavhalarini
    ko'rgan); baribir `neutralize` dan o'tadi — sarlavha teg yasay olmasin.
    """
    question = U.neutralize(str(args.get("question") or "")).strip()[:300]
    if not question:
        question = _CLARIFY_DEFAULT.get(ctx.locale, _CLARIFY_DEFAULT["uz"])
    options = [U.neutralize(o)[:120] for o in _str_list(args.get("options"), limit=3)]
    lines = [question, *(f"- {o}" for o in options)]
    return SkillResult(answer="\n".join(lines), stop=True, summary={"options": len(options)})


# ─── 2. reuse_prior ──────────────────────────────────────────────────────────


async def _reuse_prior(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    # Suhbat holatining `last_result` havolalari birinchi (u agent/stats turn'ida
    # ham to'ladi), bo'lmasa oxirgi assistant qatorining `evidence_refs` i.
    state_refs = (ctx.conv.last_result or {}).get("evidence_refs") if ctx.conv else None
    refs = list(state_refs or []) or DS.prior_refs(ctx.history)
    if not refs:
        return SkillResult(notes=["reuse_prior: no prior evidence"], summary={"hits": 0})
    async with ctx.session_factory() as session:
        hits = await DS._refs_to_hits(session, refs, account_id=ctx.account_id)
    ctx.taint.add_hits(hits)
    ctx.add_candidates(hits)
    ctx.add_evidence(hits[: DS.RERANK_KEEP])
    await DS._emit(ctx.events, "search", subquery="", hits=len(hits), reused=True)
    return SkillResult(evidence=hits, summary={"hits": len(hits)})


# ─── 3. chat_digest ──────────────────────────────────────────────────────────


class _MeterLLM:
    """`digests.digest_for_day` ning `llm.chat(..., tier=)` chaqiruvi uchun o'ram.

    Kun dayjesti hisoblansa uning narxi ham **shu yugurish** hisobiga (va
    kunlik byudjetga) tushsin: `_Meter` `tier` ni bilmaydi, shuning uchun
    u bu yerda tashlanadi (interaktiv yo'l — standart tier).
    """

    def __init__(self, meter: DS._Meter) -> None:
        self.meter = meter

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        kw.pop("tier", None)
        return await self.meter.chat(task, messages, **kw)

    async def embed(self, texts: list[str], **kw: Any) -> Any:
        """`agent_tools` ichidagi `search_messages` so'rov embedding'i — hisob shu yerda."""
        return await self.meter.embed(texts, **kw)


def _digest_target(ctx: RunState, args: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    if isinstance(args.get("chat_id"), int):
        ids.append(int(args["chat_id"]))
    ids.extend(_int_list(args.get("chat_ids")))
    if not ids:
        ids = list(ctx.plan.chats or ctx.selected or [])
    out: list[int] = []
    for c in ids:
        if c in ctx.chats_by_id and c not in out:
            out.append(c)
    return out[:DIGEST_MAX_CHATS]


async def _chat_digest(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Bitta (≤3) chatning eng arzon umumiy ko'rinishi.

    Tartib — narx bo'yicha: (1) `count(*)` (chatdagi xabarlar, meniki, davr),
    (2) keshdagi hafta tugunlari (level 3) va kun dayjestlari — LLM'siz,
    (3) kesh butunlay bo'sh bo'lsa oxirgi faol ≤`DIGEST_COMPUTE_DAYS` kun
    arzon `Task.ROUTE` bilan hisoblanadi (natija keshga yoziladi),
    (4) sitata uchun eng yangi `DIGEST_SAMPLE_MSGS` xabar.

    Dayjest matni begona xabarlardan yasalgan: `neutralize` dan o'tadi,
    da'volar blokiga (yagona konvert ichiga) tushadi va `_Taint` ro'yxatiga
    qo'shiladi — replan promptiga tushsa qo'riqchi qizaradi.
    """
    targets = _digest_target(ctx, args)
    if not targets:
        return SkillResult(notes=["chat_digest: no target chat"], summary={"chats": 0})
    since = ctx.plan.since if ctx.plan.window_from_question else None
    until = ctx.plan.until if ctx.plan.window_from_question else None
    blocks: list[str] = []
    facts: list[Fact] = []
    sample: list[S.Hit] = []
    covered = 0
    day_rows_used = 0
    week_rows_used = 0
    computed_days = 0
    first_all: datetime | None = None
    covered_from: datetime | None = None
    covered_to: datetime | None = None
    last_all: datetime | None = None
    total_all = 0
    mine_all = 0
    for chat_id in targets:
        chat = ctx.chats_by_id[chat_id]
        async with ctx.session_factory() as session:
            conds: list[Any] = [Message.chat_id == chat_id, *S._alive()]
            if since is not None:
                conds.append(Message.published_at >= since)
            if until is not None:
                conds.append(Message.published_at < until)
            row = (
                await session.execute(
                    select(
                        func.count(),
                        func.count().filter(Message.out.is_(True)),
                        func.min(Message.published_at),
                        func.max(Message.published_at),
                    ).where(*conds)
                )
            ).one()
            total, mine = int(row[0] or 0), int(row[1] or 0)
            first_at, last_at = row[2], row[3]
            node_q = select(RetrievalNode).where(
                RetrievalNode.chat_id == chat_id, RetrievalNode.level == 3
            )
            day_q = select(ChatDigest).where(ChatDigest.chat_id == chat_id)
            if since is not None:
                node_q = node_q.where(RetrievalNode.ends_at >= since)
                day_q = day_q.where(ChatDigest.day >= since - timedelta(days=1))
            if until is not None:
                node_q = node_q.where(RetrievalNode.starts_at < until)
                day_q = day_q.where(ChatDigest.day < until)
            weeks = list(
                (
                    await session.execute(
                        node_q.order_by(RetrievalNode.starts_at.desc()).limit(DIGEST_WEEK_ROWS)
                    )
                )
                .scalars()
                .all()
            )
            days = list(
                (
                    await session.execute(
                        day_q.order_by(ChatDigest.day.desc()).limit(DIGEST_DAY_ROWS)
                    )
                )
                .scalars()
                .all()
            )
            day_items: list[tuple[datetime, int, str]] = [
                (d.day, int(d.msg_count or 0), d.digest or "") for d in days if d.digest
            ]
            if not day_items and not weeks and total and last_at is not None:
                # Kesh bo'sh — oxirgi faol kunlar (ko'pi bilan DIGEST_COMPUTE_DAYS).
                from app.services import digests as DG

                day_rows = (
                    await session.execute(
                        select(func.date_trunc("day", Message.published_at).label("d"))
                        .where(*conds)
                        .group_by("d")
                        .order_by(func.date_trunc("day", Message.published_at).desc())
                        .limit(DIGEST_COMPUTE_DAYS)
                    )
                ).all()
                for (day,) in reversed(day_rows):
                    with contextlib.suppress(Exception):
                        dd = await DG.digest_for_day(
                            session, chat, day, llm=cast(LLM, _MeterLLM(ctx.meter))
                        )
                        if dd is not None and dd.text:
                            day_items.append((dd.day, dd.msg_count, dd.text))
                            computed_days += 1
            recent_q = (
                select(Message)
                .where(*conds)
                .order_by(Message.published_at.desc())
                .limit(DIGEST_SAMPLE_MSGS)
            )
            recent = list((await session.execute(recent_q)).scalars().all())
        hits = [
            S.Hit(
                message_id=m.id,
                tg_msg_id=m.tg_msg_id,
                score=1.0 / (i + 1),
                source="digest_sample",
                chat_id=chat_id,
            )
            for i, m in enumerate(recent)
        ]
        if hits:
            async with ctx.session_factory() as session:
                hits = await DS._hydrate(session, hits, account_id=ctx.account_id)
            ctx.taint.add_hits(hits)
            sample.extend(hits)
        texts = [t for _, _, t in day_items] + [w.text or "" for w in weeks]
        ctx.taint.add_texts(texts)
        day_rows_used += len(day_items)
        week_rows_used += len(weeks)
        covered += sum(n for _, n, _ in day_items) + sum(int(w.msg_count or 0) for w in weeks)
        total_all += total
        mine_all += mine
        if first_at and (first_all is None or first_at < first_all):
            first_all = first_at
        if last_at and (last_all is None or last_at > last_all):
            last_all = last_at
        lines = [
            f'### chat digest — "{U.attr(chat.title)}" (id {chat_id})',
            f"messages: {total} (mine: {mine}) · first: "
            f"{first_at.date().isoformat() if first_at else '?'} · last: "
            f"{last_at.date().isoformat() if last_at else '?'}",
        ]
        if weeks:
            lines.append("#### weekly digests (newest first)")
            for w in weeks:
                body = U.neutralize(S.compact_text(w.text or "", max_chars=DIGEST_TEXT_CHARS))
                lines.append(
                    f"- week of {w.starts_at.date().isoformat()} ({w.msg_count} msgs): {body}"
                )
        if day_items:
            lines.append("#### daily digests (newest first)")
            for day, n, text in sorted(day_items, key=lambda x: x[0], reverse=True):
                body = U.neutralize(S.compact_text(text, max_chars=DIGEST_TEXT_CHARS))
                lines.append(f"- {day.date().isoformat()} ({n} msgs): {body}")
        spans = [d for d, _, _ in day_items] + [w.starts_at for w in weeks]
        if spans:
            lo, hi = min(spans), max(spans)
            covered_from = lo if covered_from is None or lo < covered_from else covered_from
            covered_to = hi if covered_to is None or hi > covered_to else covered_to
            lines.insert(
                2,
                f"digests cover: {lo.date().isoformat()} → {hi.date().isoformat()} "
                "(the message count above is the whole history)",
            )
        if not weeks and not day_items:
            lines.append("(no digests cached for this chat — only the newest messages below)")
        blocks.append("\n".join(lines))
    ctx.claims.append("\n\n".join(blocks))
    facts.append(
        Fact("chat_total", total_all, label="messages in the chat(s)", source="chat_digest")
    )
    facts.append(
        Fact("chat_mine", mine_all, label="of them sent by the owner", source="chat_digest")
    )
    span = period_facts(since or first_all, until or last_all, source="chat_digest", tz=ctx.tz)
    facts.append(
        Fact(
            "chat_days",
            span[0].value,
            label="days from the first to the last message",
            source="chat_digest",
        )
    )
    facts.extend(relative_date_facts(first_all, last_all, tz=ctx.tz, source="chat_digest"))
    if covered_from is not None and covered_to is not None:
        facts.append(
            Fact(
                "digest_period",
                f"{covered_from.date().isoformat()} → {covered_to.date().isoformat()}",
                label="period the cached digests cover",
                source="chat_digest",
            )
        )
    sample.sort(key=lambda h: h.date or datetime.min.replace(tzinfo=UTC))
    ctx.add_candidates(sample)
    ctx.add_evidence(sample)
    ctx.digest_meta = {
        "chats": targets,
        "day_digests": day_rows_used,
        "week_digests": week_rows_used,
        "computed_days": computed_days,
        "covered_messages": covered,
        "sample": len(sample),
        "total": total_all,
        "first": first_all.isoformat() if first_all else None,
        "last": last_all.isoformat() if last_all else None,
    }
    await DS._emit(
        ctx.events,
        "digest",
        chats=len(targets),
        day_digests=day_rows_used,
        week_digests=week_rows_used,
        computed_days=computed_days,
        messages=total_all,
    )
    return SkillResult(
        evidence=sample,
        facts=facts,
        coverage_delta={"digest": dict(ctx.digest_meta)},
        summary={
            "chats": len(targets),
            "digests": day_rows_used + week_rows_used,
            "hits": len(sample),
            "messages": total_all,
        },
    )


# ─── 4. stats / window (oyna yo'li) ──────────────────────────────────────────


def _window_meta(plan: DS.DeepPlan, outcome: Any, *, stats: bool) -> dict[str, Any]:
    """`_run_pipeline` dagi oyna meta'si bilan **bir xil** shakl (qamrov shundan)."""
    spec = outcome.window
    wcount = outcome.window_count
    return {
        "retrieval_kind": "window",
        "inbox": bool(spec.inbox) if spec else plan.inbox,
        "direction": (spec.direction if spec else plan.direction) or "",
        "stats": stats,
        "since": spec.since if spec else plan.since,
        "until": spec.until if spec else plan.until,
        "tz": (spec.tz if spec else plan.window_tz) or TW.DEFAULT_TZ,
        "total": wcount.total if wcount else 0,
        "taken": (
            (wcount.total if wcount else 0)
            if stats
            else len(outcome.evidence) + outcome.stats.dropped_duplicates
        ),
        "dropped_duplicates": outcome.stats.dropped_duplicates,
        "sampled": outcome.window_sampled,
        "chats": wcount.chats if wcount else 0,
    }


def _scan_meta(ctx: RunState, outcome: Any) -> dict[str, Any]:
    from app.services import scale_search as SC

    return {
        **outcome.stats.to_dict(),
        "mode": ctx.mode,
        "q_hash": SC.question_hash(ctx.question),
        "est_cost_usd": outcome.stats.est_cost_usd,
    }


async def _stats(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Sanoq: `window_search.run_window` ning `window_stats` yo'li — skan tokeni 0.

    Faktlar: `total` (oyna + yo'nalishdagi hamma xabar), `mine`/`theirs`,
    `chats`, `days`/`weeks`/`months`, `top1_n…top5_n`, nisbiy sanalar
    (`first_at`/`last_at`, `days_since_first`/`days_since_last`). Hammasi
    `count(*)` / `min`/`max`.

    Argumentlar (planner yoki kod qo'riqchisi beradi):

    * `period` — `timewin.PERIOD_KINDS` (`this_month`, `last_month`, …):
      oyna savoldan emas, shu nomdan (`timewin.period_window`);
    * `slot` — `a` / `b`: faktlar `a_n`, `a_mine`, `a_days`, `a_period` …
      nomi bilan (ikki davr taqqoslash — `compute(op="compare_periods")`);
    * `by` — `hour` / `weekday` / `both`: taqsimot (`window_search.distribution`,
      SQL), faktlar `top_weekday`, `top_weekday_n`, `wd_<kun>_n`,
      `top_hour`, `top_hour_n`.
    """
    from app.services import window_search as WS

    plan = replace(ctx.plan, retrieval_kind="window", window_stats=True)
    period = str(args.get("period") or "")
    slot = str(args.get("slot") or "")
    slot = slot if slot in STATS_SLOTS else ""
    notes: list[str] = []
    if period and not slot and ctx.plan.window_from_question:
        # Savolning o'z oynasi (kodda, `apply_question_window`) planner nomlagan
        # davrdan ustun: "oxirgi 7 kunda" da planner `last_week` berib, oyna
        # o'tgan kalendar haftasiga siljirdi (jonli uni-12). Ikki davr
        # taqqoslashda (`slot`) esa davr nomi — yagona manba.
        notes.append(f"stats: period {period[:24]!r} ignored — the question names its own window")
        period = ""
    if period:
        win = TW.period_window(period, tz=plan.window_tz or ctx.tz)
        if win is None:
            notes.append(f"stats: unknown period {period[:24]!r} — the question window is used")
        else:
            plan.since, plan.until = win.since, win.until
            plan.window_from_question = True
            plan.window_kind = win.kind
            plan.window_tz = win.tz
    by = str(args.get("by") or "")
    dists = [d for d in WS.DISTRIBUTIONS if by in (d, "both")]
    # Yo'nalish — **kod** qarori (`detect_direction`: "kim bilan gaplashdim" =
    # `both`). Planner argumenti faqat kod hech narsa topmaganda ishlatiladi:
    # jonli eval'da planner "gaplashganman" ga `in` berib reytingni buzgan.
    if not plan.direction:
        plan.direction = DS.detect_direction(TW.fold(f"{ctx.raw_question}\n{ctx.question}"))
    direction = str(args.get("direction") or "")
    if not plan.direction and direction in DS.DIRECTIONS:
        plan.direction = direction
    plan.inbox = plan.direction == "in"
    outcome = await WS.run_window(
        ctx.session_factory,
        account_id=ctx.account_id,
        plan=plan,
        question=ctx.question,
        mode=ctx.mode,
        meter=ctx.meter,
        taint=ctx.taint,
        events=ctx.events,
        chat_titles=ctx.chat_titles,
        all_chats=sorted(ctx.chats_by_id),
        deadline=ctx.deadline(),
        confirm=True,  # sanoq yo'lida skan tokeni yo'q — tasdiq ma'nosiz
        locale=ctx.locale,
        distributions=dists,
    )
    ctx.plan.direction = plan.direction
    ctx.plan.inbox = plan.inbox
    if period and not slot:
        # Yolg'iz davr — asosiy oyna shu (qamrov, suhbat holatining `last_window`).
        ctx.plan.since, ctx.plan.until = plan.since, plan.until
        ctx.plan.window_from_question = plan.window_from_question
        ctx.plan.window_kind, ctx.plan.window_tz = plan.window_kind, plan.window_tz
    ctx.window_meta = _window_meta(plan, outcome, stats=True)
    ctx.scan_meta = ctx.scan_meta or _scan_meta(ctx, outcome)
    ctx.estimate_meta = ctx.estimate_meta or (
        outcome.estimate.to_dict() if outcome.estimate else {}
    )
    if outcome.window_stats:
        ctx.claims.append(outcome.window_stats)
    ctx.add_candidates(outcome.pool)
    ctx.add_evidence(outcome.evidence[:30])
    wcount = outcome.window_count
    spec = outcome.window
    total = int(wcount.total if wcount else 0)
    agg = outcome.window_agg
    mine = total if plan.direction == "out" else int(getattr(agg, "mine_total", 0) or 0)
    facts = [
        Fact(
            "total",
            total,
            label=f"messages in the window (direction: {plan.direction or 'all'})",
            source="stats",
        ),
        Fact("mine", mine, label="of them sent by the owner", source="stats"),
        Fact("theirs", max(0, total - mine), label="of them sent by others", source="stats"),
        Fact(
            "chats",
            int(wcount.chats if wcount else 0),
            label="chats with messages in the window",
            source="stats",
        ),
    ]
    since = spec.since if spec is not None else plan.since
    if since is None and wcount is not None:
        since = wcount.first_at
    until = spec.until if spec is not None else (plan.until or datetime.now(UTC))
    tz = plan.window_tz or ctx.tz
    facts.extend(period_facts(since, until, source="stats", tz=tz))
    days_fact = facts[4].value
    if slot:
        return _slot_result(
            ctx, outcome, facts, slot=slot, since=since, until=until, tz=tz, notes=notes
        )
    per_chat = list(getattr(agg, "per_chat", []) or [])
    titles = ctx.chat_titles
    for i, row in enumerate(per_chat[:5], 1):
        facts.append(
            Fact(
                f"top{i}_n",
                int(row.n),
                label=f'messages in #{i} "{U.attr(titles.get(row.chat_id, str(row.chat_id)))}"',
                source="stats",
            )
        )
    if wcount is not None:
        facts.extend(relative_date_facts(wcount.first_at, wcount.last_at, tz=tz, source="stats"))
    facts.extend(distribution_facts(agg))
    return SkillResult(
        evidence=list(outcome.evidence),
        facts=facts,
        notes=notes,
        coverage_delta={"window": dict(ctx.window_meta)},
        summary={"total": total, "chats": facts[3].value, "days": days_fact},
    )


#: `stats(slot=…)` — ikki davr taqqoslashda faktlar prefiksi.
STATS_SLOTS = ("a", "b")
_SLOT_RENAME = {"total": "n", "mine": "mine", "theirs": "theirs", "chats": "chats", "days": "days"}
WEEKDAY_KEYS = ("", "mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _slot_result(
    ctx: RunState,
    outcome: Any,
    facts: list[Fact],
    *,
    slot: str,
    since: datetime | None,
    until: datetime,
    tz: str,
    notes: list[str],
) -> SkillResult:
    """`stats(slot=a|b)` — faktlar `a_n`, `a_mine`, `a_days`, `a_period` nomi bilan.

    Prefikssiz `total`/`days` yozilmaydi: ikkinchi davr birinchisini jimgina
    bosib ketmasin, `compute(op="compare_periods")` esa aynan shu juftni oladi.
    """
    renamed: list[Fact] = []
    for f in facts:
        suffix = _SLOT_RENAME.get(f.name)
        if suffix is None:
            continue
        renamed.append(replace(f, name=f"{slot}_{suffix}"))
    period = TW.format_window(
        TW.TimeWindow(since=since, until=until, kind="", tz=tz or TW.DEFAULT_TZ)
    )
    renamed.append(Fact(f"{slot}_period", period, label=f"period {slot}", source="stats"))
    total = next((f.value for f in renamed if f.name == f"{slot}_n"), 0)
    return SkillResult(
        evidence=list(outcome.evidence),
        facts=renamed,
        notes=notes,
        coverage_delta={"window": dict(ctx.window_meta)},
        summary={"slot": slot, "total": total},
    )


def relative_date_facts(
    first_at: datetime | None,
    last_at: datetime | None,
    *,
    tz: str,
    source: str,
    now: datetime | None = None,
) -> list[Fact]:
    """ "Necha kun oldin / necha kundan beri" — xabar sanalaridan, kodda.

    Kun farqi **mahalliy kalendar** bo'yicha (`tz`): kecha kechqurun yozilgan
    xabar bugun ertalab "1 kun oldin" bo'ladi, "0" emas.
    """
    zi = TW.zone(tz)
    today = (now or datetime.now(UTC)).astimezone(zi).date()
    out: list[Fact] = []
    for name, when in (("first", first_at), ("last", last_at)):
        if when is None:
            continue
        local = when.astimezone(zi)
        out.append(
            Fact(
                f"{name}_at",
                local.strftime("%Y-%m-%d %H:%M"),
                label=f"{name} message in the window ({zi})",
                source=source,
            )
        )
        out.append(
            Fact(
                f"days_since_{name}",
                max(0, (today - local.date()).days),
                label=f"calendar days from the {name} message to today",
                source=source,
            )
        )
    return out


def distribution_facts(agg: Any) -> list[Fact]:
    """Soat / hafta kuni taqsimotidan faktlar — eng faol kun/soat va har kun soni."""
    out: list[Fact] = []
    by_weekday = list(getattr(agg, "by_weekday", []) or [])
    if by_weekday:
        from app.services.window_search import WEEKDAY_NAMES

        for day, n in by_weekday:
            if 1 <= day <= 7:
                out.append(
                    Fact(
                        f"wd_{WEEKDAY_KEYS[day]}_n",
                        int(n),
                        label=f"messages on {WEEKDAY_NAMES[day]}s",
                        source="stats",
                    )
                )
        top_day, top_n = max(by_weekday, key=lambda p: (p[1], -p[0]))
        if 1 <= top_day <= 7:
            out.append(
                Fact("top_weekday", WEEKDAY_NAMES[top_day], label="busiest weekday", source="stats")
            )
            out.append(
                Fact(
                    "top_weekday_n",
                    int(top_n),
                    label="messages on the busiest weekday",
                    source="stats",
                )
            )
    by_hour = list(getattr(agg, "by_hour", []) or [])
    if by_hour:
        # Har soat soni — `format_output` soat jadvali/diagrammasi shundan chiziladi
        # (`output_formats.distribution_rows`); faqat nol bo'lmagan soatlar.
        for hour, n in sorted(by_hour):
            if 0 <= int(hour) <= 23 and int(n):
                out.append(
                    Fact(
                        f"hr_{int(hour):02d}_n",
                        int(n),
                        label=f"messages at {int(hour):02d}:00-{int(hour):02d}:59 (local)",
                        source="stats",
                    )
                )
        top_hour, top_n = max(by_hour, key=lambda p: (p[1], -p[0]))
        out.append(
            Fact(
                "top_hour",
                int(top_hour),
                label="busiest hour of day (local, HH:00)",
                source="stats",
            )
        )
        out.append(
            Fact("top_hour_n", int(top_n), label="messages in the busiest hour", source="stats")
        )
    return out


async def _window(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Mavzusiz vaqt oynasi dayjesti — `window_search.run_window` (namuna + map)."""
    from app.services import window_search as WS

    plan = replace(ctx.plan, retrieval_kind="window", window_stats=False)
    notes: list[str] = []
    if plan.since is None and not plan.window_from_question:
        plan.until = plan.until or datetime.now(UTC)
        plan.since = plan.until - timedelta(days=WINDOW_DEFAULT_DAYS)
        notes.append(f"window: no period in the question, used the last {WINDOW_DEFAULT_DAYS} days")
    outcome = await WS.run_window(
        ctx.session_factory,
        account_id=ctx.account_id,
        plan=plan,
        question=ctx.question,
        mode=ctx.mode,
        meter=ctx.meter,
        taint=ctx.taint,
        events=ctx.events,
        chat_titles=ctx.chat_titles,
        all_chats=sorted(ctx.chats_by_id),
        deadline=ctx.deadline(),
        confirm=ctx.confirm,
        locale=ctx.locale,
    )
    ctx.scan_meta = _scan_meta(ctx, outcome)
    ctx.estimate_meta = outcome.estimate.to_dict() if outcome.estimate else {}
    ctx.window_meta = _window_meta(plan, outcome, stats=False)
    if outcome.stats.stopped_by == "estimate_only":
        return SkillResult(needs_confirm=True, stop=True, notes=notes)
    from app.services import scale_search as SC

    digest_lines = (
        DS.synth_budget(ctx.raw_question, follow_up=ctx.plan.intent_follow_up, window=True).kind
        == "digest"
    )
    cap = DS.DIGEST_SYNTH_LINES if digest_lines else SC.SYNTH_EVIDENCE_LINES
    ctx.add_candidates(outcome.pool)
    ctx.add_evidence(outcome.evidence[:cap])
    blocks = (outcome.window_stats, SC.render_claims_block(outcome.claims, [], []))
    block = "\n\n".join(x for x in blocks if x)
    if block:
        ctx.claims.append(block)
    wcount = outcome.window_count
    return SkillResult(
        evidence=list(outcome.evidence),
        facts=[
            Fact(
                "window_total",
                int(wcount.total if wcount else 0),
                label="messages in the window",
                source="window",
            )
        ],
        notes=notes,
        coverage_delta={"window": dict(ctx.window_meta)},
        summary={"hits": len(outcome.evidence), "total": int(wcount.total if wcount else 0)},
    )


# ─── 5. search_local / search_server / discover ──────────────────────────────


async def _search_local(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Lokal gibrid qidiruv — eski `fast` fan-out tarmog'i (reflect'siz)."""
    plan = ctx.plan
    extra = _str_list(args.get("queries"), limit=4)
    if extra:
        merged = list(plan.subqueries)
        for q in extra:
            if q.lower() not in {x.lower() for x in merged}:
                merged.append(q)
        plan.subqueries = merged[: DS.MAX_SUBQUERIES_WIDENED]
    if plan.use_hyde and not plan.hyde_text:
        plan.hyde_text = await DS.hyde_text(ctx.meter, ctx.taint, ctx.question)
    subqueries = list(plan.subqueries)
    if plan.hyde_text:
        subqueries.append(plan.hyde_text)
    candidates = await DS.fan_out(
        ctx.session_factory,
        account_id=ctx.account_id,
        plan=plan,
        subqueries=subqueries,
        meter=ctx.meter,
        events=ctx.events,
        budget_s=ctx.fan_budget(),
    )
    widened = False
    if not candidates:
        wide = DS.widen_plan(plan)
        retry = [ctx.question, *wide.subqueries][: DS.MAX_SUBQUERIES]
        await DS._emit(ctx.events, "search", subquery="", hits=0, retry=True)
        candidates = await DS.fan_out(
            ctx.session_factory,
            account_id=ctx.account_id,
            plan=wide,
            subqueries=retry,
            meter=ctx.meter,
            events=ctx.events,
            budget_s=ctx.fan_budget(),
        )
        widened = True
    ctx.taint.add_hits(candidates)
    ctx.add_candidates(candidates)
    deduped, clusters = await DS.dedup_and_cluster(
        ctx.session_factory, candidates, meter=ctx.meter, locale=ctx.locale, events=ctx.events
    )
    ranked = await DS.rank(
        ctx.question,
        deduped,
        meter=ctx.meter,
        events=ctx.events,
        dropped=len(candidates) - len(deduped),
    )
    if clusters:
        ctx.clusters = list(clusters)
    ctx.add_evidence(ranked)
    chats = len({h.chat_id for h in deduped if h.chat_id})
    return SkillResult(
        evidence=ranked,
        notes=["search_local: widened after zero hits"] if widened else [],
        summary={
            "candidates": len(candidates),
            "hits": len(ranked),
            "clusters": len(clusters),
            "chats": chats,
        },
    )


async def _search_server(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    scope = str(args.get("scope") or "")
    if scope not in ("gaps", "global"):
        scope = ctx.plan.server_scope if ctx.plan.server_scope in ("gaps", "global") else "gaps"
    return await run_escalation(ctx, scope=scope, reason="plan")


async def _discover(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    ctx.plan.server_scope = "discover"
    return await run_escalation(ctx, scope="discover", reason="plan")


# ─── 6. scan_corpus ──────────────────────────────────────────────────────────


async def _scan_corpus(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Korpus kaskadi (`scale_search.run_scale`) — baho + tasdiq qoidalari o'sha."""
    from app.services import scale_search as SC

    outcome = await SC.run_scale(
        ctx.session_factory,
        account_id=ctx.account_id,
        plan=ctx.plan,
        question=ctx.question,
        mode=ctx.mode,
        meter=ctx.meter,
        taint=ctx.taint,
        events=ctx.events,
        chat_titles=ctx.chat_titles,
        deadline=ctx.deadline(),
        confirm=ctx.confirm,
        locale=ctx.locale,
    )
    ctx.estimate_meta = outcome.estimate.to_dict() if outcome.estimate else {}
    ctx.scan_meta = _scan_meta(ctx, outcome)
    if ctx.mode == "exhaustive" and not ctx.confirm:
        return SkillResult(needs_confirm=True, stop=True)
    ctx.add_candidates(outcome.pool)
    if outcome.clusters:
        ctx.clusters = list(outcome.clusters)
    ctx.add_evidence(outcome.evidence[: SC.SYNTH_EVIDENCE_LINES])
    block = SC.render_claims_block(outcome.claims, outcome.contradictions, outcome.timeline)
    if block:
        ctx.claims.append(block)
    return SkillResult(
        evidence=list(outcome.evidence),
        coverage_delta={"scan": dict(ctx.scan_meta)},
        summary={
            "hits": len(outcome.evidence),
            "claims": len(outcome.claims),
            "buckets": int(ctx.scan_meta.get("buckets_scanned") or 0),
        },
    )


# ─── 7. compute ──────────────────────────────────────────────────────────────


async def _compute(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    items = CMP.parse_items(args)
    notes: list[str] = []
    made: list[Fact] = []
    op = str(args.get("op") or "")
    if not op and not items and {"a_n", "b_n"} <= set(ctx.facts):
        op = "compare_periods"  # ikki davr sanog'i bor — taqqoslash o'zi so'ralgan
    if op:
        made, errors = CMP.run_op(op, ctx.facts)
        notes.extend(f"compute: {e}" for e in errors)
    if items:
        pool = {**ctx.facts, **{f.name: f for f in made}}
        extra, errors = CMP.run_items(items, pool)
        made.extend(extra)
        notes.extend(f"compute: {e}" for e in errors)
    if not made:
        fallback = CMP.heuristic_items(ctx.raw_question, ctx.facts)
        if fallback:
            made, errors = CMP.run_items(fallback, ctx.facts)
            notes.extend(f"compute: {e}" for e in errors)
    if not made:
        notes.append("compute: nothing to compute from the facts at hand")
    await DS._emit(ctx.events, "compute", facts=[f.to_dict() for f in made])
    return SkillResult(facts=made, notes=notes, summary={"computed": len(made)})


# ─── 8. write_propose ────────────────────────────────────────────────────────

_WRITE_DONE = {
    "uz": "Taklif tayyor — xabar **yuborilmadi**. Amal kartasida ko'rib chiqib tasdiqlang: ",
    "ru": (
        "Предложение готово — сообщение **не отправлено**. "
        "Проверьте и подтвердите в карточке действия: "
    ),
    "en": "Proposal ready — nothing was **sent**. Review and confirm it on the action card: ",
}
_WRITE_FAIL = {
    "uz": "Yozish amalini tayyorlab bo'lmadi: ",
    "ru": "Не удалось подготовить действие: ",
    "en": "Could not prepare the write action: ",
}


async def _write_propose(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Yozish taklifi — **yagona** yo'l `write_tools.propose_or_execute` orqali.

    Tasdiq oqimi o'zgarmagan: `write_with_confirm` → `agent_actions(proposed)`,
    `autonomous` chatda o'sha funksiya o'zi bajaradi (foydalanuvchi aniq
    yoqqan bitta chat). `assert_writable`, `read_only` va delete-yo'qligi —
    `build_proposal` / `execute_action` ichida.
    """
    from app.services import tools as T
    from app.services import write_tools as W

    tool = str(args.get("tool") or "send_message")
    if tool not in W.WRITE_TOOL_NAMES:
        return SkillResult(answer=_WRITE_FAIL.get(ctx.locale, _WRITE_FAIL["uz"]) + tool, stop=True)
    call_args: dict[str, Any] = {
        k: v for k, v in args.items() if k in ("text", "reply_to", "message_id", "chat")
    }
    chat_id = args.get("chat_id")
    pinned = (
        int(chat_id)
        if isinstance(chat_id, int)
        else (ctx.selected[0] if len(ctx.selected) == 1 else None)
    )
    async with ctx.session_factory() as session:
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
        tctx = T.ToolContext(
            session=session,
            account_id=ctx.account_id,
            pinned_chat_id=pinned,
            user_id=ctx.user_id,
        )
        res = await W.propose_or_execute(
            session, tctx, run_id=ctx.run_id, tool=tool, args=call_args
        )
    action_id = res.meta.get("action_id")
    if action_id:
        ctx.action_ids.append(int(action_id))
    if not res.ok:
        text = _WRITE_FAIL.get(ctx.locale, _WRITE_FAIL["uz"]) + U.neutralize(res.text)[:300]
    elif res.meta.get("executed"):
        # `autonomous` chat — foydalanuvchi o'zi yoqqan rejim; matn rost bo'lsin.
        text = U.neutralize(res.text)[:300]
    else:
        text = _WRITE_DONE.get(ctx.locale, _WRITE_DONE["uz"]) + f"#{action_id}"
    return SkillResult(answer=text, stop=True, summary={"ok": res.ok})


# ─── 9. agent_tools ──────────────────────────────────────────────────────────

#: `agent_tools` natijasi da'volar blokida shuncha belgigacha (sintezga).
AGENT_ANSWER_CHARS = 6_000


def history_msgs(history: Sequence[DS.Turn]) -> list[Msg]:
    """Planner tarixi (`DS.Turn`) → agent tarixi (`Msg`).

    User matni — foydalanuvchining o'zi; oldingi javob esa eski `fast`
    yo'lidagi bilan bir xil: oddiy assistant turn'i, `DS.PRIOR_ANSWER_CHARS`
    gacha kesilgan (`load_history` kesadi).
    """
    out: list[Msg] = []
    for t in history:
        text = (t.text or "").strip()
        if not text:
            continue
        out.append(Msg.user(text) if t.role == "user" else Msg.assistant(text))
    return out


async def _agent_tools(ctx: RunState, args: dict[str, Any]) -> SkillResult:
    """Eski `fast` agent sikli (`agent.run_agent`) — **bitta ko'nikma** bo'lib.

    Makro-ko'nikmalar (dayjest, sanoq, oyna, qidiruv) qoplamagan ochiq tool
    ishi uchun: profil, a'zolar, reaksiyalar, izohlar, saqlangan xabarlar,
    kalendar, aniq xabar. Sikl **o'zgarishsiz** yuguradi — `AGENT_*`
    byudjetlari, suhbat keshi (`tool_cache`), takror-qo'riqchisi, "oldinga
    siljish yo'q" to'xtashi, yozish tool'lari tasdiq bilan — hammasi
    `run_agent` ichida. Narx shu yugurish hisobiga (`_MeterLLM`).

    Karantin: agent javobi begona tool natijalaridan sintez qilingan, ya'ni
    u faqat da'volar blokiga (yagona dalil konverti ichiga) va `_Taint`
    ro'yxatiga tushadi; plan/replan faqat sanoqni ko'radi.
    """
    from app.services import agent as AG

    pinned = ctx.chats_by_id.get(ctx.selected[0]) if len(ctx.selected) == 1 else None
    if pinned is None and isinstance(args.get("chat_id"), int):
        pinned = ctx.chats_by_id.get(int(args["chat_id"]))
    question = ctx.raw_question
    focus = U.neutralize(str(args.get("focus") or "")).strip()[:200]
    if focus:
        question = f"{question}\n\n(focus: {focus})"
    tools_seen: list[str] = []

    async def on_event(kind: str, data: dict[str, Any]) -> None:
        name = str(data.get("tool") or "")
        if kind == "status" and data.get("stage") == "tool" and name:
            tools_seen.append(name)
            await DS._emit(ctx.events, "tool", skill="agent_tools", name=name)

    async with ctx.session_factory() as session:
        outcome = await AG.run_agent(
            session,
            user_id=ctx.user_id,
            account_id=ctx.account_id,
            question=question,
            history=history_msgs(ctx.history),
            pinned_chat=pinned,
            locale=ctx.locale,
            llm=cast(LLM, _MeterLLM(ctx.meter)),
            on_event=on_event,
            conversation_id=ctx.conversation_id,
        )
    if ctx.run_id is None and outcome.run_id:
        ctx.run_id = int(outcome.run_id)
    ctx.action_ids.extend(int(a) for a in outcome.action_ids)
    answer = (outcome.text or "").strip()
    ctx.agent_answer = answer
    tools = [str(c.get("tool") or "") for c in outcome.tool_calls]
    ctx.agent_meta = {
        "tools": tools,
        "tool_calls": outcome.tool_calls,
        "iterations": outcome.iterations,
        "calls_executed": outcome.calls_executed,
        "stop_reason": outcome.stop_reason,
        "repeat_guard": outcome.repeat_guard,
        "cache_hits": outcome.cache_hits,
        "finish_reason": outcome.finish_reason,
        "model": outcome.model,
        "provider": outcome.provider,
    }
    if answer:
        ctx.taint.add_texts([answer])
        body = U.neutralize(answer)[:AGENT_ANSWER_CHARS]
        ctx.claims.append(
            "### tool agent findings (read-only Telegram tools; machine-written from "
            f"untrusted tool output; tools: {', '.join(dict.fromkeys(tools)) or 'none'})\n{body}"
        )
    facts = [
        Fact(
            "agent_calls",
            int(outcome.calls_executed),
            label="read-only tool calls executed by the tool agent",
            source="agent_tools",
        )
    ]
    return SkillResult(
        facts=facts,
        notes=[f"agent_tools: stop={outcome.stop_reason}"] if outcome.stop_reason else [],
        summary={
            "calls": int(outcome.calls_executed),
            "tools": len(set(tools)),
            "cache_hits": int(outcome.cache_hits),
            "answered": bool(answer),
        },
    )


# ─── registr ─────────────────────────────────────────────────────────────────

SKILLS: dict[str, Skill] = {
    s.name: s
    for s in (
        Skill(
            "answer_direct",
            "salom / rahmat / 'nimalar qila olasan' — qidiruvsiz qisqa javob",
            "greeting, thanks, 'what can you do' — a short answer with no retrieval",
            {"intent": "str"},
            0.002,
            2.0,
            "fast",
            quarantined=False,
            run=_answer_direct,
            terminal=True,
            rail=(),
        ),
        Skill(
            "clarify",
            "muhim parametr yo'q — bitta aniqlashtiruvchi savol, 2-3 variant",
            "an essential parameter is missing — one clarifying question with 2-3 options",
            {"question": "str", "options": "list[str]"},
            0.0,
            0.0,
            "fast",
            quarantined=False,
            run=_clarify,
            terminal=True,
            rail=(),
        ),
        Skill(
            "reuse_prior",
            "oldingi javob dalillarini qayta o'qish (jadval qil / tushuntir / qaysi chatdan)",
            "re-read the previous answer's evidence (reformat / explain / which chat was it)",
            {},
            0.0,
            0.5,
            "fast",
            quarantined=True,
            run=_reuse_prior,
            rail=("search",),
        ),
        Skill(
            "chat_digest",
            "bitta chatning eng arzon umumiy ko'rinishi: keshdagi kun/hafta dayjestlari + sanoq",
            "cheapest overview of one chat: cached day/week digests + counts + newest messages",
            {"chat_id": "int", "chat_ids": "list[int]"},
            0.003,
            3.0,
            "fast",
            quarantined=True,
            run=_chat_digest,
            rail=("digest",),
        ),
        Skill(
            "stats",
            "count(*) sanoqlari: nechta xabar, kim/qaysi chat eng ko'p, yo'nalish (men/menga); "
            "period=this_month|last_month|… nomlangan davr, slot=a|b taqqoslash uchun, "
            "by=weekday|hour|both — haftaning kuni / soat taqsimoti",
            "exact count(*) numbers: how many messages, top chats/senders, direction (in/out), "
            "first/last message date and days since; period=<today|yesterday|this_week|"
            "last_week|this_month|last_month|this_year|last_year> counts that named period, "
            "slot=a|b stores the facts as a_n/a_days/a_period (for comparing two periods), "
            "by=weekday|hour|both adds the day-of-week / hour-of-day distribution",
            {"direction": "str", "period": "str", "slot": "str", "by": "str"},
            0.0,
            2.0,
            "fast",
            quarantined=False,
            run=_stats,
            rail=("stats",),
        ),
        Skill(
            "compute",
            "olingan raqamlar ustida hisob: o'rtacha kuniga/haftasiga, foiz, farq, nisbat; "
            "op=compare_periods — ikki davr (a_n, b_n): farq, foiz o'zgarish, kuniga o'rtacha",
            "arithmetic over numbers already obtained: per day/week, percent, difference, "
            "ratio; op=compare_periods turns a_n/b_n (two stats slots) into diff, pct_change, "
            "verdict and per-day averages",
            {"expr": "str", "label": "str", "items": "list[dict]", "op": "str"},
            0.0,
            0.0,
            "fast",
            quarantined=False,
            run=_compute,
            rail=("compute",),
        ),
        Skill(
            "window",
            "mavzusiz vaqt oynasi: 'bugun kimlar yozdi', 'kecha nima bo'ldi' (namuna + xulosa)",
            "a time window with no subject: 'who wrote today', 'what happened yesterday'",
            {},
            0.02,
            15.0,
            "fast",
            quarantined=True,
            run=_window,
            rail=("estimate", "pull", "select", "map"),
            confirm_modes=frozenset({"exhaustive"}),
        ),
        Skill(
            "search_local",
            "lokal indeksda mavzu qidiruvi (matn + ma'no, klaster, rerank)",
            "topic search in the local index (keyword + meaning, clusters, rerank)",
            {"queries": "list[str]"},
            0.02,
            12.0,
            "fast",
            quarantined=True,
            run=_search_local,
            rail=("search", "cluster", "rerank"),
        ),
        Skill(
            "search_server",
            "Telegram serverida qidiruv: tarixi to'liq yuklanmagan chatlar yoki global",
            "Telegram's own search: chats whose history is not fully downloaded, or global",
            {"scope": "str"},
            0.0,
            6.0,
            "fast",
            quarantined=True,
            run=_search_server,
            rail=("server",),
        ),
        Skill(
            "discover",
            "siz a'zo bo'lmagan ochiq kanallar (faqat so'ralganda)",
            "public channels the owner is not a member of (only when asked)",
            {},
            0.0,
            8.0,
            "deep",
            quarantined=True,
            run=_discover,
            rail=("server",),
        ),
        Skill(
            "scan_corpus",
            "korpus skani: ko'p chat x uzun davr bo'yicha mavzu (qimmat; to'liqda tasdiq)",
            "corpus scan: a subject across many chats over a long range (expensive)",
            {},
            0.10,
            60.0,
            "deep",
            quarantined=True,
            run=_scan_corpus,
            rail=("estimate", "buckets", "pull", "filter", "select", "map", "reduce"),
            confirm_modes=frozenset({"exhaustive"}),
            cost_by_mode=(("deep", 0.10), ("exhaustive", 0.50)),
        ),
        Skill(
            "write_propose",
            "xabar yozish/yuborish TAKLIFI (tasdiqsiz hech narsa yuborilmaydi)",
            "PROPOSE sending/editing a message (nothing is sent without confirmation)",
            {
                "tool": "str",
                "chat_id": "int",
                "text": "str",
                "reply_to": "int",
                "message_id": "int",
            },
            0.0,
            1.0,
            "fast",
            quarantined=False,
            run=_write_propose,
            terminal=True,
            rail=(),
        ),
        Skill(
            "agent_tools",
            "o'qish tool'lari agenti: profil, a'zolar, reaksiya, izohlar, saqlangan xabarlar, "
            "kalendar — boshqa ko'nikma qoplamagan ochiq ish uchun",
            "read-only tool agent: profiles, members, reactions, post comments, saved "
            "messages, calendar search, one exact message — open-ended work no other skill "
            "covers",
            {"focus": "str", "chat_id": "int"},
            0.05,
            20.0,
            "fast",
            quarantined=True,
            run=_agent_tools,
            rail=("tools",),
        ),
    )
}

#: Registrning o'z (yadro) ko'nikmalari — `skills_ext` ularni almashtira olmaydi.
CORE_SKILLS = frozenset(SKILLS)
_CORE = dict(SKILLS)


def register_extra() -> list[str]:
    """Ixtiyoriy kengaytma: `app.services.skills_ext.register(SKILLS)`.

    Modul bo'lmasa — hech narsa (bo'sh ro'yxat). Bor bo'lsa uning
    ko'nikmalari registrga qo'shiladi va planner katalogiga **avtomatik**
    tushadi (`orchestrator.ORCH_PLAN_PROMPT` registrdan import paytida
    yasaladi). Qoidalar kodda: yadro ko'nikmasini almashtirish/o'chirish
    mumkin emas (qaytariladi), `Skill` bo'lmagan yoki nomi kalitiga mos
    kelmagan yozuv tashlanadi. Kengaytma yiqilsa ilova yiqilmaydi — log.
    """
    import importlib

    try:
        mod = importlib.import_module("app.services.skills_ext")
    except ModuleNotFoundError as exc:
        if exc.name == "app.services.skills_ext":
            return []
        log.error("skills.extra_import_failed", error=str(exc)[:200])
        return []
    except Exception as exc:
        log.error("skills.extra_import_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return []
    register = getattr(mod, "register", None)
    if not callable(register):
        return []
    try:
        register(SKILLS)
    except Exception as exc:
        log.error("skills.extra_register_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
    for name, skill in list(SKILLS.items()):
        if name in _CORE:
            if skill is not _CORE[name]:
                log.warning("skills.extra_core_override_refused", skill=name)
                SKILLS[name] = _CORE[name]
            continue
        if not isinstance(skill, Skill) or skill.name != name or skill.min_mode not in MODE_RANK:
            log.warning("skills.extra_invalid_dropped", skill=str(name)[:40])
            SKILLS.pop(name, None)
    for name, skill in _CORE.items():
        SKILLS.setdefault(name, skill)
    added = [n for n in SKILLS if n not in _CORE]
    if added:
        log.info("skills.extra_registered", skills=added)
    return added


EXTRA_SKILLS: tuple[str, ...] = tuple(register_extra())

#: Karantin ko'nikmalar — chiqishi (matn) plan/replan'ga **hech qachon** tushmaydi.
QUARANTINED_SKILLS = frozenset(n for n, s in SKILLS.items() if s.quarantined)


def catalog_text(mode: str | None = None) -> str:
    """Planner system prompt'idagi katalog — registrdan, **kodda** yasaladi."""
    lines: list[str] = []
    for s in SKILLS.values():
        args = ", ".join(f"{k}?: {v}" for k, v in s.args.items()) or ""
        if s.cost_by_mode:
            cost = " / ".join(f"{m} ~${c:.2f}" for m, c in s.cost_by_mode)
        else:
            cost = f"~${s.cost_usd:.3f}"
        flags = []
        if s.min_mode != "fast":
            flags.append(f"mode ≥ {s.min_mode}")
        if s.confirm_modes:
            flags.append("asks for confirmation in " + ", ".join(sorted(s.confirm_modes)))
        if s.quarantined:
            flags.append("quarantined")
        if s.terminal:
            flags.append("single step")
        tail = f" [{'; '.join(flags)}]" if flags else ""
        lines.append(
            f"- {s.name}({args}) — {cost}, ~{s.est_latency_s:g}s. {s.description_en}. "
            f"(uz: {s.description_uz}){tail}"
        )
    return "\n".join(lines)


def allowed_skills(mode: str) -> list[str]:
    return [n for n, s in SKILLS.items() if s.allowed(mode)]


def fold_question(text: str) -> str:
    return TN.uz_fold(text or "")


__all__ = [
    "CORE_SKILLS",
    "EXTRA_SKILLS",
    "MODE_RANK",
    "QUARANTINED_SKILLS",
    "SKILLS",
    "WRITE_INTENT_RE",
    "RunState",
    "Skill",
    "SkillResult",
    "allowed_skills",
    "catalog_text",
    "coerce_args",
    "history_msgs",
    "period_facts",
    "register_extra",
    "run_escalation",
    "server_calls_left",
]
