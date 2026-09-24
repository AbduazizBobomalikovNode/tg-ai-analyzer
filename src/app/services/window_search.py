"""Oyna bo'yicha olish (`retrieval_kind="window"`) — mavzu skani **emas**.

Har savol korpus bo'ylab mavzu qidirish emas. "Bugun menga kimlar yozgan?",
"kecha qaysi chatlar faol edi?", "oxirgi 2 yilda eng faol chatlar qaysi?" —
bularda reyting qiladigan **mavzu yo'q**: kerakli narsa vaqt oynasining
o'zidagi xabarlar. Kaskad (`scale_search`) bunday savolda bucket rejasi
quradi, tirsak bilan kesadi va LLM filtriga to'lab, oxirida "1 hafta
rejalashtirildi" deb ogohlantirardi — savol aynan "bugun" degan bo'lsa ham.

Shuning uchun bu yo'l:

    plan → (baho) → pull (oynadagi hamma xabar) → select → (map) → write

* **Bucket rejasi yo'q, tirsak yo'q.** Oyna — `plan.since/until`
  (`timewin.parse_window` savoldan hisoblaydi), nishon chatlar —
  `plan.chats` yoki hamma sinxronlangan chat.
* **Sanoq aniq va bepul**: `count(*)`. Baho shundan quriladi, taxmindan emas.
* **Kesish qoidasi bitta**: oynada `SOFT_MAX_EVIDENCE` (1000) yoki undan kam
  xabar bo'lsa — **hammasi** dalil; ko'p bo'lsa `(chat, kun)` kvotasi bilan
  eng yangisidan olinadi va qamrovda "M dan N ko'rildi" deb **aytiladi**.
  Savolda mavzu ham bo'lsa (kesilgan holatda) arzon filtr qo'shimcha
  tartiblaydi — aks holda filtr umuman chaqirilmaydi.
* **"Menga yozgan"** (`inbox`) — sxemada bor narsa bilan ta'riflanadi:
  `messages.out = false` **va** (chat shaxsiy **yoki** xabar mening
  xabarimga javob). `@username` eslatmalari saqlanmaydi, shuning uchun ular
  qamrovga kirmaydi — buni UI ham, qamrov ham yashirmaydi.
* Dalillar map bosqichiga **yuboruvchi bo'yicha** guruhlanadi (bitta chat +
  bitta yuboruvchi = bitta bo'lak), chunki javob ham shunday o'qiladi:
  kim, nechta xabar, nima maqsadda.

Invariantlar o'zgarmadi: dalillar bitta `<untrusted_data kind="evidence">`
konvertida (`deep_search.build_evidence_envelope`), `plan`/`reflect` bu
yerda umuman chaqirilmaydi, telefon raqami hech qayerda yo'q.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import Select, and_, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import Chat, Message
from app.llm import Task
from app.services import deep_search as DS
from app.services import scale_search as SC
from app.services import search as S
from app.services import timewin as TW
from app.services import untrusted as U
from app.services.prompts_deep import (
    DEEP_SYNTH_PROMPT,
    SCALE_FILTER_PROMPT,
    SCALE_MAP_PROMPT,
)

log = structlog.get_logger(__name__)

#: `exhaustive` tasdiq so'raydigan chegara — **skan** narxi bo'yicha
#: (`est_cost_usd`), sintez bilan birga emas. Nega: sintez har rejimda
#: to'lanadi va u oynaning kattaligi haqida hech narsa demaydi; "bugun
#: menga kim yozdi" uchun baho ekranini ko'rsatish — foydasiz bosqich.
#: Ikkinchi shart pastda: oyna qopqoqqa **sig'masa** (kesiladi) tasdiq
#: baribir so'raladi — foydalanuvchi qisqartirilgan skanni tanlayapti.
WINDOW_CONFIRM_MIN_USD = 0.05

#: Sanoq shu chegaradan oshsa `(chat, kun)` kvotasi ishlaydi.
WINDOW_SOFT_MAX = SC.SOFT_MAX_EVIDENCE

#: Map bo'lagi — bitta chat + bitta yuboruvchi, shundan katta bo'lsa bo'linadi.
WINDOW_CHUNK_MSGS = SC.MAP_CHUNK_MSGS
WINDOW_MAX_CHUNKS = SC.MAP_MAX_CHUNKS

#: Kvota yo'lida filtr faqat savolda mavzu bo'lganda chaqiriladi.
WINDOW_FILTER_MIN_TERMS = 1


@dataclass(slots=True)
class WindowSpec:
    """Oyna so'rovining barcha kirishlari — bitta joyda, qayta hisoblanmaydi."""

    account_id: int
    chats: list[int] | None
    since: datetime | None
    until: datetime
    inbox: bool = False
    tz: str = TW.DEFAULT_TZ

    def to_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat(),
            "inbox": self.inbox,
            "chats": len(self.chats) if self.chats else 0,
            "tz": self.tz,
        }


@dataclass(slots=True)
class WindowCount:
    """`count(*)` — aniq va bepul. Baho shundan quriladi."""

    total: int = 0
    chats: int = 0
    #: `(chat, kun)` juftliklari soni — kvota shundan hisoblanadi.
    groups: int = 0
    first_at: datetime | None = None
    last_at: datetime | None = None
    per_chat: list[tuple[int, int]] = field(default_factory=list)


def _conds(spec: WindowSpec) -> list[Any]:
    """Oyna sharti — `_scope` bilan bir mantiq, lekin `inbox` filtri bilan."""
    conds: list[Any] = [Chat.account_id == spec.account_id, *S._alive()]
    if spec.chats:
        conds.append(Message.chat_id.in_(list(spec.chats)))
    if spec.since is not None:
        conds.append(Message.published_at >= spec.since)
    conds.append(Message.published_at < spec.until)
    if spec.inbox:
        # "Menga yozgan" = meniki emas **va** (shaxsiy chat **yoki** mening
        # xabarimga javob). Eslatmalar (`@username`) sxemada saqlanmaydi —
        # ular bu ta'rifga kirmaydi va qamrovda shunday deyiladi.
        mine = aliased(Message)
        replied = exists().where(
            and_(
                mine.chat_id == Message.chat_id,
                mine.tg_msg_id == Message.reply_to_msg_id,
                mine.out.is_(True),
            )
        )
        conds.append(Message.out.is_(False))
        conds.append(or_(Chat.type == "private", replied))
    return conds


def _base(spec: WindowSpec) -> Select[Any]:
    return select(Message).join(Chat, Chat.id == Message.chat_id).where(*_conds(spec))


async def count_window(session: AsyncSession, spec: WindowSpec) -> WindowCount:
    """Oynadagi xabarlar sanog'i — **skan tokenisiz**, aniq.

    `groups` = alohida `(chat, kun)` juftliklari: kvota shundan bo'linadi,
    ya'ni 2 yillik oyna ham, 1 kunlik oyna ham bir xil mantiq bilan
    kesiladi (kvota oynaning **shakliga** moslashadi, konstantaga emas).
    """
    day = func.date_trunc("day", Message.published_at)
    stmt = (
        select(
            func.count().label("n"),
            func.count(func.distinct(Message.chat_id)).label("chats"),
            func.count(func.distinct(func.concat(Message.chat_id, ":", day))).label("groups"),
            func.min(Message.published_at),
            func.max(Message.published_at),
        )
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_conds(spec))
    )
    row = (await session.execute(stmt)).one()
    out = WindowCount(
        total=int(row[0] or 0),
        chats=int(row[1] or 0),
        groups=int(row[2] or 0),
        first_at=row[3],
        last_at=row[4],
    )
    per = (
        select(Message.chat_id, func.count().label("n"))
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_conds(spec))
        .group_by(Message.chat_id)
        .order_by(func.count().desc())
        .limit(200)
    )
    out.per_chat = [(int(c), int(n)) for c, n in (await session.execute(per)).all()]
    return out


def _hit(m: Message, title: str) -> S.Hit:
    return S.Hit(
        message_id=m.id,
        tg_msg_id=m.tg_msg_id,
        score=1.0,
        source="window",
        chat_id=m.chat_id,
        chat_title=title,
        date=m.published_at,
        text=m.text or "",
    )


async def pull_window(
    session: AsyncSession,
    spec: WindowSpec,
    *,
    count: WindowCount,
    cap: int = WINDOW_SOFT_MAX,
    titles: dict[int, str] | None = None,
) -> tuple[list[S.Hit], bool]:
    """Oynadagi xabarlar → `(hitlar, kesilganmi)`.

    `total <= cap` — **hammasi** (vaqt bo'yicha). Aks holda `(chat, kun)`
    kvotasi: har juftlikdan eng yangi `q` ta, `q = max(1, cap // groups)`.
    Bitta gapiradigan chat butun byudjetni yeb qo'ymasin, lekin jim chat
    ham ko'rinmay qolmasin.
    """
    names = dict(titles or {})
    if count.total <= cap:
        rows = (
            await session.execute(_base(spec).order_by(Message.published_at.desc()).limit(cap))
        ).scalars()
        hits = [_hit(m, names.get(m.chat_id, "")) for m in rows]
        return hits, False

    groups = max(1, count.groups)
    quota = max(1, cap // groups)
    day = func.date_trunc("day", Message.published_at)
    rn = (
        func.row_number()
        .over(
            partition_by=(Message.chat_id, day),
            order_by=Message.published_at.desc(),
        )
        .label("rn")
    )
    inner = (
        select(Message.id, rn)
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_conds(spec))
        .subquery()
    )
    stmt = (
        select(Message)
        .join(inner, inner.c.id == Message.id)
        .where(inner.c.rn <= quota)
        .order_by(Message.published_at.desc())
        .limit(cap)
    )
    rows = (await session.execute(stmt)).scalars()
    hits = [_hit(m, names.get(m.chat_id, "")) for m in rows]
    log.info(
        "window.quota",
        total=count.total,
        groups=groups,
        quota=quota,
        taken=len(hits),
    )
    return hits, True


def estimate_window(
    count: WindowCount,
    *,
    mode: str = "deep",
    cap: int = WINDOW_SOFT_MAX,
    window_from_question: bool = True,
    spec: WindowSpec | None = None,
    filtered: bool = False,
) -> SC.Estimate:
    """Oyna bahosi — **aniq sanoqdan**, bucket taxminidan emas.

    Skan qismi: kesilgan holatda arzon filtr (aks holda 0) + map. Sintez
    alohida (`est_total_usd`), `scale_search.estimate` bilan bir uslubda.
    """
    prices = {
        "filter": SC._price_for(Task.ROUTE),
        "map": SC._price_for(Task.ROUTE),
        "synth": SC._price_for(Task.DEEP),
    }
    taken = min(count.total, cap)
    n_filter = taken if filtered else 0
    n_batches = math.ceil(n_filter / SC.FILTER_BATCH) if n_filter else 0
    filter_in = n_batches * (
        int(len(SCALE_FILTER_PROMPT) / S.CHARS_PER_TOKEN)
        + int(SC.FILTER_BATCH * (SC.FILTER_LINE_CHARS + 45) / S.CHARS_PER_TOKEN)
    )
    filter_out = n_batches * 120

    chunks = (
        0
        if taken <= SC.MAP_SKIP_TOTAL
        else min(WINDOW_MAX_CHUNKS, math.ceil(taken / WINDOW_CHUNK_MSGS))
    )
    map_in = chunks * (
        int(len(SCALE_MAP_PROMPT) / S.CHARS_PER_TOKEN)
        + int(WINDOW_CHUNK_MSGS * (SC.MAP_LINE_CHARS + 45) / S.CHARS_PER_TOKEN)
    )
    map_out = chunks * SC.EST_MAP_OUT_TOKENS

    synth_lines = min(SC.SYNTH_EVIDENCE_LINES, taken)
    synth_in = int(len(DEEP_SYNTH_PROMPT) / S.CHARS_PER_TOKEN) + int(
        synth_lines * (SC.MAP_LINE_CHARS + 45) / S.CHARS_PER_TOKEN
    )

    def cost(stage: str, tin: int, tout: int) -> float:
        price_in, price_out = prices[stage]
        return (tin * price_in + tout * price_out) / 1_000_000

    breakdown = {
        "filter": round(cost("filter", filter_in, filter_out), 6),
        "map": round(cost("map", map_in, map_out), 6),
        "synth": round(cost("synth", synth_in, SC.EST_SYNTH_OUT_TOKENS), 6),
    }
    scan = breakdown["filter"] + breakdown["map"]
    return SC.Estimate(
        mode=mode,
        buckets=max(count.groups, 0),
        est_messages=count.total,
        est_cost_usd=round(scan, 6),
        est_total_usd=round(scan + breakdown["synth"], 6),
        ceiling_usd=SC.COST_CEILING.get(mode, SC.COST_CEILING["deep"]),
        weeks=0,
        range_from=(spec.since if spec else None) or count.first_at,
        range_to=(spec.until if spec else None) or count.last_at,
        breakdown=breakdown,
        retrieval_kind="window",
        window_from_question=window_from_question,
    )


#: Sanoq blokida ko'rsatiladigan chat/yuboruvchi soni.
STATS_TOP_CHATS = 20
STATS_TOP_SENDERS = 20


def render_window_stats(
    count: WindowCount,
    *,
    titles: dict[int, str] | None = None,
    evidence: Sequence[S.Hit] = (),
    taken: int = 0,
) -> str:
    """Sanoqlar **kodda** hisoblanadi — briefing bilan bir qoida.

    "Qaysi chatlar eng faol edi" degan savolning javobi `count(*)` dan
    keladi, dalil satrlaridan emas: model ko'rgan ≤1000 satr oynaning
    **hammasi** bo'lmasligi mumkin va u shunda "hammasi bitta chatda edi"
    deb yozardi (jonli regressiya). Blok yagona `kind="evidence"`
    konvertining ichiga tushadi (`build_evidence_envelope(claims_block=…)`),
    chat sarlavhalari `untrusted.attr` dan o'tgan.
    """
    names = dict(titles or {})
    lines = [
        "## window totals (counted in code, exact — trust these over the lines below)",
        f"messages in window: {count.total}; chats: {count.chats}; "
        f"evidence lines shown: {taken or len(evidence)}",
    ]
    if count.per_chat:
        lines.append("messages per chat (descending):")
        for chat_id, n in count.per_chat[:STATS_TOP_CHATS]:
            title = U.attr(names.get(chat_id, "")) or f"chat {chat_id}"
            lines.append(f"- {title}: {n}")
        if len(count.per_chat) > STATS_TOP_CHATS:
            lines.append(f"- (+{len(count.per_chat) - STATS_TOP_CHATS} more chats)")
    senders: dict[str, int] = {}
    for hit in evidence:
        key = (hit.sender or "?").strip() or "?"
        senders[key] = senders.get(key, 0) + 1
    if senders:
        lines.append("messages per sender (within the evidence shown):")
        for name, n in sorted(senders.items(), key=lambda p: -p[1])[:STATS_TOP_SENDERS]:
            lines.append(f"- {U.attr(name)}: {n}")
    return "\n".join(lines)


def window_chunks(
    evidence: list[S.Hit],
    *,
    chunk_size: int = WINDOW_CHUNK_MSGS,
    max_chunks: int = WINDOW_MAX_CHUNKS,
    skip_below: int = SC.MAP_SKIP_TOTAL,
) -> list[list[S.Hit]]:
    """Bo'laklar **yuboruvchi bo'yicha**: bitta chat + bitta yuboruvchi.

    Nega: javob ham shu shaklda o'qiladi ("Nurbek — 12 xabar, ijara
    haqida"). `map_chunks` esa `(chat, hafta)` bo'yicha bo'ladi — u mavzu
    skani uchun, bu yerda esa kim yozgani muhim.
    """
    if len(evidence) <= max(0, skip_below):
        return []
    groups: dict[tuple[int, str], list[S.Hit]] = {}
    order: dict[tuple[int, str], int] = {}
    for i, hit in enumerate(evidence):
        key = (hit.chat_id, (hit.sender or "?").strip().lower())
        groups.setdefault(key, []).append(hit)
        order.setdefault(key, i)
    out: list[list[S.Hit]] = []
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        rows = sorted(groups[key], key=lambda h: h.date or datetime.min.replace(tzinfo=UTC))
        for i in range(0, len(rows), chunk_size):
            out.append(rows[i : i + chunk_size])
    return out[:max_chunks]


async def run_window(
    session_factory: DS.SessionFactory,
    *,
    account_id: int,
    plan: DS.DeepPlan,
    question: str,
    mode: str,
    meter: DS._Meter,
    taint: DS._Taint,
    events: DS.EventSink | None = None,
    chat_titles: dict[int, str] | None = None,
    all_chats: list[int] | None = None,
    deadline: float | None = None,
    confirm: bool = True,
    locale: str = "uz",
) -> SC.ScaleOutcome:
    """Oyna yo'lining to'liq yugurishi — `deep_search._run_pipeline` shu yerga kiradi.

    `confirm=False` va `mode="exhaustive"` bo'lsa **faqat qimmat** oyna
    tasdiq so'raydi (`WINDOW_CONFIRM_MIN_USD`): "bugun kim yozdi" uchun
    baho ekranini ko'rsatish foydasiz bosqich.
    """
    started = time.monotonic()
    stats = SC.ScanStats()
    outcome = SC.ScaleOutcome(stats=stats)
    ceiling = SC.COST_CEILING.get(mode, SC.COST_CEILING["deep"])
    wall = deadline if deadline is not None else started + SC.EXHAUSTIVE_WALL_S
    budget = SC._Budget(meter, ceiling=ceiling, deadline=wall)
    titles = dict(chat_titles or {})

    spec = WindowSpec(
        account_id=account_id,
        chats=list(plan.chats) if plan.chats else (list(all_chats) if all_chats else None),
        since=plan.since,
        until=plan.until or datetime.now(UTC),
        inbox=plan.inbox,
        tz=plan.window_tz or TW.DEFAULT_TZ,
    )
    outcome.window = spec

    async with session_factory() as session:
        count = await count_window(session, spec)
    outcome.window_count = count
    stats.buckets_planned = max(count.groups, 0)

    # Savolda mavzu bormi: kesilgan holatda **faqat shunda** arzon filtr
    # chaqiriladi. Mezon — `plan.entities` (nom, joy, summa), `subqueries`
    # **emas**: planner har savolga subquery yasaydi, ya'ni u mavzu borligi
    # haqida hech narsa demaydi. Jonli regressiya: "kecha qaysi chatlar eng
    # faol edi?" savolida filtr 2468 xabardan 19 tasini qoldirib, erta
    # to'xtadi (`yield`) va javob "hammasi bitta chatda edi" dedi — savolda
    # esa reyting qiladigan mavzu umuman yo'q edi.
    will_filter = count.total > WINDOW_SOFT_MAX and bool(
        [e for e in plan.entities if e and e.strip()]
    )

    est = estimate_window(
        count,
        mode=mode,
        window_from_question=plan.window_from_question,
        spec=spec,
        filtered=will_filter,
    )
    outcome.estimate = est
    stats.est_cost_usd = est.est_cost_usd
    stats.est_total_usd = est.est_total_usd
    await DS._emit(events, "estimate", **est.to_dict())

    if not count.total:
        stats.note_stop("filter", "complete")
        return outcome
    too_big = count.total > WINDOW_SOFT_MAX
    if (
        mode == "exhaustive"
        and not confirm
        and (est.est_cost_usd > WINDOW_CONFIRM_MIN_USD or too_big)
    ):
        stats.stopped_by = "estimate_only"
        log.info(
            "window.needs_confirm",
            total=count.total,
            est_cost_usd=est.est_cost_usd,
            too_big=too_big,
        )
        return outcome

    # ── (b) tortish: oynadagi xabarlar, bucket rejasisiz ────────────────────
    async with session_factory() as session:
        pool, quota_cut = await pull_window(session, spec, count=count, titles=titles)
    stats.buckets_scanned = stats.buckets_planned
    stats.pulled = count.total
    stats.elbow = len(pool)
    stats.candidates = len(pool)
    stats.pool_capped = quota_cut
    if quota_cut:
        stats.truncated = True
        stats.note_stop("pull", "yield")
    await DS._emit(
        events,
        "pull",
        pulled=count.total,
        kept=len(pool),
        chats=count.chats,
        buckets=stats.buckets_planned,
        done=stats.buckets_planned,
        total=stats.buckets_planned,
    )
    if not pool:
        stats.note_stop("filter", "complete")
        return outcome

    async with session_factory() as session:
        pool = await DS._hydrate(session, pool, account_id=account_id)
    taint.add_hits(pool)
    outcome.pool = pool

    # ── (c) tanlov ──────────────────────────────────────────────────────────
    if will_filter:
        day_of = {h.message_id: (h.date.strftime("%Y-%m-%d") if h.date else "") for h in pool}
        scored = await SC.filter_batched(
            session_factory,
            account_id=account_id,
            question=question,
            pool=pool,
            bucket_of=day_of,
            meter=meter,
            budget=budget,
            taint=taint,
            events=events,
            mode=mode,
            stats=stats,
        )
        evidence = [h for h, _s in sorted(scored, key=lambda p: (-p[1], -(p[0].score or 0)))]
        evidence = evidence[:WINDOW_SOFT_MAX]
    else:
        evidence = list(pool)
        stats.kept = len(evidence)

    # Dalil vaqt bo'yicha o'qiladi: eng eskisidan eng yangisigacha.
    evidence.sort(key=lambda h: h.date or datetime.min.replace(tzinfo=UTC))
    stats.evidence = len(evidence)
    outcome.evidence = evidence
    await DS._emit(
        events,
        "select",
        evidence=len(evidence),
        chats=len({h.chat_id for h in evidence}),
        weeks=len({SC.week_key(h) for h in evidence}),
        quota_trimmed=max(0, count.total - len(evidence)),
    )

    # ── (d) map → da'volar (yuboruvchi bo'yicha) ────────────────────────────
    chunks = window_chunks(evidence)
    claims = await SC.map_claims(
        question=question,
        chunks=chunks,
        meter=meter,
        budget=budget,
        taint=taint,
        events=events,
        stats=stats,
    )
    outcome.claims = claims
    # Sanoqlar kodda: "qaysi chatlar eng faol edi" javobi `count(*)` dan
    # keladi, model ko'rgan satrlardan emas.
    outcome.window_stats = render_window_stats(
        count, titles=titles, evidence=evidence, taken=len(evidence)
    )
    if budget.stop_reason():
        stats.note_stop("map", budget.stop_reason())
        stats.truncated = True
    stats.scan_cost_usd = round(budget.spent, 6)
    log.info(
        "window.done",
        mode=mode,
        inbox=spec.inbox,
        total=count.total,
        taken=len(evidence),
        chats=count.chats,
        groups=count.groups,
        chunks=len(chunks),
        claims=len(claims),
        stopped_by=stats.stopped_by,
        cost_usd=stats.scan_cost_usd,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    return outcome


__all__ = [
    "WINDOW_CONFIRM_MIN_USD",
    "WINDOW_SOFT_MAX",
    "WindowCount",
    "WindowSpec",
    "count_window",
    "estimate_window",
    "pull_window",
    "run_window",
    "window_chunks",
]
