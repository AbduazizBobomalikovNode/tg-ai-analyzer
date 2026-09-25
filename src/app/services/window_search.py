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

from app.db.models import Chat, Message, Sender
from app.llm import Task
from app.services import cluster as C
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
    #: `in | out | both | ""` — `deep_search.DIRECTIONS`. `inbox=True` eski
    #: yozuv (moslik taxallusi) va `direction="in"` bilan bir xil.
    direction: str = ""

    def __post_init__(self) -> None:
        if self.inbox and not self.direction:
            self.direction = "in"
        self.inbox = self.direction == "in"

    def to_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat(),
            "inbox": self.inbox,
            "direction": self.direction,
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
    if spec.direction == "out":
        # "Men kimga yozdim" — faqat mening xabarlarim.
        conds.append(Message.out.is_(True))
    elif spec.direction == "both":
        # "Kim bilan gaplashdim" — men **shu oynada** kamida bitta xabar
        # yozgan chatlardagi hamma xabar. Aks holda men hech yozmagan katta
        # kanal/guruh "suhbatdosh" bo'lib reytingni egallardi.
        # `IN (SELECT DISTINCT chat_id …)`, korrelyatsiyali `EXISTS` emas:
        # jonli korpusda EXISTS har qator uchun nested-loop semi-join bo'lib
        # 31 s oldi, bu esa bitta hash semi-join (≈0.1 s).
        mine = aliased(Message)
        mine_chat = aliased(Chat)
        window = [mine.out.is_(True), mine_chat.account_id == spec.account_id]
        if spec.chats:
            window.append(mine.chat_id.in_(list(spec.chats)))
        if spec.since is not None:
            window.append(mine.published_at >= spec.since)
        window.append(mine.published_at < spec.until)
        talked = (
            select(mine.chat_id)
            .join(mine_chat, mine_chat.id == mine.chat_id)
            .where(*window)
            .distinct()
        )
        conds.append(Message.chat_id.in_(talked))
    elif spec.inbox:
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
    lo = (spec.since if spec else None) or count.first_at
    hi = (spec.until if spec else None) or count.last_at
    days = span_days(lo, hi) if count.total else 0
    return SC.Estimate(
        mode=mode,
        buckets=max(count.groups, 0),
        buckets_total=max(count.groups, 0),
        est_messages=count.total,
        est_cost_usd=round(scan, 6),
        est_total_usd=round(scan + breakdown["synth"], 6),
        ceiling_usd=SC.COST_CEILING.get(mode, SC.COST_CEILING["deep"]),
        # Oyna yo'lida bo'lak = `(chat, kun)` juftligi; oyna uzunligi esa
        # hafta (yoki qisqa oynada kun) bilan aytiladi — "0 hafta" emas.
        weeks=math.ceil(days / 7) if days else 0,
        days=days,
        grain="day" if days and days <= SC.SHORT_WINDOW_DAYS else "week",
        chats=count.chats,
        chats_total=count.chats,
        range_from=lo,
        range_to=hi,
        breakdown=breakdown,
        retrieval_kind="window",
        window_from_question=window_from_question,
    )


def span_days(lo: datetime | None, hi: datetime | None) -> int:
    """Oyna uzunligi butun kunlarda (yuqoriga yaxlitlangan, kamida 1)."""
    if lo is None or hi is None or hi <= lo:
        return 0
    return max(1, math.ceil((hi - lo).total_seconds() / 86400.0 - 1e-9))


# ─── sanoq yo'li (`plan.window_stats`) ───────────────────────────────────────
# "Kimga eng ko'p yozganman", "bu oyda nechta xabar yozdim", "kim bilan eng
# ko'p gaplashganman" — javob **raqam**. 667k xabarni tortib, filtrlab, map
# qilish o'rniga DB `count(*)` agregatlari bilan javob beriladi: aniq,
# to'liq va deyarli bepul. Model faqat tayyor jadvalni so'zga aylantiradi.

#: Sanoq blokida ko'rsatiladigan chat/yuboruvchi soni.
STATS_TOP_CHATS = 20
STATS_TOP_SENDERS = 20

#: Reytingda ko'rsatiladigan chat soni (namunalar ham shulardan olinadi).
#: Sintez raqamlardan qiladi (`STATS_SYNTH_NOTE`), namuna faqat illyustratsiya —
#: shuning uchun arzon: 5 chat x 2 xabar = ≤10 satr (narx o'lchovi, 2-vazifa).
STATS_TOP_N = 5
#: Har top chatdan illyustratsiya uchun eng yangi xabarlar soni.
STATS_SAMPLE_PER_CHAT = 2


@dataclass(slots=True)
class ChatStat:
    """Bitta chat bo'yicha aniq sanoq (tanlangan yo'nalishda)."""

    chat_id: int
    n: int
    mine: int = 0  # shundan men yozganlarim (`messages.out`)
    theirs: int = 0  # shundan ular yozganlari (`n - mine`)
    chat_type: str = ""
    first_at: datetime | None = None
    last_at: datetime | None = None


@dataclass(slots=True)
class WindowStats:
    """Sanoq yo'lining natijasi — hammasi `count(*)` dan."""

    direction: str = ""
    per_chat: list[ChatStat] = field(default_factory=list)
    #: `(ko'rinadigan nom, son)` — faqat boshqalar yozgani (`out = false`).
    per_sender: list[tuple[str, int]] = field(default_factory=list)
    mine_total: int = 0


async def stats_window(
    session: AsyncSession, spec: WindowSpec, *, top: int = STATS_TOP_CHATS
) -> WindowStats:
    """Chat va yuboruvchi bo'yicha agregatlar — **xabar tortilmaydi**."""
    out = WindowStats(direction=spec.direction)
    mine = func.count().filter(Message.out.is_(True))
    theirs = func.count().filter(Message.out.is_(False))
    per = (
        select(
            Message.chat_id,
            func.count().label("n"),
            mine.label("mine"),
            theirs.label("theirs"),
            Chat.type,
            func.min(Message.published_at),
            func.max(Message.published_at),
        )
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_conds(spec))
        .group_by(Message.chat_id, Chat.type)
    )
    if spec.direction == "both":
        # "Kim bilan gaplashdim" — reyting **suhbat**, ya'ni ikki tomon ham
        # yozgan chatlar bo'yicha bo'lishi kerak: bitta xabar yozib qo'ygan
        # 13 mingta xabarli kanal "suhbatdosh" bo'lib chiqmasin. Ball —
        # `min(mine, theirs)` (mutuallik), teng bo'lsa `mine`, keyin jami
        # bo'yicha. `mine == 0` chatlar `_conds` semi-join'i bilan allaqachon
        # tashlangan (invariant: faqat men yozgan chatlar keladi); qo'shimcha
        # `theirs > 0` shartisiz esa "javobsiz yozganlarim" ham reytingga
        # kirar edi. Bu chatlarning xabarlari **jami sanoqda** qoladi
        # (`count_window`), faqat shu reytingdan chiqadi.
        score = func.least(mine, theirs)
        per = per.having(and_(mine > 0, theirs > 0)).order_by(
            score.desc(), mine.desc(), func.count().desc(), Message.chat_id
        )
    else:
        per = per.order_by(func.count().desc(), Message.chat_id)
    per = per.limit(top)
    for row in (await session.execute(per)).all():
        out.per_chat.append(
            ChatStat(
                chat_id=int(row[0]),
                n=int(row[1] or 0),
                mine=int(row[2] or 0),
                theirs=int(row[3] or 0),
                chat_type=str(row[4] or ""),
                first_at=row[5],
                last_at=row[6],
            )
        )
    if spec.direction != "out":
        # Kim yozgani — faqat **boshqalar** (mening xabarlarim shaxsiy chatda
        # `sender_id` siz keladi va "?" bo'lib reytingni buzardi). Ism
        # `senders` jadvalidan; telefon raqami u yerda yo'q.
        name = func.max(func.coalesce(func.nullif(Sender.display_name, ""), Sender.username))
        per_sender = (
            select(Message.sender_id, name, func.count().label("n"))
            .select_from(Message)
            .join(Chat, Chat.id == Message.chat_id)
            .outerjoin(
                Sender,
                and_(
                    Sender.account_id == spec.account_id,
                    Sender.tg_user_id == Message.sender_id,
                ),
            )
            .where(*_conds(spec), Message.out.is_(False), Message.sender_id.is_not(None))
            .group_by(Message.sender_id)
            .order_by(func.count().desc(), Message.sender_id)
            .limit(STATS_TOP_SENDERS)
        )
        for sender_id, display, n in (await session.execute(per_sender)).all():
            label = (str(display).strip() if display else "") or f"user:{int(sender_id)}"
            out.per_sender.append((label, int(n or 0)))
    if spec.direction != "out":
        total_mine = (
            select(func.count())
            .select_from(Message)
            .join(Chat, Chat.id == Message.chat_id)
            .where(*_conds(spec), Message.out.is_(True))
        )
        out.mine_total = int((await session.execute(total_mine)).scalar() or 0)
    return out


async def stats_samples(
    session: AsyncSession,
    spec: WindowSpec,
    chat_ids: Sequence[int],
    *,
    per_chat: int = STATS_SAMPLE_PER_CHAT,
    titles: dict[int, str] | None = None,
) -> list[S.Hit]:
    """Top chatlardan eng yangi `per_chat` ta xabar — faqat **namuna** (sitata uchun)."""
    ids = [int(c) for c in chat_ids if c]
    if not ids or per_chat <= 0:
        return []
    rn = (
        func.row_number()
        .over(partition_by=Message.chat_id, order_by=Message.published_at.desc())
        .label("rn")
    )
    inner = (
        select(Message.id, rn)
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_conds(spec), Message.chat_id.in_(ids))
        .subquery()
    )
    stmt = (
        select(Message)
        .join(inner, inner.c.id == Message.id)
        .where(inner.c.rn <= per_chat)
        .order_by(Message.published_at.desc())
        .limit(per_chat * len(ids))
    )
    names = dict(titles or {})
    rows = (await session.execute(stmt)).scalars()
    return [_hit(m, names.get(m.chat_id, "")) for m in rows]


def render_window_stats(
    count: WindowCount,
    *,
    titles: dict[int, str] | None = None,
    evidence: Sequence[S.Hit] = (),
    taken: int = 0,
    stats: WindowStats | None = None,
    spec: WindowSpec | None = None,
) -> str:
    """Sanoqlar **kodda** hisoblanadi — briefing bilan bir qoida.

    `stats` berilsa (sanoq yo'li) — yo'nalish, har chat/qabul qiluvchi
    bo'yicha **to'liq** reyting (ulush %, birinchi/oxirgi xabar), boshqalar
    bo'yicha yuboruvchi reytingi; dalil satrlari u holda faqat namuna.

    "Qaysi chatlar eng faol edi" degan savolning javobi `count(*)` dan
    keladi, dalil satrlaridan emas: model ko'rgan ≤1000 satr oynaning
    **hammasi** bo'lmasligi mumkin va u shunda "hammasi bitta chatda edi"
    deb yozardi (jonli regressiya). Blok yagona `kind="evidence"`
    konvertining ichiga tushadi (`build_evidence_envelope(claims_block=…)`),
    chat sarlavhalari `untrusted.attr` dan o'tgan.
    """
    names = dict(titles or {})
    if stats is not None:
        return _render_stats(count, stats, names=names, spec=spec, samples=len(evidence))
    lines = [
        "## window totals (counted in code, exact — trust these over the lines below)",
        f"period: {_period_text(spec)}",
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


_DIRECTION_TEXT = {
    "out": "messages the owner SENT (messages.out = true); per chat = per recipient",
    "in": (
        "messages written TO the owner (not the owner's; private chat or a reply to "
        "the owner's message)"
    ),
    "both": (
        "all messages (sent and received) in chats where the owner wrote at least once "
        "in this window; the ranking below is by MUTUAL conversation "
        "(min(sent, received)), not by raw message count, so a chat with many messages "
        "but only a one-sided reply does not outrank an actual back-and-forth"
    ),
    "": "all messages in the window (no direction filter)",
}


def _day(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "—"


def _period_text(spec: WindowSpec | None) -> str:
    """`period: <since → until | butun davr>` — model o'qigan aniq oraliq.

    Sintez prompti javobning **birinchi gapida** shu oraliqni aytishni talab
    qiladi (`STATS_SYNTH_NOTE`/`WINDOW_SYNTH_NOTE`): oyna savoldan kelmagan
    bo'lsa (`window_from_question=False`), model "eng faol chat" kabi
    javobni qaysi davr bo'yicha berganini aytmay qo'yardi (jonli regressiya,
    `daily-04`).
    """
    if spec is None:
        return "butun davr"
    until = _day(spec.until)
    if spec.since is None:
        return f"butun davr … → {until}"
    return f"{_day(spec.since)} → {until}"


def _render_stats(
    count: WindowCount,
    stats: WindowStats,
    *,
    names: dict[int, str],
    spec: WindowSpec | None,
    samples: int,
) -> str:
    """Sanoq yo'lining bloki — reyting jadvali tayyor, model faqat o'qiydi."""
    total = max(count.total, 0)
    window = ""
    if spec is not None:
        window = f"{_day(spec.since) if spec.since else '…'} → {_day(spec.until)}"
    lines = [
        "## window stats (counted in code with count(*) over EVERY message in the window "
        "— exact and complete; use these numbers verbatim, do not recount)",
        f"period: {_period_text(spec)}",
        f"direction: {_DIRECTION_TEXT.get(stats.direction, _DIRECTION_TEXT[''])}",
        f"window: {window or '—'}; first message: {_day(count.first_at)}; "
        f"last message: {_day(count.last_at)}",
        f"total messages: {total}; chats: {count.chats}",
    ]
    if stats.direction != "out" and stats.mine_total:
        lines.append(f"of these sent by the owner: {stats.mine_total}")
    label = "recipient chat" if stats.direction == "out" else "chat"
    if stats.per_chat:
        if stats.direction == "both":
            lines.append(
                f"rank by chat (mutual conversation — ranked by min(sent, received), not "
                f"by total message count; chats where only one side ever wrote are left "
                f"out of this ranking even though their messages still count in the "
                f"window total; top {len(stats.per_chat)}):"
            )
        else:
            lines.append(f"rank by {label} (descending, top {len(stats.per_chat)}):")
        for i, row in enumerate(stats.per_chat, 1):
            title = U.attr(names.get(row.chat_id, "")) or f"chat {row.chat_id}"
            share = (100.0 * row.n / total) if total else 0.0
            kind = f" [{U.attr(row.chat_type)}]" if row.chat_type else ""
            if stats.direction == "both":
                mine = f"; sent by owner: {row.mine}, received: {row.theirs}"
            elif stats.direction not in ("out", "in") and row.mine:
                mine = f"; sent by owner: {row.mine}"
            else:
                mine = ""
            lines.append(
                f"{i}. {title}{kind}: {row.n} ({share:.1f}%){mine}; "
                f"{_day(row.first_at)} … {_day(row.last_at)}"
            )
        rest = count.chats - len(stats.per_chat)
        if rest > 0:
            shown = sum(r.n for r in stats.per_chat)
            lines.append(f"(+{rest} more chats with {max(0, total - shown)} messages)")
    if stats.per_sender:
        lines.append("rank by sender (messages from others, descending):")
        for i, (name, n) in enumerate(stats.per_sender, 1):
            share = (100.0 * n / total) if total else 0.0
            lines.append(f"{i}. {U.attr(name)}: {n} ({share:.1f}%)")
    lines.append(
        f"sample lines below: {samples} (newest ≤{STATS_SAMPLE_PER_CHAT} per top chat, "
        "illustration only — not a basis for counting)"
    )
    return "\n".join(lines)


#: Oyna yo'lida dedup faqat shundan uzun (normallashtirilgan) matnga
#: qo'llanadi — `cluster.dedup_simhash` ning qisqa-matn chegarasi bilan bir.
#: Nega: "rahmat" / "ok" ni o'nta odam yozgan bo'lsa, ular takror emas —
#: "kimlar yozgan" savolining aynan o'zi. Kross-post (vakansiya kanallari,
#: 09-20…09-25 oynada 32% aynan takror) esa har doim uzun.
WINDOW_DEDUP_MIN_CHARS = 24


def dedup_window(pool: list[S.Hit]) -> tuple[list[S.Hit], int]:
    """Aynan va deyarli bir xil **uzun** matnlarni bitta vakilga yig'adi — O(n).

    `scale_search.dedup_fast` (sha1 + banded simhash) — kaskad bilan bitta
    implementatsiya, kosinus yo'q. Vakil — tartibda birinchisi (pool vaqt
    bo'yicha kamayib keladi, ya'ni eng yangisi); qolganlari `also_seen_in`
    ga tushadi, manba yo'qolmaydi. Qisqa matnlar tegilmaydi. Sanoqlar
    (`count(*)`) bundan mustaqil — bu faqat dalil/map kirishini tejaydi.
    Qaytadi: `(qolganlari, tashlangan soni)`.
    """
    long_ = [h for h in pool if len(C.normalize(h.text or "")) >= WINDOW_DEDUP_MIN_CHARS]
    if len(long_) < 2:
        return pool, 0
    keep = {id(h) for h in SC.dedup_fast(long_)}
    out = [
        h for h in pool if id(h) in keep or len(C.normalize(h.text or "")) < WINDOW_DEDUP_MIN_CHARS
    ]
    return out, len(pool) - len(out)


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
        direction=getattr(plan, "direction", "") or ("in" if plan.inbox else ""),
    )
    outcome.window = spec

    async with session_factory() as session:
        count = await count_window(session, spec)
    outcome.window_count = count
    if getattr(plan, "window_stats", False):
        return await _run_stats(
            session_factory,
            spec=spec,
            count=count,
            outcome=outcome,
            mode=mode,
            plan=plan,
            taint=taint,
            events=events,
            titles=titles,
            account_id=account_id,
            started=started,
        )
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
    # Kross-post / takror: bir xil uzun matn bir necha chatda (vakansiya
    # kanallari) — filtr, map va sintez uni har safar qayta o'qirdi.
    pool, dup_dropped = dedup_window(pool)
    stats.dropped_duplicates = dup_dropped
    stats.candidates = len(pool)
    if dup_dropped:
        log.info("window.dedup", dropped=dup_dropped, kept=len(pool))
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


async def _run_stats(
    session_factory: DS.SessionFactory,
    *,
    spec: WindowSpec,
    count: WindowCount,
    outcome: SC.ScaleOutcome,
    mode: str,
    plan: DS.DeepPlan,
    taint: DS._Taint,
    events: DS.EventSink | None,
    titles: dict[int, str],
    account_id: int,
    started: float,
) -> SC.ScaleOutcome:
    """Sanoq yo'li: `count(*)` agregatlari + top chatlardan bir nechta namuna.

    Tortish, filtr, map/reduce **yo'q** — skan tokeni 0. Shuning uchun
    `exhaustive` da ham tasdiq so'ralmaydi (skan narxi `WINDOW_CONFIRM_MIN_USD`
    dan har doim past); `estimate` hodisasi ham chiqmaydi — baho kartasi
    faqat tasdiq kerak bo'lganda ma'noli.
    """
    stats = outcome.stats
    est = estimate_window(
        count,
        mode=mode,
        window_from_question=plan.window_from_question,
        spec=spec,
        filtered=False,
    )
    # Skan yo'q: faqat sintez to'lanadi, u ham qisqa (≤30 namuna satri).
    synth = SC._price_for(Task.DEEP)
    synth_usd = (4_000 * synth[0] + SC.EST_SYNTH_OUT_TOKENS * synth[1]) / 1_000_000
    est.breakdown = {"filter": 0.0, "map": 0.0, "synth": round(synth_usd, 6)}
    est.est_cost_usd = 0.0
    est.est_total_usd = round(synth_usd, 6)
    outcome.estimate = est
    stats.est_cost_usd = 0.0
    stats.est_total_usd = est.est_total_usd

    agg = WindowStats(direction=spec.direction)
    samples: list[S.Hit] = []
    if count.total:
        async with session_factory() as session:
            agg = await stats_window(session, spec)
        top_ids = [row.chat_id for row in agg.per_chat[:STATS_TOP_N]]
        async with session_factory() as session:
            samples = await stats_samples(session, spec, top_ids, titles=titles)
        if samples:
            async with session_factory() as session:
                samples = await DS._hydrate(session, samples, account_id=account_id)
            taint.add_hits(samples)
    samples.sort(key=lambda h: h.date or datetime.min.replace(tzinfo=UTC))
    outcome.pool = list(samples)
    outcome.evidence = list(samples)
    stats.pulled = count.total
    stats.candidates = len(samples)
    stats.kept = len(samples)
    stats.evidence = len(samples)
    stats.note_stop("filter", "complete")
    outcome.window_stats = render_window_stats(
        count, titles=titles, evidence=samples, stats=agg, spec=spec
    )
    await DS._emit(
        events,
        "stats",
        total=count.total,
        chats=count.chats,
        direction=spec.direction,
        top=len(agg.per_chat),
        senders=len(agg.per_sender),
        samples=len(samples),
    )
    log.info(
        "window.stats",
        mode=mode,
        direction=spec.direction,
        total=count.total,
        chats=count.chats,
        top=len(agg.per_chat),
        senders=len(agg.per_sender),
        samples=len(samples),
        elapsed_s=round(time.monotonic() - started, 2),
    )
    return outcome


__all__ = [
    "STATS_SAMPLE_PER_CHAT",
    "STATS_TOP_N",
    "WINDOW_CONFIRM_MIN_USD",
    "WINDOW_SOFT_MAX",
    "ChatStat",
    "WindowCount",
    "WindowSpec",
    "WindowStats",
    "count_window",
    "estimate_window",
    "pull_window",
    "run_window",
    "stats_samples",
    "stats_window",
    "window_chunks",
]
