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
* **Mavzusiz dayjest** ("bu hafta nima muhokama qilindi?"): oynada
  `WINDOW_MAP_MAX_MSGS` (400) dan ko'p xabar bo'lsa **chat bo'yicha
  namuna** olinadi — har chatga hajmiga mutanosib kvota (≥5, ≤60), chat
  ichida eng yangisidan (`stratified_quotas`). Aniq sanoqlar (`count(*)`,
  chat bo'yicha) baribir sanoq blokida — model haqiqiy hajmni biladi,
  qamrov esa "N xabar oynada · 400 tasi o'qildi (chat bo'yicha namuna) ·
  hisoblar aniq" deydi. Jonli o'lchov (conv 285): 12,461 xabarli oynada
  eng yangi 805 tasi o'qilib, map 16 bo'lakka (qopqoq) chiqqan edi, $0.127.
* **Mavzuli oyna**: `SOFT_MAX_EVIDENCE` (1000) gacha — hammasi, ko'p bo'lsa
  `(chat, kun)` kvotasi va arzon filtr tartiblaydi; map'ga eng yuqori
  baholangan ≤400 tasi kiradi.
* **"Menga yozgan"** (`inbox`) — sxemada bor narsa bilan ta'riflanadi:
  `messages.out = false` **va** (chat shaxsiy **yoki** xabar mening
  xabarimga javob). `@username` eslatmalari saqlanmaydi, shuning uchun ular
  qamrovga kirmaydi — buni UI ham, qamrov ham yashirmaydi.
* Map bo'laklari **chat bo'yicha to'planadi** (chatlar ketma-ket tiziladi va
  80 satrlik bo'laklarga kesiladi, har chat o'z `### chat` sarlavhasi bilan):
  ilgari bo'lak =
  (chat, yuboruvchi) edi va 16 bo'lak qopqog'i eng katta 16 guruhdan
  tashqarisini map'ga umuman yetkazmasdi. Yuboruvchi har satrda turadi.

Invariantlar o'zgarmadi: dalillar bitta `<untrusted_data kind="evidence">`
konvertida (`deep_search.build_evidence_envelope`), `plan`/`reflect` bu
yerda umuman chaqirilmaydi, telefon raqami hech qayerda yo'q.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import Select, and_, case, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import Chat, Message, Sender
from app.llm import Task
from app.services import cluster as C
from app.services import deep_search as DS
from app.services import scale_search as SC
from app.services import search as S
from app.services import senders as SN
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

#: Map bo'lagi — ≤80 satr, butun chatlar to'planadi (katta chat bo'linadi).
WINDOW_CHUNK_MSGS = SC.MAP_CHUNK_MSGS
WINDOW_MAX_CHUNKS = SC.MAP_MAX_CHUNKS

#: Kvota yo'lida filtr faqat savolda mavzu bo'lganda chaqiriladi.
WINDOW_FILTER_MIN_TERMS = 1

#: Oyna yo'lida map'ga kiradigan xabarlar qopqog'i (≈5 bo'lak x 80). Jonli
#: o'lchov (conv 285, "bu hafta nima muhokama qilindi?", 12,461 xabar):
#: 805 xabar o'qildi, map 16 bo'lak (qopqoq), 17,949 chiqish tokeni, $0.127.
#: Mavzusiz dayjestda javob chat bo'yicha 1-3 satr — 400 ta namuna
#: (chat bo'yicha) va aniq sanoqlar yetadi; mavzuli oynada — eng yuqori
#: baholangan 400 tasi.
WINDOW_MAP_MAX_MSGS = 400
#: Chat bo'yicha namunada bitta chat kvotasi chegaralari: jim chat ham
#: ko'rinsin (≥5), gapiradigan chat namunani yeb qo'ymasin (≤60).
WINDOW_SAMPLE_MIN_PER_CHAT = 5
WINDOW_SAMPLE_MAX_PER_CHAT = 60
#: Baho uchun o'rtacha ixcham satr (`[#c…m…] HH:MM kim: matn`, matn o'rtacha
#: ~118 belgi, `MAP_LINE_CHARS` da kesiladi) — belgida.
WINDOW_EST_LINE_CHARS = 180
#: Baho: sanoq bloki (chat bo'yicha top 20 + yuboruvchilar) — tokenda.
WINDOW_EST_STATS_TOKENS = 600
#: Baho: bitta map da'vosi sintez kirishida — tokenda.
WINDOW_EST_CLAIM_TOKENS = 40


def stratified_quotas(
    per_chat: Sequence[tuple[int, int]],
    cap: int = WINDOW_MAP_MAX_MSGS,
    *,
    lo: int = WINDOW_SAMPLE_MIN_PER_CHAT,
    hi: int = WINDOW_SAMPLE_MAX_PER_CHAT,
) -> dict[int, int]:
    """Chat bo'yicha namuna kvotasi: hajmga mutanosib, har chatga `[lo, hi]`.

    1) hajm tartibida har chatga `min(n, lo)` (qopqoq yetguncha — chat juda
    ko'p bo'lsa kichiklari tushib qoladi); 2) qolgani hajmga mutanosib
    "suv to'ldirish" bilan, har chat `min(n, hi)` gacha; butun sonli
    qoldiq — hajm tartibida bittadan. Deterministik, jami ≤ `cap`.
    """
    rows = sorted(((int(c), int(n)) for c, n in per_chat if n > 0), key=lambda p: (-p[1], p[0]))
    quota: dict[int, int] = {}
    left = max(0, int(cap))
    for chat_id, n in rows:
        take = min(n, lo, left)
        if take <= 0:
            break
        quota[chat_id] = take
        left -= take
    while left > 0:
        room = {c: min(n, hi) - quota[c] for c, n in rows if c in quota and min(n, hi) > quota[c]}
        if not room:
            break
        weight = sum(n for c, n in rows if c in room)
        given = 0
        for chat_id, n in rows:
            if chat_id in room:
                add = min(room[chat_id], (left * n) // weight)
                quota[chat_id] += add
                given += add
        left -= given
        if given == 0:
            for chat_id, _n in rows:
                if left <= 0:
                    break
                if chat_id in room:
                    quota[chat_id] += 1
                    left -= 1
    return quota


def stratify_order(hits: Sequence[S.Hit]) -> list[S.Hit]:
    """Chatlar bo'ylab navbatma-navbat (eng yangisidan) — sintez satrlari tartibi.

    Sintezga `SYNTH_EVIDENCE_LINES` (150) satr ketadi; ilgari bu vaqt
    bo'yicha **eng eski** 150 tasi edi. Endi har chatdan navbat bilan eng
    yangisi olinadi (chat hajm tartibida), ya'ni kesim hamma chatni qamraydi.
    Konvertda chat ichidagi satrlar baribir vaqt tartibida chiqadi.
    """
    groups: dict[int, list[S.Hit]] = {}
    for h in hits:
        groups.setdefault(h.chat_id, []).append(h)
    epoch = datetime.min.replace(tzinfo=UTC)
    queues = [
        sorted(rows, key=lambda h: h.date or epoch, reverse=True)
        for rows in sorted(groups.values(), key=lambda r: -len(r))
    ]
    out: list[S.Hit] = []
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


# ─── media filtri ────────────────────────────────────────────────────────────
#: `args.media` → `messages.media_type` qiymatlari. `ingestion._media_type`
#: Telethon sinf nomidan yozadi: `MessageMediaPhoto` → `photo`,
#: `MessageMediaDocument` → `document`, `MessageMediaWebPage` → `webpage`, …
#: Telegram video, ovozli xabar, audio, stiker va fayllarni **bitta**
#: `MessageMediaDocument` bilan beradi — sxemada ular ajralmaydi, shuning uchun
#: `video` / `voice` / `document` hammasi `document` ga tushadi (qamrov va
#: sintez izohi buni ochiq aytadi). `any` — har qanday media, havola
#: ko'rinishi (`webpage`) bundan mustasno.
MEDIA_KINDS = ("photo", "video", "document", "voice", "any")
MEDIA_VALUES: dict[str, tuple[str, ...]] = {
    "photo": ("photo",),
    "video": ("document",),
    "document": ("document",),
    "voice": ("document",),
}
MEDIA_ANY_EXCLUDE = ("webpage",)
#: Savolda media turi + "yuborish" fe'li — kod darvozasi (planner arg bermasa).
_MEDIA_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("photo", re.compile(r"\brasm\w*|\bsurat\w*|\bfoto\w*|\bphoto\w*|\bpictures?\b|\bimages?\b"
                         r"|\bpics?\b|фото\w*|картин\w*|снимк\w*|снимок")),
    ("video", re.compile(r"\bvideo\w*|\bvidyo\w*|видео\w*|\bklip\w*")),
    ("voice", re.compile(r"\bovozli\b|\bvoice\b|\baudio\w*|голосов\w*|аудио\w*")),
    ("document", re.compile(r"\bfayl\w*|\bhujjat\w*|\bdocuments?\b|\bfiles?\b|\bpdf\b"
                            r"|файл\w*|документ\w*")),
    ("any", re.compile(r"\bmedia\w*|медиа\w*")),
)  # fmt: skip
_MEDIA_SEND_RE = re.compile(
    r"\byubor\w*|\btashla\w*|\bjo'nat\w*|\bjonat\w*|\bulash\w*|\bjo'natil\w*"
    r"|отправ\w*|присла\w*|присыла\w*|скину\w*|скидыва\w*|кида\w*"
    r"|\bsent\b|\bsend\w*|\bshared?\b|\bposted\b"
)


def normalize_media(value: Any) -> str:
    """Planner argumenti → `MEDIA_KINDS` dan biri yoki `""` (filtr yo'q)."""
    kind = str(value or "").strip().lower()
    aliases = {"image": "photo", "picture": "photo", "file": "document", "audio": "voice"}
    kind = aliases.get(kind, kind)
    return kind if kind in MEDIA_KINDS else ""


def detect_media(question: str) -> str:
    """ "kim rasm yubordi" / "who sent photos" — media turi (kodda) yoki `""`.

    Ataylab tor: media so'zi **va** yuborish fe'li bo'lishi shart —
    "rasm haqida nima deyishdi" mavzu savoli, media filtri emas.
    """
    text = TW.fold(question or "")
    if not _MEDIA_SEND_RE.search(text):
        return ""
    for kind, pattern in _MEDIA_WORDS:
        if pattern.search(text):
            return kind
    return ""


def media_line(kind: str) -> str:
    """Oyna blokidagi media filtri satri — nima sanalgani va nima **ajralmasligi**."""
    values = ", ".join(MEDIA_VALUES.get(kind, ())) or f"any except {', '.join(MEDIA_ANY_EXCLUDE)}"
    merged = (
        " (the archive stores video, voice, audio and files all as `document` — they "
        "cannot be told apart)"
        if kind in ("video", "voice", "document")
        else ""
    )
    return (
        f"media filter: only messages with media ({kind}; messages.media_type = {values}){merged}"
        "; the counts and lines above/below are media messages only; the file itself is "
        "never stored, so what a picture shows is unknown"
    )


def media_cond(kind: str) -> Any | None:
    """`WindowSpec.media` → SQL sharti (`messages.media_type`)."""
    if not kind:
        return None
    if kind == "any":
        return and_(
            Message.media_type.is_not(None), Message.media_type.not_in(list(MEDIA_ANY_EXCLUDE))
        )
    return Message.media_type.in_(list(MEDIA_VALUES.get(kind, (kind,))))


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
    #: `MEDIA_KINDS` dan biri yoki `""` — faqat shu turdagi media xabarlar.
    media: str = ""

    def __post_init__(self) -> None:
        if self.inbox and not self.direction:
            self.direction = "in"
        self.inbox = self.direction == "in"
        self.media = normalize_media(self.media)

    def to_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat(),
            "inbox": self.inbox,
            "direction": self.direction,
            "chats": len(self.chats) if self.chats else 0,
            "tz": self.tz,
            "media": self.media,
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
    media = media_cond(spec.media)
    if media is not None:
        conds.append(media)
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
        media_type=str(getattr(m, "media_type", None) or ""),
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


async def pull_window_sample(
    session: AsyncSession,
    spec: WindowSpec,
    *,
    count: WindowCount,
    cap: int = WINDOW_MAP_MAX_MSGS,
    titles: dict[int, str] | None = None,
) -> list[S.Hit]:
    """Mavzusiz dayjest uchun chat bo'yicha namuna — `stratified_quotas` bilan.

    SQL: chat bo'yicha `row_number()` (eng yangisi birinchi) va har chatning
    o'z kvotasi (`CASE chat_id WHEN … THEN q`). Kvota Python'da ham qayta
    qo'llanadi — SQL shakli o'zgarsa ham namuna qopqoqdan oshmasin.
    """
    quotas = stratified_quotas(count.per_chat, cap)
    if not quotas:
        return []
    rn = (
        func.row_number()
        .over(partition_by=Message.chat_id, order_by=Message.published_at.desc())
        .label("rn")
    )
    inner = (
        select(Message.id, Message.chat_id.label("chat_id"), rn)
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_conds(spec), Message.chat_id.in_(list(quotas)))
        .subquery()
    )
    limit_of = case(quotas, value=inner.c.chat_id, else_=0)
    stmt = (
        select(Message)
        .join(inner, inner.c.id == Message.id)
        .where(inner.c.rn <= limit_of)
        .order_by(Message.published_at.desc())
        .limit(cap)
    )
    names = dict(titles or {})
    used: dict[int, int] = {}
    hits: list[S.Hit] = []
    for m in (await session.execute(stmt)).scalars():
        if used.get(m.chat_id, 0) >= quotas.get(m.chat_id, 0):
            continue
        used[m.chat_id] = used.get(m.chat_id, 0) + 1
        hits.append(_hit(m, names.get(m.chat_id, "")))
        if len(hits) >= cap:
            break
    log.info(
        "window.sample",
        total=count.total,
        chats=len(quotas),
        taken=len(hits),
        max_quota=max(quotas.values()),
    )
    return hits


def estimate_window(
    count: WindowCount,
    *,
    mode: str = "deep",
    cap: int = WINDOW_SOFT_MAX,
    window_from_question: bool = True,
    spec: WindowSpec | None = None,
    filtered: bool = False,
    digest: bool = True,
) -> SC.Estimate:
    """Oyna bahosi — **aniq sanoqdan**, bucket taxminidan emas.

    Skan qismi: kesilgan holatda arzon filtr (aks holda 0) + map. Sintez
    alohida (`est_total_usd`), `scale_search.estimate` bilan bir uslubda.
    Qopqoqlar **yugurish bilan bir xil**: map kirishi ≤ `WINDOW_MAP_MAX_MSGS`
    (mavzusiz katta oynada chat bo'yicha namuna), bo'laklar to'planadi
    (`window_chunks`), sintez satrlari ≤ `DS.DIGEST_SYNTH_LINES` (`digest=True`,
    ya'ni "batafsil" so'ralmagan window/stats turn'i) yoki `SYNTH_EVIDENCE_LINES`
    (`digest=False` — "batafsil" so'ralgan), sintez chiqishi dayjest byudjeti
    bilan (`deep_search.SYNTH_BUDGET_DIGEST`).
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

    map_msgs = min(taken, WINDOW_MAP_MAX_MSGS)
    chunks = (
        0
        if map_msgs <= SC.MAP_SKIP_TOTAL
        else min(WINDOW_MAX_CHUNKS, math.ceil(map_msgs / WINDOW_CHUNK_MSGS))
    )
    map_in = (
        chunks * int(len(SCALE_MAP_PROMPT) / S.CHARS_PER_TOKEN)
        + int(map_msgs * WINDOW_EST_LINE_CHARS / S.CHARS_PER_TOKEN)
        if chunks
        else 0
    )
    map_out = chunks * SC.EST_MAP_OUT_TOKENS

    synth_cap = DS.DIGEST_SYNTH_LINES if digest else SC.SYNTH_EVIDENCE_LINES
    synth_lines = min(synth_cap, map_msgs)
    synth_in = (
        int(len(DEEP_SYNTH_PROMPT) / S.CHARS_PER_TOKEN)
        + int(synth_lines * WINDOW_EST_LINE_CHARS / S.CHARS_PER_TOKEN)
        + chunks * SC.MAP_MAX_CLAIMS * WINDOW_EST_CLAIM_TOKENS
        + WINDOW_EST_STATS_TOKENS
    )
    synth_out = min(SC.EST_SYNTH_OUT_TOKENS, DS.SYNTH_BUDGET_DIGEST)

    def cost(stage: str, tin: int, tout: int) -> float:
        price_in, price_out = prices[stage]
        return (tin * price_in + tout * price_out) / 1_000_000

    breakdown = {
        "filter": round(cost("filter", filter_in, filter_out), 6),
        "map": round(cost("map", map_in, map_out), 6),
        "synth": round(cost("synth", synth_in, synth_out), 6),
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
    #: `(soat 0..23, son)` — akkaunt mintaqasida (`spec.tz`), faqat so'ralganda.
    by_hour: list[tuple[int, int]] = field(default_factory=list)
    #: `(ISO hafta kuni 1=Du … 7=Ya, son)` — akkaunt mintaqasida, faqat so'ralganda.
    by_weekday: list[tuple[int, int]] = field(default_factory=list)


#: `stats(by=…)` — soat / hafta kuni taqsimoti (SQL `extract`, LLM yo'q).
DISTRIBUTIONS = ("hour", "weekday")
WEEKDAY_NAMES = ("", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


async def distribution(
    session: AsyncSession, spec: WindowSpec, *, by: str
) -> list[tuple[int, int]]:
    """Oynadagi xabarlar soat yoki hafta kuni bo'yicha — `count(*)`, mahalliy vaqtda.

    `timezone(tz, published_at)` — `timestamptz` ni akkaunt mintaqasidagi
    "devor soati"ga o'giradi: Toshkentda 23:30 da yozilgan xabar UTC'da
    18:30 bo'lsa ham 23-soatga tushadi. Bo'sh bucket'lar qaytmaydi.
    """
    local = func.timezone(spec.tz or TW.DEFAULT_TZ, Message.published_at)
    part = func.extract("hour" if by == "hour" else "isodow", local)
    stmt = (
        select(part.label("k"), func.count().label("n"))
        .select_from(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_conds(spec))
        .group_by(part)
        .order_by(part)
    )
    return [(int(k), int(n)) for k, n in (await session.execute(stmt)).all()]


async def stats_window(
    session: AsyncSession,
    spec: WindowSpec,
    *,
    top: int = STATS_TOP_CHATS,
    locale: str = "uz",
    distributions: Sequence[str] = (),
) -> WindowStats:
    """Chat va yuboruvchi bo'yicha agregatlar — **xabar tortilmaydi**.

    `distributions` — `hour` / `weekday`: qo'shimcha `count(*) GROUP BY
    extract(...)` (faqat so'ralganda — har biri bitta arzon so'rov).
    """
    out = WindowStats(direction=spec.direction)
    if "hour" in distributions:
        out.by_hour = await distribution(session, spec, by="hour")
    if "weekday" in distributions:
        out.by_weekday = await distribution(session, spec, by="weekday")
    owner_tg_id, owner_display = await SN.owner_info(session, spec.account_id)
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
            # Himoya qatlami: so'rov `out=False` bilan egasini chiqarib
            # tashlaydi, lekin eski qatorlarda `out` ishonchsiz bo'lishi
            # mumkin (CLAUDE.md) — `sender_id` to'g'ridan-to'g'ri
            # `tg_account_id` bilan solishtiriladi.
            if SN.is_owner_message(sender_id=sender_id, out=False, owner_tg_id=owner_tg_id):
                label = SN.owner_label(locale, display_name=owner_display)
            else:
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
    sampled: bool = False,
) -> str:
    """Sanoqlar **kodda** hisoblanadi — briefing bilan bir qoida.

    `sampled` — dalil chat bo'yicha namuna (`pull_window_sample`): blok buni
    aytadi, model esa hajmni satrlardan emas, shu sanoqlardan oladi.

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
    if spec is not None and spec.media:
        lines.append(media_line(spec.media))
    if sampled:
        lines.append(
            "evidence is a per-chat sample (share of each chat proportional to its volume, "
            "newest first within a chat) — the counts here are exact, the lines are not all "
            "messages; describe volume from these counts"
        )
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


def _day(dt: datetime | None, tz: str = "", *, end: bool = False) -> str:
    """Sana **akkaunt mintaqasida** (`tz`), UTC emas.

    Jonli uni-12: "oxirgi 7 kun" oynasi mahalliy yarim tundan boshlanadi
    (Toshkentda 19:00 UTC), UTC sanasi esa bir kun oldinni ko'rsatardi.
    `end=True` — eksklyuziv yuqori chegara: aniq yarim tun bo'lsa oldingi kun
    ko'rsatiladi ("o'tgan hafta" … → yakshanba, dushanba emas).
    """
    if dt is None:
        return "—"
    local = dt.astimezone(TW.zone(tz)) if tz and dt.tzinfo else dt
    if end and (local.hour, local.minute, local.second, local.microsecond) == (0, 0, 0, 0):
        local = local - timedelta(microseconds=1)
    return local.strftime("%Y-%m-%d")


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
    until = _day(spec.until, spec.tz, end=True)
    zone = f" ({TW.zone(spec.tz)})"
    if spec.since is None:
        return f"butun davr … → {until}{zone}"
    return f"{_day(spec.since, spec.tz)} → {until}{zone}"


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
    tz = spec.tz if spec is not None else ""
    if spec is not None:
        window = f"{_day(spec.since, tz) if spec.since else '…'} → {_day(spec.until, tz, end=True)}"
    lines = [
        "## window stats (counted in code with count(*) over EVERY message in the window "
        "— exact and complete; use these numbers verbatim, do not recount)",
        f"period: {_period_text(spec)}",
        f"direction: {_DIRECTION_TEXT.get(stats.direction, _DIRECTION_TEXT[''])}",
        f"window: {window or '—'}; first message: {_day(count.first_at, tz)}; "
        f"last message: {_day(count.last_at, tz)}",
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
                f"{_day(row.first_at, tz)} … {_day(row.last_at, tz)}"
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
    tz = (spec.tz if spec is not None else "") or TW.DEFAULT_TZ
    if stats.by_weekday:
        lines.append(f"messages by weekday (local time, {tz}; same direction filter):")
        lines.append(
            "; ".join(f"{WEEKDAY_NAMES[d]} {n}" for d, n in stats.by_weekday if 1 <= d <= 7)
        )
    if stats.by_hour:
        lines.append(f"messages by hour of day (local time, {tz}; HH = HH:00-HH:59):")
        lines.append("; ".join(f"{h:02d}h {n}" for h, n in stats.by_hour))
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
    max_msgs: int = WINDOW_MAP_MAX_MSGS,
) -> list[list[S.Hit]]:
    """Bo'laklar **chat bo'yicha to'planadi**: chatlar ketma-ket, bo'lak to'liq.

    Kirish — ustuvorlik tartibida (namuna / filtr reytingi); birinchi
    `max_msgs` tasi olinadi. Chatlar hajm tartibida, chat ichida vaqt
    bo'yicha, bitta ketma-ketlikka tiziladi va `chunk_size` lik bo'laklarga
    kesiladi — bo'lak soni `ceil(n / chunk_size)` (baho ham shunday
    hisoblaydi). Chat bo'lak chegarasida bo'linishi mumkin: `build_map_envelope`
    har bo'lakda har chat oldiga `### chat` sarlavhasini qo'yadi, yuboruvchi
    esa har satrda — "Nurbek — 12 xabar, ijara haqida" shakli uchun yetadi.

    Nega (chat, yuboruvchi) emas: 400 xabarlik namunada bunday guruhlar
    yuzlab bo'lardi va `max_chunks` (16) eng kattalaridan tashqarisini map'ga
    umuman yetkazmasdi, har kichik bo'lak esa to'liq map prompt'ini to'lardi.
    """
    if len(evidence) <= max(0, skip_below):
        return []
    epoch = datetime.min.replace(tzinfo=UTC)
    groups: dict[int, list[S.Hit]] = {}
    for hit in list(evidence)[: max(0, max_msgs)]:
        groups.setdefault(hit.chat_id, []).append(hit)
    ordered = [
        h
        for rows in sorted(groups.values(), key=lambda r: -len(r))
        for h in sorted(rows, key=lambda h: h.date or epoch)
    ]
    size = max(1, chunk_size)
    return [ordered[i : i + size] for i in range(0, len(ordered), size)][:max_chunks]


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
    distributions: Sequence[str] = (),
    media: str = "",
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
        media=media,
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
            locale=locale,
            distributions=distributions,
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

    digest = (
        DS.synth_budget(question, follow_up=plan.intent_follow_up, window=True).kind == "digest"
    )
    est = estimate_window(
        count,
        mode=mode,
        window_from_question=plan.window_from_question,
        spec=spec,
        filtered=will_filter,
        digest=digest,
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
    # Mavzusiz katta oyna — chat bo'yicha namuna (reja shu, kesish emas:
    # `truncated` yo'q, qamrov "namuna · hisoblar aniq" deydi). Mavzuli oyna
    # esa avvalgidek: hammasi yoki `(chat, kun)` kvotasi + filtr.
    sampled = not will_filter and count.total > WINDOW_MAP_MAX_MSGS
    async with session_factory() as session:
        if sampled:
            pool = await pull_window_sample(session, spec, count=count, titles=titles)
            quota_cut = False
        else:
            pool, quota_cut = await pull_window(session, spec, count=count, titles=titles)
    outcome.window_sampled = sampled
    stats.buckets_scanned = stats.buckets_planned
    stats.pulled = count.total
    stats.elbow = len(pool)
    stats.candidates = len(pool)
    stats.pool_capped = quota_cut
    if quota_cut:
        stats.truncated = True
        stats.note_stop("pull", "yield")
    elif sampled:
        stats.note_stop("pull", "complete")
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
        owner_tg_id, owner_display = await SN.owner_info(session, account_id)
    # Hydratsiya (`deep_search._hydrate`) egasini ham `user:<id>` deb
    # yozadi — bu yerda render vaqtida "Men" bilan almashtiriladi (single
    # source of truth: `services.senders`).
    SN.relabel_owner_hits(
        pool, owner_tg_id=owner_tg_id, owner_display_name=owner_display, locale=locale
    )
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

    # Map kirishi — ustuvorlik tartibida (filtr bali / namuna), ≤400.
    map_input = evidence[:WINDOW_MAP_MAX_MSGS]
    # Sintezga ketadigan birinchi `SYNTH_EVIDENCE_LINES` satr chatlar bo'ylab
    # navbat bilan (eng yangisidan) — ilgari vaqt bo'yicha eng eski 150 tasi
    # edi. Konvert chat ichida baribir vaqt tartibida chiqaradi.
    evidence = stratify_order(evidence)
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

    # ── (d) map → da'volar (chatlar to'plangan bo'laklar) ────────────────────
    chunks = window_chunks(map_input)
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
        count,
        titles=titles,
        evidence=evidence,
        taken=len(evidence),
        sampled=sampled,
        spec=spec,
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
        sampled=sampled,
        map_msgs=sum(len(c) for c in chunks),
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
    locale: str = "uz",
    distributions: Sequence[str] = (),
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
            agg = await stats_window(session, spec, locale=locale, distributions=distributions)
        top_ids = [row.chat_id for row in agg.per_chat[:STATS_TOP_N]]
        async with session_factory() as session:
            samples = await stats_samples(session, spec, top_ids, titles=titles)
        if samples:
            async with session_factory() as session:
                samples = await DS._hydrate(session, samples, account_id=account_id)
                owner_tg_id, owner_display = await SN.owner_info(session, account_id)
            SN.relabel_owner_hits(
                samples, owner_tg_id=owner_tg_id, owner_display_name=owner_display, locale=locale
            )
            taint.add_hits(samples)
    samples.sort(key=lambda h: h.date or datetime.min.replace(tzinfo=UTC))
    outcome.window_agg = agg
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
    "WINDOW_MAP_MAX_MSGS",
    "WINDOW_SOFT_MAX",
    "ChatStat",
    "WindowCount",
    "WindowSpec",
    "WindowStats",
    "count_window",
    "estimate_window",
    "pull_window",
    "pull_window_sample",
    "run_window",
    "stats_samples",
    "stats_window",
    "stratified_quotas",
    "stratify_order",
    "window_chunks",
]
