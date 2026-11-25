"""Epizod segmentatsiyasi — LLM'siz, deterministik (arxitektura v2, 5.2-band).

Nega kerak: "ha, roziman" xabari o'zicha embed qilinsa hech nimaga mos
kelmaydi. Retrieval birligi bitta xabar emas, **epizod** bo'lishi kerak —
bir necha daqiqa/soat ichidagi bitta suhbat bo'lagi. Segmentatsiya sof kod
bilan qilinadi (3-tamoyil: deterministik ish — koddan, sub'ektiv ish —
modeldan): model chaqirilmaydi, natija bir xil kirishda har doim bir xil.

Chegara qo'yiladigan uch shart (``tg_msg_id`` tartibida yurilib):

  * oldingi xabardan **gap > 45 daqiqa** (kanal uchun 6 soat — kanalda post
    oqimi tabiiy siyrak, 45 daqiqa har postni alohida epizod qilib yuborardi);
  * joriy epizodda **≥40 xabar**;
  * joriy epizodda **≥6000 belgi**.

Qat'iy istisno: **reply-zanjir hech qachon bo'linmaydi.** Avval
``reply_to_msg_id`` bo'yicha union-find quriladi, keyin chegara faqat
komponentlar **orasidagi** joyga tushadi. Shuning uchun qopqoqlardan katta
komponent bitta epizod bo'lib qoladi — bu ataylab: yarim savolli, yarim
javobli epizod retrieval uchun qopqoqdan ham yomon.

Maqsad — median 8-15 xabar (``tests/test_episodes.py`` 500 xabarlik sintetik
oqimda [5, 25] oralig'ini qo'riqlaydi).

Epizod matni ``digests._line`` formatida: digest va retrieval **bitta**
render yo'lidan foydalanadi, ikkinchisini sinxron saqlashga hojat yo'q.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, ChatType, Message, RetrievalNode, RetrievalNodeEmbedding
from app.logging import get_logger
from app.services.textnorm import simhash64

log = get_logger(__name__)

# ── segmentatsiya qopqoqlari (v2, 5.2) ───────────────────────────────────────
GAP_SECONDS = 45 * 60
CHANNEL_GAP_SECONDS = 6 * 3600
MAX_EPISODE_MSGS = 40
MAX_EPISODE_CHARS = 6_000

#: `ctx_prefix` dagi so'zlovchilar ro'yxati qopqog'i — prefiks embeddingga
#: kiradi, uzun ro'yxat esa mazmunni suyultiradi.
MAX_PREFIX_SPEAKERS = 6
#: "replying to X: …" qismidagi sitata uzunligi (v2, 5.3-bandidagi namuna).
REPLY_SNIPPET_CHARS = 60

#: Bitta qurishda o'qiladigan xabarlar qopqog'i — `full=True` da ham worker
#: bitta chatda cheksiz osilib qolmasin.
MAX_MESSAGES_PER_BUILD = 50_000


@dataclass(slots=True)
class Episode:
    """Bitta epizod — DB'ga yozilishidan oldingi sof qiymat."""

    starts_at: datetime
    ends_at: datetime
    msg_lo: int
    msg_hi: int
    msg_count: int
    speakers: list[int]
    text: str
    ctx_prefix: str
    content_hash: str
    simhash: int  # belgili bigint (`to_signed64`)


@dataclass(slots=True)
class EpisodeReport:
    chat_id: int
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    messages: int = 0
    episodes: list[Episode] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return self.created + self.updated + self.deleted


# ─── sof funksiyalar ─────────────────────────────────────────────────────────


def _md5(value: str) -> str:
    return hashlib.md5(value.encode(), usedforsecurity=False).hexdigest()


def to_signed64(value: int) -> int:
    """Ishorasiz 64-bit → Postgres `bigint` (belgili) oralig'iga."""
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= (1 << 63) else value


def _gap_limit(chat: Chat) -> int:
    return CHANNEL_GAP_SECONDS if chat.type == ChatType.CHANNEL else GAP_SECONDS


def reply_components(msgs: Sequence[Message]) -> list[int]:
    """Union-find `reply_to_msg_id` bo'yicha → har xabar uchun komponent ildizi.

    Faqat **ro'yxat ichidagi** javoblar birlashtiriladi: oynadan tashqaridagi
    ota-xabar epizodni kengaytirmaydi (aks holda bitta eski postga javob butun
    tarixni bitta epizodga yopishtirardi).

    Args:
        msgs: `tg_msg_id` bo'yicha tartiblangan xabarlar.

    Returns:
        Indekslar ro'yxati: `roots[i]` — `i`-xabar komponentining ildiz indeksi.
    """
    parent = list(range(len(msgs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_msg_id = {int(m.tg_msg_id): i for i, m in enumerate(msgs)}
    for i, m in enumerate(msgs):
        parent_id = getattr(m, "reply_to_msg_id", None)
        if parent_id is None:
            continue
        j = by_msg_id.get(int(parent_id))
        if j is not None:
            union(i, j)
    return [find(i) for i in range(len(msgs))]


def allowed_cuts(msgs: Sequence[Message]) -> set[int]:
    """Chegara qo'yish mumkin bo'lgan indekslar (`i` — `i-1` va `i` orasi).

    Komponent `[lo, hi]` oralig'ini egallaydi; `lo < i <= hi` bo'lgan har
    qanday `i` **taqiqlanadi** — aks holda reply-zanjir ikkiga bo'linardi.
    """
    roots = reply_components(msgs)
    span: dict[int, tuple[int, int]] = {}
    for i, r in enumerate(roots):
        lo, hi = span.get(r, (i, i))
        span[r] = (min(lo, i), max(hi, i))
    blocked: set[int] = set()
    for lo, hi in span.values():
        blocked.update(range(lo + 1, hi + 1))
    return {i for i in range(1, len(msgs)) if i not in blocked}


def segment(msgs: Sequence[Message], *, gap_seconds: int) -> list[list[Message]]:
    """Xabarlar oqimini epizodlarga bo'ladi (sof, DB'siz).

    Args:
        msgs: `tg_msg_id` bo'yicha o'sish tartibida, o'chirilganlarsiz.
        gap_seconds: vaqt chegarasi (chat uchun 45 daqiqa, kanal uchun 6 soat).

    Returns:
        Epizodlar — har biri bo'sh bo'lmagan xabarlar ro'yxati.
    """
    if not msgs:
        return []
    cuts = allowed_cuts(msgs)
    out: list[list[Message]] = []
    start = 0
    count = 1
    chars = len(msgs[0].text or "")
    for i in range(1, len(msgs)):
        prev, cur = msgs[i - 1], msgs[i]
        gap = _seconds_between(prev.published_at, cur.published_at)
        too_big = count >= MAX_EPISODE_MSGS or chars >= MAX_EPISODE_CHARS
        if (gap > gap_seconds or too_big) and i in cuts:
            out.append(list(msgs[start:i]))
            start, count, chars = i, 0, 0
        count += 1
        chars += len(cur.text or "")
    out.append(list(msgs[start:]))
    return out


def _seconds_between(a: datetime | None, b: datetime | None) -> float:
    if a is None or b is None:
        return 0.0
    return abs((b - a).total_seconds())


def speaker_name(m: Message, chat: Chat) -> str:
    """`digests._line` bilan bir xil nom (kanalda — chat sarlavhasi)."""
    if chat.type == ChatType.CHANNEL and not m.sender_id:
        return chat.title
    return f"user:{m.sender_id}" if m.sender_id else "—"


def build_ctx_prefix(
    msgs: Sequence[Message], chat: Chat, *, reply_parent: Message | None = None
) -> str:
    """Metadatadan bepul olinadigan kontekst sarlavhasi (v2, 5.3-band).

    Format::

        [chat: Ish guruhi (supergroup) | 2026-03-14 | speakers: user:1, user:2 |
         replying to user:9: "to'lovni 20-sanaga suramizmi"]

    `replying to …` qismi **faqat** epizodning birinchi xabari epizoddan
    **tashqaridagi** xabarga javob bo'lganda qo'shiladi — ya'ni epizod o'zi
    tushuntira olmaydigan kontekst bo'lganda.
    """
    if not msgs:
        return ""
    day = msgs[0].published_at.date().isoformat() if msgs[0].published_at else "?"
    names: list[str] = []
    for m in msgs:
        name = speaker_name(m, chat)
        if name not in names:
            names.append(name)
        if len(names) >= MAX_PREFIX_SPEAKERS:
            break
    parts = [
        f"chat: {chat.title} ({chat.type})",
        day,
        "speakers: " + ", ".join(names),
    ]
    if reply_parent is not None:
        snippet = " ".join((reply_parent.text or "").split())[:REPLY_SNIPPET_CHARS]
        parts.append(f'replying to {speaker_name(reply_parent, chat)}: "{snippet}"')
    return "[" + " | ".join(parts) + "]"


def content_hash(msgs: Sequence[Message]) -> str:
    """`md5("<tg_msg_id>:<md5(text)>,…")` — qo'shilish, tahrir, o'chirishni sezadi.

    `chat_digests.src_fingerprint` bilan **bir xil shakl**: bitta 32 belgili
    hex satr. Faqat shu taqqoslash hal qiladi, tugunni qayta qurish keraksiz
    LLM/embedding xarajatiga olib kelmasin.
    """
    parts = [f"{int(m.tg_msg_id)}:{_md5(m.text or '')}" for m in msgs]
    return _md5(",".join(parts))


def episode_text(msgs: Sequence[Message], chat: Chat) -> str:
    """`digests._line` formatidagi qatorlar (yagona render yo'li)."""
    from app.services.digests import _line

    return "\n".join(_line(m, chat) for m in msgs)


def make_episode(
    msgs: Sequence[Message], chat: Chat, *, reply_parent: Message | None = None
) -> Episode:
    """Bir bo'lak xabarlardan `Episode` qiymati (DB'ga tegmaydi)."""
    text = episode_text(msgs, chat)
    speakers: list[int] = []
    for m in msgs:
        if m.sender_id and int(m.sender_id) not in speakers:
            speakers.append(int(m.sender_id))
    return Episode(
        starts_at=msgs[0].published_at,
        ends_at=msgs[-1].published_at,
        msg_lo=int(msgs[0].tg_msg_id),
        msg_hi=int(msgs[-1].tg_msg_id),
        msg_count=len(msgs),
        speakers=speakers,
        text=text,
        ctx_prefix=build_ctx_prefix(msgs, chat, reply_parent=reply_parent),
        content_hash=content_hash(msgs),
        simhash=to_signed64(simhash64(text)),
    )


def build_from_messages(msgs: Sequence[Message], chat: Chat) -> list[Episode]:
    """Sof quvur: xabarlar → epizodlar (segmentatsiya + prefiks + hash)."""
    out: list[Episode] = []
    by_id = {int(m.tg_msg_id): m for m in msgs}
    for chunk in segment(msgs, gap_seconds=_gap_limit(chat)):
        ids = {int(m.tg_msg_id) for m in chunk}
        head_parent = chunk[0].reply_to_msg_id
        parent = (
            by_id.get(int(head_parent))
            if head_parent is not None and int(head_parent) not in ids
            else None
        )
        out.append(make_episode(chunk, chat, reply_parent=parent))
    return out


# ─── DB qatlami ──────────────────────────────────────────────────────────────

LEVEL_EPISODE = 1


async def _rebuild_from(session: AsyncSession, chat_id: int, since_msg_id: int | None) -> int:
    """Qayta qurish qaysi `tg_msg_id` dan boshlanadi.

    Oxirgi **barqaror** epizod chegarasi — ya'ni `since_msg_id` ni o'z ichiga
    olgan (yoki undan oldingi) epizodning boshi. Undan keyingi hamma narsa
    qayta hisoblanadi; oldingisiga tegilmaydi.
    """
    stmt = select(RetrievalNode.msg_lo).where(
        RetrievalNode.chat_id == chat_id, RetrievalNode.level == LEVEL_EPISODE
    )
    if since_msg_id is not None:
        stmt = stmt.where(RetrievalNode.msg_lo <= since_msg_id)
    row = (await session.execute(stmt.order_by(RetrievalNode.msg_lo.desc()).limit(1))).first()
    if row is not None:
        return int(row[0])
    return 0 if since_msg_id is None else int(since_msg_id)


async def build_episodes(
    session: AsyncSession,
    chat_id: int,
    *,
    since_msg_id: int | None = None,
    full: bool = False,
    limit: int = MAX_MESSAGES_PER_BUILD,
) -> EpisodeReport:
    """Chat epizodlarini quradi/yangilaydi — idempotent.

    Idempotentlik `(chat_id, level=1, starts_at)` kaliti ustida: mavjud
    tugunning `content_hash` i o'zgarmagan bo'lsa unga **tegilmaydi**
    (embedding ham qayta hisoblanmaydi). O'zgargan bo'lsa qator yangilanadi
    va vektori o'chiriladi — `nodes.embed_nodes` uni qayta oladi.

    Args:
        session: DB sessiyasi (bitta sessiya parallel ishlatilmaydi).
        chat_id: `chats.id`.
        since_msg_id: shu `tg_msg_id` dan keyingi o'zgarishlar; qayta qurish
            uni o'z ichiga olgan epizod **boshidan** boshlanadi.
        full: butun chatni qayta qurish.
        limit: bir chaqiruvda o'qiladigan xabarlar qopqog'i.

    Returns:
        `EpisodeReport` — nechta yaratildi/yangilandi/o'chirildi.
    """
    chat = await session.get(Chat, chat_id)
    if chat is None:
        return EpisodeReport(chat_id=chat_id)

    from_msg_id = 0 if full else await _rebuild_from(session, chat_id, since_msg_id)

    rows = (
        await session.execute(
            select(Message)
            .where(
                Message.chat_id == chat_id,
                Message.tg_msg_id >= from_msg_id,
                Message.deleted_at.is_(None),
            )
            .order_by(Message.tg_msg_id)
            .limit(limit)
        )
    ).scalars()
    msgs = list(rows)
    report = EpisodeReport(chat_id=chat_id, messages=len(msgs))
    episodes = build_from_messages(msgs, chat)
    report.episodes = episodes

    existing = list(
        (
            await session.execute(
                select(RetrievalNode).where(
                    RetrievalNode.chat_id == chat_id,
                    RetrievalNode.level == LEVEL_EPISODE,
                    RetrievalNode.msg_lo >= from_msg_id,
                )
            )
        ).scalars()
    )
    by_start = {n.starts_at: n for n in existing}

    keep: set[datetime] = set()
    for ep in episodes:
        keep.add(ep.starts_at)
        node = by_start.get(ep.starts_at)
        if node is None:
            session.add(_to_node(ep, account_id=chat.account_id, chat_id=chat_id))
            report.created += 1
            continue
        if node.content_hash == ep.content_hash:
            report.unchanged += 1
            continue
        _apply(node, ep)
        await session.execute(
            delete(RetrievalNodeEmbedding).where(RetrievalNodeEmbedding.node_id == node.id)
        )
        report.updated += 1

    stale = [n.id for n in existing if n.starts_at not in keep]
    if stale:
        await session.execute(delete(RetrievalNode).where(RetrievalNode.id.in_(stale)))
        report.deleted = len(stale)

    await session.flush()
    if report.updated:
        # L6: qayta qurilgan epizodning vektori o'chirildi — tugun endi
        # qidiruvning vektor kanalidan **tushib qoladi**. Cron 15 daqiqada bir
        # keladi, lekin uni darhol turtib qo'yamiz (job id bitta — dublikat
        # yig'ilmaydi). Navbat yo'q bo'lsa (test, redis o'chiq) — jim o'tamiz.
        with contextlib.suppress(Exception):
            from app.worker.queue import enqueue

            await enqueue("embed_nodes_task", job_id="embed_nodes:after_episodes", defer_by=30)
    await link_entities(session, chat_id, from_msg_id=from_msg_id)
    log.info(
        "episodes.built",
        chat_id=chat_id,
        messages=len(msgs),
        created=report.created,
        updated=report.updated,
        unchanged=report.unchanged,
        deleted=report.deleted,
    )
    return report


#: `node_entities.node_id` ni epizod oynasi bo'yicha to'ldiruvchi **bitta**
#: `UPDATE`. Entity xabar bilan birga yoziladi (epizoddan oldin), shuning uchun
#: bog'lanish shu yerda — epizodlar qurilgach — qo'yiladi. `msg_lo/msg_hi`
#: `tg_msg_id` fazosida, `node_entities.message_id` esa `messages.id` — shu
#: sabab `messages` bo'ylab join bor.
_LINK_ENTITIES_SQL = """
UPDATE node_entities e
   SET node_id = n.id
  FROM messages m, retrieval_nodes n
 WHERE e.message_id = m.id
   AND m.chat_id = :chat_id
   AND m.tg_msg_id >= :from_msg_id
   AND n.chat_id = :chat_id
   AND n.level = :level
   AND m.tg_msg_id BETWEEN n.msg_lo AND n.msg_hi
   AND e.node_id IS DISTINCT FROM n.id
"""


async def link_entities(session: AsyncSession, chat_id: int, *, from_msg_id: int = 0) -> int:
    """`node_entities.node_id` ← epizod oynasi (phase4/A) — bitta `UPDATE`.

    Xatosi epizod qurishni **to'xtatmaydi**: bog'lanish qulaylik kanali
    (tugun darajasidagi entity statistikasi), epizodning o'zi emas.
    """
    try:
        res = await session.execute(
            text(_LINK_ENTITIES_SQL),
            {"chat_id": chat_id, "from_msg_id": int(from_msg_id), "level": LEVEL_EPISODE},
        )
    except Exception as exc:  # jadval hali yo'q bo'lsa (migratsiya 0012 gacha) ham tinch
        log.warning("episodes.link_entities_failed", chat_id=chat_id, error=str(exc)[:200])
        return 0
    return int(getattr(res, "rowcount", 0) or 0)


def _to_node(ep: Episode, *, account_id: int, chat_id: int) -> RetrievalNode:
    node = RetrievalNode(account_id=account_id, chat_id=chat_id, level=LEVEL_EPISODE)
    _apply(node, ep)
    return node


def _apply(node: RetrievalNode, ep: Episode) -> None:
    node.starts_at = ep.starts_at
    node.ends_at = ep.ends_at
    node.msg_lo = ep.msg_lo
    node.msg_hi = ep.msg_hi
    node.msg_count = ep.msg_count
    node.speakers = ep.speakers
    node.text = ep.text
    node.ctx_prefix = ep.ctx_prefix
    node.content_hash = ep.content_hash
    node.simhash = ep.simhash


__all__ = [
    "CHANNEL_GAP_SECONDS",
    "GAP_SECONDS",
    "LEVEL_EPISODE",
    "MAX_EPISODE_CHARS",
    "MAX_EPISODE_MSGS",
    "Episode",
    "EpisodeReport",
    "allowed_cuts",
    "build_ctx_prefix",
    "build_episodes",
    "build_from_messages",
    "content_hash",
    "episode_text",
    "link_entities",
    "make_episode",
    "reply_components",
    "segment",
    "speaker_name",
    "to_signed64",
]
