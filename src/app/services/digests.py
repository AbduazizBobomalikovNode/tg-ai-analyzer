"""Kunlik digest keshi — katta vaqt oynasini arzon va bir marta siqish.

Muammo: "bu oy nima muhokama qilindi?" 3000 xabar = 100k+ token. Har safar
map-reduce qilish ham sekin, ham qimmat. Yechim: har kun uchun **bitta** digest
(arzon `Task.ROUTE` model, ~150 so'z) `chat_digests` da saqlanadi:

  * savol kelganda oynadagi kunlar uchun kesh o'qiladi, yo'qlari hisoblanib yoziladi;
  * worker har kecha kechagi kunni oldindan hisoblaydi (`build_daily_digests`);
  * kun ichida xabar soni o'zgarsa (yangi sync) — `msg_count` farqi → qayta hisob;
  * < MIN_MSGS xabarli kunlar digest qilinmaydi — xom qatorlar o'zi arzon.

Natija: 30 kunlik savol ≈ 30 x 150 so'z ≈ 6k token, LLM chaqiruvsiz (kesh issiq bo'lsa).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, ChatDigest, Message
from app.llm import LLM, LLMError, Msg, Task
from app.logging import get_logger
from app.services.prompts import MAP_DIGEST_PROMPT

log = get_logger(__name__)

MIN_MSGS_FOR_DIGEST = 4  # undan kam bo'lsa xom qatorlar
MAX_DAY_CHARS = 40_000  # bir kunlik xabarlar shundan katta bo'lsa 2+ bo'lakka
CHUNK_CHARS = 14_000
MAX_DAYS_PER_QUERY = 62  # bitta savolda hisoblanadigan maksimal kun (xarajat qopqog'i)


@dataclass(slots=True)
class DayDigest:
    day: datetime
    text: str
    msg_count: int
    cached: bool
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


def _day_start(dt: datetime) -> datetime:
    dt = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _line(m: Message, chat: Chat) -> str:
    from app.services.search import compact_text

    when = m.published_at.strftime("%H:%M") if m.published_at else "?"
    who = (
        chat.title
        if chat.type == "channel" and not m.sender_id
        else (f"user:{m.sender_id}" if m.sender_id else "—")
    )
    body = compact_text(m.text or "") or (f"[{m.media_type}]" if m.media_type else "")
    views = f" (views:{m.views})" if m.views else ""
    return f"[{when}] #{m.tg_msg_id} {who}{views}: {body}"


async def day_messages(session: AsyncSession, chat_id: int, day: datetime) -> list[Message]:
    rows = await session.execute(
        select(Message)
        .where(
            Message.chat_id == chat_id,
            Message.published_at >= day,
            Message.published_at < day + timedelta(days=1),
        )
        .order_by(Message.tg_msg_id)
    )
    return list(rows.scalars().all())


async def _summarize(
    lines: list[str], *, title: str, day: datetime, llm: LLM
) -> tuple[str, int, int, str]:
    """Bir kunlik qatorlarni (kerak bo'lsa bo'laklab) digest qiladi."""
    chunks: list[list[str]] = [[]]
    size = 0
    for line in lines:
        if size + len(line) > CHUNK_CHARS and chunks[-1]:
            chunks.append([])
            size = 0
        chunks[-1].append(line)
        size += len(line) + 1
    parts: list[str] = []
    t_in = t_out = 0
    model = ""
    for i, chunk in enumerate(chunks, 1):
        body = "\n".join(chunk)
        head = f'<untrusted_data source="telegram" chat="{title}" day="{day.date()}"'
        if len(chunks) > 1:
            head += f' part="{i}/{len(chunks)}"'
        res = await llm.chat(
            Task.ROUTE,
            [Msg.system(MAP_DIGEST_PROMPT), Msg.user(f"{head}>\n{body}\n</untrusted_data>")],
            max_tokens=350,
        )
        parts.append(res.text.strip())
        t_in += res.usage.tokens_in
        t_out += res.usage.tokens_out
        model = res.model
    return "\n".join(parts), t_in, t_out, model


async def digest_for_day(
    session: AsyncSession, chat: Chat, day: datetime, *, llm: LLM | None = None, force: bool = False
) -> DayDigest | None:
    """Bir kun uchun digest (kesh → yo'q bo'lsa hisoblab yozadi). Xabar bo'lmasa None."""
    day = _day_start(day)
    msgs = await day_messages(session, chat.id, day)
    if not msgs:
        return None
    cached = (
        await session.execute(
            select(ChatDigest).where(ChatDigest.chat_id == chat.id, ChatDigest.day == day)
        )
    ).scalar_one_or_none()
    if cached is not None and not force and cached.msg_count == len(msgs):
        return DayDigest(day, cached.digest, cached.msg_count, True)

    lines = [_line(m, chat) for m in msgs]
    if len(msgs) < MIN_MSGS_FOR_DIGEST:
        # arzon: xom qatorlar (keshlanmaydi — o'zi qisqa)
        return DayDigest(day, "\n".join(lines), len(msgs), False)

    try:
        text, t_in, t_out, model = await _summarize(
            lines, title=chat.title, day=day, llm=llm or LLM()
        )
    except LLMError as exc:
        log.warning("digest.failed", chat_id=chat.id, day=str(day.date()), error=str(exc)[:200])
        return DayDigest(day, "\n".join(lines)[:6000], len(msgs), False)

    if cached is None:
        cached = ChatDigest(chat_id=chat.id, day=day)
        session.add(cached)
    cached.digest = text
    cached.msg_count = len(msgs)
    cached.model = model
    cached.tokens_in = t_in
    cached.tokens_out = t_out
    await session.flush()
    log.info(
        "digest.built", chat_id=chat.id, day=str(day.date()), msgs=len(msgs), tokens=t_in + t_out
    )
    return DayDigest(day, text, len(msgs), False, t_in, t_out, model)


async def digests_for_window(
    session: AsyncSession,
    chat: Chat,
    *,
    since: datetime,
    until: datetime | None = None,
    llm: LLM | None = None,
) -> list[DayDigest]:
    """Oynadagi har kun uchun digest (kesh + kerakli hisob). Eng yangi kunlar ustuvor."""
    until = until or datetime.now(UTC)
    days: list[datetime] = []
    d = _day_start(until)
    start = _day_start(since)
    while d >= start and len(days) < MAX_DAYS_PER_QUERY:
        days.append(d)
        d -= timedelta(days=1)
    out: list[DayDigest] = []
    for day in reversed(days):
        dd = await digest_for_day(session, chat, day, llm=llm)
        if dd is not None:
            out.append(dd)
    return out


async def chats_needing_digest(
    session: AsyncSession, day: datetime, *, limit: int = 200
) -> list[int]:
    """Kechagi kun uchun digest yo'q, lekin ≥ MIN_MSGS xabar bor chatlar."""
    day = _day_start(day)
    cnt = func.count(Message.id).label("n")
    have = select(ChatDigest.chat_id).where(ChatDigest.day == day)
    rows = await session.execute(
        select(Message.chat_id, cnt)
        .where(
            Message.published_at >= day,
            Message.published_at < day + timedelta(days=1),
            Message.chat_id.notin_(have),
        )
        .group_by(Message.chat_id)
        .having(cnt >= MIN_MSGS_FOR_DIGEST)
        .order_by(cnt.desc())
        .limit(limit)
    )
    return [int(r[0]) for r in rows.all()]
