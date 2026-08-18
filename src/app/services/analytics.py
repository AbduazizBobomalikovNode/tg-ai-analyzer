"""Chat statistikasi — DB'dan, LLM'siz (4-bosqichning yadrosi, tool uchun).

Ikki xil savolni ajratamiz (rejaning 4.6-bandi):
  * "davrda chop etilgan postlar" — `messages.published_at` bo'yicha (views/reactions
    oxirgi ma'lum qiymat);
  * "davrda qo'shilgan ko'rishlar" (o'sish) — faqat `message_metric_snapshots`
    (davr boshidagi va oxiridagi snapshot farqi).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, Message, MessageMetricSnapshot


@dataclass(slots=True)
class TopPost:
    msg_id: int
    date: str
    views: int | None
    forwards: int | None
    reactions: int | None
    replies: int | None
    text: str
    link: str | None


@dataclass(slots=True)
class ChatStats:
    chat: str
    period_days: int
    since: str
    until: str
    posts: int
    posts_per_day: float
    views_sum: int
    views_avg: float | None
    views_median: float | None
    reactions_sum: int
    forwards_sum: int
    replies_sum: int
    views_delta_period: int | None  # snapshot'lardan: davrda qo'shilgan ko'rishlar
    snapshot_coverage: int  # nechta post uchun ≥2 snapshot bor
    by_weekday: dict[str, int] = field(default_factory=dict)
    by_hour: dict[str, int] = field(default_factory=dict)
    top_by_views: list[TopPost] = field(default_factory=list)
    top_by_reactions: list[TopPost] = field(default_factory=list)
    top_by_forwards: list[TopPost] = field(default_factory=list)
    media_share: dict[str, int] = field(default_factory=dict)
    prev_period: dict[str, Any] | None = None  # taqqoslash: oldingi shu uzunlikdagi davr

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def msg_link(chat: Chat, msg_id: int) -> str | None:
    return f"https://t.me/{chat.username}/{msg_id}" if chat.username else None


async def chat_stats(
    session: AsyncSession,
    chat: Chat,
    *,
    days: int = 7,
    until: datetime | None = None,
    top_n: int = 5,
    compare_previous: bool = True,
) -> ChatStats:
    until = until or datetime.now(UTC)
    since = until - timedelta(days=days)

    base = (Message.chat_id == chat.id, Message.published_at >= since, Message.published_at < until)
    agg = (
        await session.execute(
            select(
                func.count(Message.id),
                func.coalesce(func.sum(Message.views), 0),
                func.avg(Message.views),
                func.percentile_cont(0.5).within_group(Message.views),
                func.coalesce(func.sum(Message.reactions_total), 0),
                func.coalesce(func.sum(Message.forwards), 0),
                func.coalesce(func.sum(Message.replies_count), 0),
            ).where(*base)
        )
    ).one()
    posts = int(agg[0] or 0)

    wd = func.extract("dow", Message.published_at)  # 0 = Sunday
    hr = func.extract("hour", Message.published_at)
    wd_rows = (await session.execute(select(wd, func.count()).where(*base).group_by(wd))).all()
    hr_rows = (await session.execute(select(hr, func.count()).where(*base).group_by(hr))).all()
    by_weekday = {_WD[(int(r[0]) - 1) % 7]: int(r[1]) for r in wd_rows}
    by_hour = {f"{int(r[0]):02d}": int(r[1]) for r in hr_rows}

    media_rows = (
        await session.execute(
            select(func.coalesce(Message.media_type, "text"), func.count())
            .where(*base)
            .group_by(Message.media_type)
        )
    ).all()
    media_share = {str(r[0]): int(r[1]) for r in media_rows}

    async def top(col: Any) -> list[TopPost]:
        rows = await session.execute(
            select(Message).where(*base, col.isnot(None)).order_by(col.desc()).limit(top_n)
        )
        return [
            TopPost(
                msg_id=m.tg_msg_id,
                date=m.published_at.strftime("%Y-%m-%d") if m.published_at else "",
                views=m.views,
                forwards=m.forwards,
                reactions=m.reactions_total,
                replies=m.replies_count,
                text=(m.text or "").replace("\n", " ")[:140],
                link=msg_link(chat, m.tg_msg_id),
            )
            for m in rows.scalars().all()
        ]

    # ko'rishlar o'sishi — davr ichidagi snapshot'lar (barcha postlar, sanadan qat'i nazar)
    first_last = (
        select(
            MessageMetricSnapshot.message_id.label("mid"),
            func.min(MessageMetricSnapshot.captured_at).label("t0"),
            func.max(MessageMetricSnapshot.captured_at).label("t1"),
            func.count().label("n"),
        )
        .join(Message, Message.id == MessageMetricSnapshot.message_id)
        .where(
            Message.chat_id == chat.id,
            MessageMetricSnapshot.captured_at >= since,
            MessageMetricSnapshot.captured_at < until,
        )
        .group_by(MessageMetricSnapshot.message_id)
        .having(func.count() >= 2)
        .subquery()
    )
    s0 = MessageMetricSnapshot.__table__.alias("s0")
    s1 = MessageMetricSnapshot.__table__.alias("s1")
    delta_row = (
        await session.execute(
            select(
                func.coalesce(func.sum(s1.c.views - s0.c.views), 0),
                func.count(),
            )
            .select_from(first_last)
            .join(s0, (s0.c.message_id == first_last.c.mid) & (s0.c.captured_at == first_last.c.t0))
            .join(s1, (s1.c.message_id == first_last.c.mid) & (s1.c.captured_at == first_last.c.t1))
        )
    ).one()
    coverage = int(delta_row[1] or 0)
    views_delta = int(delta_row[0] or 0) if coverage else None

    prev: dict[str, Any] | None = None
    if compare_previous:
        p_since, p_until = since - timedelta(days=days), since
        prow = (
            await session.execute(
                select(
                    func.count(Message.id),
                    func.coalesce(func.sum(Message.views), 0),
                    func.coalesce(func.sum(Message.reactions_total), 0),
                    func.coalesce(func.sum(Message.forwards), 0),
                ).where(
                    Message.chat_id == chat.id,
                    Message.published_at >= p_since,
                    Message.published_at < p_until,
                )
            )
        ).one()
        prev = {
            "since": p_since.date().isoformat(),
            "until": p_until.date().isoformat(),
            "posts": int(prow[0] or 0),
            "views_sum": int(prow[1] or 0),
            "reactions_sum": int(prow[2] or 0),
            "forwards_sum": int(prow[3] or 0),
        }

    return ChatStats(
        chat=chat.title,
        period_days=days,
        since=since.date().isoformat(),
        until=until.date().isoformat(),
        posts=posts,
        posts_per_day=round(posts / days, 2) if days else float(posts),
        views_sum=int(agg[1] or 0),
        views_avg=round(float(agg[2]), 1) if agg[2] is not None else None,
        views_median=round(float(agg[3]), 1) if agg[3] is not None else None,
        reactions_sum=int(agg[4] or 0),
        forwards_sum=int(agg[5] or 0),
        replies_sum=int(agg[6] or 0),
        views_delta_period=views_delta,
        snapshot_coverage=coverage,
        by_weekday=by_weekday,
        by_hour=by_hour,
        top_by_views=await top(Message.views),
        top_by_reactions=await top(Message.reactions_total),
        top_by_forwards=await top(Message.forwards),
        media_share=media_share,
        prev_period=prev,
    )
