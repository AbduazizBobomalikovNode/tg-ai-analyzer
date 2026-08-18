"""Dashboard agregatlari — so'rovlar, tokenlar, xarajat, sifat, ingestion.

Hammasi `conversation_messages` (role=assistant) + `chats`/`messages`/
`message_metric_snapshots` ustidan SQL agregat. Bitta user doirasida
(`user_id`) — ko'p foydalanuvchili joylashuvda ham har kim o'zinikini ko'radi.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Integer, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Account,
    AgentAction,
    AgentRun,
    Chat,
    Conversation,
    ConversationMessage,
    Message,
    MessageEmbedding,
    MessageMetricSnapshot,
)

CM = ConversationMessage


def _since(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


async def overview(session: AsyncSession, user_id: int, *, days: int = 30) -> dict[str, Any]:
    since = _since(days)
    base_where = (Conversation.user_id == user_id, CM.role == "assistant", CM.created_at >= since)

    totals = (
        await session.execute(
            select(
                func.count(CM.id),
                func.coalesce(func.sum(CM.tokens_in), 0),
                func.coalesce(func.sum(CM.tokens_out), 0),
                func.coalesce(func.sum(CM.cost_usd), 0.0),
                func.avg(CM.latency_ms),
                func.percentile_cont(0.5).within_group(CM.latency_ms),
                func.count(CM.rating),
                func.sum(case((CM.rating == 1, 1), else_=0)),
                func.sum(case((CM.rating == -1, 1), else_=0)),
                func.avg(CM.auto_relevance),
                func.avg(CM.auto_usefulness),
                func.count(CM.auto_relevance),
                func.sum(case((CM.auto_grounded.is_(False), 1), else_=0)),
            )
            .select_from(CM)
            .join(Conversation, Conversation.id == CM.conversation_id)
            .where(*base_where)
        )
    ).one()
    (
        n,
        tin,
        tout,
        cost,
        lat_avg,
        lat_p50,
        n_rated,
        n_up,
        n_down,
        rel_avg,
        use_avg,
        n_auto,
        n_ungrounded,
    ) = totals

    # kunlik seriya
    day = func.date_trunc("day", CM.created_at).label("day")
    daily_rows = (
        await session.execute(
            select(
                day,
                func.count(CM.id),
                func.coalesce(func.sum(CM.tokens_in), 0),
                func.coalesce(func.sum(CM.tokens_out), 0),
                func.coalesce(func.sum(CM.cost_usd), 0.0),
                func.sum(case((CM.rating == 1, 1), else_=0)),
                func.sum(case((CM.rating == -1, 1), else_=0)),
            )
            .select_from(CM)
            .join(Conversation, Conversation.id == CM.conversation_id)
            .where(*base_where)
            .group_by(day)
            .order_by(day)
        )
    ).all()
    by_day = {r[0].date().isoformat(): r for r in daily_rows}
    days_series = []
    start = (datetime.now(UTC) - timedelta(days=days - 1)).date()
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        r = by_day.get(d)
        days_series.append(
            {
                "day": d,
                "requests": int(r[1]) if r else 0,
                "tokens_in": int(r[2]) if r else 0,
                "tokens_out": int(r[3]) if r else 0,
                "cost_usd": round(float(r[4]), 4) if r else 0.0,
                "up": int(r[5] or 0) if r else 0,
                "down": int(r[6] or 0) if r else 0,
            }
        )

    # model / provider bo'yicha
    model_rows = (
        await session.execute(
            select(
                CM.provider,
                CM.model,
                func.count(CM.id),
                func.coalesce(func.sum(CM.tokens_in), 0),
                func.coalesce(func.sum(CM.tokens_out), 0),
                func.coalesce(func.sum(CM.cost_usd), 0.0),
                func.avg(CM.latency_ms),
                func.avg(CM.auto_relevance),
            )
            .select_from(CM)
            .join(Conversation, Conversation.id == CM.conversation_id)
            .where(*base_where)
            .group_by(CM.provider, CM.model)
            .order_by(func.count(CM.id).desc())
        )
    ).all()

    # strategiya / manba bo'yicha (JSONB context)
    strat = CM.context["strategy"].astext.label("strategy")
    src = CM.context["source"].astext.label("source")
    strat_rows = (
        await session.execute(
            select(
                func.coalesce(strat, "none"),
                func.coalesce(src, "none"),
                func.count(CM.id),
                func.avg(cast(CM.context["est_tokens"].astext, Integer)),
                func.avg(CM.auto_relevance),
            )
            .select_from(CM)
            .join(Conversation, Conversation.id == CM.conversation_id)
            .where(*base_where)
            .group_by(strat, src)
            .order_by(func.count(CM.id).desc())
        )
    ).all()

    # top chatlar (kontekst sifatida)
    chat_title = CM.context["title"].astext.label("chat_title")
    top_chats = (
        await session.execute(
            select(chat_title, func.count(CM.id), func.avg(CM.auto_usefulness))
            .select_from(CM)
            .join(Conversation, Conversation.id == CM.conversation_id)
            .where(*base_where, chat_title.isnot(None))
            .group_by(chat_title)
            .order_by(func.count(CM.id).desc())
            .limit(8)
        )
    ).all()

    # ko'rib chiqish uchun: past baholangan javoblar
    low_rows = (
        await session.execute(
            select(
                CM.id,
                CM.conversation_id,
                CM.content,
                CM.rating,
                CM.rating_comment,
                CM.auto_relevance,
                CM.auto_note,
                CM.created_at,
                CM.model,
            )
            .select_from(CM)
            .join(Conversation, Conversation.id == CM.conversation_id)
            .where(*base_where)
            .where((CM.rating == -1) | (CM.auto_relevance <= 2) | (CM.auto_grounded.is_(False)))
            .order_by(CM.created_at.desc())
            .limit(10)
        )
    ).all()

    # yozish amallari (6-bosqich) — status bo'yicha
    act_rows = (
        await session.execute(
            select(AgentAction.status, func.count(AgentAction.id))
            .join(AgentRun, AgentRun.id == AgentAction.run_id)
            .where(
                AgentRun.user_id == user_id,
                AgentAction.created_at >= since,
                AgentAction.tool.in_(
                    ("send_message", "edit_message", "forward_message", "pin_message")
                ),
            )
            .group_by(AgentAction.status)
        )
    ).all()
    actions = {str(r[0]): int(r[1]) for r in act_rows}

    return {
        "days": days,
        "actions": actions,
        "totals": {
            "requests": int(n or 0),
            "tokens_in": int(tin or 0),
            "tokens_out": int(tout or 0),
            "cost_usd": round(float(cost or 0), 4),
            "latency_avg_ms": int(lat_avg or 0),
            "latency_p50_ms": int(lat_p50 or 0),
            "rated": int(n_rated or 0),
            "up": int(n_up or 0),
            "down": int(n_down or 0),
            "satisfaction": round(100 * int(n_up or 0) / int(n_rated), 1) if n_rated else None,
            "auto_relevance": round(float(rel_avg), 2) if rel_avg is not None else None,
            "auto_usefulness": round(float(use_avg), 2) if use_avg is not None else None,
            "auto_evaluated": int(n_auto or 0),
            "ungrounded": int(n_ungrounded or 0),
        },
        "daily": days_series,
        "models": [
            {
                "provider": r[0],
                "model": r[1],
                "requests": int(r[2]),
                "tokens_in": int(r[3]),
                "tokens_out": int(r[4]),
                "cost_usd": round(float(r[5] or 0), 4),
                "latency_avg_ms": int(r[6] or 0),
                "auto_relevance": round(float(r[7]), 2) if r[7] is not None else None,
            }
            for r in model_rows
        ],
        "strategies": [
            {
                "strategy": r[0],
                "source": r[1],
                "requests": int(r[2]),
                "avg_ctx_tokens": int(r[3] or 0),
                "auto_relevance": round(float(r[4]), 2) if r[4] is not None else None,
            }
            for r in strat_rows
        ],
        "top_chats": [
            {
                "title": r[0],
                "requests": int(r[1]),
                "auto_usefulness": round(float(r[2]), 2) if r[2] is not None else None,
            }
            for r in top_chats
        ],
        "review": [
            {
                "id": int(r[0]),
                "conversation_id": int(r[1]),
                "excerpt": (r[2] or "")[:200],
                "rating": r[3],
                "comment": r[4],
                "auto_relevance": r[5],
                "auto_note": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
                "model": r[8],
            }
            for r in low_rows
        ],
    }


async def ingestion_overview(session: AsyncSession, user_id: int) -> dict[str, Any]:
    acc_ids = [
        int(r[0])
        for r in (await session.execute(select(Account.id).where(Account.user_id == user_id))).all()
    ]
    if not acc_ids:
        return {
            "accounts": 0,
            "chats": 0,
            "synced_chats": 0,
            "messages": 0,
            "embedded": 0,
            "snapshots": 0,
            "running": 0,
        }
    chats_row = (
        await session.execute(
            select(
                func.count(Chat.id),
                func.sum(case((Chat.synced_total > 0, 1), else_=0)),
                func.sum(case((Chat.sync_state == "running", 1), else_=0)),
                func.coalesce(func.sum(Chat.synced_total), 0),
            ).where(Chat.account_id.in_(acc_ids))
        )
    ).one()
    embedded = (
        await session.execute(
            select(func.count(MessageEmbedding.message_id))
            .join(Message, Message.id == MessageEmbedding.message_id)
            .join(Chat, Chat.id == Message.chat_id)
            .where(Chat.account_id.in_(acc_ids))
        )
    ).scalar_one()
    snaps = (
        await session.execute(
            select(func.count(MessageMetricSnapshot.id))
            .join(Message, Message.id == MessageMetricSnapshot.message_id)
            .join(Chat, Chat.id == Message.chat_id)
            .where(Chat.account_id.in_(acc_ids), MessageMetricSnapshot.captured_at >= _since(7))
        )
    ).scalar_one()
    return {
        "accounts": len(acc_ids),
        "chats": int(chats_row[0] or 0),
        "synced_chats": int(chats_row[1] or 0),
        "running": int(chats_row[2] or 0),
        "messages": int(chats_row[3] or 0),
        "embedded": int(embedded or 0),
        "snapshots_7d": int(snaps or 0),
    }
