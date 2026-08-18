"""Limitlar (8-bosqich): foydalanuvchi bo'yicha chat tezligi va kunlik LLM byudjeti.

* `chat_rate` — daqiqasiga so'rovlar (jarayon xotirasi, api bitta worker).
* `daily_usage` — bugungi tokenlar/xarajat (`conversation_messages` assistant qatorlari).
* `assert_within_budget` — `LLM_DAILY_TOKEN_BUDGET` / `LLM_DAILY_COST_BUDGET_USD`
  (0 = cheksiz). Oshsa `ChatError("budget")` → UI 429.
Byudjet **javobdan oldin** tekshiriladi (so'nggi javob biroz oshirishi mumkin — qabul).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Conversation, ConversationMessage
from app.web.security import RateLimiter

chat_rate = RateLimiter(get_settings().chat_rate_per_minute, 60)


@dataclass(slots=True)
class Usage:
    tokens: int
    cost_usd: float
    requests: int
    token_limit: int
    cost_limit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 4),
            "requests": self.requests,
            "token_limit": self.token_limit or None,
            "cost_limit": self.cost_limit or None,
            "tokens_pct": round(100 * self.tokens / self.token_limit, 1)
            if self.token_limit
            else None,
            "cost_pct": round(100 * self.cost_usd / self.cost_limit, 1)
            if self.cost_limit
            else None,
        }

    @property
    def exceeded(self) -> str | None:
        if self.token_limit and self.tokens >= self.token_limit:
            return "tokens"
        if self.cost_limit and self.cost_usd >= self.cost_limit:
            return "cost"
        return None


async def daily_usage(session: AsyncSession, user_id: int) -> Usage:
    s = get_settings()
    since = datetime.now(UTC) - timedelta(days=1)
    row = (
        await session.execute(
            select(
                func.coalesce(
                    func.sum(ConversationMessage.tokens_in + ConversationMessage.tokens_out), 0
                ),
                func.coalesce(func.sum(ConversationMessage.cost_usd), 0.0),
                func.count(ConversationMessage.id),
            )
            .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
            .where(
                Conversation.user_id == user_id,
                ConversationMessage.role == "assistant",
                ConversationMessage.created_at >= since,
            )
        )
    ).one()
    return Usage(
        tokens=int(row[0] or 0),
        cost_usd=float(row[1] or 0),
        requests=int(row[2] or 0),
        token_limit=s.llm_daily_token_budget,
        cost_limit=s.llm_daily_cost_budget_usd,
    )


async def assert_within_budget(session: AsyncSession, user_id: int) -> Usage:
    from app.services.chat_service import ChatError

    u = await daily_usage(session, user_id)
    if u.exceeded:
        raise ChatError("budget", u.exceeded)
    return u


def check_chat_rate(user_id: int) -> bool:
    return chat_rate.check(str(user_id))
