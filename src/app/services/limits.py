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
from app.db.models import Account, Conversation, ConversationMessage
from app.web.security import RateLimiter

chat_rate = RateLimiter(get_settings().chat_rate_per_minute, 60)
# Deep search alohida, qattiqroq shift: bitta deep so'rov ≈ 10x oddiy so'rov
# narxi (arxitektura v2, 11-bo'lim), shuning uchun umumiy chat kvotasidan
# alohida hisoblanadi.
deep_rate = RateLimiter(max(get_settings().deep_rate_per_minute, 1), 60)


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


async def blocked_user_ids(session: AsyncSession) -> set[int]:
    """Kunlik byudjeti tugagan foydalanuvchilar (M3) — **bitta** so'rovda.

    Fon ishlari (embedding, hafta tugunlari) chat so'rovi emas, lekin ular ham
    LLM pulini yeydi. Har akkaunt uchun alohida `daily_usage` chaqirish qimmat,
    shuning uchun jamlama bir marta olinadi va egasi bo'yicha guruhlanadi.

    Limit qo'yilmagan bo'lsa (ikkalasi ham 0) — bo'sh to'plam.
    """
    s = get_settings()
    if not s.llm_daily_token_budget and not s.llm_daily_cost_budget_usd:
        return set()
    since = datetime.now(UTC) - timedelta(days=1)
    rows = (
        await session.execute(
            select(
                Conversation.user_id,
                func.coalesce(
                    func.sum(ConversationMessage.tokens_in + ConversationMessage.tokens_out), 0
                ),
                func.coalesce(func.sum(ConversationMessage.cost_usd), 0.0),
            )
            .join(ConversationMessage, ConversationMessage.conversation_id == Conversation.id)
            .where(
                ConversationMessage.role == "assistant",
                ConversationMessage.created_at >= since,
            )
            .group_by(Conversation.user_id)
        )
    ).all()
    blocked: set[int] = set()
    for user_id, tokens, cost in rows:
        over_tokens = bool(s.llm_daily_token_budget) and int(tokens or 0) >= (
            s.llm_daily_token_budget
        )
        over_cost = bool(s.llm_daily_cost_budget_usd) and float(cost or 0) >= (
            s.llm_daily_cost_budget_usd
        )
        if over_tokens or over_cost:
            blocked.add(int(user_id))
    return blocked


async def blocked_account_ids(session: AsyncSession) -> set[int]:
    """Byudjeti tugagan egalarning akkauntlari — fon ishlari shularni chetlab o'tadi."""
    users = await blocked_user_ids(session)
    if not users:
        return set()
    rows = (
        await session.execute(select(Account.id).where(Account.user_id.in_(sorted(users))))
    ).scalars()
    return {int(x) for x in rows}


async def account_budget_blocked(session: AsyncSession, account_id: int) -> str | None:
    """Shu akkaunt egasining byudjet holati — `"tokens"` / `"cost"` / `None`."""
    owner = (
        await session.execute(select(Account.user_id).where(Account.id == account_id))
    ).scalar_one_or_none()
    if owner is None:
        return None
    return await check_daily_budget(session, int(owner))


async def check_daily_budget(session: AsyncSession, user_id: int) -> str | None:
    """Byudjet holati — **xato ko'tarmaydi**: `"tokens"` / `"cost"` / `None`.

    Deep search uchun kerak: u so'rovni HTTP xatosi bilan emas, `error`
    hodisasi bilan to'xtatadi (SSE oqimi allaqachon ochilgan bo'lishi mumkin),
    shuning uchun tekshiruv va uni qanday e'lon qilish bir-biridan ajratilgan.
    """
    return (await daily_usage(session, user_id)).exceeded


async def assert_within_budget(session: AsyncSession, user_id: int) -> Usage:
    from app.services.chat_service import ChatError

    u = await daily_usage(session, user_id)
    if u.exceeded:
        raise ChatError("budget", u.exceeded)
    return u


def check_chat_rate(user_id: int) -> bool:
    return chat_rate.check(str(user_id))


def check_deep_rate(user_id: int) -> bool:
    """Deep so'rov shifti. `DEEP_RATE_PER_MINUTE=0` — rejim o'chiq (hech biri o'tmaydi)."""
    if get_settings().deep_rate_per_minute <= 0:
        return False
    return deep_rate.check(str(user_id))


def refund_deep_rate(user_id: int) -> None:
    """Bajarilmagan deep so'rovining shift slotini qaytaradi.

    Tasdiqsiz `exhaustive` — javob emas, **baho**: reja quriladi va yugurish
    to'xtaydi. Uni kvotaga yozish foydalanuvchini o'zi tanlamagan ish uchun
    jazolardi (uch marta baho ko'rsa, tasdiqlashga kvota qolmasdi).
    """
    if get_settings().deep_rate_per_minute <= 0:
        return
    deep_rate.refund(str(user_id))
