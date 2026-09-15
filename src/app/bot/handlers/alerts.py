"""`/alerts` — yoqilgan kuzatuv qoidalari ro'yxati (3-bosqich/C).

Javob **`bot.push.send_html`** orqali yuboriladi, `message.answer` bilan emas:
push yo'li bitta bo'lsin (rate limit, parse_mode, xato ishlovi bir joyda) va
alert xabarlari bilan bir xil ko'rinsin. Push moduli hali ulanmagan bo'lsa
(deploy oralig'i) buyruq jimgina o'tadi — bot yiqilmaydi.

Bu yerda **hech qanday yozish yo'q**: faqat `alert_rules` o'qiladi.
Qoidalarni qo'shish/tahrirlash Mini App'da (`/api/alerts`, HTTPS + CSRF).
"""

from __future__ import annotations

import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message as TgMessage
from sqlalchemy import select

from app.db.base import session_scope
from app.db.models import Account, AlertRule, Chat, User
from app.i18n import Translator
from app.logging import get_logger

log = get_logger(__name__)
router = Router(name="alerts")

MAX_LISTED = 20


def render_rules(rows: list[tuple[AlertRule, str | None]], _: Translator) -> str:
    """HTML ro'yxat. Chat sarlavhasi begona matn — `html.escape` majburiy."""
    lines = [_("alerts.bot.header", n=len(rows))]
    for i, (rule, title) in enumerate(rows[:MAX_LISTED], 1):
        where = html.escape(title, quote=False) if title else _("alerts.bot.all_chats")
        label = html.escape((rule.label or rule.pattern or "")[:64], quote=False)
        lines.append(f"{i}. <b>{label}</b> · {where} · {html.escape(str(rule.kind))}")
    return "\n".join(lines)


@router.message(Command("alerts"))
async def cmd_alerts(message: TgMessage, user: User, _: Translator) -> None:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(AlertRule, Chat.title)
                .join(Account, Account.id == AlertRule.account_id)
                .outerjoin(Chat, Chat.id == AlertRule.chat_id)
                .where(Account.user_id == user.id, AlertRule.enabled.is_(True))
                .order_by(AlertRule.id)
            )
        ).all()
    items = [(r[0], r[1]) for r in rows]
    text = render_rules(items, _) if items else _("alerts.bot.none")

    try:
        from app.bot.push import send_html
    except ImportError:  # push moduli hali ulanmagan — buyruq botni yiqitmasin
        log.warning("bot.alerts.push_missing")
        return
    await send_html(user.tg_user_id, text)
