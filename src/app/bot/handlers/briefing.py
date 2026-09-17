"""`/briefing` — bugungi nashrni egasining o'ziga qayta yuboradi (3-bosqich/B).

Handler hech narsa hisoblamaydi: briefingni worker quradi
(`worker/tasks.build_briefings`). Bu yerda faqat o'qish va `bot/push.send_html`
orqali yuborish — yagona push yo'li, jim soatlar bu yerda **qo'llanmaydi**,
chunki bu foydalanuvchining o'z so'rovi (jim oyna avtomatik push uchun).
"""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import select

from app.bot import push
from app.db.base import session_scope
from app.db.models import Account, Briefing, User
from app.i18n import Translator
from app.logging import get_logger
from app.services import briefing as BR

log = get_logger(__name__)
router = Router(name="briefing")


@router.message(Command("briefing"))
async def cmd_briefing(message: Message, user: User, _: Translator) -> None:
    async with session_scope() as session:
        accounts = list(
            (
                await session.execute(
                    select(Account).where(Account.user_id == user.id).order_by(Account.id)
                )
            )
            .scalars()
            .all()
        )
        if not accounts:
            await message.answer(_("brief.no_accounts"))
            return

        for account in accounts:
            day = BR.edition_day(BR.local_now(account.tz))
            row = (
                await session.execute(
                    select(Briefing).where(Briefing.account_id == account.id, Briefing.day == day)
                )
            ).scalar_one_or_none()
            if row is None:
                await message.answer(_("brief.not_built", day=day.isoformat()))
                continue
            html = BR.render_html(
                row.structured or {},
                title=_("brief.title", day=day.isoformat()),
                link=push.mini_app_link(f"/briefings/{day.isoformat()}"),
                link_label=_("brief.open_full"),
            )
            await push.send_html(int(user.tg_user_id), html, bot=message.bot)
