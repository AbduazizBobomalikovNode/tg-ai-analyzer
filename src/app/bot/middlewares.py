"""Bot middleware'lari: allowlist tekshiruvi + user yozib qo'yish + i18n."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, InlineQuery, Message, TelegramObject
from aiogram.types import User as TgUser
from sqlalchemy import select

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import User
from app.i18n import Translator, normalize
from app.logging import get_logger

log = get_logger(__name__)


def _tg_user(event: TelegramObject) -> TgUser | None:
    if isinstance(event, Message | CallbackQuery | InlineQuery):
        return event.from_user
    return getattr(event, "from_user", None)


class AccessMiddleware(BaseMiddleware):
    """ALLOWED_USER_IDS bo'sh bo'lmasa — faqat o'sha user'lar."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        allowed = get_settings().allowed_users
        tg_user = _tg_user(event)
        if allowed and tg_user and tg_user.id not in allowed:
            log.warning("bot.access_denied", tg_user_id=tg_user.id)
            if isinstance(event, Message):
                await event.answer(Translator(tg_user.language_code)("error.not_allowed"))
            elif isinstance(event, CallbackQuery):
                await event.answer(
                    Translator(tg_user.language_code)("error.not_allowed"), show_alert=True
                )
            return None
        return await handler(event, data)


class UserMiddleware(BaseMiddleware):
    """DB'dagi User'ni topadi/yaratadi, `user` va `_` (translator) ni inject qiladi."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        tg_user = _tg_user(event)
        if tg_user is None:
            return await handler(event, data)

        async with session_scope() as session:
            db_user = (
                await session.execute(select(User).where(User.tg_user_id == tg_user.id))
            ).scalar_one_or_none()

            if db_user is None:
                db_user = User(
                    tg_user_id=tg_user.id,
                    username=tg_user.username,
                    locale=normalize(tg_user.language_code or get_settings().default_locale),
                )
                session.add(db_user)
                await session.flush()
                log.info("user.created", tg_user_id=tg_user.id, locale=db_user.locale)
            elif db_user.username != tg_user.username:
                db_user.username = tg_user.username

            data["user"] = db_user
            data["_"] = Translator(db_user.locale)

        return await handler(event, data)
