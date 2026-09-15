from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError, TelegramUnauthorizedError
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import User
from aiohttp import ClientConnectorError

from app.bot.handlers import build_router
from app.bot.middlewares import AccessMiddleware, UserMiddleware
from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

# `get_me()` auth backoff: 30s → 300s (2x har urinishda). Konteyner
# `restart: unless-stopped` bilan ishga tushadi — noto'g'ri token'da tight
# restart loop yaratish o'rniga shu yerda kutib qayta uriniladi.
_AUTH_RETRY_INITIAL_S = 30.0
_AUTH_RETRY_MAX_S = 300.0


def build_bot() -> Bot:
    s = get_settings()
    # MarkdownV2 emas, HTML — escaping muammosi kamroq (rejaning 11.3-bandi)
    return Bot(token=s.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


def build_dispatcher() -> Dispatcher:
    s = get_settings()
    dp = Dispatcher(storage=RedisStorage.from_url(s.redis_url))

    for observer in (dp.message, dp.callback_query, dp.inline_query):
        observer.middleware(AccessMiddleware())
        observer.middleware(UserMiddleware())

    dp.include_router(build_router())
    return dp


async def _wait_for_bot_ready(bot: Bot) -> User:
    """`get_me()` muvaffaqiyatli bo'lguncha exponential backoff bilan kutadi.

    `TelegramUnauthorizedError` — token yaroqsiz/bekor qilingan; `TelegramNetworkError`
    (aiogram aiohttp xatolarini shunga o'raydi) va xom `ClientConnectorError` —
    tarmoq/DNS muammosi. Ikkalasi ham raw traceback bilan process'ni
    o'ldirmasin: bitta aniq log hodisasi (token log'ga tushmaydi, `app.logging`
    baribir tozalaydi) va qayta urinish. `KeyboardInterrupt`/`asyncio.CancelledError`
    ataylab tutilmaydi — Ctrl+C yoki shutdown signal darhol to'xtatsin.
    """
    delay = _AUTH_RETRY_INITIAL_S
    while True:
        try:
            return await bot.get_me()
        except (TelegramUnauthorizedError, TelegramNetworkError, ClientConnectorError) as exc:
            log.error(
                "bot.auth_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                retry_in=delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _AUTH_RETRY_MAX_S)


async def run_bot() -> None:
    bot = build_bot()
    dp = build_dispatcher()
    me = await _wait_for_bot_ready(bot)

    s = get_settings()
    if me.id != s.control_bot_id:
        log.warning(
            "config.control_bot_id_mismatch",
            expected=s.control_bot_id,
            actual=me.id,
            hint="CONTROL_BOT_ID ni to'g'rilang — peer deny-list shunga tayanadi",
        )

    log.info("bot.start", username=me.username, id=me.id)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)
