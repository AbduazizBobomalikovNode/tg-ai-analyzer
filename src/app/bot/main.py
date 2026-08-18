from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.redis import RedisStorage

from app.bot.handlers import build_router
from app.bot.middlewares import AccessMiddleware, UserMiddleware
from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)


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


async def run_bot() -> None:
    bot = build_bot()
    dp = build_dispatcher()
    me = await bot.get_me()

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
