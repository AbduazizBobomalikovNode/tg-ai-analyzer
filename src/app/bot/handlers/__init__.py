from aiogram import Router

from app.bot.handlers import start


def build_router() -> Router:
    router = Router(name="root")
    router.include_router(start.router)
    # 1-bosqichda qo'shiladi: auth (QR + Mini App), accounts
    return router


__all__ = ["build_router"]
