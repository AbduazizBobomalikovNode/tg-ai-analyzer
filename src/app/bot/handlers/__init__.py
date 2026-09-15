from aiogram import Router

from app.bot.handlers import alerts, briefing, start


def build_router() -> Router:
    router = Router(name="root")
    router.include_router(start.router)
    router.include_router(briefing.router)  # phase3/B: /briefing
    router.include_router(alerts.router)  # phase3/C: /alerts
    # 1-bosqichda qo'shiladi: auth (QR + Mini App), accounts
    return router


__all__ = ["build_router"]
