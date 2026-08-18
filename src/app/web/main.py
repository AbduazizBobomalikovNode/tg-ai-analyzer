"""Web UI + API (FastAPI).

Sahifalar:  /login (telefon → kod → 2FA), /chat (AI chat), /dashboard
API:        /api/auth/*, /api/me, /api/accounts/*, /api/conversations/*
Sog'liq:    /health

MUHIM: login kodi va 2FA paroli faqat shu HTTPS endpoint'lar orqali qabul
qilinadi. Chat orqali hech qachon — Telegram chat'da yuborilgan kodni bekor
qiladi (rejaning 4.1-bandi).

Auth oqimlari jarayon xotirasida (`AuthFlowStore` — har oqim o'z Telethon
klientini ushlaydi) → `api` servisi **bitta uvicorn worker** bilan ishlaydi.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.logging import get_logger, setup_logging
from app.services.auth_flow import AuthFlowStore
from app.web.routers import actions as actions_router
from app.web.routers import auth as auth_router
from app.web.routers import chat as chat_router
from app.web.routers import pages as pages_router
from app.web.routers import stats as stats_router
from app.web.routers import tg as tg_router
from app.web.security import RateLimiter

s = get_settings()
setup_logging(s.log_level, json_output=s.is_prod)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.auth_flows = AuthFlowStore()
    app.state.auth_limiter = RateLimiter(s.web_auth_rate_per_ip, 600)
    log.info("web.start", env=s.env, secure_cookies=s.web_secure_cookies)
    try:
        yield
    finally:
        from app.llm import close_all
        from app.mtproto.pool import pool

        await app.state.auth_flows.close_all()
        await pool.close_all()
        await close_all()


def create_app() -> FastAPI:
    app = FastAPI(
        title="tg-ai-analyzer",
        version="0.1.0",
        docs_url=None if s.is_prod else "/docs",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(auth_router.router)
    app.include_router(tg_router.router)
    app.include_router(chat_router.router)
    app.include_router(stats_router.router)
    app.include_router(actions_router.router)
    app.include_router(pages_router.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": s.env}

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("web.unhandled", path=request.url.path)
        return JSONResponse(status_code=500, content={"detail": {"code": "error.generic"}})

    return app


app = create_app()
