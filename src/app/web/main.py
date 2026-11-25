"""Web UI + API (FastAPI).

Sahifalar:  /login (telefon → kod → 2FA), /chat (AI chat), /dashboard
API:        /api/auth/*, /api/me, /api/accounts/*, /api/conversations/*,
            /api/deep (+ /api/deep/stream — SSE, chuqur qidiruv)
Sog'liq:    /health

MUHIM: login kodi va 2FA paroli faqat shu HTTPS endpoint'lar orqali qabul
qilinadi. Chat orqali hech qachon — Telegram chat'da yuborilgan kodni bekor
qiladi (rejaning 4.1-bandi).

Auth oqimlari jarayon xotirasida (`AuthFlowStore` — har oqim o'z Telethon
klientini ushlaydi) → `api` servisi **bitta uvicorn worker** bilan ishlaydi.
"""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import embed_dims_mismatch, get_settings
from app.logging import get_logger, setup_logging
from app.observability import http_middleware, init_sentry, metrics_payload
from app.services.auth_flow import AuthFlowStore
from app.web.routers import actions as actions_router
from app.web.routers import alerts as alerts_router
from app.web.routers import auth as auth_router
from app.web.routers import briefings as briefings_router
from app.web.routers import chat as chat_router
from app.web.routers import content as content_router
from app.web.routers import deep as deep_router
from app.web.routers import export as export_router
from app.web.routers import pages as pages_router
from app.web.routers import settings as settings_router
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
    init_sentry("api")
    # L10: noto'g'ri `EMBED_DIMS` bilan jarayon baribir ko'tariladi (o'qish
    # yo'llari ishlaydi), lekin embedding yozuvi butunlay yiqiladi — startupda
    # bir marta ogohlantiramiz.
    dims_error = embed_dims_mismatch(s)
    if dims_error:
        log.warning("web.embed_dims_mismatch", error=dims_error)
    # ui/R1b: static fayllar fingerprintini bir marta hisoblaymiz (dev'da
    # `asset()` baribir har chaqiruvda qayta o'qiydi).
    log.info("web.assets_primed", files=pages_router.prime_assets())
    log.info("web.start", env=s.env, secure_cookies=s.web_secure_cookies, version=s.app_version)
    try:
        yield
    finally:
        from app.llm import close_all
        from app.mtproto.pool import pool

        await app.state.auth_flows.close_all()
        await pool.close_all()
        await close_all()


class CachedStatic(StaticFiles):
    """Static fayllar uchun kesh sarlavhalari (ui/R1b).

    `?v=<fingerprint>` bo'lsa javob **bir yil, `immutable`**: fayl mazmuni
    o'zgarsa URL ham o'zgaradi (`pages.asset()`), ya'ni eski javob hech qachon
    noto'g'ri bo'lib qolmaydi va brauzer qayta so'ramaydi ham.

    Versiyasiz so'rov (`/static/app.css`, qo'lda yozilgan havola yoki eski
    sahifa) — `no-cache`: brauzer keshlaydi, lekin har safar ETag bilan
    tekshiradi. "Keshlamang" degan `no-store` emas.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        # `scope` — pozitsion 3-argument (starlette), nomlangan bo'lishi ham mumkin.
        scope = kwargs.get("scope") or (args[2] if len(args) > 2 else {})
        query = scope.get("query_string", b"") if isinstance(scope, dict) else b""
        versioned = b"v=" in query
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if versioned else "no-cache"
        )
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="tg-ai-analyzer",
        version="0.1.0",
        docs_url=None if s.is_prod else "/docs",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", CachedStatic(directory=str(static_dir)), name="static")
    app.middleware("http")(http_middleware())

    app.include_router(auth_router.router)
    app.include_router(tg_router.router)
    app.include_router(chat_router.router)
    app.include_router(stats_router.router)
    app.include_router(actions_router.router)
    app.include_router(content_router.router)
    app.include_router(deep_router.router)
    app.include_router(alerts_router.router)  # phase3/C
    app.include_router(briefings_router.router)  # phase3/B
    app.include_router(settings_router.router)  # ui/R1b
    app.include_router(export_router.router)  # javob eksporti
    app.include_router(pages_router.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": s.env, "version": s.app_version}

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, bool]:
        """Konteyner HEALTHCHECK uchun — auth yo'q, hech qanday ma'lumot bermaydi."""
        return {"ok": True}

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        # Prometheus scrape. METRICS_TOKEN bo'lsa — talab qilinadi (ochiq internetda qoldirmang).
        if s.metrics_token:
            auth = request.headers.get("authorization", "")
            token = request.query_params.get("token") or auth.removeprefix("Bearer ").strip()
            # Doimiy vaqtli taqqoslash — oddiy `!=` tokenni belgima-belgi
            # tiklash uchun vaqt kanalini ochadi.
            if not hmac.compare_digest(token.encode(), s.metrics_token.encode()):
                raise HTTPException(status_code=401)
        body, ctype = metrics_payload()
        return Response(content=body, media_type=ctype)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("web.unhandled", path=request.url.path)
        return JSONResponse(status_code=500, content={"detail": {"code": "error.generic"}})

    return app


app = create_app()
