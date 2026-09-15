"""HTML sahifalar (Jinja2). Ma'lumot — /api orqali, sahifa faqat qobiq."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.i18n import SUPPORTED, _load
from app.logging import get_logger
from app.web.security import LANG_COOKIE, WebIdentity, identity_of, locale_of

log = get_logger(__name__)

router = APIRouter(include_in_schema=False)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


# ─── static fingerprinting (ui/R1b) ──────────────────────────────────────────
#
# `asset("app.css")` → `/static/app.css?v=<sha1[:10]>`. Versiya fayl
# **mazmunidan** olinadi (mtime emas: konteyner qayta qurilganda mtime
# o'zgaradi, mazmun esa yo'q — foydalanuvchi keshini bekorga kuydirmaymiz).
#
# Shu bilan `web/main.py` dagi static mount `v=` bo'lsa bir yillik
# `immutable` kesh beradi: fayl o'zgarsa URL o'zgaradi, ya'ni eski javob
# hech qachon noto'g'ri bo'lmaydi.

_ASSET_CACHE: dict[str, str] = {}
ASSET_HASH_CHARS = 10


def _fingerprint(rel: str) -> str:
    """Fayl mazmunining sha1 prefiksi; fayl yo'q bo'lsa bo'sh satr."""
    path = (STATIC_DIR / rel).resolve()
    # Yo'l chegarasi: `asset("../secrets")` static papkadan chiqib ketmasin.
    if not path.is_relative_to(STATIC_DIR.resolve()) or not path.is_file():
        return ""
    digest = hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()
    return digest[:ASSET_HASH_CHARS]


def asset(path: str) -> str:
    """Jinja global: fingerprint'li static URL.

    Dev rejimida (`ENV=dev`) har chaqiruvda qayta hisoblanadi — CSS'ni
    tahrirlab sahifani yangilash yetarli bo'lsin. Prod'da bir marta
    (`prime_assets()` startupda, keyin kesh).
    """
    rel = path.lstrip("/")
    if get_settings().env == "dev":
        version = _fingerprint(rel)
    else:
        cached = _ASSET_CACHE.get(rel)
        version = cached if cached is not None else _fingerprint(rel)
        _ASSET_CACHE[rel] = version
    return f"/static/{rel}?v={version}" if version else f"/static/{rel}"


def prime_assets() -> int:
    """Startupda barcha static fayllarni keshlaydi. → keshlangan fayllar soni."""
    _ASSET_CACHE.clear()
    if not STATIC_DIR.is_dir():
        return 0
    for item in STATIC_DIR.rglob("*"):
        if item.is_file():
            rel = item.relative_to(STATIC_DIR).as_posix()
            _ASSET_CACHE[rel] = _fingerprint(rel)
    return len(_ASSET_CACHE)


templates.env.globals["asset"] = asset


def _strings(locale: str) -> dict[str, str]:
    """JS'ga beriladigan tarjimalar (faqat web.*, *.err.*, error.*)."""
    merged = {**_load("uz"), **_load(locale)}
    return {
        k: v
        for k, v in merged.items()
        if k.startswith(
            (
                "web.",
                "auth.err.",
                "pool.err.",
                "chat.err.",
                "action.err.",
                "sync.err.",
                "alert.err.",  # phase3/C
                "briefing.err.",  # phase3/B
                "error.",
            )
        )
    }


async def _locale_for(request: Request, ident: WebIdentity | None) -> str:
    """Til: `prefs.locale` (o'rnatilgan bo'lsa) → aks holda `?lang=`/cookie/header.

    Cookie mantiqi **o'z joyida qoladi** (kirmagan foydalanuvchi va til
    almashtirgich shunga tayanadi); sozlamada aniq til tanlangan bo'lsa u
    ustun turadi — foydalanuvchi buni bir marta aytgan.

    DB'ga borib bo'lmasa (test, baza yo'q) — jimgina cookie qaroriga
    tushamiz: sahifa tili tufayli butun UI yiqilmasin.
    """
    fallback = locale_of(request)
    if ident is None or request.query_params.get("lang") in SUPPORTED:
        return fallback  # aniq `?lang=` — foydalanuvchining shu safargi tanlovi
    try:
        from app.db.base import session_scope
        from app.db.models import User
        from app.services import prefs as P

        async with session_scope() as db:
            user = await db.get(User, ident.user_id)
            chosen = P.normalize(user.prefs).get("locale") if user is not None else ""
    except Exception as exc:
        log.debug("pages.prefs_locale_failed", error=str(exc)[:200])
        return fallback
    return chosen if chosen in SUPPORTED else fallback


def _ctx(request: Request, locale: str, **extra: Any) -> dict[str, Any]:
    strings = _strings(locale)
    return {
        "request": request,
        "lang": locale,
        "langs": SUPPORTED,
        "t": lambda key, **kw: (
            (strings.get(key) or key).format(**kw) if kw else strings.get(key, key)
        ),
        "strings_json": json.dumps(strings, ensure_ascii=False),
        "env": get_settings().env,
        **extra,
    }


def _finish_html(request: Request, resp: Response, locale: str) -> Response:
    """HTML javobi: `no-cache` + ETag, va `?lang=` bo'lsa cookie.

    `no-cache` "keshlamang" degani emas — "har safar serverdan so'rang"
    degani: ETag mos kelsa javob 304, ya'ni tarmoqda faqat sarlavhalar
    ketadi, lekin foydalanuvchi hech qachon eski qobiqni ko'rmaydi
    (static fayllar esa aksincha — `immutable`, chunki URL'da versiya bor).
    """
    resp.headers["Cache-Control"] = "no-cache"
    body = getattr(resp, "body", b"")
    if body:
        digest = hashlib.sha1(body, usedforsecurity=False).hexdigest()[:16]
        etag = f'"{digest}"'
        resp.headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            resp = Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
    if request.query_params.get("lang") in SUPPORTED:
        resp.set_cookie(LANG_COOKIE, locale, max_age=365 * 86400, samesite="lax")
    return resp


def _render(request: Request, name: str, locale: str, **extra: Any) -> Response:
    resp = templates.TemplateResponse(request, name, _ctx(request, locale, **extra))
    return _finish_html(request, resp, locale)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    return RedirectResponse("/chat" if await identity_of(request) else "/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Any:
    ident = await identity_of(request)
    return _render(
        request, "login.html", await _locale_for(request, ident), logged_in=ident is not None
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> Any:
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    s = get_settings()
    return _render(
        request,
        "chat.html",
        await _locale_for(request, ident),
        context_default=s.web_context_default_messages,
        context_max=s.web_context_max_messages,
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> Any:
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    return _render(request, "dashboard.html", await _locale_for(request, ident))


# ── phase3/B: shaxsiy briefing (Mini App chuqur havolasi shu sahifaga tushadi) ──


@router.get("/briefings", response_class=HTMLResponse)
@router.get("/briefings/{day}", response_class=HTMLResponse)
async def briefings_page(request: Request, day: str = "") -> Any:
    ident = await identity_of(request)
    if ident is None:
        return RedirectResponse("/login", status_code=302)
    # `day` faqat sahifa qobig'iga beriladi; ma'lumot /api/briefings dan keladi
    # va u yerda egalik tekshiriladi.
    return _render(request, "briefings.html", await _locale_for(request, ident), day=day[:10])
